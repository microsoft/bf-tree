// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{cell::RefCell, time::Duration};

use crate::{
    error::TreeError,
    nodes::PageID,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, thread, Arc, Barrier,
    },
    utils::inner_lock::ReadGuard,
    BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField,
};

thread_local! {
    static CHILD_READ_HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

pub(crate) fn before_inner_child_read() {
    if let Some(hook) = CHILD_READ_HOOK.with(|hook| hook.borrow_mut().take()) {
        hook();
    }
}

#[rstest::rstest]
fn traversal_rejects_unvalidated_inner_child(#[values(false, true)] aggressive: bool) {
    let mut config = Config::default();
    config
        .cb_size_byte(8 * 1024 * 1024)
        .cb_max_record_size(1024);
    let tree = BfTree::with_config(config, None).unwrap();
    for id in 0..20_000u64 {
        assert_eq!(
            tree.insert(&id.to_be_bytes(), &[1; 512]),
            LeafInsertResult::Success
        );
        let (root, is_leaf) = tree.get_root_page();
        if !is_leaf
            && !ReadGuard::try_read(root.as_inner_node())
                .unwrap()
                .as_ref()
                .meta
                .children_is_leaf()
        {
            break;
        }
    }
    let root = tree.get_root_page().0;
    let child = {
        let reader = ReadGuard::try_read(root.as_inner_node()).unwrap();
        assert!(!reader.as_ref().meta.children_is_leaf());
        reader.as_ref().get_value(reader.as_ref().get_kv_meta(0))
    };
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let result = thread::scope(|scope| {
        let tree = &tree;
        let worker = scope.spawn(move || {
            CHILD_READ_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    entered_tx.send(()).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                }));
            });
            matches!(
                tree.traverse_to_leaf(&0u64.to_be_bytes(), aggressive, false, None),
                Err(TreeError::Locked)
            )
        });
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let mut writer = ReadGuard::try_read(root.as_inner_node())
            .unwrap()
            .upgrade()
            .unwrap();
        // Model transient bytes observed during an optimistic inner-node read.
        writer.as_mut().update_at_pos(0, PageID::from_id(0));
        release_tx.send(()).unwrap();
        let result = worker.join();
        writer.as_mut().update_at_pos(0, child);
        result
    });
    assert!(result.unwrap());
    let mut out = [0; 512];
    assert_eq!(
        tree.read(&0u64.to_be_bytes(), &mut out),
        LeafReadResult::Found(512)
    );
    assert_eq!(out, [1; 512]);
}

const WRITERS: usize = 4;
const KEYS_PER_WRITER: usize = 64;
const STEPS: usize = 768;

fn value(key: usize, sequence: usize) -> Vec<u8> {
    let mut bytes = vec![(key ^ sequence) as u8; [32, 128, 384][sequence % 3]];
    bytes[..8].copy_from_slice(&(key as u64).to_be_bytes());
    bytes[8..16].copy_from_slice(&(sequence as u64).to_be_bytes());
    bytes
}

fn operation(writer: usize, step: usize) -> (usize, Option<Vec<u8>>) {
    let key = writer * KEYS_PER_WRITER + (step / 4) % KEYS_PER_WRITER;
    (key, (step % 4 != 2).then(|| value(key, step + 1)))
}

fn read(tree: &BfTree, key: usize) -> Option<Vec<u8>> {
    let mut out = vec![0; 384];
    match tree.read(&(key as u64).to_be_bytes(), &mut out) {
        LeafReadResult::Found(len) => {
            out.truncate(len as usize);
            let sequence = u64::from_be_bytes(out[8..16].try_into().unwrap()) as usize;
            assert_eq!(out, value(key, sequence), "torn or wrong-key read");
            Some(out)
        }
        LeafReadResult::Deleted | LeafReadResult::NotFound => None,
        other => panic!("unexpected read result: {other:?}"),
    }
}

fn scan(tree: &BfTree) -> Vec<usize> {
    let mut iter = tree
        .scan_with_count(
            &0u64.to_be_bytes(),
            WRITERS * KEYS_PER_WRITER + 1,
            ScanReturnField::KeyAndValue,
        )
        .unwrap();
    let mut out = vec![0; 392];
    let mut keys = Vec::new();
    while let Some((key_len, value_len)) = iter.next(&mut out) {
        assert_eq!(key_len, 8);
        let key = u64::from_be_bytes(out[..8].try_into().unwrap()) as usize;
        assert!(key < WRITERS * KEYS_PER_WRITER);
        assert!(keys.last().is_none_or(|previous| *previous < key));
        let sequence = u64::from_be_bytes(out[16..24].try_into().unwrap()) as usize;
        assert_eq!(&out[8..8 + value_len], value(key, sequence));
        keys.push(key);
    }
    keys
}

fn assert_snapshot_prefix(tree: &BfTree) {
    let mut expected_keys = Vec::new();
    for writer in 0..WRITERS {
        let first = writer * KEYS_PER_WRITER;
        let observed: Vec<_> = (first..first + KEYS_PER_WRITER)
            .map(|key| read(tree, key))
            .collect();
        let mut model: Vec<_> = (first..first + KEYS_PER_WRITER)
            .map(|key| Some(value(key, 0)))
            .collect();
        let mut matches_prefix = model == observed;
        for step in 0..STEPS {
            let (key, next) = operation(writer, step);
            model[key - first] = next;
            matches_prefix |= model == observed;
        }
        assert!(
            matches_prefix,
            "snapshot is not a prefix of writer {writer}"
        );
        expected_keys.extend(
            observed
                .iter()
                .enumerate()
                .filter_map(|(offset, record)| record.as_ref().map(|_| first + offset)),
        );
    }
    assert_eq!(scan(tree), expected_keys);
}

#[rstest::rstest]
fn migrated_concurrent_operations_and_snapshot_prefixes(#[values(false, true)] full_pages: bool) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::default();
    config
        .cb_size_byte(2 * 1024 * 1024)
        .use_snapshot(true)
        .read_promotion_rate(100)
        .scan_promotion_rate(100)
        .read_record_cache(!full_pages)
        .write_load_full_page(full_pages)
        .cb_min_record_size(32)
        .cb_max_record_size(512)
        .cb_max_key_len(8);
    let tree = Arc::new(BfTree::with_config(config, None).unwrap());
    for key in 0..WRITERS * KEYS_PER_WRITER {
        assert_eq!(
            tree.insert(&(key as u64).to_be_bytes(), &value(key, 0)),
            LeafInsertResult::Success
        );
    }
    let start = Barrier::new(WRITERS + 3);
    let done = AtomicBool::new(false);
    let snapshots: Vec<_> = (0..3)
        .map(|index| dir.path().join(format!("prefix-{index}.snapshot")))
        .collect();
    thread::scope(|scope| {
        let writers: Vec<_> = (0..WRITERS)
            .map(|writer| {
                let (tree, start) = (&tree, &start);
                scope.spawn(move || {
                    start.wait();
                    for step in 0..STEPS {
                        let (key, next) = operation(writer, step);
                        let encoded = (key as u64).to_be_bytes();
                        if let Some(bytes) = &next {
                            assert_eq!(tree.insert(&encoded, bytes), LeafInsertResult::Success);
                        } else {
                            tree.delete(&encoded);
                        }
                        assert_eq!(read(tree, key), next);
                    }
                })
            })
            .collect();
        scope.spawn(|| {
            start.wait();
            let mut key = 0;
            while !done.load(Ordering::Acquire) {
                read(&tree, key % (WRITERS * KEYS_PER_WRITER));
                key += 1;
            }
        });
        scope.spawn(|| {
            start.wait();
            while !done.load(Ordering::Acquire) {
                scan(&tree);
            }
        });
        start.wait();
        // Always stop readers, including when a writer or snapshot assertion fails.
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _stop = Stop(&done);
        for path in &snapshots {
            tree.cpr_snapshot(path);
        }
        for writer in writers {
            writer.join().unwrap();
        }
    });
    assert_snapshot_prefix(&tree);
    for writer in 0..WRITERS {
        let first = writer * KEYS_PER_WRITER;
        let mut model: Vec<_> = (first..first + KEYS_PER_WRITER)
            .map(|key| Some(value(key, 0)))
            .collect();
        for step in 0..STEPS {
            let (key, next) = operation(writer, step);
            model[key - first] = next;
        }
        for (offset, expected) in model.into_iter().enumerate() {
            assert_eq!(read(&tree, first + offset), expected);
        }
    }
    for path in &snapshots {
        let recovered = BfTree::new_from_cpr_snapshot(path, false, None, None, None).unwrap();
        assert_snapshot_prefix(&recovered);
    }
}
