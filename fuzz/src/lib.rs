// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::{BTreeMap, BTreeSet};

use bf_tree::{
    circular_buffer::{CircularBuffer, CircularBufferError},
    BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField, StorageBackend,
};

type Model = BTreeMap<Vec<u8>, Vec<u8>>;

fn key(id: u16) -> Vec<u8> {
    let length = [2, 8, 32][usize::from(id) % 3];
    let mut key = vec![b'k'; length];
    key[length - 2..].copy_from_slice(&id.to_be_bytes());
    key
}

fn read(tree: &BfTree, model: &Model, key: &[u8]) {
    let mut bytes = [0; 1024];
    let result = tree.read(key, &mut bytes);
    match model.get(key) {
        Some(value) => {
            assert_eq!(result, LeafReadResult::Found(value.len() as u32));
            assert_eq!(&bytes[..value.len()], value);
        }
        None => assert!(matches!(
            result,
            LeafReadResult::NotFound | LeafReadResult::Deleted
        )),
    }
}

fn scan(tree: &BfTree, model: &Model, start: &[u8], end: Option<&[u8]>, count: usize) {
    let mut actual = match end {
        Some(end) => tree.scan_with_end_key(start, end, ScanReturnField::KeyAndValue),
        None => tree.scan_with_count(start, count, ScanReturnField::KeyAndValue),
    }
    .unwrap();
    let mut bytes = [0; 1024];
    for (key, value) in model
        .range(start.to_vec()..)
        .take_while(|(key, _)| end.is_none_or(|end| key.as_slice() <= end))
        .take(count)
    {
        assert_eq!(actual.next(&mut bytes), Some((key.len(), value.len())));
        assert_eq!(&bytes[..key.len()], key);
        assert_eq!(&bytes[key.len()..key.len() + value.len()], value);
    }
    assert_eq!(actual.next(&mut bytes), None);
}

/// The first byte selects promotion policy; subsequent eight-byte chunks encode operations.
pub fn tree(data: &[u8]) {
    let Some((&policy, operations)) = data.split_first() else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let mut config = Config::new(directory.path().join("tree"), 128 * 1024);
    config
        .storage_backend(StorageBackend::Std)
        .cache_only(false)
        .use_snapshot(true)
        .cb_min_record_size(2)
        .cb_max_record_size(1024)
        .cb_max_key_len(32)
        .read_promotion_rate(if policy & 1 == 0 { 0 } else { 100 })
        .scan_promotion_rate(if policy & 1 == 0 { 0 } else { 100 });
    let tree = BfTree::with_config(config, None).unwrap();
    let mut model = Model::new();
    let mut touched = BTreeSet::new();
    for operation in operations.as_chunks::<8>().0.iter().take(1024) {
        let key = key(u16::from_le_bytes([operation[1], operation[2]]));
        touched.insert(key.clone());
        match operation[0] % 5 {
            0 => {
                let size = [0, 1, 7, 31, 63, 127, 255, 511, 992][usize::from(operation[3]) % 9];
                let value: Vec<_> = (0..size).map(|i| operation[4 + i % 4] ^ i as u8).collect();
                let result = tree.insert(&key, &value);
                if value.is_empty() {
                    assert!(matches!(result, LeafInsertResult::InvalidKV(_)));
                } else {
                    assert_eq!(result, LeafInsertResult::Success);
                    model.insert(key.clone(), value);
                }
                read(&tree, &model, &key);
            }
            1 => read(&tree, &model, &key),
            2 => {
                tree.delete(&key);
                model.remove(&key);
                read(&tree, &model, &key);
            }
            3 => scan(&tree, &model, &key, None, usize::from(operation[3]) + 1),
            _ => {
                let end = self::key(u16::from_le_bytes([operation[4], operation[5]]));
                let (start, end) = if key <= end {
                    (&key, &end)
                } else {
                    (&end, &key)
                };
                scan(&tree, &model, start, Some(end), usize::MAX);
            }
        }
    }
    for key in &touched {
        read(&tree, &model, key);
    }
    scan(&tree, &model, &[0], None, usize::MAX);
    let snapshot = directory.path().join("snapshot");
    tree.cpr_snapshot(&snapshot);
    drop(tree);
    let restored = BfTree::new_from_cpr_snapshot(&snapshot, false, None, None, None).unwrap();
    for key in &touched {
        read(&restored, &model, key);
    }
    scan(&restored, &model, &[0], None, usize::MAX);
    drop(restored);
    directory.close().unwrap();
}

fn check_allocation(ptr: *mut u8, size: usize, tag: u8) {
    // Each caller retains ownership of the initialized allocation for its modeled size.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, size) };
    assert!(bytes.iter().all(|byte| *byte == tag));
}

/// Three-byte operations cover allocation boundaries, explicit free, and eviction.
pub fn circular_buffer(data: &[u8]) {
    let buffer = CircularBuffer::new(8192, 0.1, 4, 1952, 4096, 32, None, false);
    // Classes for this configuration, also covered by the library's size-class tests.
    let classes = [64usize, 128, 256, 448, 832, 1600, 2048, 4096];
    let mut live = BTreeMap::new();
    for operation in data.as_chunks::<3>().0.iter().take(2048) {
        match operation[0] % 3 {
            0 => {
                let class = classes[usize::from(operation[1]) % classes.len()];
                let size = match operation[2] % 3 {
                    0 => (class - 1).max(classes[0]),
                    1 => class,
                    _ => (class + 1).min(4096),
                };
                match buffer.alloc(size) {
                    Ok(guard) => {
                        let ptr = guard.as_ptr();
                        assert!(live.insert(ptr, (size, operation[2])).is_none());
                        unsafe { std::slice::from_raw_parts_mut(ptr, size) }.fill(operation[2]);
                    }
                    Err(CircularBufferError::Full) => {}
                    Err(error) => panic!("unexpected single-thread allocation error: {error}"),
                }
            }
            1 => {
                if let Some((&ptr, &(size, tag))) = live
                    .iter()
                    .nth(usize::from(operation[1]) % live.len().max(1))
                {
                    let guard = unsafe { buffer.acquire_exclusive_dealloc_handle(ptr) }.unwrap();
                    check_allocation(ptr, size, tag);
                    live.remove(&ptr);
                    buffer.dealloc(guard);
                }
            }
            _ => {
                buffer.evict_one(&mut |guard| {
                    let (size, tag) = live.remove(&guard.as_ptr()).expect("unowned eviction");
                    check_allocation(guard.as_ptr(), size, tag);
                    Ok(guard)
                });
            }
        }
        for (&ptr, &(size, tag)) in &live {
            check_allocation(ptr, size, tag);
        }
    }
    buffer
        .evict_n(usize::MAX, |guard| {
            let (size, tag) = live
                .remove(&guard.as_ptr())
                .expect("duplicate or unowned eviction");
            check_allocation(guard.as_ptr(), size, tag);
            Ok(guard)
        })
        .unwrap();
    assert!(live.is_empty());
    let metrics = buffer.get_metrics();
    assert_eq!(metrics.head_tail_distance, 0);
    assert_eq!(metrics.allocated_cnt, 0);
}
