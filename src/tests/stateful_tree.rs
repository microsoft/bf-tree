// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{collections::BTreeMap, path::Path};

use proptest::{
    prelude::*,
    test_runner::{Config as PropertyConfig, TestCaseError, TestCaseResult, TestRunner},
};
use proptest_derive::Arbitrary;

use crate::{
    utils::{BfsVisitor, NodeInfo},
    BfTree, Config, LeafInsertResult, LeafReadResult, ScanIterError, ScanReturnField,
    StorageBackend,
};

pub(super) type Model = BTreeMap<Vec<u8>, Vec<u8>>;

#[derive(Clone, Debug, Arbitrary)]
enum Operation {
    Put { id: u8, size: u8, tag: u64 },
    Delete(u8),
    Read(u8),
    ScanCount { from: u8, count: u8 },
    ScanRange { from: u8, to: u8 },
}

fn key(id: u8) -> Vec<u8> {
    let id = id % 64;
    let length = [1, 8, 16, 24, 32][usize::from(id) % 5];
    let mut key = vec![b'k'; length];
    key[length - 1] = id;
    key
}

fn config(path: &Path, backend: StorageBackend, promoted: bool) -> Config {
    let mut config = Config::new(path, 128 * 1024);
    config
        .storage_backend(backend)
        .cache_only(false)
        .cb_min_record_size(2)
        .cb_max_record_size(1024)
        .cb_max_key_len(32)
        .read_promotion_rate(if promoted { 100 } else { 0 })
        .scan_promotion_rate(if promoted { 100 } else { 0 })
        .read_record_cache(!promoted)
        .write_load_full_page(promoted);
    config
}

pub(super) fn check_read(tree: &BfTree, model: &Model, key: &[u8]) -> TestCaseResult {
    let mut buffer = vec![0; model.get(key).map_or(1024, Vec::len)];
    let actual = tree.read(key, &mut buffer);
    if let Some(expected) = model.get(key) {
        prop_assert_eq!(actual, LeafReadResult::Found(expected.len() as u32));
        prop_assert_eq!(buffer.as_slice(), expected.as_slice());
    } else {
        prop_assert!(matches!(
            actual,
            LeafReadResult::NotFound | LeafReadResult::Deleted
        ));
    }
    Ok(())
}

fn check_scan(
    tree: &BfTree,
    model: &Model,
    start: &[u8],
    end: Option<&[u8]>,
    count: usize,
) -> TestCaseResult {
    let mut scan = if let Some(end) = end {
        tree.scan_with_end_key(start, end, ScanReturnField::KeyAndValue)
    } else {
        tree.scan_with_count(start, count, ScanReturnField::KeyAndValue)
    }
    .map_err(|error| TestCaseError::fail(format!("valid scan rejected: {error:?}")))?;
    let mut buffer = [0; 1024];
    for (key, value) in model
        .range(start.to_vec()..)
        .take_while(|(key, _)| end.is_none_or(|end| key.as_slice() <= end))
        .take(count)
    {
        prop_assert_eq!(scan.next(&mut buffer), Some((key.len(), value.len())));
        prop_assert_eq!(&buffer[..key.len()], key);
        prop_assert_eq!(&buffer[key.len()..key.len() + value.len()], value);
    }
    prop_assert_eq!(scan.next(&mut buffer), None, "unexpected trailing record");
    Ok(())
}

pub(super) fn check_state(tree: &BfTree, model: &Model) -> TestCaseResult {
    for key in model.keys() {
        check_read(tree, model, key)?;
    }
    check_scan(tree, model, &[0], None, usize::MAX)
}

pub(super) fn drain(tree: &BfTree) {
    tree.storage.circular_buffer.drain(|handle| {
        if crate::tree::eviction_callback(&handle, tree).is_ok() {
            Ok(handle)
        } else {
            Err(handle)
        }
    });
}

fn run_trace(operations: &[Operation], config: Config) -> TestCaseResult {
    let tree = BfTree::with_config(config, None).unwrap();
    let mut model = Model::new();
    check_state(&tree, &model)?;
    for (index, operation) in operations.iter().enumerate() {
        match *operation {
            Operation::Put { id, size, tag } => {
                let key = key(id);
                let sizes = [2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024];
                let record_size = sizes[usize::from(size) % sizes.len()].max(key.len());
                let value: Vec<_> = tag
                    .to_le_bytes()
                    .into_iter()
                    .cycle()
                    .take(record_size - key.len())
                    .collect();
                let result = tree.insert(&key, &value);
                if value.is_empty() {
                    prop_assert!(matches!(result, LeafInsertResult::InvalidKV(_)));
                } else {
                    prop_assert_eq!(result, LeafInsertResult::Success);
                    model.insert(key.clone(), value);
                }
                check_read(&tree, &model, &key)?;
            }
            Operation::Delete(id) => {
                let key = key(id);
                tree.delete(&key);
                model.remove(&key);
                check_read(&tree, &model, &key)?;
            }
            Operation::Read(id) => check_read(&tree, &model, &key(id))?,
            Operation::ScanCount { from, count } => {
                check_scan(&tree, &model, &key(from), None, usize::from(count) + 1)?;
            }
            Operation::ScanRange { from, to } => {
                let (a, b) = (key(from), key(to));
                let (start, end) = if a <= b { (a, b) } else { (b, a) };
                check_scan(&tree, &model, &start, Some(&end), usize::MAX)?;
            }
        }
        if index % 32 == 31 {
            check_state(&tree, &model)?;
        }
    }
    check_state(&tree, &model)?;
    drain(&tree);
    check_state(&tree, &model)
}

fn compare_configurations(operations: &[Operation], backend: StorageBackend) -> TestCaseResult {
    let directory = tempfile::tempdir().unwrap();
    for promoted in [false, true] {
        run_trace(
            operations,
            config(
                &directory.path().join(format!("tree-{promoted}")),
                backend.clone(),
                promoted,
            ),
        )
        .map_err(|error| TestCaseError::fail(format!("promoted={promoted}: {error}")))?;
    }
    Ok(())
}

#[rstest::rstest]
#[case::memory(StorageBackend::Memory)]
#[case::std_io(StorageBackend::Std)]
fn stateful_operations_match_ordered_model(#[case] backend: StorageBackend) {
    let mut runner = TestRunner::new(PropertyConfig {
        cases: 32,
        max_shrink_iters: 4096,
        source_file: Some(file!()),
        test_name: Some("stateful_operations_match_ordered_model"),
        ..PropertyConfig::default()
    });
    runner
        .run(
            &proptest::collection::vec(any::<Operation>(), 1..193),
            |operations| compare_configurations(&operations, backend.clone()),
        )
        .unwrap();
}

#[rstest::rstest]
#[case::memory(StorageBackend::Memory)]
#[case::std_io(StorageBackend::Std)]
fn stateful_boundary_rewrites_delete_and_reinsert(#[case] backend: StorageBackend) {
    let mut operations = Vec::new();
    for size in 0..12 {
        for id in 0..64 {
            operations.push(Operation::Put {
                id,
                size,
                tag: u64::from(size) * 64 + u64::from(id),
            });
        }
        for id in (0..64).step_by(2) {
            operations.extend([Operation::Delete(id), Operation::Delete(id)]);
        }
        operations.extend([
            Operation::ScanCount {
                from: 0,
                count: 255,
            },
            Operation::ScanRange { from: 1, to: 1 },
        ]);
    }
    compare_configurations(&operations, backend).unwrap();
}

#[test]
fn stateful_root_growth_preserves_churn() {
    let directory = tempfile::tempdir().unwrap();
    let tree = BfTree::with_config(
        config(
            &directory.path().join("live"),
            StorageBackend::Memory,
            false,
        ),
        None,
    )
    .unwrap();
    let mut model = Model::new();
    for id in 0_u64..6000 {
        let key = id.to_be_bytes().to_vec();
        let value = id.to_le_bytes().repeat(64);
        assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
        model.insert(key, value);
    }
    assert!(BfsVisitor::new_all_nodes(&tree)
        .any(|node| matches!(node, NodeInfo::Leaf { level, .. } if level >= 2)));
    check_state(&tree, &model).unwrap();
    for id in 0_u64..6000 {
        let key = id.to_be_bytes().to_vec();
        if id % 3 == 0 {
            tree.delete(&key);
            model.remove(&key);
        } else {
            let value = (id + 1)
                .to_le_bytes()
                .repeat(if id % 2 == 0 { 4 } else { 120 });
            assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
            model.insert(key, value);
        }
    }
    drain(&tree);
    check_state(&tree, &model).unwrap();
    for id in (0_u64..6000).step_by(3) {
        let key = id.to_be_bytes().to_vec();
        check_read(&tree, &model, &key).unwrap();
        let value = (id + 2).to_le_bytes().repeat(32);
        assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
        model.insert(key, value);
    }
    check_state(&tree, &model).unwrap();
}

#[test]
fn stateful_invalid_operations_preserve_contents() {
    let directory = tempfile::tempdir().unwrap();
    let tree =
        BfTree::with_config(config(directory.path(), StorageBackend::Memory, true), None).unwrap();
    let model = Model::from([(b"present".to_vec(), b"value".to_vec())]);
    assert_eq!(tree.insert(b"present", b"value"), LeafInsertResult::Success);
    // Empty values already have a dedicated regression in tests::tree.
    for (key, value) in [
        (vec![], vec![1; 16]),
        (vec![1; 33], vec![1; 16]),
        (b"present".to_vec(), vec![1; 1024]),
    ] {
        assert!(matches!(
            tree.insert(&key, &value),
            LeafInsertResult::InvalidKV(_)
        ));
        check_state(&tree, &model).unwrap();
    }
    for key in [vec![], vec![1; 33]] {
        assert_eq!(tree.read(&key, &mut [0; 1024]), LeafReadResult::InvalidKey);
        tree.delete(&key);
        check_state(&tree, &model).unwrap();
    }
    assert!(matches!(
        tree.scan_with_count(b"present", 0, ScanReturnField::KeyAndValue),
        Err(ScanIterError::InvalidCount)
    ));
    assert!(matches!(
        tree.scan_with_end_key(b"z", b"a", ScanReturnField::KeyAndValue),
        Err(ScanIterError::InvalidKeyRange)
    ));
    check_state(&tree, &model).unwrap();
}
