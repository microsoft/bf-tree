// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use bf_tree::{
    BfTree, BufferTooSmall, Config, LeafDeleteResult, LeafInsertResult, LeafReadResult,
    ScanReturnField, SnapshotError, StorageBackend,
};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

fn memory_tree() -> BfTree {
    let mut config = Config::new(":memory:", 2 * 1024 * 1024);
    config
        .storage_backend(StorageBackend::Memory)
        .cb_min_record_size(4);
    BfTree::with_config(config, None).unwrap()
}

#[test]
fn checked_read_reports_capacity_and_preserves_legacy_results() {
    let tree = memory_tree();
    assert_eq!(tree.insert(b"key", b"value"), LeafInsertResult::Success);

    assert_eq!(
        tree.read_checked(b"key", &mut []),
        Err(BufferTooSmall { required: 5 })
    );
    assert_eq!(
        tree.read_checked(b"key", &mut [0; 4]),
        Err(BufferTooSmall { required: 5 })
    );
    let mut exact = [0; 5];
    assert_eq!(
        tree.read_checked(b"key", &mut exact),
        Ok(LeafReadResult::Found(5))
    );
    assert_eq!(&exact, b"value");

    let mut larger = [0xA5; 8];
    assert_eq!(
        tree.read_checked(b"key", &mut larger),
        Ok(LeafReadResult::Found(5))
    );
    assert_eq!(&larger[..5], b"value");
    assert_eq!(&larger[5..], &[0xA5; 3]);
    assert_eq!(tree.read(b"key", &mut exact), LeafReadResult::Found(5));
    assert!(catch_unwind(AssertUnwindSafe(|| tree.read(b"key", &mut []))).is_err());
    assert_eq!(
        tree.read_checked(b"key", &mut exact),
        Ok(LeafReadResult::Found(5))
    );

    assert_eq!(
        tree.read_checked(b"missing", &mut []),
        Ok(LeafReadResult::NotFound)
    );
    assert_eq!(
        tree.read_checked(b"", &mut []),
        Ok(LeafReadResult::InvalidKey)
    );
    assert_eq!(
        tree.read_checked(&[b'x'; 1024], &mut []),
        Ok(LeafReadResult::InvalidKey)
    );
    assert!(matches!(
        tree.insert(b"empty", b""),
        LeafInsertResult::InvalidKV(_)
    ));
    assert_eq!(
        tree.read_checked(b"empty", &mut []),
        Ok(LeafReadResult::NotFound)
    );
}

#[test]
fn checked_delete_reports_validation_without_claiming_existence() {
    let tree = memory_tree();
    assert_eq!(tree.insert(b"key", b"value"), LeafInsertResult::Success);
    assert_eq!(tree.delete_checked(b""), LeafDeleteResult::InvalidKey);
    assert_eq!(
        tree.delete_checked(&[b'x'; 1024]),
        LeafDeleteResult::InvalidKey
    );
    assert_eq!(
        tree.read_checked(b"key", &mut [0; 5]),
        Ok(LeafReadResult::Found(5))
    );

    assert_eq!(tree.delete_checked(b"key"), LeafDeleteResult::Success);
    assert_eq!(
        tree.read_checked(b"key", &mut []),
        Ok(LeafReadResult::Deleted)
    );
    assert_eq!(tree.delete_checked(b"key"), LeafDeleteResult::Success);
    assert_eq!(tree.delete_checked(b"absent"), LeafDeleteResult::Success);
    assert!(!matches!(
        tree.read_checked(b"absent", &mut []),
        Ok(LeafReadResult::Found(_))
    ));
    tree.delete(b"");
    tree.delete(&[b'x'; 1024]);
}

#[test]
fn checked_scan_retries_each_field_without_consuming_the_record() {
    let tree = memory_tree();
    for key in [b"key-a", b"key-b", b"key-c"] {
        assert_eq!(tree.insert(key, b"value"), LeafInsertResult::Success);
    }
    assert_eq!(tree.delete_checked(b"key-b"), LeafDeleteResult::Success);

    for field in [
        ScanReturnField::Key,
        ScanReturnField::Value,
        ScanReturnField::KeyAndValue,
    ] {
        let expected = match field {
            ScanReturnField::Key => b"key-c".to_vec(),
            ScanReturnField::Value => b"value".to_vec(),
            ScanReturnField::KeyAndValue => b"key-cvalue".to_vec(),
        };
        let mut scan = tree.scan_with_count(b"key-b", 1, field).unwrap();
        assert_eq!(
            scan.next_checked(&mut []),
            Err(BufferTooSmall {
                required: expected.len(),
            })
        );
        let mut short = vec![0xA5; expected.len() - 1];
        assert_eq!(
            scan.next_checked(&mut short),
            Err(BufferTooSmall {
                required: expected.len(),
            })
        );
        assert!(short.iter().all(|byte| *byte == 0xA5));

        let mut output = vec![0xA5; expected.len() + 3];
        let (key_length, value_length) = scan.next_checked(&mut output).unwrap().unwrap();
        assert_eq!(key_length + value_length, expected.len());
        assert_eq!(&output[..expected.len()], expected.as_slice());
        assert_eq!(&output[expected.len()..], &[0xA5; 3]);
        assert_eq!(scan.next_checked(&mut []), Ok(None));
        assert_eq!(scan.next_checked(&mut []), Ok(None));
    }
}

#[test]
fn checked_scan_preserves_inclusive_bounds_and_legacy_next() {
    let tree = memory_tree();
    for key in [b"key-a", b"key-b", b"key-c", b"key-d"] {
        assert_eq!(tree.insert(key, b"value"), LeafInsertResult::Success);
    }
    let mut scan = tree
        .scan_with_end_key(b"key-b", b"key-c", ScanReturnField::Key)
        .unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| scan.next(&mut []))).is_err());
    let mut output = [0; 5];
    assert_eq!(scan.next_checked(&mut output), Ok(Some((5, 0))));
    assert_eq!(&output, b"key-b");
    assert_eq!(scan.next(&mut output), Some((5, 0)));
    assert_eq!(&output, b"key-c");
    assert_eq!(scan.next_checked(&mut []), Ok(None));
    assert_eq!(scan.next_checked(&mut []), Ok(None));
    drop(scan);

    let mut equal = tree
        .scan_with_end_key(b"key-b", b"key-b", ScanReturnField::Key)
        .unwrap();
    assert_eq!(equal.next_checked(&mut output), Ok(Some((5, 0))));
    assert_eq!(&output, b"key-b");
    assert_eq!(equal.next_checked(&mut []), Ok(None));
}

fn check_model(tree: &BfTree, model: &BTreeMap<Vec<u8>, Vec<u8>>) {
    for (key, value) in model {
        if !value.is_empty() {
            assert_eq!(
                tree.read_checked(key, &mut []),
                Err(BufferTooSmall {
                    required: value.len(),
                })
            );
        }
        let mut output = vec![0; value.len()];
        assert_eq!(
            tree.read_checked(key, &mut output),
            Ok(LeafReadResult::Found(value.len() as u32))
        );
        assert_eq!(&output, value);
    }

    let mut scan = tree
        .scan_with_count(
            &0_u64.to_be_bytes(),
            usize::MAX,
            ScanReturnField::KeyAndValue,
        )
        .unwrap();
    for (key, value) in model {
        let required = key.len() + value.len();
        assert_eq!(scan.next_checked(&mut []), Err(BufferTooSmall { required }));
        let mut output = vec![0; required];
        assert_eq!(
            scan.next_checked(&mut output),
            Ok(Some((key.len(), value.len())))
        );
        assert_eq!(&output[..key.len()], key);
        assert_eq!(&output[key.len()..], value);
    }
    assert_eq!(scan.next_checked(&mut []), Ok(None));
    assert_eq!(scan.next_checked(&mut []), Ok(None));
}

#[test]
fn checked_operations_match_a_model_across_splits_and_recovery() {
    for backend in [StorageBackend::Memory, StorageBackend::Std] {
        let directory = tempfile::tempdir().unwrap();
        let storage = directory.path().join("working.db");
        let snapshot = directory.path().join("checkpoint.cpr");
        let recovery = directory.path().join("recovery.cpr");
        let mut config = Config::new(&storage, 1024 * 1024);
        config
            .storage_backend(backend)
            .cb_min_record_size(4)
            .use_snapshot(true);
        let tree = BfTree::with_config(config, None).unwrap();
        let mut model = BTreeMap::new();

        for index in 0_u64..4096 {
            let key = index.to_be_bytes().to_vec();
            let value = vec![(index % 251) as u8; (index % 511 + 1) as usize];
            assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
            model.insert(key, value);
        }
        for index in (0_u64..4096).step_by(3) {
            let key = index.to_be_bytes().to_vec();
            assert_eq!(tree.delete_checked(&key), LeafDeleteResult::Success);
            model.remove(&key);
        }
        for index in (1_u64..4096).step_by(5) {
            let key = index.to_be_bytes().to_vec();
            let value = vec![0x5A; 700];
            assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
            model.insert(key, value);
        }
        check_model(&tree, &model);
        assert_eq!(tree.cpr_snapshot_checked(&snapshot), Ok(()));
        std::fs::copy(&snapshot, &recovery).unwrap();
        assert_eq!(
            tree.insert(&1_u64.to_be_bytes(), b"newer"),
            LeafInsertResult::Success
        );
        let restored = BfTree::new_from_cpr_snapshot(&recovery, false, None, None, None).unwrap();
        check_model(&restored, &model);
        drop(restored);
        drop(tree);
    }
}

#[test]
fn checked_reads_tolerate_concurrent_size_changes() {
    let tree = Arc::new(memory_tree());
    assert_eq!(tree.insert(b"key", b"short"), LeafInsertResult::Success);
    std::thread::scope(|scope| {
        let writer = Arc::clone(&tree);
        scope.spawn(move || {
            for _ in 0..1000 {
                assert_eq!(
                    writer.insert(b"key", &[0x5A; 64]),
                    LeafInsertResult::Success
                );
                assert_eq!(writer.insert(b"key", b"short"), LeafInsertResult::Success);
            }
        });
        for _ in 0..3 {
            let reader = Arc::clone(&tree);
            scope.spawn(move || {
                for _ in 0..1000 {
                    let mut output = [0; 8];
                    match reader.read_checked(b"key", &mut output) {
                        Ok(LeafReadResult::Found(5)) => assert_eq!(&output[..5], b"short"),
                        Err(BufferTooSmall { required: 64 }) => {}
                        result => panic!("unexpected checked read: {result:?}"),
                    }
                }
            });
        }
    });
}

#[test]
fn checked_snapshot_rejects_disabled_capture_without_an_artifact() {
    let tree = memory_tree();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("disabled.cpr");
    assert_eq!(
        tree.cpr_snapshot_checked(&path),
        Err(SnapshotError::Disabled)
    );
    assert!(!path.exists());
    assert!(catch_unwind(AssertUnwindSafe(|| tree.cpr_snapshot(&path))).is_err());
    assert!(!path.exists());
}
