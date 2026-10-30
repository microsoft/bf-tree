// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(all(feature = "metrics-rt", not(feature = "shuttle")))]
#[test]
fn cache_only_root_insert_is_not_repeated() {
    use crate::{
        metric::{get_tls_recorder, Counter},
        Config, LeafInsertResult,
    };
    let mut config = Config::default();
    config.cache_only(true);
    let tree = crate::BfTree::with_config(config, None).unwrap();
    get_tls_recorder().reset();
    assert_eq!(tree.insert(b"key", &[1; 64]), LeafInsertResult::Success);
    assert_eq!(
        get_tls_recorder().counters.counters[Counter::LeafInsertNew as usize],
        1
    );
    assert_eq!(
        get_tls_recorder().counters.counters[Counter::LeafInsertDuplicate as usize],
        0
    );
    assert_eq!(tree.insert(b"key", &[2; 64]), LeafInsertResult::Success);
    assert_eq!(
        get_tls_recorder().counters.counters[Counter::LeafInsertDuplicate as usize],
        1
    );
}

use std::hint::black_box;
use std::{collections::HashMap, ops::Range};

use crate::nodes::leaf_node::LeafReadResult;
use crate::{BfTree, Config as BfTreeConfig};
use proptest::{
    prelude::*,
    test_runner::{Config, TestRunner},
};
use proptest_derive::Arbitrary;

#[cfg(not(feature = "shuttle"))]
#[rstest::rstest]
fn empty_values_are_rejected_without_changing_records(#[values(false, true)] cache_only: bool) {
    let mut config = BfTreeConfig::default();
    config.cache_only(cache_only).cb_min_record_size(4);
    let tree = BfTree::with_config(config, None).unwrap();
    assert_eq!(
        tree.insert(b"present", b"value"),
        crate::LeafInsertResult::Success
    );
    for key in [b"present".as_slice(), b"absent"] {
        assert!(matches!(
            tree.insert(key, b""),
            crate::LeafInsertResult::InvalidKV(_)
        ));
    }
    let mut out = [0; 32];
    assert_eq!(tree.read(b"present", &mut out), LeafReadResult::Found(5));
    assert_eq!(&out[..5], b"value");
    assert_eq!(tree.read(b"absent", &mut out), LeafReadResult::NotFound);
    let mut scan = tree
        .scan_with_count(b"a", usize::MAX, crate::ScanReturnField::KeyAndValue)
        .unwrap();
    assert_eq!(scan.next(&mut out), Some((7, 5)));
    assert_eq!(&out[..12], b"presentvalue");
    assert_eq!(scan.next(&mut out), None);
}

#[derive(Clone, Arbitrary, Debug)]
enum TreeOp {
    Insert,
    Delete,
    Read,
}

fn tree_insert_read_delete_scan(
    input: Vec<(Vec<u8>, Vec<u8>, TreeOp)>,
    bf_tree_config: BfTreeConfig,
) {
    let mut model = HashMap::<Vec<u8>, Vec<u8>>::new();
    let tree = BfTree::with_config(bf_tree_config, None).unwrap();

    let mut out_buffer = vec![0u8; tree.config.cb_max_record_size]; // Buffer for reading from tree

    let mut current_idx = 0;

    for (k, v, op) in input.iter() {
        match op {
            TreeOp::Insert => {
                tree.insert(k, v);
                model.insert(k.to_owned(), v.to_owned());
            }
            TreeOp::Delete => {
                let _ = tree.delete(k);
                model.remove(k);
            }
            TreeOp::Read => {
                let rt = tree.read(k, &mut out_buffer);
                match model.get(k) {
                    Some(model_v) => match rt {
                        LeafReadResult::Found(v) => {
                            assert_eq!(model_v.len(), v as usize);
                            assert_eq!(&out_buffer[0..model_v.len()], model_v);
                        }
                        _ => {
                            if !tree.cache_only {
                                panic!("Missing key");
                            }
                        }
                    },
                    None => match rt {
                        LeafReadResult::Found(_) => {
                            panic!("Phantom key");
                        }
                        _ => {}
                    },
                }
            }
        }
        current_idx += 1;
    }
    black_box(current_idx);
}

fn leaf_insert_read_inner(len_range: Range<usize>, bf_tree_config: &BfTreeConfig) {
    let config = Config {
        cases: 1,
        max_local_rejects: 1,
        max_global_rejects: 1,
        source_file: Some("src/tests/tree.rs"),
        verbose: 0,
        ..Config::default()
    };

    let strategy = if bf_tree_config.cb_min_record_size <= bf_tree_config.cb_max_record_size - 16 {
        proptest::collection::vec(
            (
                proptest::collection::vec(any::<u8>(), 1..16), // Key
                proptest::collection::vec(
                    any::<u8>(),
                    bf_tree_config.cb_min_record_size..(bf_tree_config.cb_max_record_size - 16),
                ), // Value
                any::<TreeOp>(),
            ),
            len_range, // Length of the list
        )
    } else {
        proptest::collection::vec(
            (
                proptest::collection::vec(any::<u8>(), 16), // Key
                proptest::collection::vec(any::<u8>(), bf_tree_config.cb_max_record_size - 16), // Value
                any::<TreeOp>(),
            ),
            len_range, // Length of the list
        )
    };

    let test = |input: Vec<(Vec<u8>, Vec<u8>, TreeOp)>| {
        tree_insert_read_delete_scan(input, bf_tree_config.clone());
        Ok(())
    };

    let mut runner = TestRunner::new(config);
    runner.run(&strategy, test).expect("Test Failed");
}

#[test]
fn test_tree_insert_read_1() {
    let mut config = BfTreeConfig::default();
    config.cb_min_record_size = 3072;
    config.cb_max_record_size = 3072;
    config.leaf_page_size = 8192;
    config.cache_only = true;

    leaf_insert_read_inner(1000..10000, &config);
}

#[test]
fn test_tree_insert_read_2() {
    let mut config = BfTreeConfig::default();
    config.cb_min_record_size = 32;
    config.cb_max_record_size = 2016;
    config.leaf_page_size = 4096;
    config.cache_only = true;

    leaf_insert_read_inner(500..5000, &config);
}

#[test]
fn test_tree_insert_read_3() {
    let mut config = BfTreeConfig::default();
    config.cb_min_record_size = 64;
    config.cb_max_record_size = 2016;
    config.leaf_page_size = 4096;
    config.cache_only = true;

    leaf_insert_read_inner(1000..10000, &config);
}

#[test]
fn test_tree_insert_read_4() {
    let mut config = BfTreeConfig::default();
    config.cb_min_record_size = 64;
    config.cb_max_record_size = 2048;
    config.leaf_page_size = 16384;
    config.cache_only = false;

    leaf_insert_read_inner(500..5000, &config);
}

#[test]
fn test_tree_insert_read_5() {
    let mut config = BfTreeConfig::default();
    config.cb_min_record_size = 64;
    config.cb_max_record_size = 2048;
    config.leaf_page_size = 16384;
    config.cache_only = false;

    leaf_insert_read_inner(1000..10000, &config);
}

#[test]
fn test_tree_insert_read_6() {
    let mut config = BfTreeConfig::default();
    config.cb_min_record_size = 11268;
    config.cb_max_record_size = 11268;
    config.leaf_page_size = 32768;
    config.cache_only = false;

    leaf_insert_read_inner(500..5000, &config);
}

#[test]
fn test_tree_insert_read_7() {
    let mut config = BfTreeConfig::default();
    config.cb_size_byte(8 * 1024);
    config.cb_min_record_size = 32;
    config.cb_max_record_size = 1952;
    config.leaf_page_size = 4096;
    config.cache_only = false;

    leaf_insert_read_inner(1000..10000, &config);
}

#[test]
fn test_tree_insert_read_8() {
    let mut config = BfTreeConfig::default();
    config.cb_size_byte(16 * 1024);
    config.cb_min_record_size = 32;
    config.cb_max_record_size = 1952;
    config.leaf_page_size = 4096;
    config.cache_only = true;

    config.validate().expect("Invalid configuration");
    leaf_insert_read_inner(1000..10000, &config);
}

#[test]
fn test_tree_insert_read_9() {
    let mut config = BfTreeConfig::default();
    config.cb_size_byte(16 * 1024);
    config.cb_min_record_size = 2024;
    config.cb_max_record_size = 2024;
    config.leaf_page_size = 4096;
    config.cache_only = true;

    leaf_insert_read_inner(1000..10000, &config);
}

#[test]
fn aggressive_traversal_preserves_root_leaf_splitting() {
    let mut config = BfTreeConfig::default();
    config
        .storage_backend(crate::StorageBackend::Memory)
        .cache_only(false)
        .use_snapshot(false)
        .cb_min_record_size(64)
        .cb_max_record_size(256);
    let tree = BfTree::with_config(config, None).unwrap();
    for key in 0..4u64 {
        assert_eq!(
            tree.insert(&key.to_be_bytes(), &[0x11; 200]),
            crate::LeafInsertResult::Success
        );
    }
    let (root, is_leaf) = tree.get_root_page();
    assert!(is_leaf);
    tree.mapping_table()
        .get_mut(&root)
        .load_base_page_mut()
        .set_split_flag();
    let (found, parent) = tree
        .traverse_to_leaf(&0u64.to_be_bytes(), false, false, None)
        .unwrap();
    assert_eq!(found, root);
    assert!(parent.is_none());
    assert!(matches!(
        tree.traverse_to_leaf(&0u64.to_be_bytes(), true, false, None),
        Err(crate::error::TreeError::NeedRestart)
    ));
    assert!(!tree.get_root_page().1);
    let (found, parent) = tree
        .traverse_to_leaf(&0u64.to_be_bytes(), true, false, None)
        .unwrap();
    assert_eq!(found, root);
    assert!(parent.is_some());
    for key in 0..4u64 {
        let mut bytes = [0; 200];
        assert_eq!(
            tree.read(&key.to_be_bytes(), &mut bytes),
            LeafReadResult::Found(200)
        );
        assert_eq!(bytes, [0x11; 200]);
    }
}
