// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::hint::black_box;
use std::{collections::HashMap, ops::Range};

use crate::nodes::leaf_node::LeafReadResult;
use crate::{BfTree, Config as BfTreeConfig};
use proptest::{
    prelude::*,
    test_runner::{Config, TestRunner},
};
use proptest_derive::Arbitrary;

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

    leaf_insert_read_inner(1000..10000, &config);
}

#[test]
fn test_tree_insert_read_9() {
    let mut config = BfTreeConfig::default();
    config.cb_size_byte(16 * 1024);
    config.cb_min_record_size = 2028;
    config.cb_max_record_size = 2028;
    config.leaf_page_size = 4096;
    config.cache_only = true;

    leaf_insert_read_inner(1000..10000, &config);
}
