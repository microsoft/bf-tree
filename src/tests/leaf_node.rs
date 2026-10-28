// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use proptest::prelude::*;
use proptest::test_runner::{Config, FileFailurePersistence, TestRunner};
use proptest_derive::Arbitrary;
use std::collections::HashMap;

use crate::nodes::leaf_node::{LeafNode, LeafReadResult, OpType};

#[test]
fn empty_prefix_and_infinite_fence_do_not_use_sentinel_offsets() {
    use crate::nodes::leaf_node::{LeafKVMeta, MiniPageNextLevel};
    let ptr = LeafNode::make_base_page(4096, 0);
    let leaf = unsafe { &mut *ptr };
    assert!(leaf.get_prefix().is_empty());
    leaf.initialize(
        &[],
        &[],
        4096,
        MiniPageNextLevel::new_null(),
        true,
        false,
        0,
    );
    assert!(leaf.get_low_fence_full_key().is_empty());
    assert!(leaf.get_prefix().is_empty());
    let empty = LeafKVMeta::make_prefixed_meta(u16::MAX, 0, &[], 0, OpType::Delete);
    assert!(leaf.get_remaining_key(&empty).is_empty());
    leaf.initialize(
        b"prefix-a",
        b"prefix-z",
        4096,
        MiniPageNextLevel::new_null(),
        true,
        false,
        0,
    );
    assert_eq!(leaf.get_prefix(), b"prefix-");
    let prefix_only = LeafKVMeta::make_prefixed_meta(u16::MAX, 0, b"prefix-", 7, OpType::Delete);
    assert!(leaf.get_remaining_key(&prefix_only).is_empty());
    LeafNode::free_base_page(ptr);
}

#[derive(Clone, Arbitrary, Debug)]
enum LeafTestOp {
    Insert,
    Delete,
    Read,
}

fn leaf_insert_read(input: Vec<(Vec<u8>, Vec<u8>, LeafTestOp)>) {
    let mut model = HashMap::<Vec<u8>, Vec<u8>>::new();
    let leaf =
        unsafe { &mut *LeafNode::make_base_page(4096, crate::snapshot::INVALID_SNAPSHOT_VERSION) };
    let mut out_buffer = vec![0u8; 1024]; // Buffer for reading from LeafNode

    for (k, v, op) in input.iter() {
        match op {
            LeafTestOp::Insert => {
                let rt = leaf.insert(k, v, OpType::Insert, 60);
                assert!(rt);

                model.insert(k.to_owned(), v.to_owned());
            }
            LeafTestOp::Delete => {
                let _ = leaf.insert(k, &[], OpType::Delete, 60);
                model.remove(k);
            }
            LeafTestOp::Read => {
                let rt = leaf.read_by_key(k, &mut out_buffer);
                match model.get(k) {
                    Some(v) => {
                        assert_eq!(rt, LeafReadResult::Found(v.len() as u32));
                        assert_eq!(&out_buffer[0..v.len()], v);
                    }
                    None => {
                        assert!(rt == LeafReadResult::NotFound || rt == LeafReadResult::Deleted);
                    }
                }
            }
        }
    }

    let model_cnt = model.len();
    // Now sanity check every value
    for (k, v) in model {
        let rt = leaf.read_by_key(&k, &mut out_buffer);
        assert_eq!(rt, LeafReadResult::Found(v.len() as u32));
        if &out_buffer[0..v.len()] != v {
            let rt = leaf.read_by_key(&k, &mut out_buffer);
            assert_eq!(rt, LeafReadResult::Found(v.len() as u32));
        }
        assert_eq!(&out_buffer[0..v.len()], v);
    }

    leaf.consolidate(crate::snapshot::INVALID_SNAPSHOT_VERSION);
    let leaf_cnt = leaf.meta.meta_count_without_fence();
    assert_eq!(model_cnt, leaf_cnt as usize);

    LeafNode::free_base_page(leaf);
}

#[test]
fn test_leaf_insert_read() {
    let config = Config {
        cases: 1000,
        failure_persistence: Some(Box::new(FileFailurePersistence::SourceParallel(
            "proptest-regressions",
        ))),
        source_file: Some("src/prop_tests/leaf_node.rs"),
        ..Config::default()
    };

    let strategy = proptest::collection::vec(
        (
            proptest::collection::vec(any::<u8>(), 1..30), // Key
            proptest::collection::vec(any::<u8>(), 1..30), // Value
            any::<LeafTestOp>(),
        ),
        1..50, // Length of the list
    );

    let test = |input: Vec<(Vec<u8>, Vec<u8>, LeafTestOp)>| {
        leaf_insert_read(input);
        Ok(())
    };

    let mut runner = TestRunner::new(config);
    runner.run(&strategy, test).unwrap();
}
