// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use proptest::prelude::*;
use proptest::test_runner::{Config, FileFailurePersistence, TestRunner};
use proptest_derive::Arbitrary;
use std::collections::HashMap;

use crate::nodes::{InnerNode, InnerNodeBuilder, PageID};
use crate::storage::DiskOffsetGuard;
use crate::utils::TestVfs;

#[derive(Clone, Arbitrary, Debug)]
enum InnerTestOp {
    Insert,
    Read,
}

fn inner_insert_read(input: Vec<(Vec<u8>, u64, InnerTestOp)>) {
    let mut model = HashMap::<Vec<u8>, PageID>::new();

    let test_vfs = TestVfs::new();
    let mut inner_builder = InnerNodeBuilder::new();
    inner_builder
        .set_disk_offset(DiskOffsetGuard::new(0, &test_vfs))
        .set_children_is_leaf(true)
        .set_left_most_page_id(PageID::new(0));
    let inner = unsafe { &mut *inner_builder.build() };

    for (k, v, op) in input.iter() {
        match op {
            InnerTestOp::Insert => {
                let id = PageID::from_id(*v);
                let rt = inner.insert(k, id);
                assert!(rt);

                model.insert(k.to_owned(), id);
            }
            InnerTestOp::Read => {
                let pos = inner.lower_bound(k);

                match model.get(k) {
                    Some(v) => {
                        let meta = inner.get_kv_meta(pos as u16);
                        let key = inner.get_full_key(&meta);
                        assert_eq!(&key, k);

                        let value = inner.get_value(&meta);
                        assert_eq!(value, *v);
                    }
                    None => {
                        if pos < inner.meta.value_count_inner() as u64 {
                            let meta = inner.get_kv_meta(pos as u16);
                            let key = inner.get_full_key(&meta);
                            assert_ne!(&key, k);
                        }
                    }
                }
            }
        }
    }

    let model_cnt = model.len();
    // Now sanity check every value
    for (k, v) in model {
        let meta = inner.get_kv_meta(inner.lower_bound(&k) as u16);
        let key = inner.get_full_key(&meta);
        assert_eq!(&key, &k);
        let value = inner.get_value(&meta);
        assert_eq!(value, v);
    }

    inner.consolidate();
    let inner_cnt = inner.meta.value_count_inner();
    assert_eq!(model_cnt, inner_cnt as usize);

    InnerNode::free_node(inner);
}

#[test]
fn test_inner_insert_read() {
    let config = Config {
        cases: 1000,
        failure_persistence: Some(Box::new(FileFailurePersistence::SourceParallel(
            "proptest-regressions",
        ))),
        source_file: Some("src/prop_tests/inner_node.rs"),
        ..Config::default()
    };

    let strategy = proptest::collection::vec(
        (
            proptest::collection::vec(any::<u8>(), 1..30), // Key
            any::<u64>(),                                  // Value
            any::<InnerTestOp>(),
        ),
        1..50, // Length of the list
    );

    let test = |input: Vec<(Vec<u8>, u64, InnerTestOp)>| {
        inner_insert_read(input);
        Ok(())
    };

    let mut runner = TestRunner::new(config);

    match runner.run(&strategy, test) {
        Ok(_) => println!("All tests passed!"),
        Err(e) => {
            println!("Test failed! Seed: {:?}", runner.rng());
            panic!("Test failed: {:?}", e);
        }
    }
    // runner.run(&strategy, test).unwrap();
}
