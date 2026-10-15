// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use proptest::prelude::*;
use proptest::test_runner::{Config, FileFailurePersistence, TestRunner};
use proptest_derive::Arbitrary;
use std::collections::{BTreeMap, HashMap};
use std::ops::{Deref, DerefMut};

use crate::nodes::{InnerNode, InnerNodeBuilder, PageID, INNER_NODE_SIZE};
use crate::storage::DiskOffsetGuard;
use crate::utils::TestVfs;

#[derive(Clone, Arbitrary, Debug)]
enum InnerTestOp {
    Insert,
    Read,
}

struct TestInnerNode(*mut InnerNode);

impl Deref for TestInnerNode {
    type Target = InnerNode;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl DerefMut for TestInnerNode {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

impl Drop for TestInnerNode {
    fn drop(&mut self) {
        InnerNode::free_node(self.0);
    }
}

fn make_inner_node() -> TestInnerNode {
    let test_vfs = TestVfs::new();
    let mut inner_builder = InnerNodeBuilder::new();
    inner_builder
        .set_disk_offset(DiskOffsetGuard::new(0, &test_vfs))
        .set_children_is_leaf(true)
        .set_left_most_page_id(PageID::new(0));
    TestInnerNode(inner_builder.build(crate::snapshot::INVALID_SNAPSHOT_VERSION))
}

fn inner_insert_read(input: Vec<(Vec<u8>, u64, InnerTestOp)>) {
    let mut model = HashMap::<Vec<u8>, PageID>::new();
    let mut inner = make_inner_node();

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

    inner.consolidate(crate::snapshot::INVALID_SNAPSHOT_VERSION);
    let inner_cnt = inner.meta.value_count_inner();
    assert_eq!(model_cnt, inner_cnt as usize);
}

#[test]
fn test_inner_insert_read() {
    let config = Config {
        cases: 1000,
        failure_persistence: Some(Box::new(FileFailurePersistence::SourceParallel(
            "proptest-regressions",
        ))),
        source_file: Some(file!()),
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

#[test]
fn test_inner_lower_bound_with_same_prefix() {
    let mut inner = make_inner_node();

    assert!(inner.insert(b"bbbb", PageID::from_id(1)));
    assert!(inner.insert(b"test", PageID::from_id(2)));
    assert!(inner.insert(b"teste", PageID::from_id(3)));

    let pos = inner.lower_bound(b"teste");
    let meta = inner.get_kv_meta(pos as u16);
    assert_eq!(inner.get_full_key(meta), b"teste");
    assert_eq!(inner.get_value(meta), PageID::from_id(3));

    let pos = inner.lower_bound(b"test");
    let meta = inner.get_kv_meta(pos as u16);
    assert_eq!(inner.get_full_key(meta), b"test");
    assert_eq!(inner.get_value(meta), PageID::from_id(2));

    let pos = inner.lower_bound(b"aaaa");
    assert_eq!(pos, 0);

    let pos = inner.lower_bound(b"testz");
    let meta = inner.get_kv_meta(pos as u16);
    assert_eq!(inner.get_full_key(meta), b"teste");
    assert_eq!(inner.get_value(meta), PageID::from_id(3));
}

fn assert_inner_records(inner: &InnerNode, model: &BTreeMap<Vec<u8>, PageID>, left_most: PageID) {
    assert_eq!(inner.meta.value_count_inner() as usize, model.len());
    assert_eq!(inner.get_value(inner.get_kv_meta(0)), left_most);
    for (index, (key, value)) in model.iter().enumerate() {
        let meta = inner.get_kv_meta(index as u16 + 1);
        assert_eq!(inner.get_full_key(meta), *key);
        assert_eq!(inner.get_value(meta), *value);
        assert_eq!(inner.lower_bound(key), index as u64 + 1);
    }
    // This also checks compacted offsets against the metadata and free-space count.
    inner.current_lowest_offset();
}

#[test]
fn test_inner_update_full_node_and_reject_oversized_keys() {
    let mut inner = make_inner_node();
    let mut model = BTreeMap::new();
    for value in 0u32.. {
        let key = value.to_be_bytes();
        let child = PageID::from_id(value as u64);
        if !inner.insert(&key, child) {
            break;
        }
        model.insert(key.to_vec(), child);
    }
    let remaining = inner.meta.remaining_size;
    assert!(!inner.have_space_for(&0u32.to_be_bytes()));
    let replacement = PageID::from_id(100_000);
    assert!(inner.insert(&0u32.to_be_bytes(), replacement));
    model.insert(0u32.to_be_bytes().to_vec(), replacement);
    assert_eq!(inner.meta.remaining_size, remaining);

    for len in [u16::MAX as usize, 1 << 16, (1 << 16) + 4, 1 << 17] {
        let key = vec![0xff; len];
        assert!(!inner.have_space_for(&key));
        assert!(!inner.insert(&key, PageID::from_id(99)));
        assert_eq!(inner.meta.remaining_size, remaining);
    }
    assert_inner_records(&inner, &model, PageID::new(0));
}

#[test]
fn test_inner_oversized_keys_cannot_wrap_capacity_on_empty_node() {
    let mut inner = make_inner_node();
    let remaining = inner.meta.remaining_size;
    for len in [u16::MAX as usize - 12, u16::MAX as usize, 1 << 16, 1 << 17] {
        let key = vec![0; len];
        assert!(!inner.have_space_for(&key));
        assert!(!inner.insert(&key, PageID::from_id(1)));
        assert_eq!(inner.meta.remaining_size, remaining);
        assert_eq!(inner.meta.meta_count_with_fence(), 1);
    }
}

#[test]
fn test_inner_consolidate_and_split_binary_keys() {
    // A deterministic counterpart to the property test also runs under Miri
    // without requiring a random-number or filesystem source.
    let keys: &[&[u8]] = &[
        b"",
        b"\0",
        b"\0\0",
        b"a",
        b"abc",
        b"abcd",
        b"abcde",
        b"same",
        b"same\0",
        b"same\0\xff",
        b"same\xfflonger suffix",
        b"\xff\xff\xff\xff\0",
    ];
    let mut inner = make_inner_node();
    let mut model = BTreeMap::new();
    for (value, key) in keys.iter().rev().enumerate() {
        let child = PageID::from_id(value as u64 + 1);
        assert!(inner.insert(key, child));
        model.insert(key.to_vec(), child);
    }
    inner.set_root(true);
    let remaining = inner.meta.remaining_size;
    inner.consolidate(17);
    assert_inner_records(&inner, &model, PageID::new(0));
    assert_eq!(inner.meta.remaining_size, remaining);

    let mut builder = InnerNodeBuilder::new();
    let split_key = inner.split(&mut builder, 18);
    let sibling = TestInnerNode(builder.build(18));
    let mut right = model.split_off(&split_key);
    let promoted_child = right.remove(&split_key).unwrap();
    assert_inner_records(&inner, &model, PageID::new(0));
    assert_inner_records(&sibling, &right, promoted_child);
    assert!(inner.is_root());
    assert_eq!(inner.get_clean_snapshot_version(), 18);
    assert!(!sibling.is_root());
    assert!(inner.insert(&split_key, promoted_child));
    model.insert(split_key, promoted_child);
    inner.consolidate(19);
    assert_inner_records(&inner, &model, PageID::new(0));
}

#[test]
#[should_panic(expected = "inner node image is too short")]
fn test_inner_restore_rejects_short_image() {
    InnerNodeBuilder::new().build_from_slice(&[0; INNER_NODE_SIZE - 1]);
}

#[cfg(not(feature = "shuttle"))]
#[test]
fn test_inner_serialization_initializes_unused_bytes_and_resets_lock() {
    use crate::sync::atomic::Ordering;

    let mut inner = make_inner_node();
    // A fresh page has one metadata entry and one child value; its gap and
    // the alignment padding between NodeMeta and AtomicU64 remain zero.
    let header_size = INNER_NODE_SIZE - InnerNode::max_data_size();
    let gap_end = header_size + inner.current_lowest_offset() as usize;
    assert!(inner.as_slice()[6..8].iter().all(|byte| *byte == 0));
    assert!(inner.as_slice()[header_size + 8..gap_end]
        .iter()
        .all(|byte| *byte == 0));
    assert!(inner.insert(b"same\0suffix", PageID::from_id(41)));
    inner.set_snapshot_version(42);
    inner.set_root(true);
    inner.version_lock.store(123, Ordering::Relaxed);
    let restored = TestInnerNode(InnerNodeBuilder::new().build_from_slice(inner.as_slice()));
    assert_eq!(restored.version_lock.load(Ordering::Relaxed), 0);
    assert_eq!(restored.get_clean_snapshot_version(), 42);
    assert!(restored.is_root());
    assert_eq!(restored.disk_offset, inner.disk_offset);
    let mut model = BTreeMap::new();
    model.insert(b"same\0suffix".to_vec(), PageID::from_id(41));
    assert_inner_records(&restored, &model, PageID::new(0));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn test_inner_binary_keys_consolidate_and_split(
        keys in proptest::collection::vec(
            prop_oneof![
                proptest::collection::vec(any::<u8>(), 0..24),
                proptest::collection::vec(any::<u8>(), 0..20)
                    .prop_map(|suffix| [b"same".as_slice(), suffix.as_slice()].concat()),
                proptest::collection::vec(Just(0u8), 0..8),
            ],
            0..90,
        ),
        queries in proptest::collection::vec(proptest::collection::vec(any::<u8>(), 0..28), 0..40),
    ) {
        let mut inner = make_inner_node();
        let mut model = BTreeMap::new();
        for (value, key) in keys.into_iter().enumerate() {
            let child = PageID::from_id(value as u64 + 1);
            assert!(inner.insert(&key, child));
            model.insert(key, child);
        }
        assert_inner_records(&inner, &model, PageID::new(0));
        let remaining = inner.meta.remaining_size;
        inner.set_root(true);
        inner.meta.set_split_flag();
        inner.consolidate(17);
        assert_inner_records(&inner, &model, PageID::new(0));
        assert_eq!(inner.meta.remaining_size, remaining);
        assert_eq!(inner.get_clean_snapshot_version(), 17);
        assert!(inner.is_root());
        assert!(!inner.meta.get_split_flag());
        assert!(inner.meta.children_is_leaf());
        assert_eq!(inner.disk_offset, 0);

        for query in queries {
            let expected_pos = model.keys().take_while(|key| key.as_slice() <= query.as_slice()).count();
            assert_eq!(inner.lower_bound(&query), expected_pos as u64);
        }

        if model.len() >= 3 {
            let mut builder = InnerNodeBuilder::new();
            let split_key = inner.split(&mut builder, 18);
            let sibling = TestInnerNode(builder.build(18));
            let mut right = model.split_off(&split_key);
            let promoted_child = right.remove(&split_key).unwrap();
            assert_inner_records(&inner, &model, PageID::new(0));
            assert_inner_records(&sibling, &right, promoted_child);
            assert_eq!(inner.get_clean_snapshot_version(), 18);
            assert_eq!(sibling.get_clean_snapshot_version(), 18);
            assert!(inner.is_root());
            assert!(!sibling.is_root());
            assert!(sibling.meta.children_is_leaf());
            // The reclaimed payload capacity must also be usable by later inserts.
            assert!(inner.insert(&split_key, promoted_child));
            model.insert(split_key, promoted_child);
            assert_inner_records(&inner, &model, PageID::new(0));
        }
    }
}
