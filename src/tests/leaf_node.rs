// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use proptest::prelude::*;
use proptest::test_runner::{Config, FileFailurePersistence, TestRunner};
use proptest_derive::Arbitrary;
use std::collections::{BTreeMap, HashMap};

use crate::nodes::leaf_node::{
    LeafNode, LeafNodeHeader, LeafReadResult, MiniPageNextLevel, OpType,
};

struct TestBasePage(*mut LeafNodeHeader, usize);

impl TestBasePage {
    fn new(size: usize) -> Self {
        Self(
            LeafNode::make_base_page(size, crate::snapshot::INVALID_SNAPSHOT_VERSION),
            size,
        )
    }

    fn page(&mut self) -> &mut LeafNode {
        // Preserve the original allocation provenance and carry its entire
        // extent into the reference, rather than extending a header reference.
        unsafe { &mut *LeafNode::from_raw_parts(self.0, self.1) }
    }
}

impl Drop for TestBasePage {
    fn drop(&mut self) {
        // A test may reinitialize this allocation as a mini page. Ownership and
        // allocation layout remain the same regardless of its current metadata.
        let layout =
            std::alloc::Layout::from_size_align(self.1, std::mem::align_of::<LeafNodeHeader>())
                .unwrap();
        unsafe { std::alloc::dealloc(self.0.cast::<u8>(), layout) };
    }
}

#[test]
fn leaf_consolidation_keeps_clean_tombstones_absent() {
    for skip_tombstone in [false, true] {
        let mut allocation = TestBasePage::new(4096);
        let leaf = allocation.page();
        leaf.initialize(&[], &[], 4096, MiniPageNextLevel::new(0), false, false, 0);
        assert!(leaf.insert(b"deleted-with-value", b"old", OpType::Insert, 0));
        assert!(leaf.insert(b"deleted-with-value", &[], OpType::Delete, 0));
        assert!(leaf.insert(b"deleted-empty", &[], OpType::Delete, 0));
        assert!(leaf.insert(b"retained", b"value", OpType::Insert, 0));
        leaf.covert_insert_records_to_cache();

        leaf.consolidate_inner(OpType::Cache, None, skip_tombstone, false, None, 1);
        let mut out = [0; 16];
        for key in [b"deleted-with-value".as_slice(), b"deleted-empty"] {
            let expected = if skip_tombstone {
                LeafReadResult::NotFound
            } else {
                LeafReadResult::Deleted
            };
            assert_eq!(leaf.read_by_key(key, &mut out), expected);
        }
        assert_eq!(
            leaf.read_by_key(b"retained", &mut out),
            LeafReadResult::Found(5)
        );
        assert_eq!(&out[..5], b"value");
        assert_eq!(
            leaf.meta.meta_count_without_fence(),
            if skip_tombstone { 1 } else { 3 }
        );
        if !skip_tombstone {
            assert!(leaf
                .meta_iter()
                .take(2)
                .all(|meta| meta.op_type() == OpType::Phantom));
        }
    }
}

#[test]
fn leaf_upgrade_preserves_prefix_keys_and_filters_only_cold_cache() {
    for with_prefix in [false, true] {
        for discard_cold_cache in [false, true] {
            let mut source = TestBasePage::new(4096);
            let leaf = source.page();
            let (low, high) = if with_prefix {
                (b"tenant/a".as_slice(), b"tenant/z".as_slice())
            } else {
                (&[][..], &[][..])
            };
            leaf.initialize(
                low,
                high,
                4096,
                MiniPageNextLevel::new(128),
                with_prefix,
                false,
                0,
            );
            let records = [
                (
                    b"tenant/a".as_slice(),
                    b"one".as_slice(),
                    OpType::Cache,
                    false,
                ),
                (b"tenant/b", b"two", OpType::Cache, true),
                (b"tenant/c", b"", OpType::Phantom, false),
                (b"tenant/d", b"", OpType::Phantom, true),
                (b"tenant/e", b"five", OpType::Insert, false),
                (b"tenant/f", b"", OpType::Delete, false),
            ];
            for (key, value, op, referenced) in records {
                assert!(leaf.insert(key, value, op, 0));
                if referenced {
                    leaf.get_kv_meta(leaf.lower_bound(key) as usize)
                        .mark_as_ref();
                }
            }
            let mut destination = TestBasePage::new(8192);
            leaf.copy_initialize_to(destination.0, 8192, discard_cold_cache, 91);
            let destination = destination.page();
            assert!(destination.get_prefix().is_empty());
            assert_eq!(destination.next_level.as_offset(), 128);
            assert_eq!(destination.get_clean_snapshot_version(), 91);
            let retained: Vec<_> = records
                .iter()
                .filter(|(_, _, op, referenced)| {
                    !discard_cold_cache || op.is_dirty() || *referenced
                })
                .collect();
            assert_eq!(
                destination.meta.meta_count_without_fence() as usize,
                retained.len()
            );
            for (meta, (key, value, op, _)) in destination.meta_iter().zip(retained) {
                assert_eq!(destination.get_full_key(meta), *key);
                assert_eq!(destination.get_value(meta), *value);
                assert_eq!(meta.op_type(), *op);
                assert!(!meta.is_referenced());
            }
        }
    }
}

/// Run in an optimized build, with other tests/benchmarks stopped. Copy this
/// harness unchanged to the compared revision; setup and checks are not timed.
#[test]
#[ignore = "manual leaf rebuild benchmark"]
fn leaf_rebuild_microbench() {
    use std::hint::black_box;
    use std::time::Instant;

    const ITERATIONS: usize = 2_000;
    const SAMPLES: usize = 9;
    for (key_len, count) in [(8, 96usize), (128, 20)] {
        let keys: Vec<_> = (0..count)
            .map(|index| {
                let mut key = vec![b'x'; key_len];
                key[key_len - 4..].copy_from_slice(&(index as u32).to_be_bytes());
                key
            })
            .collect();
        let value = [7; 16];
        for operation in ["consolidate", "upgrade"] {
            let mut samples = Vec::with_capacity(SAMPLES);
            for sample in 0..=SAMPLES {
                let mut source = TestBasePage::new(4096);
                let source = source.page();
                source.initialize(&[], &[], 4096, MiniPageNextLevel::new(0), false, false, 0);
                for key in &keys {
                    assert!(source.insert(key, &value, OpType::Insert, 0));
                }
                let mut destination = TestBasePage::new(8192);
                let start = Instant::now();
                if operation == "consolidate" {
                    for _ in 0..ITERATIONS {
                        black_box(&mut *source).consolidate(1);
                    }
                } else {
                    for _ in 0..ITERATIONS {
                        black_box(&*source).copy_initialize_to(
                            black_box(destination.0),
                            8192,
                            false,
                            1,
                        );
                    }
                }
                let elapsed = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
                let result = if operation == "consolidate" {
                    source
                } else {
                    destination.page()
                };
                assert_eq!(result.meta.meta_count_without_fence() as usize, count);
                for (meta, key) in result.meta_iter().zip(&keys) {
                    assert_eq!(result.get_full_key(meta), *key);
                    assert_eq!(result.get_value(meta), value);
                }
                if sample != 0 {
                    samples.push(elapsed);
                }
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "leaf_{operation}_k{key_len},records={count},median_ns={:.2},min_ns={:.2},max_ns={:.2}",
                samples[SAMPLES / 2],
                samples[0],
                samples[SAMPLES - 1],
            );
        }
    }
}

#[derive(Clone, Arbitrary, Debug)]
enum LeafTestOp {
    Insert,
    Delete,
    Read,
}

fn leaf_insert_read(input: Vec<(Vec<u8>, Vec<u8>, LeafTestOp)>) {
    let mut model = HashMap::<Vec<u8>, Vec<u8>>::new();
    let mut allocation = TestBasePage::new(4096);
    let leaf = allocation.page();
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
}

fn collision_key(prefix: &[u8], id: u8) -> Vec<u8> {
    let mut key = prefix.to_vec();
    match id {
        0 => {}
        1 => key.extend_from_slice(&[0]),
        2 => key.extend_from_slice(&[0, 0]),
        3 => key.extend_from_slice(&[0, 0, 0]),
        4 => key.extend_from_slice(&[0, 0, 1]),
        5 => key.extend_from_slice(&[0, 1]),
        6 => key.extend_from_slice(&[1]),
        7 => key.extend_from_slice(&[1, 0]),
        _ => key.extend_from_slice(&[2, 2, id]),
    }
    key
}

#[test]
fn leaf_short_preview_search_matches_btree_order() {
    // Cover empty suffixes, strict zero-byte prefixes, signed-byte boundaries,
    // and suffixes longer than the two-byte metadata preview.
    let mut suffixes = vec![Vec::new()];
    let mut previous = vec![Vec::new()];
    for _ in 0..3 {
        let mut next = Vec::new();
        for suffix in &previous {
            for byte in [0, 1, 0x80, 0xFF] {
                let mut key = suffix.clone();
                key.push(byte);
                next.push(key);
            }
        }
        suffixes.extend(next.iter().cloned());
        previous = next;
    }

    for prefix in [b"".as_slice(), b"\0tenant/\0".as_slice()] {
        for stored_parity in 0..2 {
            let mut allocation = TestBasePage::new(4096);
            let leaf = allocation.page();
            let has_fence = !prefix.is_empty();
            let mut high_fence = prefix.to_vec();
            high_fence.extend_from_slice(&[0xFF; 4]);
            leaf.initialize(
                if has_fence { prefix } else { &[] },
                if has_fence { &high_fence } else { &[] },
                4096,
                MiniPageNextLevel::new_null(),
                has_fence,
                false,
                crate::snapshot::INVALID_SNAPSHOT_VERSION,
            );
            assert_eq!(leaf.get_prefix(), prefix);
            let first_record = if has_fence { 2 } else { 0 };
            let mut model = BTreeMap::<Vec<u8>, Vec<u8>>::new();
            // Alternate which short keys are absent, and insert in reverse
            // generation order so insertion exercises the same lower bound.
            for (index, suffix) in suffixes.iter().enumerate().rev() {
                if index % 2 != stored_parity {
                    continue;
                }
                let mut key = prefix.to_vec();
                key.extend_from_slice(suffix);
                let value = vec![index as u8, 0, 0xA5];
                assert!(leaf.insert(&key, &value, OpType::Insert, 0));
                model.insert(key, value);
            }
            // Initialized bytes outside a zero/one-byte preview are irrelevant
            // to slice ordering, even when they are deliberately nonzero.
            for index in first_record..leaf.meta.meta_count_with_fence() as usize {
                leaf.get_kv_meta_mut(index)
                    .fill_unused_preview_bytes(prefix.len(), 0xA5);
            }
            let stored: Vec<_> = leaf
                .meta_iter()
                .map(|meta| leaf.get_full_key(meta))
                .collect();
            assert_eq!(stored, model.keys().cloned().collect::<Vec<_>>());

            let mut queries: Vec<_> = suffixes
                .iter()
                .map(|suffix| {
                    let mut key = prefix.to_vec();
                    key.extend_from_slice(suffix);
                    key
                })
                .collect();
            for suffix in [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0xFF, 0xFF, 0xFF, 0],
            ] {
                let mut key = prefix.to_vec();
                key.extend_from_slice(&suffix);
                queries.push(key);
            }
            queries.extend((0..prefix.len()).map(|len| prefix[..len].to_vec()));
            queries.extend([vec![0xFF], high_fence]);

            for query in queries {
                let expected_position = first_record + model.range(..query.clone()).count();
                assert_eq!(leaf.lower_bound(&query) as usize, expected_position);
                assert_eq!(leaf.linear_lower_bound(&query) as usize, expected_position);
                for binary_search in [true, false] {
                    let mut out = [0x7E; 3];
                    let result = leaf.read_by_key_inner(&query, &mut out, binary_search);
                    if let Some(expected) = model.get(&query) {
                        assert_eq!(result, LeafReadResult::Found(expected.len() as u32));
                        assert_eq!(out.as_slice(), expected);
                    } else {
                        assert_eq!(result, LeafReadResult::NotFound);
                        assert_eq!(out, [0x7E; 3]);
                    }
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(Config::with_cases(256))]

    #[test]
    fn leaf_search_update_and_consolidation_match_model(
        prefix in proptest::collection::vec(any::<u8>(), 0..32),
        operations in proptest::collection::vec(
            (0u8..16, proptest::collection::vec(any::<u8>(), 1..80), 0u8..4),
            1..64,
        ),
    ) {
        let mut allocation = TestBasePage::new(8192);
        let leaf = allocation.page();
        let mut high_fence = prefix.clone();
        high_fence.push(u8::MAX);
        leaf.initialize(
            &prefix,
            &high_fence,
            8192,
            MiniPageNextLevel::new_null(),
            true,
            false,
            crate::snapshot::INVALID_SNAPSHOT_VERSION,
        );
        let mut model = BTreeMap::<Vec<u8>, Vec<u8>>::new();
        let mut out = [0u8; 80];
        for (id, value, operation) in operations {
            let key = collision_key(&prefix, id);
            if operation == 0 {
                prop_assert!(leaf.insert(&key, &[], OpType::Delete, 0));
                model.remove(&key);
            } else {
                prop_assert!(leaf.insert(&key, &value, OpType::Insert, 0));
                model.insert(key, value);
            }

            // Small key spaces deliberately force overwrite, growth, deletion,
            // resurrection, and collisions in the two-byte metadata preview.
            for query_id in 0..17 {
                let query = collision_key(&prefix, query_id);
                let binary = leaf.read_by_key_inner(&query, &mut out, true);
                if let Some(expected) = model.get(&query) {
                    prop_assert_eq!(&binary, &LeafReadResult::Found(expected.len() as u32));
                    prop_assert_eq!(&out[..expected.len()], expected);
                } else {
                    prop_assert!(matches!(binary, LeafReadResult::NotFound | LeafReadResult::Deleted));
                }
                let linear = leaf.read_by_key_inner(&query, &mut out, false);
                prop_assert_eq!(binary, linear);
                if let Some(expected) = model.get(&query) {
                    prop_assert_eq!(&out[..expected.len()], expected);
                }
            }
        }

        let mut queries: Vec<_> = (0..17).map(|id| collision_key(&prefix, id)).collect();
        queries.extend((0..prefix.len()).map(|len| prefix[..len].to_vec()));
        queries.extend([vec![0], vec![u8::MAX], high_fence.clone()]);
        let stored: Vec<_> = leaf.meta_iter().map(|meta| leaf.get_full_key(meta)).collect();
        for query in queries {
            let expected = 2 + stored.partition_point(|key| key < &query) as u16;
            prop_assert_eq!(leaf.lower_bound(&query), expected);
            prop_assert_eq!(leaf.linear_lower_bound(&query), expected);
            prop_assert_eq!(leaf.get_kv_num_below_key(&query), expected - 2);
        }

        leaf.consolidate(37);
        prop_assert_eq!(leaf.get_low_fence_key(), prefix);
        prop_assert_eq!(leaf.get_high_fence_key(), high_fence);
        prop_assert_eq!(leaf.get_clean_snapshot_version(), 37);
        prop_assert_eq!(leaf.meta.meta_count_without_fence() as usize, model.len());
        for (meta, (key, value)) in leaf.meta_iter().zip(&model) {
            prop_assert_eq!(&leaf.get_full_key(meta), key);
            prop_assert_eq!(leaf.get_value(meta), value);
            prop_assert_eq!(meta.op_type(), OpType::Insert);
            prop_assert!(!meta.is_referenced());
        }
    }
}

#[test]
fn leaf_consolidation_changes_prefix_and_skips_requested_key() {
    let mut allocation = TestBasePage::new(4096);
    let leaf = allocation.page();
    leaf.initialize(
        b"tenant/a",
        b"tenant/z",
        4096,
        MiniPageNextLevel::new_null(),
        true,
        false,
        crate::snapshot::INVALID_SNAPSHOT_VERSION,
    );
    assert!(leaf.insert(b"tenant/aa", b"short", OpType::Insert, 0));
    assert!(leaf.insert(b"tenant/ab", &[3; 96], OpType::Insert, 0));
    assert!(leaf.insert(b"tenant/ac", b"delete", OpType::Insert, 0));
    assert!(leaf.insert(b"tenant/ac", &[], OpType::Delete, 0));
    assert!(leaf.insert(b"tenant/ad", b"skip", OpType::Insert, 0));
    leaf.lsn = 42;
    leaf.consolidate_inner(
        OpType::Insert,
        Some(b"tenant/az"),
        true,
        false,
        Some(b"tenant/ad"),
        73,
    );

    assert_eq!(leaf.get_prefix(), b"tenant/a");
    assert_eq!(leaf.get_low_fence_key(), b"tenant/a");
    assert_eq!(leaf.get_high_fence_key(), b"tenant/az");
    assert_eq!(leaf.meta.meta_count_without_fence(), 2);
    assert_eq!(leaf.lsn, 42);
    assert_eq!(leaf.get_clean_snapshot_version(), 73);
    let mut out = [0; 96];
    assert_eq!(
        leaf.read_by_key(b"tenant/aa", &mut out),
        LeafReadResult::Found(5)
    );
    assert_eq!(&out[..5], b"short");
    assert_eq!(
        leaf.read_by_key(b"tenant/ab", &mut out),
        LeafReadResult::Found(96)
    );
    assert_eq!(out, [3; 96]);
    for key in [b"tenant/ac", b"tenant/ad"] {
        assert_eq!(leaf.read_by_key(key, &mut out), LeafReadResult::NotFound);
    }
}

#[test]
fn leaf_insert_rejects_unrepresentable_lengths_without_mutation() {
    let mut allocation = TestBasePage::new(4096);
    let leaf = allocation.page();
    leaf.initialize(
        b"tenant/a",
        b"tenant/z",
        4096,
        MiniPageNextLevel::new_null(),
        true,
        false,
        0,
    );
    assert!(leaf.insert(b"tenant/m", b"original", OpType::Insert, 0));
    let remaining = leaf.meta.remaining_size;
    let count = leaf.meta.meta_count_with_fence();

    for len in [1 << 14, 1 << 16, (1 << 16) + 8] {
        assert!(!leaf.insert(&vec![b'a'; len], b"value", OpType::Insert, 0));
    }
    for len in [1 << 15, 1 << 16, (1 << 16) + 8] {
        assert!(!leaf.insert(b"tenant/m", &vec![1; len], OpType::Insert, 0));
    }
    assert!(!leaf.insert(b"short", b"value", OpType::Insert, 0));

    assert_eq!(leaf.meta.remaining_size, remaining);
    assert_eq!(leaf.meta.meta_count_with_fence(), count);
    let mut out = [0; 8];
    assert_eq!(
        leaf.read_by_key(b"tenant/m", &mut out),
        LeafReadResult::Found(8)
    );
    assert_eq!(&out, b"original");
}

#[test]
fn leaf_base_page_unused_storage_is_initialized() {
    let allocation = TestBasePage::new(4096);
    // Whole pages are serialized by storage. This read also lets Miri check
    // that the header, fence metadata, and unused capacity are initialized.
    let bytes = unsafe { std::slice::from_raw_parts(allocation.0.cast::<u8>(), 4096) };
    let unused_start = std::mem::size_of::<LeafNodeHeader>() + 2 * crate::nodes::KV_META_SIZE;
    assert!(bytes[unused_start..].iter().all(|byte| *byte == 0));
}

#[test]
fn leaf_variable_allocations_preserve_bounds_and_shared_reference_bits() {
    for page_size in [64, 128, 448, 832, 4096] {
        for has_fence in [false, true] {
            let mut allocation = TestBasePage::new(page_size);
            {
                let leaf = allocation.page();
                assert_eq!(std::mem::size_of_val(leaf), page_size);
                leaf.initialize(
                    &[],
                    &[],
                    page_size,
                    MiniPageNextLevel::new_null(),
                    has_fence,
                    false,
                    0,
                );
                let record_index = if has_fence { 2 } else { 0 };
                let value_len = page_size
                    - std::mem::size_of::<LeafNodeHeader>()
                    - (record_index + 1) * crate::nodes::KV_META_SIZE
                    - 1;
                let value = vec![0xA5; value_len];
                assert!(leaf.insert(b"k", &value, OpType::Insert, 0));
                assert_eq!(leaf.meta.remaining_size, 0);

                // The record reaches the last allocated payload byte. Shared
                // page/value borrows must still allow the metadata's atomic
                // reference bit to be updated by a matching lookup.
                let shared_leaf = &*leaf;
                let meta = shared_leaf.get_kv_meta(record_index);
                let borrowed_value = shared_leaf.get_value(meta);
                let mut out = vec![0; value_len];
                assert!(!meta.is_referenced());
                assert_eq!(
                    shared_leaf.read_by_key(b"j", &mut out),
                    LeafReadResult::NotFound
                );
                assert!(!meta.is_referenced());
                assert_eq!(
                    shared_leaf.read_by_key(b"k", &mut out),
                    LeafReadResult::Found(value_len as u32)
                );
                assert!(meta.is_referenced());
                assert_eq!(borrowed_value, value);
                assert_eq!(out, value);

                // Rebuild both a full page and an empty one through the same
                // allocation-backed DST borrow.
                leaf.consolidate(2);
                assert_eq!(leaf.meta.remaining_size, 0);
                assert!(!leaf.get_kv_meta(record_index).is_referenced());
                assert!(leaf.insert(b"k", &[], OpType::Delete, 0));
                leaf.consolidate(4);
                assert_eq!(leaf.meta.meta_count_without_fence(), 0);
                assert!(leaf.insert(b"k", &value, OpType::Insert, 0));
            }

            // End the first mutable borrow before reconstructing a second
            // full-extent reference from the owner-held allocation pointer.
            let leaf = allocation.page();
            assert_eq!(std::mem::size_of_val(leaf), page_size);
            let mut out = vec![0; page_size];
            assert!(matches!(
                leaf.read_by_key(b"k", &mut out),
                LeafReadResult::Found(_)
            ));
            leaf.consolidate(6);
        }
    }
}

#[test]
fn leaf_variable_allocations_upgrade_into_uninitialized_storage() {
    for page_size in [64, 128, 448, 832, 4096] {
        let mut source = TestBasePage::new(page_size);
        let leaf = source.page();
        leaf.initialize(
            &[],
            &[],
            page_size,
            MiniPageNextLevel::new(128),
            false,
            false,
            0,
        );
        assert!(leaf.insert(b"a", b"first", OpType::Insert, 0));
        assert!(leaf.insert(b"z", b"last", OpType::Cache, 0));

        let destination_size = page_size * 2;
        let layout = std::alloc::Layout::from_size_align(
            destination_size,
            std::mem::align_of::<LeafNodeHeader>(),
        )
        .unwrap();
        // Mini-page storage comes from an allocator and need not be zeroed.
        // Only initialized metadata and record ranges may subsequently be read.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let mut destination = TestBasePage(ptr.cast(), destination_size);
        leaf.copy_initialize_to(destination.0, destination_size, false, 7);
        let upgraded = destination.page();
        assert_eq!(std::mem::size_of_val(upgraded), destination_size);
        assert_eq!(upgraded.get_clean_snapshot_version(), 7);
        assert_eq!(upgraded.next_level.as_offset(), 128);
        let gap_start = std::mem::size_of::<LeafNodeHeader>()
            + upgraded.meta.meta_count_with_fence() as usize * crate::nodes::KV_META_SIZE;
        let gap_end = gap_start + upgraded.meta.remaining_size as usize;
        // Read every serialized byte under Miri, including genuinely
        // uninitialized allocator capacity that snapshot_bytes must clear.
        let snapshot = upgraded.snapshot_bytes().to_vec();
        assert_eq!(snapshot.len(), destination_size);
        assert!(snapshot[gap_start..gap_end].iter().all(|byte| *byte == 0));
        let mut out = [0; 5];
        assert_eq!(
            upgraded.read_by_key(b"a", &mut out),
            LeafReadResult::Found(5)
        );
        assert_eq!(&out, b"first");
        assert_eq!(
            upgraded.read_by_key(b"z", &mut out),
            LeafReadResult::Found(4)
        );
        assert_eq!(&out[..4], b"last");
        upgraded.consolidate(8);
        assert_eq!(upgraded.meta.meta_count_without_fence(), 2);
        assert_eq!(
            upgraded.read_by_key(b"z", &mut out),
            LeafReadResult::Found(4)
        );
        assert_eq!(&out[..4], b"last");
    }
}

fn check_leaf_allocations_through_tree(cache_only: bool, record_count: u64) {
    use crate::{BfTree, Config as TreeConfig, LeafInsertResult, ScanReturnField};

    let mut config = TreeConfig::default();
    config
        .cache_only(cache_only)
        .cb_size_byte(128 * 1024)
        .cb_max_record_size(256)
        .cb_max_key_len(8)
        .read_promotion_rate(100)
        .scan_promotion_rate(100)
        .read_record_cache(false)
        .write_load_full_page(true);
    let tree = BfTree::with_config(config, None).unwrap();
    let mut model = BTreeMap::new();
    for id in 0..record_count {
        let key = (id * 2).to_be_bytes().to_vec();
        let value = vec![id as u8; 128];
        assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
        model.insert(key, value);
    }
    if record_count == 96 {
        // More than a page of records exercises allocator-backed splitting
        // and inner-node traversal in addition to the root-leaf lifecycle.
        assert_eq!(
            tree.root_page_id
                .load(crate::sync::atomic::Ordering::Relaxed)
                & BfTree::ROOT_IS_LEAF_MASK,
            0
        );
    }
    let mut out = [0; 264];
    for (key, value) in &model {
        assert_eq!(tree.read(key, &mut out), LeafReadResult::Found(128));
        assert_eq!(&out[..128], value);
    }
    for id in 0..record_count {
        let absent_key = (id * 2 + 1).to_be_bytes();
        assert_eq!(tree.read(&absent_key, &mut out), LeafReadResult::NotFound);
        let key = (id * 2).to_be_bytes();
        if id % 7 == 0 {
            tree.delete(&key);
            model.remove(key.as_slice());
            assert!(matches!(
                tree.read(&key, &mut out),
                LeafReadResult::NotFound | LeafReadResult::Deleted
            ));
        } else if id % 3 == 0 {
            let value = vec![255 - id as u8; 192];
            assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
            assert_eq!(tree.read(&key, &mut out), LeafReadResult::Found(192));
            assert_eq!(&out[..192], value);
            model.insert(key.to_vec(), value);
        }
    }
    {
        let start = 0u64.to_be_bytes();
        let mut scan = tree
            .scan_with_count(
                &start,
                record_count as usize + 1,
                ScanReturnField::KeyAndValue,
            )
            .unwrap();
        for (key, value) in &model {
            assert_eq!(scan.next(&mut out), Some((key.len(), value.len())));
            assert_eq!(&out[..key.len()], key);
            assert_eq!(&out[key.len()..key.len() + value.len()], value);
        }
        assert_eq!(scan.next(&mut out), None);
    }
    // Drop also traverses the real circular buffer's allocation metadata.
    drop(tree);
}

#[test]
fn leaf_allocation_cache_only_tree_lifecycle() {
    check_leaf_allocations_through_tree(true, 96);
}

#[test]
fn leaf_allocation_memory_backed_tree_lifecycle() {
    check_leaf_allocations_through_tree(false, 96);
}

#[test]
fn leaf_allocation_single_leaf_cache_only_lifecycle() {
    check_leaf_allocations_through_tree(true, 16);
}

#[test]
fn leaf_allocation_single_leaf_memory_backed_lifecycle() {
    check_leaf_allocations_through_tree(false, 16);
}

#[test]
fn leaf_circular_buffer_growth_preserves_allocation_metadata() {
    use crate::circular_buffer::CircularBuffer;

    let sizes = crate::BfTree::create_mem_page_size_classes(2, 256, 4096, 16, true);
    let buffer = CircularBuffer::new(16 * 1024, 0.1, 2, 256, 4096, 16, None, true);
    let mut page_size = sizes[0];
    let mut allocation = buffer.alloc(page_size).unwrap();
    LeafNode::initialize_mini_page(
        &allocation,
        page_size,
        MiniPageNextLevel::new_null(),
        true,
        0,
    );
    {
        let leaf = unsafe { &mut *LeafNode::from_raw_parts(allocation.as_ptr().cast(), page_size) };
        assert!(leaf.insert(b"key", b"value", OpType::Insert, 0));
    }
    for (version, &next_size) in sizes[1..].iter().enumerate() {
        let next = buffer.alloc(next_size).unwrap();
        {
            let leaf = unsafe { &*LeafNode::from_raw_parts(allocation.as_ptr().cast(), page_size) };
            leaf.copy_initialize_to(next.as_ptr().cast(), next_size, false, version as u64);
        }
        // Keep the guard's original allocation pointer. A leaf reference only
        // covers page bytes and cannot be used to reach preceding AllocMeta.
        let old_ptr = allocation.as_ptr();
        drop(allocation);
        let handle = unsafe { buffer.acquire_exclusive_dealloc_handle(old_ptr).unwrap() };
        buffer.dealloc(handle);
        allocation = next;
        page_size = next_size;

        let leaf = unsafe { &mut *LeafNode::from_raw_parts(allocation.as_ptr().cast(), page_size) };
        assert_eq!(std::mem::size_of_val(leaf), page_size);
        assert_eq!(leaf.get_clean_snapshot_version(), version as u64);
        let mut out = [0; 5];
        assert_eq!(leaf.read_by_key(b"key", &mut out), LeafReadResult::Found(5));
        assert_eq!(&out, b"value");
        leaf.consolidate(version as u64);
        assert_eq!(leaf.snapshot_bytes().to_vec().len(), page_size);
    }
    let ptr = allocation.as_ptr();
    drop(allocation);
    let handle = unsafe { buffer.acquire_exclusive_dealloc_handle(ptr).unwrap() };
    buffer.dealloc(handle);
    // CircularBuffer::drop verifies every real allocation is tombstoned or
    // freelisted, and Miri validates the payload/metadata borrow separation.
}

#[test]
fn leaf_empty_prefix_and_empty_consolidation_are_valid() {
    let mut allocation = TestBasePage::new(4096);
    let leaf = allocation.page();
    assert_eq!(leaf.lsn, 0);
    assert!(leaf.get_prefix().is_empty());
    leaf.consolidate(1);
    assert_eq!(leaf.meta.meta_count_without_fence(), 0);

    // A fresh fence-less page has no initialized record metadata to inspect.
    leaf.initialize(
        &[],
        &[],
        4096,
        MiniPageNextLevel::new_null(),
        false,
        false,
        2,
    );
    assert!(leaf.get_prefix().is_empty());
    leaf.consolidate(3);
    assert_eq!(leaf.meta.meta_count_without_fence(), 0);
    assert!(leaf.get_prefix().is_empty());
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
