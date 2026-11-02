// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

use crate::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, thread, Arc,
    },
    utils::{BfsVisitor, NodeInfo},
    BfTree, Config, LeafInsertResult, ScanReturnField, StorageBackend,
};

type Rows = Vec<(u64, Vec<u8>)>;
type Versions = BTreeMap<u64, BTreeSet<Vec<u8>>>;
type NextRecord<'a> = dyn FnMut(&mut [u8]) -> Option<(usize, usize)> + 'a;

fn check(
    rows: &Rows,
    allowed: &Versions,
    stable: &BTreeSet<u64>,
    bounds: (u64, u64),
) -> Result<(), &'static str> {
    let mut previous = None;
    let mut seen = BTreeSet::new();
    for (key, value) in rows {
        if *key < bounds.0 || *key > bounds.1 {
            return Err("out-of-range key");
        }
        if previous.is_some_and(|last| last >= *key) {
            return Err("unordered or duplicate key");
        }
        if !allowed
            .get(key)
            .is_some_and(|values| values.contains(value))
        {
            return Err("invented, future or torn key/value pair");
        }
        previous = Some(*key);
        seen.insert(*key);
    }
    if !stable.is_subset(&seen) {
        return Err("unchanged key omitted");
    }
    Ok(())
}

#[test]
fn scan_oracle_accepts_mixed_versions_and_rejects_broken_results() {
    let allowed = Versions::from([
        (1, BTreeSet::from([vec![1, 1], vec![2, 2]])),
        (2, BTreeSet::from([vec![1, 1], vec![2, 2]])),
    ]);
    let stable = BTreeSet::from([1]);
    assert_eq!(
        check(
            &vec![(1, vec![1, 1]), (2, vec![2, 2])],
            &allowed,
            &stable,
            (1, 2)
        ),
        Ok(())
    );
    for invalid in [
        vec![(1, vec![1, 1]), (1, vec![1, 1])],
        vec![(2, vec![2, 2]), (1, vec![1, 1])],
        vec![(0, vec![1, 1]), (1, vec![1, 1])],
        vec![(2, vec![2, 2])],
        vec![(1, vec![1, 2])],
        vec![(1, vec![3, 3])],
        vec![(1, vec![1, 1]), (3, vec![1, 1])],
    ] {
        assert!(check(&invalid, &allowed, &stable, (1, 2)).is_err());
    }
}

fn value(key: u64, version: u8) -> Vec<u8> {
    let mut result = vec![version; [128, 248, 504][usize::from(version)]];
    result[..8].copy_from_slice(&key.to_be_bytes());
    for (index, byte) in result[8..].iter_mut().enumerate() {
        *byte = (key as u8).wrapping_add(index as u8) ^ version;
    }
    result
}

fn tick(clock: &AtomicUsize) -> usize {
    clock.fetch_add(1, Ordering::SeqCst)
}

#[rstest::rstest]
fn weak_scan_histories(#[values(false, true)] exclusive: bool) {
    let mut config = Config::default();
    config
        .storage_backend(StorageBackend::Memory)
        .cache_only(false)
        .cb_size_byte(2 * 1024 * 1024)
        .cb_min_record_size(16)
        .cb_max_record_size(1024)
        .cb_max_key_len(8)
        .read_promotion_rate(100);
    let tree = Arc::new(BfTree::with_config(config, None).unwrap());
    let initial: BTreeMap<_, _> = (0..256).map(|key| (key, value(key, 0))).collect();
    for (key, value) in &initial {
        assert_eq!(
            tree.insert(&key.to_be_bytes(), value),
            LeafInsertResult::Success
        );
    }
    let count_leaves = || {
        BfsVisitor::new_all_nodes(&tree)
            .filter(|node| matches!(node, NodeInfo::Leaf { .. }))
            .count()
    };
    let leaves_before = count_leaves();
    assert!(
        leaves_before > 1,
        "low scan leaf must be separate from high mutation region"
    );
    let (start_tx, start_rx) = mpsc::channel();
    let (batch_tx, batch_rx) = mpsc::channel();
    let clock = Arc::new(AtomicUsize::new(0));
    let writer = {
        let (tree, clock) = (tree.clone(), clock.clone());
        thread::spawn(move || {
            start_rx.recv_timeout(Duration::from_secs(30)).unwrap();
            let mut events = Vec::new();
            let mut put = |key: u64, version| {
                let bytes = value(key, version);
                let invoked = tick(&clock);
                assert_eq!(
                    tree.insert(&key.to_be_bytes(), &bytes),
                    LeafInsertResult::Success
                );
                events.push((invoked, key, Some(bytes)));
                thread::yield_now();
            };
            // These writes cannot wait on the scanner's held low leaf.
            put(240, 1);
            for key in 256..560 {
                put(key, 0);
            }
            batch_tx.send(()).unwrap();
            for key in (128_u64..256).step_by(8) {
                let invoked = tick(&clock);
                tree.delete(&key.to_be_bytes());
                events.push((invoked, key, None));
                thread::yield_now();
                let bytes = value(key, 2);
                let invoked = tick(&clock);
                assert_eq!(
                    tree.insert(&key.to_be_bytes(), &bytes),
                    LeafInsertResult::Success
                );
                events.push((invoked, key, Some(bytes)));
                thread::yield_now();
            }
            events
        })
    };
    let scan_start = tick(&clock);
    let collect = |next: &mut NextRecord<'_>| {
        let mut rows = Vec::new();
        let mut output = [0; 2048];
        while let Some((key_len, value_len)) = next(&mut output) {
            assert_eq!(key_len, 8);
            let key = u64::from_be_bytes(output[..8].try_into().unwrap());
            rows.push((key, output[8..8 + value_len].to_vec()));
            if rows.len() == 1 {
                start_tx.send(()).unwrap();
                batch_rx.recv_timeout(Duration::from_secs(30)).unwrap();
            }
            thread::yield_now();
        }
        rows
    };
    let start_key = 16_u64.to_be_bytes();
    let end_key = 511_u64.to_be_bytes();
    let rows = if exclusive {
        let mut scan = tree
            .scan_mut_with_end_key(&start_key, &end_key, ScanReturnField::KeyAndValue)
            .unwrap();
        collect(&mut |out| scan.next(out))
    } else {
        let mut scan = tree
            .scan_with_end_key(&start_key, &end_key, ScanReturnField::KeyAndValue)
            .unwrap();
        collect(&mut |out| scan.next(out))
    };
    let scan_end = tick(&clock);
    let events = writer.join().unwrap();
    assert!(events
        .iter()
        .any(|(invoked, _, _)| scan_start < *invoked && *invoked < scan_end));
    assert!(
        count_leaves() > leaves_before,
        "overlap must include structural growth"
    );
    let mut allowed: Versions = initial
        .iter()
        .map(|(&key, value)| (key, BTreeSet::from([value.clone()])))
        .collect();
    let mut stable: BTreeSet<_> = initial
        .keys()
        .copied()
        .filter(|key| (16..=511).contains(key))
        .collect();
    let mut final_model = initial;
    for (invoked, key, bytes) in events {
        if invoked < scan_end {
            stable.remove(&key);
            if let Some(bytes) = &bytes {
                allowed.entry(key).or_default().insert(bytes.clone());
            }
        }
        if let Some(bytes) = bytes {
            final_model.insert(key, bytes);
        } else {
            final_model.remove(&key);
        }
    }
    assert_eq!(
        check(&rows, &allowed, &stable, (16, 511)),
        Ok(()),
        "rows={rows:?}"
    );
    let model = final_model
        .into_iter()
        .map(|(key, value)| (key.to_be_bytes().to_vec(), value))
        .collect();
    super::stateful_tree::check_state(&tree, &model).unwrap();
}
