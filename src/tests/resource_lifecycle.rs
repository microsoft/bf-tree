// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use super::stateful_tree::{check_read, check_state, drain, Model};
use crate::{
    sync::{thread, Arc, Barrier},
    BfTree, Config, LeafInsertResult, StorageBackend,
};

fn resources() -> (usize, usize) {
    (
        std::fs::read_dir("/proc/self/task").unwrap().count(),
        std::fs::read_dir("/proc/self/fd").unwrap().count(),
    )
}

#[test]
fn resource_lifecycle_cycles() {
    // Exact process counts require a serial test process, not parallel harness workers.
    let baseline = std::env::var_os("BFTREE_RESOURCE_CHECK").map(|_| resources());
    for cycle in 0..3 {
        let directory = tempfile::tempdir().unwrap();
        let mut config = Config::new(directory.path().join("live"), 128 * 1024);
        config
            .storage_backend(StorageBackend::Std)
            .use_snapshot(true)
            .cb_min_record_size(16)
            .cb_max_record_size(1024)
            .cb_max_key_len(8)
            .read_promotion_rate(100)
            .scan_promotion_rate(100);
        let tree = Arc::new(BfTree::with_config(config, None).unwrap());
        let mut model = Model::new();
        for generation in 0..4 {
            let start = Arc::new(Barrier::new(4));
            let workers: Vec<_> = (0..4)
                .map(|worker| {
                    let (tree, start) = (tree.clone(), start.clone());
                    thread::spawn(move || {
                        start.wait();
                        let mut changes = Vec::new();
                        for index in (worker..512).step_by(4) {
                            let id = (generation % 3) * 512 + index;
                            let key = (id as u64).to_be_bytes().to_vec();
                            let size = [24, 120, 504, 1000][(index + generation) % 4];
                            let mut value = vec![generation as u8; size];
                            value[..8].copy_from_slice(&(id as u64).to_be_bytes());
                            assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
                            check_read(&tree, &Model::from([(key.clone(), value.clone())]), &key)
                                .unwrap();
                            if (index + generation) % 3 == 0 {
                                tree.delete(&key);
                                check_read(&tree, &Model::new(), &key).unwrap();
                                changes.push((key, None));
                            } else {
                                changes.push((key, Some(value)));
                            }
                        }
                        changes
                    })
                })
                .collect();
            let results: Vec<_> = workers.into_iter().map(|worker| worker.join()).collect();
            for result in results {
                for (key, value) in result.unwrap() {
                    if let Some(value) = value {
                        model.insert(key, value);
                    } else {
                        model.remove(&key);
                    }
                }
            }
            check_state(&tree, &model).unwrap();
        }
        assert!(
            model
                .iter()
                .map(|(key, value)| key.len() + value.len())
                .sum::<usize>()
                > 128 * 1024
        );
        let occupied = tree.storage.circular_buffer.get_metrics();
        assert!(occupied.head_tail_distance <= occupied.capacity);
        assert_eq!(occupied.not_ready_cnt, 0);
        assert_eq!(occupied.begin_tombstone_cnt, 0);
        drain(&tree);
        let drained = tree.storage.circular_buffer.get_metrics();
        assert_eq!(drained.head_tail_distance, 0);
        assert_eq!(drained.allocated_cnt, 0);
        check_state(&tree, &model).unwrap();
        let snapshot = directory.path().join("snapshot");
        tree.cpr_snapshot(&snapshot);
        assert_eq!(Arc::strong_count(&tree), 1);
        drop(tree);
        let recovered = BfTree::new_from_cpr_snapshot(&snapshot, false, None, None, None).unwrap();
        check_state(&recovered, &model).unwrap();
        drain(&recovered);
        assert_eq!(
            recovered
                .storage
                .circular_buffer
                .get_metrics()
                .allocated_cnt,
            0
        );
        drop(recovered);
        directory.close().unwrap();
        if let Some(baseline) = baseline {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while resources() != baseline {
                assert!(
                    std::time::Instant::now() < deadline,
                    "resource drift after cycle {cycle}: {baseline:?} -> {:?}",
                    resources()
                );
                thread::yield_now();
            }
        }
    }
}
