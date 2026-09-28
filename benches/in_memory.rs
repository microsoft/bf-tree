// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Portable, deterministic public-API workloads. See doc/performance.md.
use std::{hint::black_box, time::Instant};

use bf_tree::{BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

const RECORDS: usize = 16_384;
const READ_PASSES: usize = 16;
const UPDATE_PASSES: usize = 4;

fn tree(key_len: usize) -> BfTree {
    let mut config = Config::default();
    config
        .cache_only(true)
        .cb_size_byte(64 * 1024 * 1024)
        .cb_max_key_len(key_len)
        .read_promotion_rate(0)
        .scan_promotion_rate(0);
    BfTree::with_config(config, None).unwrap()
}

fn keys(len: usize, common_prefix: bool) -> Vec<Vec<u8>> {
    (0..RECORDS)
        .map(|i| {
            let mut key = vec![b'p'; len];
            let offset = if common_prefix { len - 8 } else { 0 };
            key[offset..offset + 8].copy_from_slice(&(2 * i as u64).to_be_bytes());
            key
        })
        .collect()
}

fn measure(name: &str, samples: usize, mut run: impl FnMut() -> f64) {
    // Discard one complete warmup; each update/insert sample starts with a fresh tree.
    black_box(run());
    let mut times: Vec<_> = (0..samples).map(|_| run()).collect();
    times.sort_by(f64::total_cmp);
    println!(
        "{name},{:.3},{:.3},{:.3}",
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
}

fn main() {
    let samples = std::env::var("BFTREE_BENCH_SAMPLES")
        .map(|s| {
            s.parse::<usize>()
                .expect("BFTREE_BENCH_SAMPLES must be an integer")
        })
        .unwrap_or(9);
    assert!(samples > 0);
    let filter = std::env::var("BFTREE_BENCH_FILTER").unwrap_or_default();
    let mut order: Vec<_> = (0..RECORDS).collect();
    order.shuffle(&mut StdRng::seed_from_u64(0x000B_F7EE));
    println!("workload,median_ns_per_op,min_ns_per_op,max_ns_per_op");

    for (len, prefix) in [(8, false), (32, true), (128, true)] {
        let keys = keys(len, prefix);
        let mut missing = keys.clone();
        for key in &mut missing {
            let offset = if prefix { len - 1 } else { 7 };
            key[offset] |= 1;
        }
        let label = format!("k{len}");
        let value = [0xA5; 32];
        let populated = || {
            let t = tree(len);
            for &i in &order {
                assert_eq!(t.insert(&keys[i], &value), LeafInsertResult::Success);
            }
            t
        };

        let name = format!("insert/{label}");
        if name.contains(&filter) {
            measure(&name, samples, || {
                let t = tree(len);
                let start = Instant::now();
                for &i in &order {
                    black_box(t.insert(black_box(&keys[i]), black_box(&value)));
                }
                let elapsed = start.elapsed().as_nanos() as f64 / RECORDS as f64;
                let mut out = [0; 32];
                for key in &keys {
                    assert_eq!(t.read(key, &mut out), LeafReadResult::Found(32));
                    assert_eq!(out, value);
                }
                elapsed
            });
        }

        for (kind, queries, expected) in [
            ("read_hit", &keys, LeafReadResult::Found(32)),
            ("read_miss", &missing, LeafReadResult::NotFound),
        ] {
            let name = format!("{kind}/{label}");
            if name.contains(&filter) {
                let t = populated();
                let mut out = [0; 32];
                for key in queries {
                    assert_eq!(t.read(key, &mut out), expected);
                }
                measure(&name, samples, || {
                    let start = Instant::now();
                    for _ in 0..READ_PASSES {
                        for &i in &order {
                            black_box(t.read(black_box(&queries[i]), black_box(&mut out)));
                        }
                    }
                    start.elapsed().as_nanos() as f64 / (RECORDS * READ_PASSES) as f64
                });
            }
        }

        let name = format!("update_grow/{label}");
        if name.contains(&filter) {
            measure(&name, samples, || {
                let t = populated();
                let start = Instant::now();
                for pass in 0..UPDATE_PASSES {
                    let value = vec![pass as u8; 64 + 32 * pass];
                    for &i in &order {
                        black_box(t.insert(black_box(&keys[i]), black_box(&value)));
                    }
                }
                let elapsed = start.elapsed().as_nanos() as f64 / (RECORDS * UPDATE_PASSES) as f64;
                let mut out = [0; 160];
                for key in &keys {
                    assert_eq!(t.read(key, &mut out), LeafReadResult::Found(160));
                    assert_eq!(out, [3; 160]);
                }
                elapsed
            });
        }

        let name = format!("scan/{label}");
        if name.contains(&filter) {
            let t = populated();
            let mut out = vec![0; len + value.len()];
            measure(&name, samples, || {
                let start = Instant::now();
                let mut count = 0;
                for _ in 0..READ_PASSES {
                    let mut iter = t
                        .scan_with_count(&keys[0], RECORDS, ScanReturnField::KeyAndValue)
                        .unwrap();
                    while let Some(result) = iter.next(black_box(&mut out)) {
                        black_box(result);
                        count += 1;
                    }
                }
                let elapsed = start.elapsed().as_nanos() as f64 / count as f64;
                assert_eq!(count, RECORDS * READ_PASSES);
                elapsed
            });
        }
    }
}
