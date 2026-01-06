// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::sync::atomic::{AtomicUsize, Ordering};

use bf_tree::{
    circular_buffer::CircularBufferMetrics, metric::Timer, timer, BfTree, LeafReadResult,
    ScanReturnField,
};
use rand::{rngs::SmallRng, SeedableRng};
use shumai::{config, ShumaiBench};

use crate::{
    bench_e2e::install_value_to_buffer,
    common::{Distribution, MicroBenchResult, Sampler, StorageBackend, Workload, WorkloadMix},
};

#[config(path = "bench_bftree.toml")]
pub struct BfTreeBench {
    pub name: String,
    pub threads: Vec<usize>,
    pub time: usize,
    pub repeat: usize,
    pub record_cnt: usize,
    #[matrix]
    pub distribution: Distribution,
    #[matrix]
    pub workload_mix: WorkloadMix,
    #[matrix]
    pub memory_size_mb: usize,
    pub file_path: String,
    pub key_len: usize, // must be multiple of 8
    pub scan_cnt: usize,
    #[matrix]
    pub storage: StorageBackend,
    #[matrix]
    pub read_promotion_rate: u64,
    #[matrix]
    pub copy_on_access_ratio: f64,
}

struct TestBench {
    bftree: BfTree,
    config: BfTreeBench,
    positive_sampler: Sampler, // sample only records that are inserted in load phase
    all_sampler: Sampler,      // sample records from all possible records
}

impl TestBench {
    fn new(c: &BfTreeBench) -> Self {
        let memory_size = c.memory_size_mb * 1024 * 1024;
        assert!(
            c.key_len.is_multiple_of(8),
            "key_len must be multiple of 8!"
        );

        _ = std::fs::remove_file(&c.file_path);
        _ = std::fs::remove_dir_all(&c.file_path);

        let positive_sampler = Sampler::from(&c.distribution, 0..c.record_cnt);
        let all_sampler = Sampler::from(&c.distribution, 0..usize::MAX);

        let mut config = bf_tree::Config::new(&c.file_path, memory_size);
        config.read_promotion_rate(c.read_promotion_rate as usize);
        config.storage_backend(c.storage.into());
        config.cb_copy_on_access_ratio(c.copy_on_access_ratio);
        let bf_tree = BfTree::with_config(config, None).unwrap();

        Self {
            bftree: bf_tree,
            config: c.clone(),
            positive_sampler,
            all_sampler,
        }
    }
}

impl ShumaiBench for TestBench {
    type Config = BfTreeBench;
    type Result = MicroBenchResult;

    fn run(&self, context: shumai::Context<Self::Config>) -> MicroBenchResult {
        let mut small_rng = SmallRng::from_os_rng();
        let mut key_buffer = vec![0; self.config.key_len / 8];
        let mut value_buffer: Vec<u8> = vec![0; self.config.key_len];
        let mut op_cnt = 0;

        bf_tree::metric::get_tls_recorder().reset();

        context.wait_for_start();

        while context.is_running() {
            let op = self.config.workload_mix.gen(&mut small_rng);
            timer!(Timer::Read);
            match op {
                Workload::Read => {
                    let key_id = self.positive_sampler.sample(&mut small_rng);
                    let key = install_value_to_buffer(&mut key_buffer, key_id);

                    let cnt = self.bftree.read(key, &mut value_buffer);
                    match cnt {
                        LeafReadResult::Found(v) => {
                            assert_eq!(v as usize, key.len());
                            assert_eq!(key, &value_buffer);
                        }
                        _ => {
                            panic!("Missing key");
                        }
                    }
                    op_cnt += 1;
                }
                Workload::NegativeRead => {}
                Workload::Scan => {
                    let key_id = self.positive_sampler.sample(&mut small_rng);
                    let key = install_value_to_buffer(&mut key_buffer, key_id);

                    let mut iter = self
                        .bftree
                        .scan_with_count(key, self.config.scan_cnt, ScanReturnField::Value)
                        .expect("Failed to create scan iterator");

                    while let Some((key_len, value_len)) = iter.next(&mut value_buffer) {
                        assert!(value_len <= value_buffer.len());
                        assert!(key_len == 0);
                    }

                    op_cnt += 1;
                }
                Workload::Update => {
                    let key_id = self.positive_sampler.sample(&mut small_rng);
                    let key = install_value_to_buffer(&mut key_buffer, key_id);

                    self.bftree.insert(key, key);
                    op_cnt += 1;
                }
                Workload::Insert => {
                    let key_id = self.all_sampler.sample(&mut small_rng);
                    let key = install_value_to_buffer(&mut key_buffer, key_id);

                    self.bftree.insert(key, key);
                    op_cnt += 1;
                }
            }
        }

        let metric = if cfg!(feature = "metrics-rt") {
            Some(bf_tree::metric::get_tls_recorder().clone())
        } else {
            None
        };

        MicroBenchResult::new(op_cnt, metric)
    }

    fn cleanup(&mut self) -> Option<serde_json::Value> {
        None
    }

    fn on_thread_finished(&mut self, _cur_thread: usize) -> Option<serde_json::Value> {
        let metrics = self.bftree.get_buffer_metrics();
        Some(serde_json::json!({
            "circular_buffer_metrics": metrics,
        }))
    }

    fn load(&mut self) -> Option<serde_json::Value> {
        let mut metrics = bf_tree::metric::TlsRecorder::default();
        let loading_thread = 16;

        let total_record = self.config.record_cnt;
        let record_per_thread = total_record / loading_thread;
        assert_eq!(total_record % loading_thread, 0);

        let loaded = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let mut handles = vec![];

            for t in 0..loading_thread {
                let tree = &self.bftree;
                let key_len = self.config.key_len;
                let loaded_ref = &loaded;
                let h = s.spawn(move || {
                    bf_tree::metric::get_tls_recorder().reset();
                    let mut buffer = vec![0; key_len / 8];

                    let start = t * record_per_thread;
                    let end = start + record_per_thread;
                    let print_step = record_per_thread / 4;
                    for i in start..end {
                        let key = install_value_to_buffer(&mut buffer, i);
                        tree.insert(key, key);

                        if i % print_step == 0 {
                            loaded_ref.fetch_add(print_step, Ordering::Relaxed);
                            let loaded = loaded_ref.load(Ordering::Relaxed);
                            println!("Loading: {loaded}/{total_record}");
                        }
                    }
                    bf_tree::metric::get_tls_recorder().clone()
                });
                handles.push(h);
            }

            for handle in handles {
                metrics += handle.join().unwrap();
            }
        });

        let metrics = if cfg!(feature = "metrics-rt") {
            Some(metrics)
        } else {
            None
        };

        let circular_buffer_metrics: Option<CircularBufferMetrics> = {
            #[cfg(feature = "metrics-rt")]
            {
                Some(self.bftree.get_buffer_metrics())
            }
            #[cfg(not(feature = "metrics-rt"))]
            {
                None
            }
        };
        Some(serde_json::json!({
            "metrics": metrics,
            "circular_buffer_metrics": circular_buffer_metrics,
        }))
    }
}

pub fn run_bftree_bench(c: BfTreeBench) {
    let mut bench = TestBench::new(&c);
    let results = shumai::run(&mut bench, &c, c.repeat);
    results.write_json().expect("Failed to write json!");

    // Output debug metrics, if any
    let metrics = bench.bftree.get_metrics();
    if let Some(t) = metrics {
        println!("Metrics: {}", t);
    }
}
