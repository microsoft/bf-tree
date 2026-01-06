// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    common::{murmur64, Distribution, MicroBenchResult, Sampler, Workload, WorkloadMix},
    wrappers::{BfTreeWrapper, BwTreeWrapper, PageCacheWrapper},
    KvStore,
};
use bf_tree::{metric::Timer, timer};
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use shumai::{config, ShumaiBench};

#[derive(Serialize, Clone, Debug, Deserialize)]
pub enum SystemUnderTest {
    BfTree,
    RocksDB,
    ReDB,
    Sled,
    Lmdb,
    Splinterdb,
    PageCache,
    BwTree,
}

#[config(path = "bench_e2e.toml")]
pub struct E2EBench {
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
    pub sut: SystemUnderTest,
}

struct TestBench<V: KvStore> {
    store: V,
    config: E2EBench,
    rand_sampler: Sampler,
}

impl<V: KvStore> TestBench<V> {
    fn new(c: &E2EBench) -> Self {
        let memory_size = c.memory_size_mb * 1024 * 1024;
        assert!(
            c.key_len.is_multiple_of(8),
            "key_len must be multiple of 8!"
        );

        _ = std::fs::remove_file(&c.file_path);
        _ = std::fs::remove_dir_all(&c.file_path);

        let sampler = Sampler::from(&c.distribution, 0..c.record_cnt);

        Self {
            store: V::new(&c.file_path, memory_size),
            config: c.clone(),
            rand_sampler: sampler,
        }
    }
}

pub(crate) fn install_value_to_buffer(buffer: &mut [usize], key_id: usize) -> &[u8] {
    let murmured = murmur64(key_id);
    for i in buffer.iter_mut() {
        *i = murmured;
    }

    unsafe {
        let ptr = buffer.as_ptr();
        std::slice::from_raw_parts(ptr as *const u8, buffer.len() * 8)
    }
}

impl<V: KvStore> ShumaiBench for TestBench<V> {
    type Config = E2EBench;
    type Result = MicroBenchResult;

    fn run(&self, context: shumai::Context<Self::Config>) -> MicroBenchResult {
        let mut small_rng = SmallRng::from_os_rng();
        let mut key_buffer = vec![0; self.config.key_len / 8];
        let mut value_buffer: Vec<u8> = vec![0; self.config.key_len];
        let mut op_cnt = 0;

        self.store.register_thread();

        context.wait_for_start();

        while context.is_running() {
            let op = self.config.workload_mix.gen(&mut small_rng);
            match op {
                Workload::Read => {
                    timer!(Timer::Read);
                    let key_id = self.rand_sampler.sample(&mut small_rng);
                    let key = install_value_to_buffer(&mut key_buffer, key_id);

                    let cnt = self.store.read(key, &mut value_buffer);
                    assert_eq!(cnt as usize, key.len());
                    assert_eq!(key, &value_buffer);
                    op_cnt += 1;
                }
                Workload::NegativeRead => {}
                Workload::Scan => {
                    let key_id = self.rand_sampler.sample(&mut small_rng);
                    let key = install_value_to_buffer(&mut key_buffer, key_id);

                    self.store
                        .scan(key, self.config.scan_cnt, &mut value_buffer);
                    op_cnt += 1;
                }
                Workload::Update => {
                    let key_id = self.rand_sampler.sample(&mut small_rng);
                    let key = install_value_to_buffer(&mut key_buffer, key_id);

                    self.store.update(key, key);
                    op_cnt += 1;
                }
                Workload::Insert => {
                    panic!("insert is not implemented for e2e benchmark")
                }
            }
        }

        self.store.deregister_thread();

        MicroBenchResult::new(op_cnt, None)
    }

    fn cleanup(&mut self) -> Option<serde_json::Value> {
        None
    }

    fn load(&mut self) -> Option<serde_json::Value> {
        let loading_thread = 16;

        let total_record = self.config.record_cnt;
        let record_per_thread = total_record / loading_thread;
        assert_eq!(total_record % loading_thread, 0);

        let loaded = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let mut handles = vec![];

            for t in 0..loading_thread {
                let tree = &self.store;
                let key_len = self.config.key_len;
                let loaded_ref = &loaded;
                let h = s.spawn(move || {
                    tree.register_thread();
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
                    tree.deregister_thread();
                });
                handles.push(h);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });

        Some(serde_json::json!({}))
    }
}

pub fn run_e2e_bench(c: E2EBench) {
    let results = match c.sut {
        SystemUnderTest::BfTree => {
            let mut bench = TestBench::<BfTreeWrapper>::new(&c);
            shumai::run(&mut bench, &c, c.repeat)
        }
        SystemUnderTest::RocksDB => {
            #[cfg(feature = "rocksdb")]
            {
                use crate::wrappers::RocksDbWrapper;
                let mut bench = TestBench::<RocksDbWrapper>::new(&c);
                shumai::run(&mut bench, &c, c.repeat)
            }
            #[cfg(not(feature = "rocksdb"))]
            {
                panic!("RocksDB is not enabled in the build, run with --features `rocksdb`!")
            }
        }
        SystemUnderTest::ReDB => {
            #[cfg(feature = "redb")]
            {
                use crate::wrappers::RedbWrapper;
                let mut bench = TestBench::<RedbWrapper>::new(&c);
                shumai::run(&mut bench, &c, c.repeat)
            }
            #[cfg(not(feature = "redb"))]
            {
                panic!("ReDB is not enabled in the build, run with --features `redb`!")
            }
        }
        SystemUnderTest::Sled => {
            #[cfg(feature = "sled")]
            {
                use crate::wrappers::SledWrapper;
                let mut bench = TestBench::<SledWrapper>::new(&c);
                shumai::run(&mut bench, &c, c.repeat)
            }
            #[cfg(not(feature = "sled"))]
            {
                panic!("Sled is not enabled in the build, run with --features `sled`!")
            }
        }
        SystemUnderTest::Lmdb => {
            #[cfg(feature = "lmdb")]
            {
                use crate::wrappers::LmdbWrapper;
                let mut bench = TestBench::<LmdbWrapper>::new(&c);
                shumai::run(&mut bench, &c, c.repeat)
            }
            #[cfg(not(feature = "lmdb"))]
            {
                panic!("LMDB is not enabled in the build, run with --features `lmdb`!")
            }
        }
        SystemUnderTest::Splinterdb => {
            // let mut bench = TestBench::<SplinterdbWrapper>::new(&c);
            // shumai::run(&mut bench, &c, c.repeat)
            todo!("Splinterdb is not implemented yet!")
        }
        SystemUnderTest::PageCache => {
            let mut bench = TestBench::<PageCacheWrapper>::new(&c);
            shumai::run(&mut bench, &c, c.repeat)
        }
        SystemUnderTest::BwTree => {
            let mut bench = TestBench::<BwTreeWrapper>::new(&c);
            shumai::run(&mut bench, &c, c.repeat)
        }
    };
    results.write_json().expect("Failed to write results!");
}
