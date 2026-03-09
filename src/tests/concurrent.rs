// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use rand_test::{
    distr::{Distribution, StandardUniform, Uniform},
    rngs::StdRng,
    SeedableRng,
};
use std::hint::black_box;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::nodes::leaf_node::LeafReadResult;
use crate::sync::thread;
use crate::LeafInsertResult;
use crate::{BfTree, Config, StorageBackend};

#[cfg(feature = "shuttle")]
use shuttle::rand::{thread_rng, Rng};

#[cfg(not(feature = "shuttle"))]
use rand::{thread_rng, Rng};

fn make_key(key: u32, len: usize) -> Vec<u8> {
    let bytes = key.to_ne_bytes();
    bytes.into_iter().cycle().take(len).collect::<Vec<_>>()
}

fn sanity_check_value(value: &[u8], expected_len: usize) {
    assert!(value.len() == expected_len);
    let ptr = value.as_ptr() as *const u32;
    let first_v = unsafe { *ptr };
    let u32_cnt = expected_len / std::mem::size_of::<u32>();
    for i in 0..u32_cnt {
        let v = unsafe { *ptr.add(i) };
        assert_eq!(v, first_v);
    }
}
#[test]
fn concurrent_ops() {
    let mut bf_tree_config = Config::default();
    bf_tree_config.cb_size_byte(4096 * 2);
    bf_tree_config.cb_max_key_len(400);
    bf_tree_config.cb_max_record_size(800);

    let bf_tree = Arc::new(BfTree::with_config(bf_tree_config, None).unwrap());

    const OP_RANGE: usize = 4;

    let config = ConcurrentConfig::new_large();

    let mut join_handles = Vec::with_capacity(config.thread_cnt);

    for _ in 0..config.thread_cnt {
        let tree_clone = bf_tree.clone();
        let handle = thread::spawn(move || {
            let mut buffer = vec![0u8; 4096];

            let mut rng = thread_rng();
            let current_tid = thread::current().id();
            black_box(current_tid);

            for op_n in 0..config.op_cnt_per_thread {
                black_box(op_n);
                match rng.gen_range(0..OP_RANGE) {
                    0..=1 => {
                        // insert
                        let key = rng.gen_range(0..config.key_range);
                        let kv = make_key(key, config.key_len);
                        let _unused = tree_clone.insert(&kv, &kv);
                    }
                    2 => {
                        // read
                        let key = rng.gen_range(0..config.key_range);
                        let kv = make_key(key, config.key_len);
                        let cnt = tree_clone.read(&kv, &mut buffer);

                        match cnt {
                            LeafReadResult::Found(v) => {
                                sanity_check_value(&buffer[..v as usize], kv.len());
                            }
                            _ => {}
                        }
                    }
                    3 => {
                        // delete
                        let key = rng.gen_range(0..config.key_range);
                        let kv = make_key(key, config.key_len);
                        tree_clone.delete(&kv);
                    }
                    _ => unreachable!(),
                }
            }
        });
        join_handles.push(handle);
    }

    for h in join_handles {
        h.join().unwrap();
    }
}

struct ConcurrentConfig {
    thread_cnt: usize,
    op_cnt_per_thread: usize,
    key_range: u32,
    key_len: usize,
}

impl ConcurrentConfig {
    #[allow(dead_code)]
    fn new_large() -> Self {
        Self {
            thread_cnt: 3,
            op_cnt_per_thread: 400,
            key_range: 1_000,
            key_len: 400,
        }
    }

    #[allow(dead_code)]
    fn new_small() -> Self {
        Self {
            thread_cnt: 2,
            op_cnt_per_thread: 200,
            key_range: 200,
            key_len: 400,
        }
    }
}

#[test]
fn concurrent_stress_test_cache_only_mode() {
    let finish = Arc::new(AtomicBool::new(false));

    let mut config = Config::default();
    config
        .storage_backend(StorageBackend::Memory)
        .cache_only(true)
        .cb_size_byte(16384) // 16KB, 4 leaf pages
        .file_path(":memory:");

    let cache = Arc::new(BfTree::with_config(config, None).unwrap());

    let num_threads = 32;
    let read_rate = 0.8;
    let incremental_delete_rate = 0.9;
    let id_range = 1000000;
    let dim = 128;
    let duration = std::time::Duration::from_secs(20);

    // Spawn threads to emit mixed read + write + delete workloads.
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let finish_clone = finish.clone();
            let cache_clone = cache.clone();
            std::thread::spawn(move || {
                let mut rng = StdRng::from_os_rng();
                let id_range = Uniform::new(0, id_range).unwrap();

                let mut v = vec![0usize; dim];
                while !finish_clone.load(Ordering::Relaxed) {
                    let i: f64 = StandardUniform {}.sample(&mut rng);
                    let id: usize = id_range.sample(&mut rng);

                    // Fill the cache if:
                    //
                    // 1. We were to emit a read and the key to be read is not in the cache.
                    // 2. The random number generated is between `incremental_delete_rate` and
                    //    1.0.
                    //
                    let fill = if i < read_rate {
                        match cache_clone.read(
                            bytemuck::bytes_of(&id),
                            bytemuck::must_cast_slice_mut::<usize, u8>(&mut v),
                        ) {
                            // Check contents for consistency
                            LeafReadResult::Found(bytes) => {
                                assert_eq!(bytes as usize, std::mem::size_of_val(&*v));
                                assert!(v.iter().all(|&i| i == id));
                                false
                            }
                            LeafReadResult::InvalidKey => panic!("invalid key: {}", id),
                            LeafReadResult::Deleted | LeafReadResult::NotFound => true,
                        }
                    } else if i < incremental_delete_rate {
                        cache_clone.delete(bytemuck::bytes_of(&id));
                        false
                    } else {
                        true
                    };

                    if fill {
                        v.fill(id);
                        match cache_clone.insert(
                            bytemuck::bytes_of(&id),
                            bytemuck::must_cast_slice::<usize, u8>(&v),
                        ) {
                            LeafInsertResult::Success => {}
                            LeafInsertResult::InvalidKV(message) => {
                                panic!("failed with \"{}\"", message)
                            }
                        }
                    }
                }
            })
        })
        .collect();

    println!("starting");
    let tic = std::time::Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let toc = std::time::Instant::now();
        let time_since_start = toc - tic;
        if time_since_start > duration {
            println!("stopping");
            break;
        } else {
            println!(
                "Elapsed {} of {}",
                time_since_start.as_secs(),
                duration.as_secs()
            );
        }
    }

    // Join all threads.
    finish.store(true, Ordering::Relaxed);
    for h in handles.into_iter() {
        h.join().unwrap();
    }
    println!("joined");

    let mut hit_count = 0;
    let mut v = vec![0usize; dim];
    for id in 0..id_range {
        match cache.read(
            bytemuck::bytes_of(&id),
            bytemuck::must_cast_slice_mut::<usize, u8>(&mut v),
        ) {
            LeafReadResult::Found(bytes) => {
                assert_eq!(bytes as usize, std::mem::size_of_val(&*v));
                assert!(v.iter().all(|&i| i == id));
                hit_count += 1;
            }
            LeafReadResult::Deleted | LeafReadResult::NotFound => {}
            LeafReadResult::InvalidKey => panic!("invalid key: {}", id),
        }
    }
    println!("hit count = {} of {}", hit_count, id_range);
}

#[test]
fn concurrent_stress_test_non_cache_only_mode() {
    let finish = Arc::new(AtomicBool::new(false));

    let mut config = Config::default();
    config
        .storage_backend(StorageBackend::Memory)
        .cache_only(false)
        .cb_size_byte(16384) // 16KB, 4 leaf pages
        .file_path(":memory:");

    let cache = Arc::new(BfTree::with_config(config, None).unwrap());

    let num_threads = 32;
    let read_rate = 0.8;
    let incremental_delete_rate = 0.9;
    let id_range = 1000000;
    let dim = 128;
    let duration = std::time::Duration::from_secs(20);

    // Spawn threads to emit mixed read + write + delete workloads.
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let finish_clone = finish.clone();
            let cache_clone = cache.clone();
            std::thread::spawn(move || {
                let mut rng = StdRng::from_os_rng();
                let id_range = Uniform::new(0, id_range).unwrap();

                let mut v = vec![0usize; dim];
                while !finish_clone.load(Ordering::Relaxed) {
                    let i: f64 = StandardUniform {}.sample(&mut rng);
                    let id: usize = id_range.sample(&mut rng);

                    // Fill the cache if:
                    //
                    // 1. We were to emit a read and the key to be read is not in the cache.
                    // 2. The random number generated is between `incremental_delete_rate` and
                    //    1.0.
                    //
                    let fill = if i < read_rate {
                        match cache_clone.read(
                            bytemuck::bytes_of(&id),
                            bytemuck::must_cast_slice_mut::<usize, u8>(&mut v),
                        ) {
                            // Check contents for consistency
                            LeafReadResult::Found(bytes) => {
                                assert_eq!(bytes as usize, std::mem::size_of_val(&*v));
                                assert!(v.iter().all(|&i| i == id));
                                false
                            }
                            LeafReadResult::InvalidKey => panic!("invalid key: {}", id),
                            LeafReadResult::Deleted | LeafReadResult::NotFound => true,
                        }
                    } else if i < incremental_delete_rate {
                        cache_clone.delete(bytemuck::bytes_of(&id));
                        false
                    } else {
                        true
                    };

                    if fill {
                        v.fill(id);
                        match cache_clone.insert(
                            bytemuck::bytes_of(&id),
                            bytemuck::must_cast_slice::<usize, u8>(&v),
                        ) {
                            LeafInsertResult::Success => {}
                            LeafInsertResult::InvalidKV(message) => {
                                panic!("failed with \"{}\"", message)
                            }
                        }
                    }
                }
            })
        })
        .collect();

    println!("starting");
    let tic = std::time::Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let toc = std::time::Instant::now();
        let time_since_start = toc - tic;
        if time_since_start > duration {
            println!("stopping");
            break;
        } else {
            println!(
                "Elapsed {} of {}",
                time_since_start.as_secs(),
                duration.as_secs()
            );
        }
    }

    // Join all threads.
    finish.store(true, Ordering::Relaxed);
    for h in handles.into_iter() {
        h.join().unwrap();
    }
    println!("joined");

    let mut hit_count = 0;
    let mut v = vec![0usize; dim];
    for id in 0..id_range {
        match cache.read(
            bytemuck::bytes_of(&id),
            bytemuck::must_cast_slice_mut::<usize, u8>(&mut v),
        ) {
            LeafReadResult::Found(bytes) => {
                assert_eq!(bytes as usize, std::mem::size_of_val(&*v));
                assert!(v.iter().all(|&i| i == id));
                hit_count += 1;
            }
            LeafReadResult::Deleted | LeafReadResult::NotFound => {}
            LeafReadResult::InvalidKey => panic!("invalid key: {}", id),
        }
    }
    println!("hit count = {} of {}", hit_count, id_range);
}

#[cfg(feature = "shuttle")]
#[test]
fn shuttle_bf_tree_concurrent_operations() {
    use std::{path::PathBuf, str::FromStr};

    tracing_subscriber::fmt()
        .with_ansi(true)
        .with_thread_names(false)
        .with_target(false)
        .init();

    let mut config = shuttle::Config::default();
    config.max_steps = shuttle::MaxSteps::None;
    config.failure_persistence =
        shuttle::FailurePersistence::File(Some(PathBuf::from_str("target").unwrap()));
    let mut runner = shuttle::PortfolioRunner::new(true, config);

    let available_cores = std::thread::available_parallelism().unwrap().get().min(4);

    for _i in 0..available_cores {
        runner.add(shuttle::scheduler::PctScheduler::new(10, 4_000));
    }

    runner.run(concurrent_ops);
}

#[cfg(feature = "shuttle")]
#[test]
fn shuttle_ht_replay() {
    // install global collector configured based on RUST_LOG env var.
    tracing_subscriber::fmt()
        .with_ansi(true)
        .with_thread_names(false)
        .with_target(false)
        .init();

    shuttle::replay_from_file(concurrent_ops, "target/schedule000.txt");
}
