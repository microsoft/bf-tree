// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::{
    sync::{
        atomic::{AtomicU64, Ordering},
        thread, Arc, Barrier, Mutex,
    },
    BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField, StorageBackend,
};

const MAX_VALUE: usize = 1536;

#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct StressCase {
    seed: u64,
    records: usize,
    threads: usize,
    seconds: u64,
    read_pct: u32,
    scan_pct: u32,
    scan_length: usize,
    memory_bytes: usize,
    backend: String,
    contended: bool,
    full_pages: bool,
    size_profile: usize,
}

impl Default for StressCase {
    fn default() -> Self {
        Self {
            seed: 42,
            records: 4096,
            threads: 4,
            seconds: 30,
            read_pct: 40,
            scan_pct: 20,
            scan_length: 32,
            memory_bytes: 256 * 1024,
            backend: "memory".into(),
            contended: false,
            full_pages: false,
            size_profile: 1,
        }
    }
}

fn key(id: usize) -> Vec<u8> {
    let mut bytes = vec![(id.wrapping_mul(17) >> 3) as u8; [8, 16, 24, 32][id % 4]];
    bytes[..8].copy_from_slice(&(id as u64).to_be_bytes());
    bytes
}

fn value(id: usize, version: u64, profile: usize) -> Vec<u8> {
    let sizes: &[usize] = match profile {
        0 => &[16, 24, 32, 48, 64, 96, 128],
        1 => &[16, 24, 56, 120, 248, 504, 1016, MAX_VALUE],
        _ => &[31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025],
    };
    let len = sizes[(id.wrapping_add(version as usize)) % sizes.len()];
    let salt = (id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ version.rotate_left(23);
    let mut bytes: Vec<_> = (0..len)
        .map(|offset| (salt.rotate_left((offset % 64) as u32) as u8) ^ offset as u8)
        .collect();
    bytes[..8].copy_from_slice(&(id as u64).to_be_bytes());
    bytes[8..16].copy_from_slice(&version.to_be_bytes());
    bytes
}

fn verify_value(id: usize, bytes: &[u8], profile: usize, issued: &[AtomicU64]) -> u64 {
    assert!(bytes.len() >= 16);
    let version = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
    assert!(
        version <= issued[id].load(Ordering::Acquire),
        "unissued version"
    );
    assert_eq!(
        bytes,
        value(id, version, profile),
        "torn value for key {id}"
    );
    version
}

fn read(tree: &BfTree, id: usize, profile: usize, issued: &[AtomicU64]) -> u64 {
    let mut bytes = [0; MAX_VALUE];
    match tree.read(&key(id), &mut bytes) {
        LeafReadResult::Found(len) => verify_value(id, &bytes[..len as usize], profile, issued),
        other => panic!("loaded key {id} disappeared: {other:?}"),
    }
}

fn scan(
    tree: &BfTree,
    start: usize,
    count: usize,
    profile: usize,
    issued: &[AtomicU64],
    expected: Option<&[u64]>,
    records: usize,
) {
    let mut scan = tree
        .scan_with_count(&key(start), count, ScanReturnField::KeyAndValue)
        .unwrap();
    let mut bytes = [0; 32 + MAX_VALUE];
    let mut copied = 0;
    while let Some((key_len, value_len)) = scan.next(&mut bytes) {
        assert!(key_len >= 8);
        let id = u64::from_be_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(id, start + copied, "scan skipped or duplicated a live key");
        assert!(id < records);
        assert_eq!(&bytes[..key_len], key(id));
        let version = verify_value(id, &bytes[key_len..key_len + value_len], profile, issued);
        if let Some(expected) = expected {
            assert_eq!(version, expected[id], "stale scan value");
        }
        copied += 1;
        assert!(copied <= count);
    }
    assert_eq!(copied, count.min(records.saturating_sub(start)));
}

#[rstest::rstest]
#[case(StorageBackend::Memory)]
#[case(StorageBackend::Std)]
#[cfg_attr(target_os = "linux", case(StorageBackend::StdDirect))]
#[cfg_attr(target_os = "linux", case(StorageBackend::IoUringBlocking))]
fn production_mixed_record_stress(
    #[case] backend: StorageBackend,
    #[values(false, true)] full_pages: bool,
    #[values(false, true)] contended: bool,
    #[values(0, 1, 2)] profile: usize,
) {
    let case = StressCase {
        backend: format!("{backend:?}"),
        full_pages,
        contended,
        size_profile: profile,
        ..StressCase::default()
    };
    run_case(&case, backend, Some(400));
}

#[test]
#[ignore = "production matrix; run with BF_PRODUCTION_CASE under the supervised runner"]
fn production_matrix_case() {
    let case: StressCase = serde_json::from_str(
        &std::env::var("BF_PRODUCTION_CASE").expect("BF_PRODUCTION_CASE is required"),
    )
    .expect("invalid production case");
    let backend = match case.backend.as_str() {
        "memory" => StorageBackend::Memory,
        "std" => StorageBackend::Std,
        #[cfg(target_os = "linux")]
        "direct" => StorageBackend::StdDirect,
        #[cfg(target_os = "linux")]
        "blocking" => StorageBackend::IoUringBlocking,
        #[cfg(target_os = "linux")]
        "polling" => StorageBackend::IoUringPolling,
        other => panic!("unsupported backend: {other}"),
    };
    run_case(&case, backend, None);
}

fn run_case(case: &StressCase, backend: StorageBackend, steps: Option<usize>) {
    assert!(case.records >= 128 && case.threads > 0 && case.seconds > 0);
    assert!(case.read_pct <= 100 && case.scan_pct <= 100 - case.read_pct);
    assert!(case.scan_length > 0 && case.size_profile <= 2);
    let (profile, contended) = (case.size_profile, case.contended);
    let temporary = tempfile::tempdir().unwrap();
    let directory = std::env::var_os("BF_PRODUCTION_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| temporary.path().to_owned());
    std::fs::create_dir_all(&directory).unwrap();
    let mut config = Config::new(directory.join("tree"), case.memory_bytes);
    config
        .storage_backend(backend)
        .cb_min_record_size(24)
        .cb_max_record_size(1600)
        .cb_max_key_len(32)
        .read_promotion_rate(100)
        .scan_promotion_rate(100)
        .read_record_cache(!case.full_pages)
        .write_load_full_page(case.full_pages);
    let tree = Arc::new(BfTree::with_config(config, None).unwrap());
    let issued: Vec<_> = (0..case.records).map(|_| AtomicU64::new(0)).collect();
    let model: Vec<_> = (0..case.records).map(|_| Mutex::new(0u64)).collect();
    let completed = AtomicU64::new(0);
    eprintln!("PRODUCTION_CASE {}", serde_json::to_string(case).unwrap());
    eprintln!("PHASE preload");
    thread::scope(|scope| {
        for tid in 0..case.threads {
            let tree = &tree;
            scope.spawn(move || {
                for id in (tid..case.records).step_by(case.threads) {
                    assert_eq!(
                        tree.insert(&key(id), &value(id, 0, profile)),
                        LeafInsertResult::Success
                    );
                }
            });
        }
    });
    if steps.is_none() {
        drain(&tree);
    }
    eprintln!("PHASE workload");
    let ready = Barrier::new(case.threads + 1);
    let counts = thread::scope(|scope| {
        let mut workers = Vec::new();
        for tid in 0..case.threads {
            let (tree, ready, issued, model, completed) =
                (&tree, &ready, &issued, &model, &completed);
            workers.push(scope.spawn(move || {
                let mut rng = SmallRng::seed_from_u64(case.seed.wrapping_add(tid as u64));
                let mut counts = [0u64; 3];
                ready.wait();
                let deadline = Instant::now() + Duration::from_secs(case.seconds);
                let mut step = 0;
                while steps.map_or_else(|| Instant::now() < deadline, |limit| step < limit) {
                    let id = if contended && rng.random_ratio(4, 5) {
                        rng.random_range(0..64)
                    } else {
                        rng.random_range(0..case.records)
                    };
                    let choice = rng.random_range(0..100);
                    if choice < case.read_pct {
                        if contended {
                            read(tree, id, profile, issued);
                        } else {
                            let expected = model[id].lock().unwrap();
                            assert_eq!(read(tree, id, profile, issued), *expected);
                        }
                        counts[0] += 1;
                    } else if choice < case.read_pct + case.scan_pct {
                        scan(
                            tree,
                            id,
                            case.scan_length,
                            profile,
                            issued,
                            None,
                            case.records,
                        );
                        counts[2] += 1;
                    } else {
                        let guard = (!contended).then(|| model[id].lock().unwrap());
                        let version = issued[id].fetch_add(1, Ordering::AcqRel) + 1;
                        assert_eq!(
                            tree.insert(&key(id), &value(id, version, profile)),
                            LeafInsertResult::Success
                        );
                        if let Some(mut expected) = guard {
                            *expected = version;
                        }
                        counts[1] += 1;
                    }
                    step += 1;
                    completed.fetch_add(1, Ordering::Relaxed);
                }
                counts
            }));
        }
        ready.wait();
        if steps.is_none() {
            let begin = Instant::now();
            while begin.elapsed() < Duration::from_secs(case.seconds) {
                thread::sleep(Duration::from_secs(1));
                eprintln!("PROGRESS {}", completed.load(Ordering::Relaxed));
            }
        }
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .fold([0u64; 3], |mut total, counts| {
                for (total, count) in total.iter_mut().zip(counts) {
                    *total += count;
                }
                total
            })
    });
    eprintln!("PHASE verify");
    let versions: Vec<_> = model
        .iter()
        .enumerate()
        .map(|(id, expected)| {
            let actual = read(&tree, id, profile, &issued);
            if !contended {
                assert_eq!(
                    actual,
                    *expected.lock().unwrap(),
                    "lost acknowledged update"
                );
            }
            actual
        })
        .collect();
    scan(
        &tree,
        0,
        case.records + 1,
        profile,
        &issued,
        Some(&versions),
        case.records,
    );
    eprintln!("PHASE drain");
    drain(&tree);
    scan(
        &tree,
        0,
        case.records + 1,
        profile,
        &issued,
        Some(&versions),
        case.records,
    );
    drop(tree);
    eprintln!(
        "PRODUCTION_RESULT {}",
        serde_json::json!({
            "case": case, "reads": counts[0], "writes": counts[1], "scans": counts[2], "status": "passed"
        })
    );
}

fn drain(tree: &BfTree) {
    tree.storage.circular_buffer.drain(|handle| {
        if crate::tree::eviction_callback(&handle, tree).is_ok() {
            Ok(handle)
        } else {
            Err(handle)
        }
    });
}
