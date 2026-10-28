// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::{
    sync::{
        atomic::{AtomicU64, Ordering},
        thread, Arc, Barrier, Mutex,
    },
    BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField, StorageBackend,
};

const RECORDS: usize = 4096;
const MAX_VALUE: usize = 1536;

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
        assert!(id < RECORDS);
        assert_eq!(&bytes[..key_len], key(id));
        let version = verify_value(id, &bytes[key_len..key_len + value_len], profile, issued);
        if let Some(expected) = expected {
            assert_eq!(version, expected[id], "stale scan value");
        }
        copied += 1;
        assert!(copied <= count);
    }
    assert_eq!(copied, count.min(RECORDS.saturating_sub(start)));
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
    let temporary = tempfile::tempdir().unwrap();
    let mut config = Config::new(temporary.path().join("tree"), 256 * 1024);
    config
        .storage_backend(backend)
        .cb_min_record_size(24)
        .cb_max_record_size(1600)
        .cb_max_key_len(32)
        .read_promotion_rate(100)
        .scan_promotion_rate(100)
        .read_record_cache(!full_pages)
        .write_load_full_page(full_pages);
    let tree = Arc::new(BfTree::with_config(config, None).unwrap());
    let issued: Vec<_> = (0..RECORDS).map(|_| AtomicU64::new(0)).collect();
    let model: Vec<_> = (0..RECORDS).map(|_| Mutex::new(0u64)).collect();
    thread::scope(|scope| {
        for tid in 0..4 {
            let tree = &tree;
            scope.spawn(move || {
                for id in (tid..RECORDS).step_by(4) {
                    assert_eq!(
                        tree.insert(&key(id), &value(id, 0, profile)),
                        LeafInsertResult::Success
                    );
                }
            });
        }
    });
    let ready = Barrier::new(4);
    thread::scope(|scope| {
        for tid in 0..4 {
            let (tree, ready, issued, model) = (&tree, &ready, &issued, &model);
            scope.spawn(move || {
                let mut rng = SmallRng::seed_from_u64(42 + tid);
                ready.wait();
                for _ in 0..400 {
                    let id = if contended && rng.random_ratio(4, 5) {
                        rng.random_range(0..64)
                    } else {
                        rng.random_range(0..RECORDS)
                    };
                    match rng.random_range(0..100) {
                        0..40 => {
                            if contended {
                                read(tree, id, profile, issued);
                            } else {
                                let expected = model[id].lock().unwrap();
                                assert_eq!(read(tree, id, profile, issued), *expected);
                            }
                        }
                        40..60 => scan(tree, id, 32, profile, issued, None),
                        _ => {
                            let guard = (!contended).then(|| model[id].lock().unwrap());
                            let version = issued[id].fetch_add(1, Ordering::AcqRel) + 1;
                            assert_eq!(
                                tree.insert(&key(id), &value(id, version, profile)),
                                LeafInsertResult::Success
                            );
                            if let Some(mut expected) = guard {
                                *expected = version;
                            }
                        }
                    }
                }
            });
        }
    });
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
    scan(&tree, 0, RECORDS + 1, profile, &issued, Some(&versions));
    tree.storage.circular_buffer.drain(|handle| {
        if crate::tree::eviction_callback(&handle, &tree).is_ok() {
            Ok(handle)
        } else {
            Err(handle)
        }
    });
    scan(&tree, 0, RECORDS + 1, profile, &issued, Some(&versions));
}
