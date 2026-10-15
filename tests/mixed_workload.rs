// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::BTreeMap;

use bf_tree::{BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

fn check_scan(tree: &BfTree, model: &BTreeMap<Vec<u8>, Vec<u8>>, start: &[u8]) {
    let mut scan = tree
        .scan_with_count(start, model.len() + 1, ScanReturnField::KeyAndValue)
        .unwrap();
    let mut out = [0; 512];
    for (key, value) in model.range(start.to_vec()..) {
        assert_eq!(scan.next(&mut out), Some((key.len(), value.len())));
        assert_eq!(&out[..key.len()], key);
        assert_eq!(&out[key.len()..key.len() + value.len()], value);
    }
    assert_eq!(scan.next(&mut out), None);
}

#[test]
fn mixed_updates_deletes_and_scans_match_btree_map() {
    for cache_only in [false, true] {
        for key_len in [8, 128] {
            let mut config = Config::default();
            config
                .cache_only(cache_only)
                .cb_size_byte(8 * 1024 * 1024)
                .cb_max_record_size(1024)
                .cb_max_key_len(key_len);
            let tree = BfTree::with_config(config, None).unwrap();
            let mut model = BTreeMap::new();
            let mut rng = StdRng::seed_from_u64(0x5AFE_2026);
            let mut keys: Vec<_> = (0u64..256)
                .map(|i| {
                    let mut key = vec![0xF0; key_len];
                    key[key_len - 8..].copy_from_slice(&i.to_be_bytes());
                    key
                })
                .collect();
            let first = keys[0].clone();
            keys.shuffle(&mut rng);

            // Force page splits before exercising growth, shrinkage and tombstones.
            for key in &keys {
                let value = vec![0xAB; 128];
                assert_eq!(tree.insert(key, &value), LeafInsertResult::Success);
                model.insert(key.clone(), value);
            }
            for step in 0..2_000 {
                let key = &keys[rng.random_range(0..keys.len())];
                if rng.random_range(0..4) == 0 {
                    tree.delete(key);
                    model.remove(key);
                } else {
                    let len = [1, 8, 32, 96, 160, 256][rng.random_range(0..6)];
                    let value = vec![(step % 251) as u8; len];
                    assert_eq!(tree.insert(key, &value), LeafInsertResult::Success);
                    model.insert(key.clone(), value);
                }
                let mut out = [0; 256];
                match model.get(key) {
                    Some(value) => {
                        assert_eq!(
                            tree.read(key, &mut out),
                            LeafReadResult::Found(value.len() as u32)
                        );
                        assert_eq!(&out[..value.len()], value);
                    }
                    None => assert!(matches!(
                        tree.read(key, &mut out),
                        LeafReadResult::NotFound | LeafReadResult::Deleted
                    )),
                }
                if step % 100 == 0 {
                    check_scan(&tree, &model, &first);
                    check_scan(&tree, &model, key);
                }
            }
            check_scan(&tree, &model, &first);
        }
    }
}
