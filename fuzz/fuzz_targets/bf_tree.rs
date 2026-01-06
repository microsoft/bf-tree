// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#![no_main]

use arbitrary::Arbitrary;
use bf_tree::{BfTree, Config, LeafReadResult, ScanReturnField};
use libfuzzer_sys::fuzz_target;
use mimalloc::MiMalloc;
use std::{
    env,
    hash::{DefaultHasher, Hash, Hasher},
    ops::Bound::{Included, Unbounded},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Arbitrary, Debug)]
enum Methods {
    Insert(u16),
    Read(u16),
    Delete(u16),
    Scan((u16, u8)),
}

fn make_key(value: u16) -> Vec<u8> {
    const LEN_BITS: u16 = 0x1f;
    let len = (value & LEN_BITS) as usize;
    let val = value >> 6;
    let bytes = val.to_ne_bytes();
    bytes
        .into_iter()
        .cycle()
        .take(len * bytes.len())
        .collect::<Vec<_>>()
}

fuzz_target!(|methods: Vec<Methods>| {
    let pid = std::process::id();
    let tid = {
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        hasher.finish()
    };
    let tmp_dir = env::temp_dir();
    let tmp_file_path = tmp_dir.join(format!("bf_tree_fuzz_{pid}_{tid}.db"));
    let mut config = Config::new(&tmp_file_path, 8192);
    config.storage_backend(bf_tree::StorageBackend::Std);
    let bf_tree = BfTree::with_config(config.clone(), None).unwrap();

    let mut std_btree = std::collections::BTreeSet::<Vec<u8>>::new();

    let mut buffer = vec![0; 4096];

    for m in methods {
        match m {
            Methods::Insert(v) => {
                let k = make_key(v);
                if k.is_empty() {
                    continue;
                }

                bf_tree.insert(&k, &k);
                std_btree.insert(k);
            }
            Methods::Read(k) => {
                let k = make_key(k);
                if k.is_empty() {
                    continue;
                }
                let std_v = std_btree.get(&k);
                let bf_v = bf_tree.read(&k, &mut buffer);
                let read_len = match bf_v {
                    LeafReadResult::Found(v) => v as usize,
                    _ => 0,
                };

                match std_v {
                    Some(v) => {
                        assert_eq!(v[..], buffer[0..read_len as usize]);
                    }
                    None => {
                        assert_eq!(read_len, 0);
                    }
                }
            }
            Methods::Delete(k) => {
                let k = make_key(k);
                if k.is_empty() {
                    continue;
                }
                std_btree.remove(&k);
                bf_tree.delete(&k);
            }
            Methods::Scan((s, len)) => {
                let k = make_key(s);
                if k.is_empty() {
                    continue;
                }

                let mut bf_v = bf_tree
                    .scan_with_count(&k, len as usize, ScanReturnField::Value)
                    .expect("Failed to create scan iterator");

                let std_v = std_btree
                    .range::<Vec<u8>, _>((Included(&k), Unbounded))
                    .take(len as usize);

                for v in std_v {
                    if let Some((_key_len, value_len)) = bf_v.next(&mut buffer) {
                        assert_eq!(v[..], buffer[0..value_len])
                    } else {
                        panic!("Missing key in scan");
                    }
                }
                let v = bf_v.next(&mut buffer);
                assert!(v.is_none());
            }
        }
    }

    bf_tree.snapshot();
    drop(bf_tree);
    let bf_tree = BfTree::new_from_snapshot(config.clone(), None).unwrap();

    for k in std_btree.iter() {
        let bf_v = bf_tree.read(k, &mut buffer);

        let read_len = match bf_v {
            LeafReadResult::Found(v) => v as usize,
            _ => 0,
        };
        assert_eq!(k[..], buffer[0..read_len]);
    }
    drop(bf_tree);
    std::fs::remove_file(tmp_file_path).unwrap();
});
