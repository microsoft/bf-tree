// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::path::Path;

use bf_tree::{BfTree, LeafReadResult, ScanReturnField};

use crate::KvStore;

pub struct BwTreeWrapper {
    db: BfTree,
}

impl BwTreeWrapper {
    pub fn new(file_path: impl AsRef<Path>, cache_size: usize) -> Self {
        let mut config = bf_tree::Config::new(file_path, cache_size);

        config
            .read_record_cache(false)
            .max_mini_page_size(256)
            .mini_page_binary_search(false)
            .storage_backend(bf_tree::StorageBackend::IoUringBlocking);

        Self {
            db: BfTree::with_config(config, None).unwrap(),
        }
    }
}

impl KvStore for BwTreeWrapper {
    fn new(file_path: impl AsRef<Path>, memory_size: usize) -> Self
    where
        Self: Sized,
    {
        let cache_size = memory_size;
        Self::new(file_path, cache_size)
    }

    fn insert(&self, key: &[u8], value: &[u8]) {
        self.db.insert(key, value);
    }

    fn read(&self, key: &[u8], value: &mut [u8]) -> usize {
        let len = self.db.read(key, value);
        match len {
            LeafReadResult::Found(v) => v as usize,
            _ => 0,
        }
    }

    fn update(&self, key: &[u8], value: &[u8]) {
        self.db.insert(key, value);
    }

    fn scan(&self, key: &[u8], cnt: usize, out_buffer: &mut [u8]) {
        let mut iter = self
            .db
            .scan_with_count(key, cnt, ScanReturnField::Value)
            .expect("Failed to create scan iterator");

        while let Some((_key_len, value_len)) = iter.next(out_buffer) {
            assert!(value_len <= out_buffer.len());
        }
    }
}
