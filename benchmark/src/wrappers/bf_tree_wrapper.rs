// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::path::Path;

use bf_tree::{BfTree, LeafReadResult, ScanReturnField};

use crate::KvStore;

pub struct BfTreeWrapper {
    db: BfTree,
}

impl BfTreeWrapper {
    pub fn new(file_path: impl AsRef<Path>, cache_size: usize) -> Self {
        let file_path = file_path.as_ref().to_str().unwrap();
        let prefix = file_path.strip_suffix(":memory");
        let (path, in_memory) = match prefix {
            Some(prefix) => (prefix.to_string(), true),
            None => (file_path.to_owned(), false),
        };

        let mut config = bf_tree::Config::new(path, cache_size);

        // compare file_path if matches ":memory:"
        if in_memory {
            // a trick to shorten the load time
            config.read_record_cache(false);
            config.read_promotion_rate(100);
        } else {
            #[cfg(feature = "spdk")]
            {
                config.storage_backend(bf_tree::StorageBackend::Spdk);
            }
            #[cfg(not(feature = "spdk"))]
            {
                config.storage_backend(bf_tree::StorageBackend::IoUringBlocking);
            }
        }

        let tree = BfTree::with_config(config, None).unwrap();
        Self { db: tree }
    }
}

impl KvStore for BfTreeWrapper {
    fn new(file_path: impl AsRef<Path>, memory_size: usize) -> Self
    where
        Self: Sized,
    {
        let cache_size = memory_size;
        Self::new(file_path, cache_size)
    }

    #[inline(always)]
    fn insert(&self, key: &[u8], value: &[u8]) {
        self.db.insert(key, value);
    }

    #[inline(always)]
    fn read(&self, key: &[u8], value: &mut [u8]) -> usize {
        let len = self.db.read(key, value);
        match len {
            LeafReadResult::Found(v) => v as usize,
            _ => 0,
        }
    }

    #[inline(always)]
    fn update(&self, key: &[u8], value: &[u8]) {
        self.db.insert(key, value);
    }

    #[inline(always)]
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
