// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::path::Path;

use bf_tree::BfTree;

use crate::KvStore;

/// Basically bf-tree, but page cache.
/// we do page cache by scanning items with record count of 1.
pub struct PageCacheWrapper {
    db: BfTree,
    scan_buffer: *mut u8,
}

unsafe impl Send for PageCacheWrapper {}
unsafe impl Sync for PageCacheWrapper {}

impl PageCacheWrapper {
    pub fn new(file_path: impl AsRef<Path>, cache_size: usize) -> Self {
        let ptr =
            unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(4096, 4096).unwrap()) };

        let mut config = bf_tree::Config::new(file_path, cache_size);
        config.storage_backend(bf_tree::StorageBackend::IoUringBlocking);
        let tree = BfTree::with_config(config, None);
        Self {
            db: tree,
            scan_buffer: ptr,
        }
    }
}

impl KvStore for PageCacheWrapper {
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
        let mut iter = self.db.scan(key, 1);
        iter.next(value).unwrap()
    }

    fn update(&self, key: &[u8], _value: &[u8]) {
        let mut iter = self.db.scan_mut(key, 1);
        let buf = unsafe { std::slice::from_raw_parts_mut(self.scan_buffer, 4096) };
        let _len = iter.next(buf).unwrap();
    }

    fn scan(&self, key: &[u8], cnt: usize, out_buffer: &mut [u8]) {
        let mut iter = self.db.scan(key, cnt);

        while let Some(len) = iter.next(out_buffer) {
            assert!(len <= out_buffer.len());
        }
    }
}
