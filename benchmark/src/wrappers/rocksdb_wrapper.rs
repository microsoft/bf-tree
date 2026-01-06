// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::path::Path;

use rocksdb::IteratorMode;

use crate::KvStore;

pub struct RocksDbWrapper {
    db: rocksdb::DB,
    write_options: rocksdb::WriteOptions,
}

/// Split into 3 parts: row cache, block cache, write buffer,
fn divide_cache_size(cache_size: usize) -> (usize, usize, usize) {
    let read_cache = cache_size / 2;
    let write_buffer_size = cache_size - read_cache;
    (read_cache / 2, read_cache / 2, write_buffer_size)
}

impl KvStore for RocksDbWrapper {
    fn new(file_path: impl AsRef<Path>, memory_size: usize) -> Self
    where
        Self: Sized,
    {
        let cache_size = memory_size;
        let (row_cache_size, block_cache_size, write_buffer_size) = divide_cache_size(cache_size);

        let cache = rocksdb::Cache::new_lru_cache(row_cache_size);
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        options.set_use_fsync(false);
        options.set_use_direct_reads(true);
        options.set_use_direct_io_for_flush_and_compaction(true);

        options.optimize_for_point_lookup(block_cache_size as u64 / 1024 / 1024);
        options.set_use_adaptive_mutex(true);

        options.set_db_write_buffer_size(write_buffer_size);
        options.set_row_cache(&cache);

        let mut write_options = rocksdb::WriteOptions::default();
        write_options.set_sync(false);
        write_options.disable_wal(true);

        let db = rocksdb::DB::open(&options, file_path).unwrap();
        Self { db, write_options }
    }

    fn insert(&self, key: &[u8], value: &[u8]) {
        self.db.put_opt(key, value, &self.write_options).unwrap()
        // self.db.put(key, value).unwrap()
    }

    fn read(&self, key: &[u8], value: &mut [u8]) -> usize {
        let v = self.db.get_pinned(key).unwrap().expect("key not found");
        let data = v.as_ref();
        value.clone_from_slice(data);
        data.len()
    }

    fn update(&self, key: &[u8], value: &[u8]) {
        self.db.put_opt(key, value, &self.write_options).unwrap();
        // self.db.put(key, value).unwrap()
    }

    fn scan(&self, key: &[u8], cnt: usize, out_buffer: &mut [u8]) {
        let mut iter = self
            .db
            .iterator(IteratorMode::From(key, rocksdb::Direction::Forward));

        let mut cur_cnt = 0;
        while let Some(v) = iter.next() {
            cur_cnt += 1;
            let kv = v.unwrap();
            out_buffer.clone_from_slice(&kv.0);
            if cur_cnt == cnt {
                break;
            }
        }
    }
}
