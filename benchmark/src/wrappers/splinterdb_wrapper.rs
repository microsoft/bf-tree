// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use splinterdb_sys::{DefaultSdb, LookupResult};

use crate::KvStore;

pub struct SplinterdbWrapper {
    db: splinterdb_sys::SplinterDB,
}

impl KvStore for SplinterdbWrapper {
    fn new(file_path: impl AsRef<std::path::Path>, memory_size: usize) -> Self
    where
        Self: Sized,
    {
        let cache_size = memory_size * 3 / 4;
        let config = splinterdb_sys::DBConfig {
            cache_size_bytes: cache_size,
            disk_size_bytes: 10 * 1024 * 1024 * 1024,
            max_key_size: 10,
            max_value_size: 10,
        };
        let mut db = splinterdb_sys::SplinterDB::new::<DefaultSdb>();
        db.db_create(&file_path, &config).unwrap();

        Self { db }
    }

    fn register_thread(&self) {
        self.db.register_thread();
    }

    fn deregister_thread(&self) {
        self.db.deregister_thread();
    }

    fn insert(&self, key: &[u8], value: &[u8]) {
        self.db.insert(key, value).unwrap();
    }

    fn read(&self, key: &[u8], value: &mut [u8]) -> usize {
        let v = self.db.lookup(key).unwrap();
        match v {
            LookupResult::Found(v) => {
                let data = v.as_ref();
                value.clone_from_slice(data);
                data.len()
            }
            _ => {
                panic!("key not found");
            }
        }
    }

    fn update(&self, key: &[u8], value: &[u8]) {
        self.db.update(key, value).unwrap();
    }

    fn scan(&self, _key: &[u8], _cnt: usize, _out_buffer: &mut [u8]) {
        todo!()
    }
}
