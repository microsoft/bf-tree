// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::KvStore;

pub struct SledWrapper {
    db: sled::Db,
}

impl KvStore for SledWrapper {
    fn new(file_path: impl AsRef<std::path::Path>, memory_size: usize) -> Self
    where
        Self: Sized,
    {
        println!("sled path: {:?}", file_path.as_ref());
        let cache_size = memory_size / 5;
        let config = sled::Config::default()
            .cache_capacity(cache_size as u64)
            .path(file_path.as_ref())
            .mode(sled::Mode::HighThroughput)
            .temporary(true)
            .create_new(true);

        Self {
            db: config.open().unwrap(),
        }
    }

    fn insert(&self, key: &[u8], value: &[u8]) {
        self.db.insert(key, value).unwrap();
    }

    fn read(&self, key: &[u8], value: &mut [u8]) -> usize {
        let v = self.db.get(key).unwrap().unwrap();
        value.clone_from_slice(&v);
        v.len()
    }

    fn update(&self, key: &[u8], value: &[u8]) {
        self.db.insert(key, value).unwrap();
    }

    fn scan(&self, _key: &[u8], _cnt: usize, _out_buffer: &mut [u8]) {
        todo!()
    }
}
