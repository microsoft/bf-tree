// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use lmdb::Transaction;

use crate::KvStore;

pub struct LmdbWrapper {
    db: lmdb::Database,
    env: lmdb::Environment,
}

impl KvStore for LmdbWrapper {
    fn new(file_path: impl AsRef<std::path::Path>, cache_size: usize) -> Self
    where
        Self: Sized,
    {
        _ = std::fs::create_dir(file_path.as_ref());
        let flags = lmdb::EnvironmentFlags::NO_SYNC;
        let env = lmdb::Environment::new()
            .set_flags(flags)
            .open(file_path.as_ref())
            .unwrap();
        env.set_map_size(cache_size).unwrap();
        let db = env.open_db(None).unwrap();
        Self { db, env }
    }

    fn insert(&self, key: &[u8], value: &[u8]) {
        let mut txn = self.env.begin_rw_txn().unwrap();
        txn.put(self.db, &key, &value, lmdb::WriteFlags::empty())
            .unwrap();
        txn.commit().unwrap();
    }

    fn read(&self, key: &[u8], value: &mut [u8]) -> usize {
        let txn = self.env.begin_ro_txn().unwrap();
        let v = txn.get(self.db, &key).unwrap();
        value.clone_from_slice(&v);
        v.len()
    }

    fn update(&self, key: &[u8], value: &[u8]) {
        let mut txn = self.env.begin_rw_txn().unwrap();
        txn.put(self.db, &key, &value, lmdb::WriteFlags::empty())
            .unwrap();
        txn.commit().unwrap();
    }

    fn scan(&self, _key: &[u8], _cnt: usize, _out_buffer: &mut [u8]) {
        todo!()
    }
}
