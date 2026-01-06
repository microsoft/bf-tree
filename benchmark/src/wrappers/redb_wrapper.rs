// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::path::Path;

use redb::{ReadableTable, TableDefinition};

use crate::KvStore;

pub struct RedbWrapper {
    db: redb::Database,
}

const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("benchmark_table");

impl KvStore for RedbWrapper {
    fn new(file_path: impl AsRef<Path>, cache_size: usize) -> Self
    where
        Self: Sized,
    {
        let db = redb::Database::builder()
            .set_cache_size(cache_size)
            .create(file_path)
            .unwrap();
        Self { db }
    }

    fn insert(&self, key: &[u8], value: &[u8]) {
        let txn = self.db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TABLE).unwrap();
            table.insert(key, value).unwrap();
        }
        txn.commit().unwrap();
    }

    fn read(&self, key: &[u8], value: &mut [u8]) -> usize {
        let txn = self.db.begin_read().unwrap();
        let table = txn.open_table(TABLE).unwrap();
        let val_guard = table.get(key).unwrap().unwrap();
        let val = val_guard.value();
        value.clone_from_slice(val);
        val.len()
    }

    fn update(&self, key: &[u8], value: &[u8]) {
        let txn = self.db.begin_write().unwrap();
        {
            let mut table = txn.open_table(TABLE).unwrap();
            table.insert(key, value).unwrap();
        }
        txn.commit().unwrap();
    }

    fn scan(&self, _key: &[u8], _cnt: usize, _out_buffer: &mut [u8]) {
        todo!()
    }
}
