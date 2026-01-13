// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{cell::UnsafeCell, mem::MaybeUninit};

use crate::sync::{Mutex, MutexGuard};

// we support at most 128 batches, which is 128M pages.
const MAX_BATCHES: usize = 128;

struct RecordBatch<T> {
    data: Vec<MaybeUninit<T>>,
}

impl<T> RecordBatch<T> {
    fn new(record_per_batch: usize) -> Self {
        let mut data = Vec::with_capacity(record_per_batch);

        data.extend((0..record_per_batch).map(|_| MaybeUninit::uninit()));

        Self { data }
    }

    /// # Safety
    /// (1) The record must be initialized.
    /// (2) The record must be smaller than `RECORD_PER_BATCH`.
    unsafe fn get_record(&self, id: usize) -> &T {
        let record = unsafe { self.data.get_unchecked(id) };
        unsafe { record.assume_init_ref() }
    }
}

struct States {
    next_id: u64,
    current_initialized_batch: usize,
}

/// A mapping table that allows you to insert a record and later retrieve it by id.
/// Insertion and retrieval only takes exactly one (not O(1)!) memory access.
pub struct MappingTable<T> {
    states: Mutex<States>,
    batches: UnsafeCell<Vec<MaybeUninit<RecordBatch<T>>>>,
    record_per_batch: usize,
}

impl<T> Default for MappingTable<T> {
    fn default() -> Self {
        Self::new(DEFAULT_RECORD_PER_BATCH)
    }
}

impl<T> Drop for MappingTable<T> {
    fn drop(&mut self) {
        let mut states = self.states.lock().unwrap();
        let initialized_records = states.next_id - 1;
        for i in 0..initialized_records {
            let batch_id = self.get_batch_id(i);
            let record_id = self.get_record_id(i);
            let batch = unsafe { self.get_batch_mut(batch_id, &mut states) };
            let record = unsafe { batch.data.get_unchecked_mut(record_id) };
            unsafe {
                record.as_mut_ptr().drop_in_place();
            }
        }
        let batch_cnt = states.current_initialized_batch + 1;
        for i in 0..batch_cnt {
            let record_batch_vec = unsafe { &mut *self.batches.get() };
            let batch = unsafe { record_batch_vec.get_unchecked_mut(i) };
            unsafe {
                batch.as_mut_ptr().drop_in_place();
            }
        }
    }
}

const DEFAULT_RECORD_PER_BATCH: usize = 1024 * 1024;

impl<T> MappingTable<T> {
    pub fn new(record_per_batch: usize) -> Self {
        let mut batches = Vec::new();
        for i in 0..MAX_BATCHES {
            if i == 0 {
                batches.push(MaybeUninit::new(RecordBatch::new(record_per_batch)));
            } else {
                batches.push(MaybeUninit::uninit());
            }
        }

        Self {
            states: Mutex::new(States {
                next_id: 0,
                current_initialized_batch: 0,
            }),
            batches: UnsafeCell::new(batches),
            record_per_batch,
        }
    }

    pub(crate) fn new_from_iter(mapping: impl Iterator<Item = (u64, T)>) -> Self {
        let mt = Self::default();
        let mut states = mt.states.lock().unwrap();

        for (id, val) in mapping {
            states.next_id = id + 1;
            mt.set(id, val, &mut states);
        }
        drop(states);
        mt
    }

    /// # Safety
    /// (1) The batch must be initialized.
    /// (2) The batch must be smaller than `MAX_BATCHES`.
    unsafe fn get_batch(&self, batch_id: usize) -> &RecordBatch<T> {
        let record_batch_vec = unsafe { &mut *self.batches.get() };
        let batch = unsafe { record_batch_vec.get_unchecked_mut(batch_id) };
        let batch = unsafe { batch.assume_init_ref() };
        batch
    }

    /// # Safety
    /// (1) The batch must be initialized.
    /// (2) The batch must be smaller than `MAX_BATCHES`.
    /// (3) Mutex lock must be held.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_batch_mut(
        &self,
        batch_id: usize,
        _lock: &mut MutexGuard<'_, States>,
    ) -> &mut RecordBatch<T> {
        let record_batch_vec = unsafe { &mut *self.batches.get() };
        let batch = unsafe { record_batch_vec.get_unchecked_mut(batch_id) };
        let batch = unsafe { batch.assume_init_mut() };
        batch
    }

    fn get_batch_id(&self, id: u64) -> usize {
        (id / self.record_per_batch as u64) as usize
    }

    fn get_record_id(&self, id: u64) -> usize {
        (id % self.record_per_batch as u64) as usize
    }

    /// Peek next id, used for iterating all entries.
    pub(crate) fn peek_next_id(&self) -> u64 {
        let states = self.states.lock().unwrap();
        states.next_id
    }

    /// Get the record by id.
    /// The id must be returned by `insert`.
    pub fn get(&self, id: u64) -> &T {
        let batch_id = self.get_batch_id(id);
        let record_id = self.get_record_id(id);

        let batch = unsafe { self.get_batch(batch_id) };
        let record = unsafe { batch.get_record(record_id) };
        record
    }

    fn set(&self, id: u64, val: T, states: &mut MutexGuard<States>) {
        let batch_id = self.get_batch_id(id);
        let record_id = self.get_record_id(id);

        if batch_id > states.current_initialized_batch {
            if batch_id >= MAX_BATCHES {
                panic!("Reached max batches!");
            }
            let batches = unsafe { &mut *self.batches.get() };
            let batch = unsafe { batches.get_unchecked_mut(batch_id) };
            unsafe {
                batch
                    .as_mut_ptr()
                    .write(RecordBatch::new(self.record_per_batch));
            }
            states.current_initialized_batch = batch_id;
        }

        let batch = unsafe { self.get_batch_mut(batch_id, states) };
        let record = unsafe { batch.data.get_unchecked_mut(record_id) };
        unsafe {
            record.as_mut_ptr().write(val);
        }
    }

    /// Insert a record into the mapping table.
    /// Returns the id of the record and the reference to the record.
    ///
    /// The id can be used to retrieve the record later using `get`.
    pub fn insert(&self, val: T) -> (u64, &T) {
        let mut states = self.states.lock().unwrap();

        let page_id = states.next_id;
        states.next_id += 1;

        self.set(page_id, val, &mut states);

        (page_id, self.get(page_id))
    }
}
