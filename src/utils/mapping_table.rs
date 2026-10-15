// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{cell::UnsafeCell, mem::MaybeUninit, ptr};

use crate::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

// At the default batch size this supports 128M pages.
const MAX_BATCHES: usize = 128;
const DEFAULT_RECORD_PER_BATCH: usize = 1024 * 1024;

struct RecordBatch<T> {
    data: Box<[UnsafeCell<MaybeUninit<T>>]>,
    initialized: UnsafeCell<usize>,
}

impl<T> RecordBatch<T> {
    fn new(record_per_batch: usize) -> Self {
        let data = Box::<[UnsafeCell<MaybeUninit<T>>]>::new_uninit_slice(record_per_batch);
        Self {
            // SAFETY: UnsafeCell has the same validity requirements as its inner
            // value, and MaybeUninit<T> accepts uninitialized storage. No T is
            // read until insertion has initialized and published its slot.
            data: unsafe { data.assume_init() },
            initialized: UnsafeCell::new(0),
        }
    }

    /// # Safety
    /// The caller must have acquired publication of this initialized record.
    unsafe fn get_record(&self, id: usize) -> &T {
        // SAFETY: The published prefix bounds the index, and an initialized
        // slot is never overwritten. Other slots have their own UnsafeCells.
        unsafe { (&*self.data.get_unchecked(id).get()).assume_init_ref() }
    }

    /// # Safety
    /// The insertion mutex must be held, and id must be the next uninitialized
    /// slot within this batch.
    unsafe fn insert_record(&self, id: usize, val: T) {
        // SAFETY: The caller exclusively owns this uninitialized slot. Writing
        // through its UnsafeCell does not invalidate references to other slots.
        unsafe {
            self.data
                .get_unchecked(id)
                .get()
                .write(MaybeUninit::new(val));
            self.initialized.get().write(id + 1);
        }
    }
}

impl<T> Drop for RecordBatch<T> {
    fn drop(&mut self) {
        let initialized = *self.initialized.get_mut();
        // SAFETY: UnsafeCell<MaybeUninit<T>> has T's size and alignment. This
        // prefix contains initialized, uniquely owned T values. Slice drop
        // glue also drops remaining values if a destructor panics; the Box
        // then frees the entire allocation, including its uninitialized tail.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.data.as_mut_ptr().cast::<T>(),
                initialized,
            ));
        }
    }
}

/// An append-only mapping table with stable record addresses and lock-free
/// retrieval. Insertion is serialized; records are immutable once published.
pub struct MappingTable<T> {
    next_id: Mutex<u64>,
    initialized: AtomicU64,
    batches: [UnsafeCell<Option<RecordBatch<T>>>; MAX_BATCHES],
    record_per_batch: usize,
}

// SAFETY: A single writer initializes each batch and record under next_id's
// mutex. Readers acquire the published initialized prefix before accessing
// immutable slots. T must be Send for insertion from another thread and Sync
// because references to initialized records may be shared between threads.
unsafe impl<T: Send + Sync> Sync for MappingTable<T> {}

impl<T> Default for MappingTable<T> {
    fn default() -> Self {
        Self::new(DEFAULT_RECORD_PER_BATCH)
    }
}

impl<T> MappingTable<T> {
    pub fn new(record_per_batch: usize) -> Self {
        assert!(record_per_batch > 0, "record_per_batch must be positive");
        Self {
            next_id: Mutex::new(0),
            initialized: AtomicU64::new(0),
            batches: std::array::from_fn(|_| UnsafeCell::new(None)),
            record_per_batch,
        }
    }

    pub(crate) fn new_from_iter(mapping: impl Iterator<Item = (u64, T)>) -> Self {
        let mt = Self::default();
        let mut next_id = mt.next_id.lock().unwrap();
        for (id, val) in mapping {
            assert_eq!(
                id, *next_id,
                "mapping IDs must be contiguous and start at zero"
            );
            mt.insert_locked(val, &mut next_id);
        }
        drop(next_id);
        mt
    }

    /// # Safety
    /// The batch must have been initialized under the insertion mutex, or
    /// observed through an acquire load of the published initialized prefix.
    unsafe fn get_batch(&self, batch_id: usize) -> &RecordBatch<T> {
        // SAFETY: This slot is in bounds, initialized once, and never replaced.
        unsafe {
            (&*self.batches.get_unchecked(batch_id).get())
                .as_ref()
                .unwrap_unchecked()
        }
    }

    /// Peek the number of fully initialized entries for iteration.
    pub(crate) fn peek_next_id(&self) -> u64 {
        self.initialized.load(Ordering::Acquire)
    }

    /// Get the record by an id returned by `insert`.
    ///
    /// # Panics
    /// Panics if the id has not been inserted into this table.
    #[inline]
    pub fn get(&self, id: u64) -> &T {
        assert!(
            id < self.initialized.load(Ordering::Acquire),
            "invalid mapping ID"
        );
        let batch_id = (id / self.record_per_batch as u64) as usize;
        let record_id = (id % self.record_per_batch as u64) as usize;
        // SAFETY: The acquire load makes both the batch and this record's
        // initialization visible. The assertion bounds both indices.
        unsafe { self.get_batch(batch_id).get_record(record_id) }
    }

    fn insert_locked(&self, val: T, next_id: &mut u64) -> (u64, &T) {
        let id = *next_id;
        let next = id.checked_add(1).expect("mapping IDs exhausted");
        let batch_id = id / self.record_per_batch as u64;
        let record_id = (id % self.record_per_batch as u64) as usize;
        // Check before advancing the initialized prefix or touching storage,
        // so a caught capacity panic leaves every existing record valid.
        assert!(batch_id < MAX_BATCHES as u64, "Reached max batches!");
        let batch_id = batch_id as usize;

        if record_id == 0 {
            let batch = RecordBatch::new(self.record_per_batch);
            // SAFETY: Insertions are serialized, and no record in this batch
            // has been published, so no reader can reference the batch yet.
            unsafe { self.batches[batch_id].get().write(Some(batch)) };
        }

        // SAFETY: The insertion mutex is held, this batch is initialized, and
        // id is the next uninitialized slot. Prior records remain untouched.
        let record = unsafe {
            let batch = self.get_batch(batch_id);
            batch.insert_record(record_id, val);
            batch.get_record(record_id)
        };
        *next_id = next;
        self.initialized.store(*next_id, Ordering::Release);
        (id, record)
    }

    /// Insert a record and return its id and a stable reference to it.
    ///
    /// # Panics
    /// Panics if all 128 batches have been filled.
    pub fn insert(&self, val: T) -> (u64, &T) {
        // A capacity or allocation panic cannot publish an incomplete record.
        // Recover the mutex so a caught panic does not poison later attempts.
        let mut next_id = self.next_id.lock().unwrap_or_else(|err| err.into_inner());
        self.insert_locked(val, &mut next_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{atomic::AtomicUsize, thread, Arc};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn check(test: impl Fn() + Send + Sync + 'static) {
        #[cfg(feature = "shuttle")]
        shuttle::check_random(test, 1);
        #[cfg(not(feature = "shuttle"))]
        test();
    }

    struct DropRecord(Arc<AtomicUsize>);

    impl Drop for DropRecord {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn stable_references_across_insertions_and_batches() {
        check(|| {
            let table = MappingTable::new(2);
            let (_, first) = table.insert(String::from("first"));
            let first_address = ptr::from_ref(first);
            for id in 1..9 {
                let (actual_id, record) = table.insert(id.to_string());
                assert_eq!(actual_id, id);
                assert_eq!(record, &id.to_string());
                assert_eq!(first, "first");
            }
            assert_eq!(ptr::from_ref(table.get(0)), first_address);
            assert_eq!(table.peek_next_id(), 9);
        });
    }

    #[test]
    fn invalid_ids_panic_without_affecting_records() {
        check(|| {
            let table = MappingTable::new(2);
            assert!(catch_unwind(AssertUnwindSafe(|| table.get(0))).is_err());
            table.insert(42);
            for id in [1, 2, 256, u64::MAX] {
                assert!(catch_unwind(AssertUnwindSafe(|| table.get(id))).is_err());
            }
            assert_eq!(*table.get(0), 42);
            assert_eq!(table.insert(43).0, 1);
        });
    }

    #[test]
    fn zero_batch_size_is_rejected() {
        check(|| {
            assert!(catch_unwind(|| MappingTable::<u64>::new(0)).is_err());
        });
    }

    #[test]
    fn capacity_panic_preserves_initialized_prefix_and_drops() {
        check(|| {
            let drops = Arc::new(AtomicUsize::new(0));
            let table = MappingTable::new(1);
            for id in 0..MAX_BATCHES {
                assert_eq!(table.insert(DropRecord(drops.clone())).0, id as u64);
            }
            for expected_drops in 1..=2 {
                assert!(catch_unwind(AssertUnwindSafe(|| {
                    table.insert(DropRecord(drops.clone()));
                }))
                .is_err());
                assert_eq!(table.peek_next_id(), MAX_BATCHES as u64);
                assert!(Arc::ptr_eq(&table.get((MAX_BATCHES - 1) as u64).0, &drops));
                assert_eq!(drops.load(Ordering::Relaxed), expected_drops);
            }
            drop(table);
            assert_eq!(drops.load(Ordering::Relaxed), MAX_BATCHES + 2);
        });
    }

    #[test]
    fn drops_only_initialized_records_in_partial_batches() {
        check(|| {
            let drops = Arc::new(AtomicUsize::new(0));
            for count in [0, 1, 2, 3, 7] {
                let table = MappingTable::new(3);
                for _ in 0..count {
                    table.insert(DropRecord(drops.clone()));
                }
                drops.store(0, Ordering::Relaxed);
                drop(table);
                assert_eq!(drops.load(Ordering::Relaxed), count);
            }
        });
    }

    #[test]
    fn allocation_layout_panic_does_not_publish_or_leak_a_record() {
        check(|| {
            let drops = Arc::new(AtomicUsize::new(0));
            let table = MappingTable::new(usize::MAX);
            for expected_drops in 1..=2 {
                assert!(catch_unwind(AssertUnwindSafe(|| {
                    table.insert(DropRecord(drops.clone()));
                }))
                .is_err());
                assert_eq!(table.peek_next_id(), 0);
                assert_eq!(drops.load(Ordering::Relaxed), expected_drops);
            }
            drop(table);
            assert_eq!(drops.load(Ordering::Relaxed), 2);
        });
    }

    #[test]
    fn zero_sized_and_overaligned_records() {
        check(|| {
            let table = MappingTable::new(1);
            for id in 0..4 {
                assert_eq!(table.insert(()).0, id);
                assert_eq!(*table.get(id), ());
            }

            #[repr(align(256))]
            struct Aligned(u64);
            let table = MappingTable::new(2);
            for id in 0..5 {
                table.insert(Aligned(id));
                let record = table.get(id);
                assert_eq!(record.0, id);
                assert_eq!(ptr::from_ref(record).addr() % 256, 0);
            }
        });
    }

    #[test]
    fn restore_contiguous_ids_and_continue_inserting() {
        check(|| {
            let table = MappingTable::new_from_iter((0..4).map(|id| (id, id * 2)));
            for id in 0..4 {
                assert_eq!(*table.get(id), id * 2);
            }
            assert_eq!(table.insert(8).0, 4);
            assert_eq!(table.peek_next_id(), 5);
        });
    }

    #[test]
    fn invalid_restore_drops_every_input_once() {
        check(|| {
            for ids in [[1, 2, 3], [0, 2, 3], [0, 0, 1], [0, 1, u64::MAX]] {
                let drops = Arc::new(AtomicUsize::new(0));
                let records = ids.map(|id| (id, DropRecord(drops.clone())));
                assert!(catch_unwind(AssertUnwindSafe(|| {
                    MappingTable::new_from_iter(records.into_iter());
                }))
                .is_err());
                assert_eq!(drops.load(Ordering::Relaxed), ids.len());
            }
        });
    }

    #[test]
    fn panicking_destructor_still_drops_remaining_batches() {
        check(|| {
            struct PanickingRecord {
                drops: Arc<AtomicUsize>,
                panic: bool,
            }
            impl Drop for PanickingRecord {
                fn drop(&mut self) {
                    self.drops.fetch_add(1, Ordering::Relaxed);
                    assert!(!self.panic, "intentional destructor panic");
                }
            }
            let drops = Arc::new(AtomicUsize::new(0));
            let table = MappingTable::new(2);
            for id in 0..5 {
                table.insert(PanickingRecord {
                    drops: drops.clone(),
                    panic: id == 0,
                });
            }
            assert!(catch_unwind(AssertUnwindSafe(|| drop(table))).is_err());
            assert_eq!(drops.load(Ordering::Relaxed), 5);
        });
    }

    #[test]
    fn concurrent_insertions_publish_fully_initialized_records() {
        let test = || {
            let table = Arc::new(MappingTable::new(2));
            table.insert([0_u64; 8]);
            let mut workers = Vec::new();
            for worker in 1..=2_u64 {
                let table = table.clone();
                workers.push(thread::spawn(move || {
                    let first = table.get(0);
                    for _ in 0..4 {
                        let (id, record) = table.insert([worker; 8]);
                        assert_eq!(*record, [worker; 8]);
                        assert_eq!(record, table.get(id));
                        for id in 0..table.peek_next_id() {
                            let record = table.get(id);
                            assert_eq!(*record, [record[0]; 8]);
                            assert!(record[0] <= 2);
                        }
                        assert_eq!(*first, [0; 8]);
                    }
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
            assert_eq!(table.peek_next_id(), 9);
        };
        #[cfg(feature = "shuttle")]
        shuttle::check_random(test, 100);
        #[cfg(not(feature = "shuttle"))]
        test();
    }
}
