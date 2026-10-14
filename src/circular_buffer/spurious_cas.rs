// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::cell::Cell;
use std::sync::atomic::{AtomicU8 as StdAtomicU8, Ordering};

use super::{CircularBuffer, MetaState, CB_ALLOC_META_SIZE};

thread_local! {
    static FAIL_NEXT_WEAK: Cell<bool> = const { Cell::new(false) };
}

// Keep the native allocation metadata layout while injecting only into unit tests.
#[repr(transparent)]
pub(super) struct AtomicU8(StdAtomicU8);

impl AtomicU8 {
    pub(super) fn new(value: u8) -> Self {
        Self(StdAtomicU8::new(value))
    }

    pub(super) fn load(&self, order: Ordering) -> u8 {
        self.0.load(order)
    }

    pub(super) fn compare_exchange(
        &self,
        current: u8,
        new: u8,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u8, u8> {
        self.0.compare_exchange(current, new, success, failure)
    }

    #[allow(dead_code)]
    pub(super) fn compare_exchange_weak(
        &self,
        current: u8,
        new: u8,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u8, u8> {
        let observed = self.0.load(failure);
        if observed == current && FAIL_NEXT_WEAK.with(|armed| armed.replace(false)) {
            return Err(observed);
        }
        self.0.compare_exchange_weak(current, new, success, failure)
    }
}

struct SpuriousFailure;

impl SpuriousFailure {
    fn arm() -> Self {
        FAIL_NEXT_WEAK.with(|armed| assert!(!armed.replace(true)));
        Self
    }
}

impl Drop for SpuriousFailure {
    fn drop(&mut self) {
        FAIL_NEXT_WEAK.with(|armed| armed.set(false));
    }
}

#[test]
fn injector_models_a_permitted_spurious_failure() {
    let state = AtomicU8::new(MetaState::FreeListed.into());
    let _failure = SpuriousFailure::arm();
    assert_eq!(
        state.compare_exchange_weak(
            MetaState::FreeListed.into(),
            MetaState::NotReady.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ),
        Err(MetaState::FreeListed.into())
    );
    assert_eq!(
        state.load(Ordering::Acquire),
        u8::from(MetaState::FreeListed)
    );
    assert_eq!(
        state.compare_exchange(
            MetaState::FreeListed.into(),
            MetaState::NotReady.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ),
        Ok(MetaState::FreeListed.into())
    );
    assert_eq!(
        state.compare_exchange(
            MetaState::FreeListed.into(),
            MetaState::NotReady.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ),
        Err(MetaState::NotReady.into())
    );
}

#[test]
fn reuse_does_not_orphan_a_freelisted_entry() {
    let buffer = CircularBuffer::new(65536, 0.1, 64, 1952, 4096, 32, None, false);
    let size = buffer.free_list.size_classes[0];
    let allocation = buffer.alloc(size).unwrap();
    let original = allocation.as_ptr();
    drop(allocation);
    buffer.dealloc(unsafe { buffer.acquire_exclusive_dealloc_handle(original).unwrap() });
    assert_eq!(buffer.get_metrics().free_listed_cnt, 1);

    let _failure = SpuriousFailure::arm();
    let allocation = buffer.alloc(size).unwrap();
    let reused = allocation.as_ptr();
    drop(allocation);
    buffer.dealloc(unsafe { buffer.acquire_exclusive_dealloc_handle(reused).unwrap() });

    // Fail before eviction on the old code, rather than hanging on its orphan.
    assert_eq!(
        reused, original,
        "a spurious CAS failure lost the free entry"
    );
    let metrics = buffer.get_metrics();
    assert_eq!(metrics.allocated_cnt, 1);
    assert_eq!(metrics.free_listed_cnt, 1);
    assert_eq!(metrics.ready_cnt, 0);
    assert_eq!(metrics.not_ready_cnt, 0);
    assert_eq!(
        buffer
            .evict_n(1, |_| panic!(
                "free-list eviction must not call the callback"
            ))
            .unwrap() as usize,
        size + CB_ALLOC_META_SIZE
    );
    let metrics = buffer.get_metrics();
    assert_eq!(metrics.head_tail_distance, 0);
    assert_eq!(metrics.allocated_cnt, 0);
    assert_eq!(metrics.free_listed_cnt, 0);
}
