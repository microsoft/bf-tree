// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

mod sync {
    pub mod atomic {
        pub use std::sync::atomic::Ordering;
    }
}

mod error {
    #[derive(Debug, PartialEq, Eq)]
    pub enum TreeError {
        Locked,
    }
}

#[derive(Debug)]
struct InjectedAtomic {
    value: AtomicU64,
    fail_next_weak: Cell<bool>,
}

impl InjectedAtomic {
    fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
            fail_next_weak: Cell::new(true),
        }
    }

    fn load(&self, order: Ordering) -> u64 {
        self.value.load(order)
    }

    fn fetch_add(&self, value: u64, order: Ordering) -> u64 {
        self.value.fetch_add(value, order)
    }

    fn compare_exchange(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        self.value.compare_exchange(current, new, success, failure)
    }

    fn compare_exchange_weak(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        let observed = self.value.load(failure);
        if self.fail_next_weak.replace(false) && observed == current {
            return Err(observed);
        }
        self.value
            .compare_exchange_weak(current, new, success, failure)
    }
}

mod nodes {
    #[derive(Debug)]
    pub struct InnerNode {
        pub(crate) version_lock: super::InjectedAtomic,
    }
}

// Compile the production helper, not a copy, against the only node field it
// accesses. Fault injection stays out of the library and is independent of ISA.
#[allow(dead_code)]
#[path = "../src/utils/inner_lock.rs"]
mod inner_lock;

fn node() -> nodes::InnerNode {
    nodes::InnerNode {
        version_lock: InjectedAtomic::new(),
    }
}

#[test]
fn injector_models_a_permitted_spurious_failure() {
    let atomic = InjectedAtomic::new();
    assert_eq!(
        atomic.compare_exchange_weak(0, 2, Ordering::Release, Ordering::Relaxed),
        Err(0)
    );
    assert_eq!(atomic.load(Ordering::Relaxed), 0);
    assert_eq!(
        atomic.compare_exchange(0, 2, Ordering::Release, Ordering::Relaxed),
        Ok(0)
    );
}

#[test]
fn private_node_upgrade_does_not_fail_spuriously() {
    let node = node();
    let reader = inner_lock::ReadGuard::try_read(&node).unwrap();
    let writer = reader
        .upgrade()
        .expect("an unchanged private node must upgrade without caller retries");
    assert_eq!(node.version_lock.load(Ordering::Relaxed), 2);
    drop(writer);
    assert_eq!(node.version_lock.load(Ordering::Relaxed), 4);
}

#[test]
fn held_write_lock_still_rejects_read_and_upgrade() {
    let node = node();
    let stale = inner_lock::ReadGuard::try_read(&node).unwrap();
    let writer = inner_lock::ReadGuard::try_read(&node)
        .unwrap()
        .upgrade()
        .unwrap();
    assert!(matches!(
        inner_lock::ReadGuard::try_read(&node),
        Err(error::TreeError::Locked)
    ));
    let (stale, error) = stale.upgrade().unwrap_err();
    assert_eq!(error, error::TreeError::Locked);
    assert_eq!(stale.check_version(), Err(error::TreeError::Locked));
    assert_eq!(node.version_lock.load(Ordering::Relaxed), 2);
    drop(writer);
    assert_eq!(node.version_lock.load(Ordering::Relaxed), 4);
}

#[test]
fn completed_writer_still_invalidates_a_stale_reader() {
    let node = node();
    let stale = inner_lock::ReadGuard::try_read(&node).unwrap();
    drop(
        inner_lock::ReadGuard::try_read(&node)
            .unwrap()
            .upgrade()
            .unwrap(),
    );
    let (_, error) = stale.upgrade().unwrap_err();
    assert_eq!(error, error::TreeError::Locked);
    assert_eq!(node.version_lock.load(Ordering::Relaxed), 4);
}

#[test]
fn downgrade_retains_version_and_allows_a_fresh_upgrade() {
    let node = node();
    let reader = inner_lock::ReadGuard::try_read(&node)
        .unwrap()
        .upgrade()
        .unwrap()
        .downgrade();
    assert_eq!(reader.check_version().unwrap(), 4);
    let writer = reader.upgrade().unwrap();
    assert_eq!(node.version_lock.load(Ordering::Relaxed), 6);
    drop(writer);
    assert_eq!(node.version_lock.load(Ordering::Relaxed), 8);
}
