// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::cell::Cell;

thread_local! {
    static FAIL_NEXT_WEAK: Cell<bool> = const { Cell::new(false) };
}

mod sync {
    pub mod atomic {
        pub use std::sync::atomic::Ordering;

        pub struct AtomicU32(std::sync::atomic::AtomicU32);

        impl AtomicU32 {
            pub fn new(value: u32) -> Self {
                Self(std::sync::atomic::AtomicU32::new(value))
            }

            pub fn compare_exchange_weak(
                &self,
                current: u32,
                new: u32,
                success: Ordering,
                failure: Ordering,
            ) -> Result<u32, u32> {
                let observed = self.0.load(failure);
                if observed == current && crate::FAIL_NEXT_WEAK.with(|armed| armed.replace(false)) {
                    return Err(observed);
                }
                self.0.compare_exchange_weak(current, new, success, failure)
            }
        }

        impl std::ops::Deref for AtomicU32 {
            type Target = std::sync::atomic::AtomicU32;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    }
}

mod utils {
    pub mod atomic_wait {
        use crate::sync::atomic::AtomicU32;
        pub fn wait(_: &AtomicU32, _: u32) {
            panic!("nonblocking operations must not wait");
        }
        pub fn wake_one(_: *const AtomicU32) {}
        pub fn wake_all(_: *const AtomicU32) {}
    }
}

#[allow(dead_code)]
#[path = "../src/utils/rw_lock.rs"]
mod rw_lock;

#[test]
fn try_read_and_write_ignore_spurious_failures_but_reject_contention() {
    let lock = rw_lock::RwLock::new(7);
    FAIL_NEXT_WEAK.with(|armed| armed.set(true));
    let reader = lock
        .try_read()
        .expect("uncontended read must not fail spuriously");
    assert_eq!(*reader, 7);
    assert!(lock.try_write().is_err());
    drop(reader);
    FAIL_NEXT_WEAK.with(|armed| armed.set(true));
    let mut writer = lock
        .try_write()
        .expect("uncontended write must not fail spuriously");
    *writer = 9;
    assert!(lock.try_read().is_err());
    assert!(lock.try_write().is_err());
    drop(writer);
    FAIL_NEXT_WEAK.with(|armed| armed.set(false));
    assert_eq!(*lock.try_read().unwrap(), 9);
}
