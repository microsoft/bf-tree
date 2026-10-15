// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(all(feature = "shuttle", test))]
use crate::sync::atomic::AtomicU32;

#[cfg(not(all(feature = "shuttle", test)))]
pub(crate) use ::atomic_wait::{wait, wake_all, wake_one};

/// If the value is `value`, wait until woken up.
///
/// This function might also return spuriously,
/// without a corresponding wake operation.
#[cfg(all(feature = "shuttle", test))]
#[inline]
pub(crate) fn wait(_atomic: &AtomicU32, _value: u32) {
    shuttle::thread::yield_now();
}

/// Wake one thread that is waiting on this atomic.
///
/// It's okay if the pointer dangles or is null.
#[cfg(all(feature = "shuttle", test))]
#[inline]
pub(crate) fn wake_one(_atomic: *const AtomicU32) {
    shuttle::thread::yield_now();
}

/// Wake all threads that are waiting on this atomic.
///
/// It's okay if the pointer dangles or is null.
#[cfg(all(feature = "shuttle", test))]
#[inline]
pub(crate) fn wake_all(_atomic: *const AtomicU32) {
    shuttle::thread::yield_now();
}
