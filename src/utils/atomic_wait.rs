// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/// Adapted from https://github.com/m-ou-se/atomic-wait
use crate::sync::atomic::AtomicU32;

#[cfg(all(target_os = "linux", not(feature = "shuttle")))]
mod platform {
    use core::sync::atomic::AtomicU32;
    #[inline]
    pub fn wait(a: &AtomicU32, expected: u32) {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                a,
                libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                expected,
                core::ptr::null::<libc::timespec>(),
            );
        };
    }

    #[inline]
    pub fn wake_one(ptr: *const AtomicU32) {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                ptr,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1i32,
            );
        };
    }

    #[inline]
    pub fn wake_all(ptr: *const AtomicU32) {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                ptr,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                i32::MAX,
            );
        };
    }
}

#[cfg(all(feature = "shuttle", test))]
mod platform {
    use shuttle::sync::atomic::AtomicU32;

    #[inline]
    pub fn wait(_a: &AtomicU32, _expected: u32) {
        shuttle::thread::yield_now();
    }

    #[inline]
    pub fn wake_one(_ptr: *const AtomicU32) {
        shuttle::thread::yield_now();
    }

    #[inline]
    pub fn wake_all(_ptr: *const AtomicU32) {
        shuttle::thread::yield_now();
    }
}

#[cfg(all(target_os = "macos", not(feature = "shuttle")))]
/// We don't do anything for macOS, just to make sure it compiles and correct.
mod platform {
    #[inline]
    pub fn wait(_a: &std::sync::atomic::AtomicU32, _expected: u32) {}

    #[inline]
    pub fn wake_one(_ptr: *const std::sync::atomic::AtomicU32) {}

    #[inline]
    pub fn wake_all(_ptr: *const std::sync::atomic::AtomicU32) {}
}

#[cfg(all(target_os = "windows", not(feature = "shuttle")))]
mod platform {
    use core::sync::atomic::AtomicU32;
    use windows_sys::Win32::System::Threading::{
        WaitOnAddress, WakeByAddressAll, WakeByAddressSingle, INFINITE,
    };

    #[inline]
    pub fn wait(a: &AtomicU32, expected: u32) {
        let ptr: *const AtomicU32 = a;
        let expected_ptr: *const u32 = &expected;
        unsafe { WaitOnAddress(ptr.cast(), expected_ptr.cast(), 4, INFINITE) };
    }

    #[inline]
    pub fn wake_one(ptr: *const AtomicU32) {
        unsafe { WakeByAddressSingle(ptr.cast()) };
    }

    #[inline]
    pub fn wake_all(ptr: *const AtomicU32) {
        unsafe { WakeByAddressAll(ptr.cast()) };
    }
}

/// If the value is `value`, wait until woken up.
///
/// This function might also return spuriously,
/// without a corresponding wake operation.
#[inline]
pub fn wait(atomic: &AtomicU32, value: u32) {
    platform::wait(atomic, value)
}

/// Wake one thread that is waiting on this atomic.
///
/// It's okay if the pointer dangles or is null.
#[inline]
pub fn wake_one(atomic: *const AtomicU32) {
    platform::wake_one(atomic);
}

/// Wake all threads that are waiting on this atomic.
///
/// It's okay if the pointer dangles or is null.
#[inline]
pub fn wake_all(atomic: *const AtomicU32) {
    platform::wake_all(atomic);
}
