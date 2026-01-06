// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
};

use crate::sync::atomic::{AtomicU32, Ordering};
use crate::utils::atomic_wait;

/// A RWLock with upgrade operation.
/// We use this because std RWLock don't allow lock upgrade.
///
/// Borrowed from https://marabos.nl/atomics/building-locks.html#reader-writer-lock
pub struct RwLock<T> {
    val: UnsafeCell<T>,
    lock_val: AtomicU32,
    writer_wake_counter: AtomicU32,
}

impl<T> RwLock<T> {
    pub fn new(val: T) -> Self {
        Self {
            val: UnsafeCell::new(val),
            lock_val: AtomicU32::new(0),
            writer_wake_counter: AtomicU32::new(0),
        }
    }

    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        let mut v = self.lock_val.load(Ordering::Relaxed);
        loop {
            if v.is_multiple_of(2) {
                match self.lock_val.compare_exchange_weak(
                    v,
                    v + 2,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return RwLockReadGuard { lock: self },
                    Err(e) => v = e,
                }
            }

            if v % 2 == 1 {
                atomic_wait::wait(&self.lock_val, v);
                v = self.lock_val.load(Ordering::Relaxed);
            }
        }
    }

    pub fn try_read(&self) -> Result<RwLockReadGuard<'_, T>, ()> {
        let v = self.lock_val.load(Ordering::Relaxed);

        if v.is_multiple_of(2) {
            let new_v = v + 2;

            match self.lock_val.compare_exchange_weak(
                v,
                new_v,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => Ok(RwLockReadGuard { lock: self }),
                Err(_) => Err(()),
            }
        } else {
            Err(())
        }
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        let mut s = self.lock_val.load(Ordering::Relaxed);
        loop {
            if s <= 1 {
                match self.lock_val.compare_exchange(
                    s,
                    u32::MAX,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return RwLockWriteGuard { lock: self },
                    Err(e) => {
                        s = e;
                        continue;
                    }
                }
            }

            if s.is_multiple_of(2) {
                match self
                    .lock_val
                    .compare_exchange(s, s + 1, Ordering::Relaxed, Ordering::Relaxed)
                {
                    Ok(_) => {}
                    Err(e) => {
                        s = e;
                        continue;
                    }
                }
            }

            let w = self.writer_wake_counter.load(Ordering::Acquire);
            s = self.lock_val.load(Ordering::Relaxed);
            if s >= 2 {
                atomic_wait::wait(&self.writer_wake_counter, w);
                s = self.lock_val.load(Ordering::Relaxed);
            }
        }
    }

    pub fn try_write(&self) -> Result<RwLockWriteGuard<'_, T>, ()> {
        let s = self.lock_val.load(Ordering::Relaxed);
        if s <= 1 {
            match self.lock_val.compare_exchange_weak(
                s,
                u32::MAX,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(RwLockWriteGuard { lock: self }),
                Err(_) => return Err(()),
            }
        }
        Err(())
    }
}

pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        if self.lock.lock_val.fetch_sub(2, Ordering::Release) == 3 {
            self.lock
                .writer_wake_counter
                .fetch_add(1, Ordering::Release);
            atomic_wait::wake_one(&self.lock.writer_wake_counter);
        }
    }
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T> RwLockReadGuard<'a, T> {
    pub fn try_upgrade(self) -> Result<RwLockWriteGuard<'a, T>, RwLockReadGuard<'a, T>> {
        let old_v = 2;

        match self.lock.lock_val.compare_exchange_weak(
            old_v,
            u32::MAX,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let lock = self.lock;
                std::mem::forget(self);
                Ok(RwLockWriteGuard { lock })
            }
            Err(_e) => Err(self),
        }
    }

    pub(crate) fn as_ref(&self) -> &T {
        unsafe { &*self.lock.val.get() }
    }
}

pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.lock_val.store(0, Ordering::Release);
        self.lock
            .writer_wake_counter
            .fetch_add(1, Ordering::Release);
        atomic_wait::wake_one(&self.lock.writer_wake_counter);
        atomic_wait::wake_all(&self.lock.lock_val);
    }
}

impl<T> RwLockWriteGuard<'_, T> {
    pub(crate) fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.val.get() }
    }

    pub(crate) fn as_ref(&self) -> &T {
        unsafe { &*self.lock.val.get() }
    }
}
