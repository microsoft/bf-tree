// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::sync::atomic::Ordering;
use std::cell::UnsafeCell;

use crate::{error::TreeError, nodes::InnerNode};

pub(crate) fn is_locked(version: u16) -> bool {
    (version & 0b10) == 0b10
}

#[derive(Debug)]
pub(crate) struct ReadGuard<'a> {
    version: u16,
    node: &'a UnsafeCell<InnerNode>,
}

impl<'a> ReadGuard<'a> {
    pub(crate) fn new(v: u16, node: &'a InnerNode) -> Self {
        Self {
            version: v,
            node: unsafe { &*(node as *const InnerNode as *const UnsafeCell<InnerNode>) }, // todo: the caller should pass UnsafeCell<BaseNode> instead
        }
    }
    pub(crate) fn try_read(ptr: *const InnerNode) -> Result<ReadGuard<'a>, TreeError> {
        let node = unsafe { &*ptr };
        let v = node.version_lock.load(Ordering::Acquire);
        if is_locked(v) {
            Err(TreeError::Locked)
        } else {
            Ok(Self::new(v, node))
        }
    }

    pub(crate) fn check_version(&self) -> Result<u16, TreeError> {
        let v = self.as_ref().version_lock.load(Ordering::Acquire);

        if v == self.version {
            Ok(v)
        } else {
            Err(TreeError::Locked)
        }
    }

    pub(crate) fn as_ref(&self) -> &InnerNode {
        unsafe { &*self.node.get() }
    }

    pub(crate) fn upgrade(self) -> Result<WriteGuard<'a>, (Self, TreeError)> {
        let new_version = self.version + 0b10;
        match self.as_ref().version_lock.compare_exchange_weak(
            self.version,
            new_version,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(WriteGuard { node: self.node }),
            Err(_v) => Err((self, TreeError::Locked)),
        }
    }
}

#[derive(Debug)]
pub struct WriteGuard<'a> {
    pub(crate) node: &'a UnsafeCell<InnerNode>,
}

impl<'a> WriteGuard<'a> {
    pub(crate) fn as_ref(&self) -> &'a InnerNode {
        unsafe { &*self.node.get() }
    }

    pub(crate) fn as_mut(&mut self) -> &'a mut InnerNode {
        unsafe { &mut *self.node.get() }
    }

    #[allow(dead_code)]
    pub(crate) fn mark_obsolete(&mut self) {
        self.as_mut()
            .version_lock
            .fetch_add(0b01, Ordering::Release);
    }

    pub(crate) fn downgrade(self) -> ReadGuard<'a> {
        let new_v = self
            .as_ref()
            .version_lock
            .fetch_add(0b10, Ordering::Release)
            + 0b10;
        let n = self.node.get();
        let rt = ReadGuard::new(new_v, unsafe { &*n });
        std::mem::forget(self);
        rt
    }
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        self.as_mut()
            .version_lock
            .fetch_add(0b10, Ordering::Release);
    }
}
