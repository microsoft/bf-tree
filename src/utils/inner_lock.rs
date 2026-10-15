// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::sync::atomic::Ordering;
use std::cell::UnsafeCell;

use crate::{error::TreeError, nodes::InnerNode};

pub(crate) fn is_locked(version: u64) -> bool {
    (version & 0b10) == 0b10
}

#[derive(Debug)]
pub(crate) struct ReadGuard<'a> {
    version: u64,
    node: &'a UnsafeCell<InnerNode>,
}

impl<'a> ReadGuard<'a> {
    pub(crate) fn new(v: u64, node: &'a InnerNode) -> Self {
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

    pub(crate) fn check_version(&self) -> Result<u64, TreeError> {
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
    /// The returned reference cannot outlive the borrow of this lock guard.
    pub(crate) fn as_ref(&self) -> &InnerNode {
        unsafe { &*self.node.get() }
    }

    /// Keep the mutable borrow tied to the guard, so it cannot be borrowed
    /// again or released while the returned reference is still in use.
    pub(crate) fn as_mut(&mut self) -> &mut InnerNode {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{InnerNodeBuilder, PageID};

    #[test]
    fn write_guard_reborrows_and_releases_the_version_lock() {
        let test = || {
            let mut builder = InnerNodeBuilder::new();
            builder
                .set_children_is_leaf(true)
                .set_left_most_page_id(PageID::from_id(0));
            let ptr = builder.build(crate::snapshot::INVALID_SNAPSHOT_VERSION);
            // SAFETY: The newly allocated node is exclusively owned here, and
            // no reference survives the test's final free_node call.
            let node = unsafe { &*ptr.cast::<UnsafeCell<InnerNode>>() };
            unsafe { &*std::ptr::addr_of!((*ptr).version_lock) }.store(2, Ordering::Relaxed);
            {
                let mut guard = WriteGuard { node };
                guard.as_mut().disk_offset = 17;
                assert_eq!(guard.as_ref().disk_offset, 17);
                guard.as_mut().disk_offset = 23;
                assert_eq!(guard.as_ref().disk_offset, 23);
            }
            assert_eq!(unsafe { &*ptr }.version_lock.load(Ordering::Acquire), 4);
            assert_eq!(unsafe { &*ptr }.disk_offset, 23);
            InnerNode::free_node(ptr);
        };
        #[cfg(feature = "shuttle")]
        shuttle::check_random(test, 1);
        #[cfg(not(feature = "shuttle"))]
        test();
    }
}
