// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::{
    sync::{Mutex, MutexGuard},
    BfTree,
};
use std::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::TryLockError,
};

/// Default block sizes to use
const DEFAULT_FREE_LIST_SIZE_CLASSES: &[usize] = &[4096, 2048, 1024, 512, 256, 64];

#[derive(Debug)]
pub(crate) struct ListNode {
    pub next: *mut ListNode,
}

impl ListNode {
    /// casting a *u8 to *ListNode can be UB because of alignment
    pub(crate) fn from_u8_ptr_unchecked(addr: *mut u8) -> *mut ListNode {
        debug_assert!((addr as usize).is_multiple_of(std::mem::align_of::<ListNode>()));
        addr as *mut ListNode
    }
}

#[derive(Debug)]
pub enum FreeListError {
    WouldBlock,
    SizeTooSmall,
    // Add more error types as needed
}

#[derive(Debug)]
pub(super) struct FreeList {
    pub(crate) size_classes: Vec<usize>,
    list_heads: Vec<Mutex<*mut ListNode>>,
}

unsafe impl Send for FreeList {}
unsafe impl Sync for FreeList {}

impl Default for FreeList {
    fn default() -> Self {
        let size_classes = DEFAULT_FREE_LIST_SIZE_CLASSES.to_vec();
        let mut list_heads: Vec<Mutex<*mut ListNode>> =
            Vec::with_capacity(DEFAULT_FREE_LIST_SIZE_CLASSES.len());

        for _ in 0..size_classes.len() {
            list_heads.push(Mutex::new(std::ptr::null_mut()));
        }

        Self {
            size_classes,
            list_heads,
        }
    }
}

impl FreeList {
    pub(super) fn new(
        min_record_size: usize,
        max_record_size: usize,
        leaf_page_size: usize,
        max_fence_len: usize,
        cache_only: bool,
    ) -> Self {
        let size_classes = Self::create_free_list_size_classes(
            min_record_size,
            max_record_size,
            leaf_page_size,
            max_fence_len,
            cache_only,
        );
        let mut list_heads: Vec<Mutex<*mut ListNode>> = Vec::with_capacity(size_classes.len());
        for _ in 0..size_classes.len() {
            list_heads.push(Mutex::new(std::ptr::null_mut()));
        }

        Self {
            size_classes,
            list_heads,
        }
    }

    /// Create the corresponding free list size classes based on mini-page size classes
    fn create_free_list_size_classes(
        min_record_size: usize,
        max_record_size: usize,
        leaf_page_size: usize,
        max_fence_len: usize,
        cache_only: bool,
    ) -> Vec<usize> {
        let mut size_classes = BfTree::create_mem_page_size_classes(
            min_record_size,
            max_record_size,
            leaf_page_size,
            max_fence_len,
            cache_only,
        );

        // Reverse the order
        size_classes.reverse();

        size_classes
    }

    fn size_class_smaller_than(&self, size: usize) -> usize {
        self.size_classes
            .iter()
            .position(|&s| s <= size)
            .expect("size too small")
    }

    fn size_class_larger_than(&self, size: usize) -> usize {
        let pos = self
            .size_classes
            .iter()
            .rev()
            .position(|&s| s >= size)
            .expect("size too large");
        self.size_classes.len() - 1 - pos
    }
    /// Returns the ptr and the size class.
    /// return size >= requested size.
    #[cfg_attr(feature = "tracing", tracing::instrument)]
    pub(super) fn remove(&self, size: usize) -> Option<NonNull<u8>> {
        let size_class_idx = self.size_class_larger_than(size);
        let mut node = self.list_heads[size_class_idx].lock().unwrap();

        if node.is_null() {
            return None;
        }
        let old = *node.deref();
        let new = unsafe { (*(*node.deref())).next };
        *node.deref_mut() = new;
        Some(NonNull::new(old as *mut u8).unwrap())
    }

    /// Error if can't acquire lock
    ///
    /// Returns the lock guard so that caller can set appropriate metadata.
    #[cfg_attr(feature = "tracing", tracing::instrument)]
    pub(super) fn try_add(
        &self,
        ptr: *mut u8,
        size: usize,
    ) -> Result<MutexGuard<'_, *mut ListNode>, FreeListError> {
        if size < *self.size_classes.last().unwrap() {
            return Err(FreeListError::SizeTooSmall);
        }

        let size_class_idx = self.size_class_smaller_than(size);
        let mut head = match self.list_heads[size_class_idx].try_lock() {
            Ok(v) => v,
            Err(TryLockError::WouldBlock) => return Err(FreeListError::WouldBlock),
            Err(TryLockError::Poisoned(_)) => panic!("poisoned lock"),
        };
        debug_assert!(std::mem::size_of::<ListNode>() <= self.size_classes[size_class_idx]);
        debug_assert!(std::mem::align_of::<ListNode>() <= self.size_classes[size_class_idx]);

        let node = ListNode::from_u8_ptr_unchecked(ptr);
        unsafe { (*node).next = *head };
        *head = node;
        Ok(head)
    }

    /// Returns false if not found
    ///
    /// Assumption:
    ///   No two threads may call this function at the same time!
    ///
    #[cfg_attr(feature = "tracing", tracing::instrument)]
    pub(super) fn find_and_remove(&self, ptr: *mut u8, size: usize) -> bool {
        let size_class_idx = self.size_class_smaller_than(size);
        let mut node_guard = self.list_heads[size_class_idx].lock().unwrap();
        let mut node = *node_guard.deref_mut();
        let mut prev: *mut ListNode = std::ptr::null_mut();
        loop {
            if node.is_null() {
                return false;
            }
            if node as *mut u8 == ptr {
                if prev.is_null() {
                    *node_guard.deref_mut() = unsafe { (*node).next };
                } else {
                    unsafe { (*prev).next = (*node).next };
                }
                return true;
            }
            prev = node;
            node = unsafe { (*node).next };
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(64, 1952, 4096)] // 1 leaf page = 1 disk page
    #[case(3072, 3072, 8192)] // 1 leaf page = 1 disk page, uniform record size
    #[case(64, 2048, 16384)] // 1 leaf page = 4 disk page
    fn test_new_initialization(
        #[case] min_record_size: usize,
        #[case] max_record_size: usize,
        #[case] leaf_page_size: usize,
    ) {
        let free_list = FreeList::new(min_record_size, max_record_size, leaf_page_size, 32, false);
        // Verify each list head is initialized to null.
        for head in free_list.list_heads.iter() {
            assert!(head.lock().unwrap().is_null());
        }
    }

    #[test]
    fn test_remove_empty() {
        let free_list = FreeList::default();
        assert!(free_list.remove(64).is_none());
    }

    #[test]
    fn test_add_and_remove() {
        let free_list = FreeList::new(32, 1952, 4096, 32, false);
        let block = Box::into_raw(Box::new([0u8; 64])); // Allocate a block of 64 bytes.
        let lock_guard = free_list.try_add(block as *mut u8, 64).unwrap();
        drop(lock_guard);
        let removed = free_list.remove(64).unwrap();
        assert_eq!(removed.as_ptr(), block as *mut u8);
        // Cleanup
        unsafe {
            _ = Box::from_raw(block);
        }
    }

    #[test]
    fn test_find_and_remove() {
        let free_list = FreeList::new(32, 1952, 4096, 32, false);
        let block = Box::into_raw(Box::new([0u8; 64]));
        let lock_guard = free_list.try_add(block as *mut u8, 64).unwrap();
        drop(lock_guard);
        assert!(free_list.find_and_remove(block as *mut u8, 64));
        // Verify removal
        assert!(!free_list.find_and_remove(block as *mut u8, 64));
        // Cleanup
        unsafe {
            _ = Box::from_raw(block);
        }
    }

    use crate::sync::thread;
    use crate::sync::{Arc, Barrier};

    #[test]
    fn test_multithreaded_access() {
        let free_list = Arc::new(FreeList::default());
        let n_threads = 10;
        let barrier = Arc::new(Barrier::new(n_threads));
        let mut handles = vec![];

        for _ in 0..n_threads {
            let fl = Arc::clone(&free_list);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait(); // Ensure all threads start simultaneously.
                let block = Box::into_raw(Box::new([0u8; 64]));
                if let Ok(lock_guard) = fl.try_add(block as *mut u8, 64) {
                    drop(lock_guard);
                    let removed = fl.find_and_remove(block as *mut u8, 64);
                    assert!(removed);
                }
                // Cleanup
                unsafe {
                    _ = Box::from_raw(block);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
