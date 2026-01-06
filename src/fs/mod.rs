// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

mod memory_vfs;
mod std_vfs;

#[cfg(target_os = "linux")]
mod std_direct_vfs;
use std::sync::atomic::Ordering;

#[cfg(target_os = "linux")]
pub(crate) use std_direct_vfs::StdDirectVfs;

#[cfg(target_os = "linux")]
mod io_uring_vfs;
#[cfg(target_os = "linux")]
pub(crate) use io_uring_vfs::IoUringVfs;

#[cfg(all(target_os = "linux", feature = "spdk"))]
mod spdk_vfs;
#[cfg(all(target_os = "linux", feature = "spdk"))]
pub(crate) use spdk_vfs::SpdkVfs;

pub(crate) use memory_vfs::MemoryVfs;
pub(crate) use std_vfs::StdVfs;

use crate::nodes::DISK_PAGE_SIZE;

/// Similar to `std::io::Write` and `std::io::Read`, but without &mut self, i.e., no locking
pub(crate) trait VfsImpl: Send + Sync {
    fn read(&self, offset: usize, buf: &mut [u8]);

    fn write(&self, offset: usize, buf: &[u8]);

    /// Allocate a new page returns the physical offset of the page.
    /// The size of the page is a multiple of DISK_PAGE_SIZE
    fn alloc_offset(&self, size: usize) -> usize;

    /// When we no longer need a page, we let fs know so it can be reused.
    fn dealloc_offset(&self, offset: usize);

    /// Flush the data to disk, similar to fsync on Linux.
    fn flush(&self);

    fn open(path: impl AsRef<std::path::Path>) -> Self
    where
        Self: Sized;
}

/// We need these pair of function because spdk don't work with arbitrary memory, it needs memory that is pinned.
/// Which essentially requires allocating memory from spdk, not from us.
pub(crate) fn buffer_alloc(layout: std::alloc::Layout) -> *mut u8 {
    #[cfg(feature = "spdk")]
    {
        use crate::fs::spdk_vfs::spdk_alloc_queue;
        _ = layout;

        // SPDK malloc is very expensive, we need to initialize it only once and keep it around.
        let ptr = spdk_alloc_queue()
            .pop()
            .expect("Unable to allocate memory")
            .into_ptr();

        ptr
    }

    #[cfg(not(feature = "spdk"))]
    unsafe {
        std::alloc::alloc(layout)
    }
}

/// We need these pair of function because spdk don't work with any memory, it needs memory that is pinned.
/// Which essentially requires allocating memory from spdk, not from us.
pub(crate) fn buffer_dealloc(ptr: *mut u8, layout: std::alloc::Layout) {
    #[cfg(feature = "spdk")]
    {
        use crate::fs::spdk_vfs::{spdk_alloc_queue, SpdkAllocGuard};
        _ = layout;
        let guard = SpdkAllocGuard::from_ptr(ptr);
        spdk_alloc_queue().push(guard).unwrap();
    }

    #[cfg(not(feature = "spdk"))]
    unsafe {
        std::alloc::dealloc(ptr, layout)
    }
}

/// A simple page allocator for disk.
/// TODO: maybe too simple, we should at least implement a free list, and potentially persist a free list.
pub(crate) struct OffsetAlloc {
    next_available_offset: crate::sync::atomic::AtomicUsize,
}

impl OffsetAlloc {
    pub(crate) fn new_with(mut offset: usize) -> Self {
        if offset < DISK_PAGE_SIZE {
            // the file was empty, we start from second page
            offset = DISK_PAGE_SIZE;
        }
        Self {
            next_available_offset: crate::sync::atomic::AtomicUsize::new(offset),
        }
    }

    pub(crate) fn alloc(&self, size: usize) -> usize {
        self.next_available_offset.fetch_add(size, Ordering::AcqRel)
    }

    pub(crate) fn dealloc_offset(&self, _offset: usize) {
        // We don't need to do anything here.
    }
}
