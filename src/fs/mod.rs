// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

mod memory_vfs;
mod std_vfs;

#[cfg(target_os = "linux")]
mod std_direct_vfs;
use std::sync::atomic::Ordering;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

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

    fn reset(&self) {}

    fn open(path: impl AsRef<std::path::Path>) -> Self
    where
        Self: Sized;
}

pub(crate) fn read_exact_at(
    file: &std::fs::File,
    mut buf: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    while !buf.is_empty() {
        #[cfg(unix)]
        let bytes_read = retry_interrupted(|| file.read_at(buf, offset))?;
        #[cfg(windows)]
        let bytes_read = retry_interrupted(|| file.seek_read(buf, offset))?;

        if bytes_read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "failed to fill the positioned read buffer",
            ));
        }
        offset += bytes_read as u64;
        buf = &mut buf[bytes_read..];
    }
    Ok(())
}

pub(crate) fn write_all_at(
    file: &std::fs::File,
    mut buf: &[u8],
    mut offset: u64,
) -> std::io::Result<()> {
    while !buf.is_empty() {
        #[cfg(unix)]
        let bytes_written = retry_interrupted(|| file.write_at(buf, offset))?;
        #[cfg(windows)]
        let bytes_written = retry_interrupted(|| file.seek_write(buf, offset))?;

        if bytes_written == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "failed to write the positioned buffer",
            ));
        }
        offset += bytes_written as u64;
        buf = &buf[bytes_written..];
    }
    Ok(())
}

#[inline]
fn retry_interrupted<T>(mut operation: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    loop {
        match operation() {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => return result,
        }
    }
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
    {
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        ptr
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

    pub(crate) fn reset(&self, mut offset: usize) {
        if offset < DISK_PAGE_SIZE {
            offset = DISK_PAGE_SIZE;
        }
        self.next_available_offset.store(offset, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

    #[test]
    fn positioned_io_retries_interruptions_and_preserves_other_errors() {
        let mut attempts = 0;
        let result = retry_interrupted(|| {
            attempts += 1;
            if attempts < 3 {
                Err(std::io::Error::from(ErrorKind::Interrupted))
            } else {
                Ok(7)
            }
        });
        assert_eq!(result.unwrap(), 7);
        assert_eq!(attempts, 3);

        let mut attempts = 0;
        let result: std::io::Result<()> = retry_interrupted(|| {
            attempts += 1;
            Err(std::io::Error::from(ErrorKind::PermissionDenied))
        });
        assert_eq!(result.unwrap_err().kind(), ErrorKind::PermissionDenied);
        assert_eq!(attempts, 1);
    }

    #[test]
    fn positioned_io_reads_and_writes_at_requested_offsets() {
        let file = tempfile::tempfile().unwrap();
        write_all_at(&file, b"abcdef", 5).unwrap();
        write_all_at(&file, b"xy", 7).unwrap();
        let mut bytes = [0; 6];
        read_exact_at(&file, &mut bytes, 5).unwrap();
        assert_eq!(&bytes, b"abxyef");
        assert_eq!(
            read_exact_at(&file, &mut [0; 7], 5).unwrap_err().kind(),
            ErrorKind::UnexpectedEof
        );
        read_exact_at(&file, &mut [], u64::MAX).unwrap();
        write_all_at(&file, &[], u64::MAX).unwrap();
    }
}
