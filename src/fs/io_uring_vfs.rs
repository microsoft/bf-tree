// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    cell::RefCell,
    ffi::CString,
    fs::File,
    io::Write,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::ffi::OsStrExt,
    },
    path::{Path, PathBuf},
};

use crate::{counter, utils};
use io_uring::{opcode, IoUring};

use super::{OffsetAlloc, VfsImpl};

/// The purpose of this struct is to create a group of rings that share the same kernel poll thread.
/// Checkout this test case to learn how to setup this: https://github.com/axboe/liburing/blob/7ad5e52d4d2f91203615cd738e56aba10ad8b8f6/test/sq-poll-share.c
struct IoUringInstance {
    ring: Vec<RefCell<IoUring>>,
}

impl IoUringInstance {
    fn new(poll: bool) -> Self {
        let parallelism: usize = std::thread::available_parallelism().unwrap().into();
        let thread_cnt = 32.max(parallelism * 4);
        let mut ring = Vec::with_capacity(thread_cnt);

        for i in 0..thread_cnt {
            let mut r = IoUring::builder();

            if poll {
                r.setup_sqpoll(50000);
                r.setup_iopoll();

                if i >= 1 {
                    let pre_r: &RefCell<IoUring> = &ring[i - 1];
                    r.setup_attach_wq(pre_r.borrow().as_raw_fd());
                }
            }

            let r = r.build(8).expect("Failed to create io_uring");
            ring.push(RefCell::new(r));
        }

        Self { ring }
    }

    fn get_current_ring(&self) -> &RefCell<IoUring> {
        // TODO: this is unstable feature, we rely on a implementation detail
        let v = utils::thread_id_to_u64(std::thread::current().id());
        let idx = v % self.ring.len() as u64;
        let ring = self.get_ring(idx);
        ring
    }

    fn get_ring(&self, thread_id: u64) -> &RefCell<IoUring> {
        &self.ring[thread_id as usize]
    }
}

pub(crate) struct IoUringVfs {
    pub(crate) file: File,
    offset_alloc: OffsetAlloc,
    rings: IoUringInstance,
    _path: PathBuf,
    polling: bool,
}

unsafe impl Send for IoUringVfs {}
unsafe impl Sync for IoUringVfs {}

impl IoUringVfs {
    pub(crate) fn new_blocking(path: impl AsRef<Path>) -> Self {
        Self::new_inner(path, false)
    }

    fn wait_cnt(&self) -> usize {
        if self.polling {
            0
        } else {
            1
        }
    }

    fn new_inner(path: impl AsRef<Path>, use_poll: bool) -> Self {
        let path = path.as_ref();

        let parent = path.parent().unwrap();
        _ = std::fs::create_dir_all(parent);

        let path_cstr = CString::new(path.as_os_str().as_bytes()).unwrap();
        let raw_fd = unsafe {
            libc::open(
                path_cstr.as_ptr(),
                libc::O_DIRECT | libc::O_RDWR | libc::O_CREAT,
                libc::S_IRUSR | libc::S_IWUSR,
            )
        };
        assert!(
            raw_fd >= 0,
            "Failed to open file {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        );

        let mut file = unsafe { File::from_raw_fd(raw_fd) };
        file.flush().unwrap();
        let offset = file.metadata().unwrap().len();

        IoUringVfs {
            rings: IoUringInstance::new(use_poll),
            file,
            _path: path.to_path_buf(),
            offset_alloc: OffsetAlloc::new_with(offset as usize),
            polling: use_poll,
        }
    }
}

impl VfsImpl for IoUringVfs {
    fn alloc_offset(&self, size: usize) -> usize {
        self.offset_alloc.alloc(size)
    }

    fn dealloc_offset(&self, offset: usize) {
        self.offset_alloc.dealloc_offset(offset)
    }

    fn open(path: impl AsRef<Path>) -> Self {
        Self::new_inner(path, true)
    }

    fn flush(&self) {
        self.file.sync_all().unwrap();
    }

    fn read(&self, offset: usize, buf: &mut [u8]) {
        counter!(IOReadRequest);
        let read_e = opcode::Read::new(
            io_uring::types::Fd(self.file.as_raw_fd()),
            buf.as_mut_ptr(),
            buf.len() as _,
        )
        .offset(offset as u64)
        .build()
        .user_data(0x42);

        let ring = self.rings.get_current_ring();
        let mut ring_mut = ring.borrow_mut(); // no body will borrow our ring at the same time
        unsafe {
            let mut sq = ring_mut.submission();
            sq.push(&read_e).unwrap();
            sq.sync();
        }

        ring_mut.submit_and_wait(self.wait_cnt()).unwrap();

        let mut cq = ring_mut.completion();

        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    assert_eq!(cqe.user_data(), 0x42);
                    assert_eq!(
                        cqe.result(),
                        buf.len() as i32,
                        "Read cqe result error: {}",
                        std::io::Error::last_os_error()
                    );
                    break;
                }
                None => {
                    continue;
                }
            };
        }
    }

    /// Note that both buf len and buf ptr need to be aligned to 512 bytes:
    /// https://stackoverflow.com/questions/55447218/what-does-o-direct-512-byte-aligned-mean
    fn write(&self, offset: usize, buf: &[u8]) {
        counter!(IOWriteRequest);
        let write_e = opcode::Write::new(
            io_uring::types::Fd(self.file.as_raw_fd()),
            buf.as_ptr(),
            buf.len() as _,
        )
        .offset(offset as u64)
        .build()
        .user_data(0x42);

        let ring = self.rings.get_current_ring();

        let mut ring_mut = ring.borrow_mut();
        unsafe {
            let mut sq = ring_mut.submission();
            sq.push(&write_e).expect("submission queue is full");
            sq.sync();
        }

        ring_mut.submit_and_wait(self.wait_cnt()).unwrap();

        let mut cq = ring_mut.completion();

        loop {
            cq.sync();
            match cq.next() {
                Some(cqe) => {
                    assert_eq!(cqe.user_data(), 0x42);
                    assert_eq!(
                        cqe.result(),
                        buf.len() as i32,
                        "Write cqe result error: {}",
                        std::io::Error::last_os_error()
                    );
                    break;
                }
                None => {
                    continue;
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ioring_vfs_open() {
        let file_path = PathBuf::from_str("target")
            .unwrap()
            .join("test_io_uring_vfs_open.db");

        let vfs = IoUringVfs::open(&file_path);
        assert!(vfs.file.metadata().is_ok());

        std::fs::remove_file(&file_path).expect("Failed to remove test file");
    }

    use std::{alloc::Layout, str::FromStr};

    #[test]
    fn test_read_write_operations() {
        let file_path = PathBuf::from_str("target")
            .unwrap()
            .join("test_read_write_operations.db");

        // make and write file to file_path
        std::fs::write(&file_path, "This is a test file for IoUringVfs.").unwrap();

        let vfs = IoUringVfs::open(&file_path);

        let buf_size = 512;
        let buf_layout = Layout::from_size_align(buf_size, buf_size).unwrap();
        let write_buf_ptr = unsafe { std::alloc::alloc(buf_layout) };
        let write_buf = unsafe { std::slice::from_raw_parts_mut(write_buf_ptr, buf_size) };

        for i in 0..write_buf.len() {
            write_buf[i] = i as u8;
        }

        let offset = 0;
        vfs.write(offset, write_buf);

        // we need to alloc zeroed here, ow the memory sanitizer will complain (I believe it is a false positive)
        let read_buf_ptr = unsafe { std::alloc::alloc_zeroed(buf_layout) };
        let read_buf = unsafe { std::slice::from_raw_parts_mut(read_buf_ptr, buf_size) };
        vfs.read(offset, read_buf);

        assert_eq!(read_buf, write_buf, "Read data does not match written data");

        unsafe { std::alloc::dealloc(write_buf_ptr, buf_layout) };
        unsafe { std::alloc::dealloc(read_buf_ptr, buf_layout) };

        std::fs::remove_file(file_path).expect("Failed to remove test file");
    }
}
