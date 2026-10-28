// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    ffi::CString,
    fs::File,
    io::Write,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::ffi::OsStrExt,
    },
    path::{Path, PathBuf},
};

use crate::{counter, sync::Mutex};
use io_uring::{opcode, IoUring};
use thread_local::ThreadLocal;

use super::{OffsetAlloc, VfsImpl};

fn submit_and_wait(ring: &mut IoUring, wait: usize) {
    loop {
        #[cfg(all(test, not(feature = "shuttle")))]
        let result = tests::submit(ring, wait);
        #[cfg(not(all(test, not(feature = "shuttle"))))]
        let result = ring.submit_and_wait(wait);
        match result {
            Ok(_) => return,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => {
                // Submission may have reached the kernel. Unwinding could free its buffers.
                std::process::abort();
            }
        }
    }
}

/// The purpose of this struct is to create a group of rings that share the same kernel poll thread.
/// Checkout this test case to learn how to setup this: https://github.com/axboe/liburing/blob/7ad5e52d4d2f91203615cd738e56aba10ad8b8f6/test/sq-poll-share.c
struct IoUringInstance {
    rings: ThreadLocal<Mutex<IoUring>>,
    poll: bool,
}

impl IoUringInstance {
    fn new(poll: bool) -> Self {
        Self {
            rings: ThreadLocal::new(),
            poll,
        }
    }

    fn get_current_ring(&self) -> &Mutex<IoUring> {
        // Never hash thread IDs into a shared RefCell: collisions race the ring.
        self.rings.get_or(|| {
            let mut r = IoUring::builder();

            if self.poll {
                r.setup_sqpoll(50000);
                r.setup_iopoll();

                if let Some(first) = self.rings.iter().next() {
                    r.setup_attach_wq(first.lock().unwrap().as_raw_fd());
                }
            }

            Mutex::new(r.build(8).expect("Failed to create io_uring"))
        })
    }
}

pub(crate) struct IoUringVfs {
    pub(crate) file: File,
    offset_alloc: OffsetAlloc,
    rings: IoUringInstance,
    _path: PathBuf,
    polling: bool,
}

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

    fn requires_aligned_io(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.file.set_len(0).unwrap();
        self.offset_alloc.reset(0);
    }

    fn read(&self, offset: usize, buf: &mut [u8]) {
        counter!(IOReadRequest);
        let mut done = 0;
        while done < buf.len() {
            let read_e = opcode::Read::new(
                io_uring::types::Fd(self.file.as_raw_fd()),
                buf[done..].as_mut_ptr(),
                (buf.len() - done) as _,
            )
            .offset((offset + done) as u64)
            .build()
            .user_data(0x42);

            let ring = self.rings.get_current_ring();
            let mut ring_mut = ring.lock().unwrap();
            unsafe {
                let mut sq = ring_mut.submission();
                sq.push(&read_e).unwrap();
                sq.sync();
            }

            submit_and_wait(&mut ring_mut, self.wait_cnt());

            let mut cq = ring_mut.completion();

            loop {
                cq.sync();
                match cq.next() {
                    Some(cqe) => {
                        assert_eq!(cqe.user_data(), 0x42);
                        let result = cqe.result();
                        #[cfg(all(test, not(feature = "shuttle")))]
                        let result = tests::completion(result);
                        if result == -libc::EINTR {
                            break;
                        }
                        assert!(
                            result >= 0,
                            "io_uring read: {}",
                            std::io::Error::from_raw_os_error(-result)
                        );
                        assert!(
                            result > 0,
                            "unexpected EOF reading at offset {}",
                            offset + done
                        );
                        done += result as usize;
                        break;
                    }
                    None => {
                        continue;
                    }
                };
            }
        }
    }

    /// Note that both buf len and buf ptr need to be aligned to 512 bytes:
    /// https://stackoverflow.com/questions/55447218/what-does-o-direct-512-byte-aligned-mean
    fn write(&self, offset: usize, buf: &[u8]) {
        counter!(IOWriteRequest);
        let mut done = 0;
        while done < buf.len() {
            let write_e = opcode::Write::new(
                io_uring::types::Fd(self.file.as_raw_fd()),
                buf[done..].as_ptr(),
                (buf.len() - done) as _,
            )
            .offset((offset + done) as u64)
            .build()
            .user_data(0x42);

            let ring = self.rings.get_current_ring();

            let mut ring_mut = ring.lock().unwrap();
            unsafe {
                let mut sq = ring_mut.submission();
                sq.push(&write_e).expect("submission queue is full");
                sq.sync();
            }

            submit_and_wait(&mut ring_mut, self.wait_cnt());

            let mut cq = ring_mut.completion();

            loop {
                cq.sync();
                match cq.next() {
                    Some(cqe) => {
                        assert_eq!(cqe.user_data(), 0x42);
                        let result = cqe.result();
                        #[cfg(all(test, not(feature = "shuttle")))]
                        let result = tests::completion(result);
                        if result == -libc::EINTR {
                            break;
                        }
                        assert!(
                            result >= 0,
                            "io_uring write: {}",
                            std::io::Error::from_raw_os_error(-result)
                        );
                        assert!(result > 0, "zero-length write at offset {}", offset + done);
                        done += result as usize;
                        break;
                    }
                    None => {
                        continue;
                    }
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "shuttle"))]
    type SubmitHook = Box<dyn FnMut(&mut IoUring, usize) -> std::io::Result<usize>>;
    #[cfg(not(feature = "shuttle"))]
    type CompletionHook = Box<dyn FnMut(i32) -> i32>;
    #[cfg(not(feature = "shuttle"))]
    thread_local! {
        static SUBMIT: std::cell::RefCell<Option<SubmitHook>> = std::cell::RefCell::new(None);
        static COMPLETE: std::cell::RefCell<Option<CompletionHook>> = std::cell::RefCell::new(None);
    }

    #[cfg(not(feature = "shuttle"))]
    pub(super) fn submit(ring: &mut IoUring, wait: usize) -> std::io::Result<usize> {
        SUBMIT.with(|hook| match hook.borrow_mut().as_mut() {
            Some(hook) => hook(ring, wait),
            None => ring.submit_and_wait(wait),
        })
    }

    #[cfg(not(feature = "shuttle"))]
    pub(super) fn completion(result: i32) -> i32 {
        COMPLETE.with(|hook| {
            hook.borrow_mut()
                .as_mut()
                .map_or(result, |hook| hook(result))
        })
    }

    #[cfg(not(feature = "shuttle"))]
    struct ResetHooks;

    #[cfg(not(feature = "shuttle"))]
    impl Drop for ResetHooks {
        fn drop(&mut self) {
            SUBMIT.with(|hook| *hook.borrow_mut() = None);
            COMPLETE.with(|hook| *hook.borrow_mut() = None);
        }
    }

    #[cfg(not(feature = "shuttle"))]
    fn buffered_vfs() -> IoUringVfs {
        IoUringVfs {
            file: tempfile::tempfile().unwrap(),
            offset_alloc: OffsetAlloc::new_with(0),
            rings: IoUringInstance::new(false),
            _path: PathBuf::new(),
            polling: false,
        }
    }

    #[cfg(not(feature = "shuttle"))]
    #[rstest::rstest]
    fn interrupted_and_short_completions_preserve_exact_progress(
        #[values(false, true)] read: bool,
    ) {
        use crate::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::os::unix::fs::FileExt;
        let _reset = ResetHooks;
        let vfs = buffered_vfs();
        vfs.file.write_all_at(b"abcdefgh", 0).unwrap();
        let file = vfs.file.try_clone().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        COMPLETE.with(|hook| {
            let calls = calls.clone();
            *hook.borrow_mut() = Some(Box::new(move |result| {
                let call = calls.fetch_add(1, Ordering::Relaxed) + 1;
                assert_eq!(result, if call <= 2 { 8 } else { 4 });
                match call {
                    1 => {
                        if !read {
                            file.write_all_at(b"--------", 0).unwrap();
                        }
                        -libc::EINTR
                    }
                    2 => {
                        if read {
                            file.write_all_at(b"XXXX", 0).unwrap();
                        } else {
                            file.write_all_at(b"----", 4).unwrap();
                        }
                        4
                    }
                    3 => {
                        if !read {
                            file.write_all_at(b"----", 4).unwrap();
                        }
                        -libc::EINTR
                    }
                    4 => result,
                    _ => panic!("exact I/O did not finish after retry"),
                }
            }));
        });
        let mut bytes = [0; 8];
        if read {
            vfs.read(0, &mut bytes);
        } else {
            vfs.write(0, b"abcdefgh");
            vfs.file.read_exact_at(&mut bytes, 0).unwrap();
        }
        assert_eq!(&bytes, b"abcdefgh");
        assert_eq!(calls.load(Ordering::Relaxed), 4);
    }

    #[cfg(not(feature = "shuttle"))]
    #[rstest::rstest]
    fn interrupted_submission_is_retried(#[values(false, true)] read: bool) {
        use std::os::unix::fs::FileExt;
        let _reset = ResetHooks;
        let vfs = buffered_vfs();
        vfs.file.write_all_at(b"abcdefgh", 0).unwrap();
        SUBMIT.with(|hook| {
            let mut first = true;
            *hook.borrow_mut() = Some(Box::new(move |ring, wait| {
                if std::mem::take(&mut first) {
                    Err(std::io::Error::from_raw_os_error(libc::EINTR))
                } else {
                    ring.submit_and_wait(wait)
                }
            }));
        });
        let mut bytes = [0; 8];
        if read {
            vfs.read(0, &mut bytes);
        } else {
            vfs.write(0, b"abcdefgh");
            vfs.file.read_exact_at(&mut bytes, 0).unwrap();
        }
        assert_eq!(&bytes, b"abcdefgh");
    }

    #[cfg(not(feature = "shuttle"))]
    #[test]
    fn interrupted_wait_keeps_an_accepted_read_alive() {
        use crate::sync::{mpsc, thread};
        use std::{
            os::{fd::OwnedFd, unix::net::UnixStream},
            time::Duration,
        };
        let _reset = ResetHooks;
        let (reader, mut writer) = UnixStream::pair().unwrap();
        let mut vfs = buffered_vfs();
        let reader: OwnedFd = reader.into();
        vfs.file = reader.into();
        let (resumed, resume) = mpsc::channel();
        SUBMIT.with(|hook| {
            let mut first = true;
            *hook.borrow_mut() = Some(Box::new(move |ring, wait| {
                if std::mem::take(&mut first) {
                    ring.submit()?;
                    Err(std::io::Error::from_raw_os_error(libc::EINTR))
                } else {
                    resumed.send(()).unwrap();
                    ring.submit_and_wait(wait)
                }
            }));
        });
        thread::scope(|scope| {
            scope.spawn(move || {
                resume.recv_timeout(Duration::from_secs(10)).unwrap();
                writer.write_all(b"abcdefgh").unwrap();
            });
            let mut bytes = [0; 8];
            vfs.read(0, &mut bytes);
            assert_eq!(&bytes, b"abcdefgh");
        });
    }

    #[cfg(not(feature = "shuttle"))]
    #[test]
    fn hard_submission_failure_aborts_instead_of_releasing_buffers() {
        use std::os::{
            fd::OwnedFd,
            unix::{net::UnixStream, process::ExitStatusExt},
        };
        const CHILD: &str = "BFTREE_TEST_PENDING_IO_ABORT";
        if std::env::var_os(CHILD).is_some() {
            let (reader, _writer) = UnixStream::pair().unwrap();
            let mut vfs = buffered_vfs();
            let reader: OwnedFd = reader.into();
            vfs.file = reader.into();
            SUBMIT.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(|ring, _| {
                    ring.submit()?;
                    eprintln!("accepted-read-pending");
                    Err(std::io::Error::from_raw_os_error(libc::EIO))
                }));
            });
            vfs.read(0, &mut [0; 8]);
            panic!("hard submission failure returned");
        }
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "fs::io_uring_vfs::tests::hard_submission_failure_aborts_instead_of_releasing_buffers", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("accepted-read-pending"),
            "{result:?}"
        );
        assert_eq!(result.status.signal(), Some(libc::SIGABRT), "{result:?}");
    }

    #[cfg(not(feature = "shuttle"))]
    #[test]
    fn concurrent_blocking_io_uses_distinct_rings_and_exact_reads() {
        use crate::sync::{thread, Barrier};

        #[repr(align(512))]
        struct Buffer([u8; 1024]);

        let dir = tempfile::tempdir().unwrap();
        let vfs = IoUringVfs::new_blocking(dir.path().join("concurrent.db"));
        let barrier = Barrier::new(8);
        let mut rings = thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|worker| {
                    let vfs = &vfs;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        let ring = vfs.rings.get_current_ring() as *const _ as usize;
                        barrier.wait();
                        let offset = vfs.alloc_offset(1024);
                        let mut written = Buffer([worker; 1024]);
                        let mut read = Buffer([0; 1024]);
                        for iteration in 0..32 {
                            written.0[0] = iteration;
                            vfs.write(offset, &written.0);
                            vfs.read(offset, &mut read.0);
                            assert_eq!(read.0, written.0);
                        }
                        ring
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        rings.sort_unstable();
        rings.dedup();
        assert_eq!(rings.len(), 8);

        let mut read = Buffer([0; 1024]);
        let length = vfs.file.metadata().unwrap().len() as usize;
        let short_read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            vfs.read(length - 512, &mut read.0);
        }));
        assert!(
            short_read.is_err(),
            "a partial read must not silently succeed"
        );
        vfs.reset();
        assert_eq!(vfs.file.metadata().unwrap().len(), 0);
        assert_eq!(vfs.alloc_offset(512), crate::nodes::DISK_PAGE_SIZE);
    }

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
