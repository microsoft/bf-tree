// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::path::Path;
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

mod operations;

use crate::config::WalConfig;
use crate::fs::VfsImpl;
use crate::storage::make_vfs;
use crate::sync::{atomic::AtomicBool, Condvar, Mutex};

pub(crate) use operations::{LogEntry, WriteOp};

const BLOCK_SIZE: usize = 512;

pub(crate) trait LogEntryImpl<'a> {
    fn log_size(&self) -> usize;
    fn write_to_buffer(&self, buffer: &mut [u8]);
    fn read_from_buffer(buffer: &'a [u8]) -> Self;
}

/// Ptr aligned to block size, so that it can be directly write to storage device
struct RawBuffer {
    buffer_size: usize,
    ptr: *mut u8,
}

impl RawBuffer {
    fn new(buffer_size: usize) -> RawBuffer {
        let layout = std::alloc::Layout::from_size_align(buffer_size, BLOCK_SIZE).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        RawBuffer { ptr, buffer_size }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.buffer_size) }
    }

    unsafe fn as_mut_slice_at_exact(&mut self, offset: usize, size: usize) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), size) }
    }
}

unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.buffer_size, BLOCK_SIZE).unwrap();
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

struct WriteAheadLogInner {
    buffer: RawBuffer,
    file_handle: Arc<dyn VfsImpl>,
    buffer_cursor: usize,
    file_offset: usize,
    next_lsn: u64,
    flushed_lsn: u64,
    need_flush: bool,
}

impl WriteAheadLogInner {
    fn flush(&mut self) {
        if self.buffer_cursor == 0 {
            // nothing to flush
            return;
        }

        self.clear_next_header();
        self.file_handle
            .write(self.file_offset, self.buffer.as_slice());

        if !self.should_inplace_flush() {
            self.file_offset += self.buffer.buffer_size;
            self.buffer_cursor = 0;
        }

        self.flushed_lsn = self.next_lsn - 1;
        self.need_flush = false;
    }

    fn clear_next_header(&mut self) {
        if self.buffer_cursor + 8 <= self.buffer.buffer_size {
            let slice = unsafe { self.buffer.as_mut_slice_at_exact(self.buffer_cursor, 8) };
            slice.copy_from_slice(&[0u8; 8]);
        }
    }

    unsafe fn alloc_buffer(&mut self, size: usize) -> &mut [u8] {
        debug_assert!(
            self.buffer_cursor + size <= self.buffer.buffer_size,
            "buffer overflow"
        );
        let cursor = self.buffer_cursor;
        self.buffer_cursor += size;
        unsafe { self.buffer.as_mut_slice_at_exact(cursor, size) }
    }

    /// if buffer is less than half full, we should not create a new buffer,
    /// instead inplace flush the buffer
    fn should_inplace_flush(&self) -> bool {
        self.buffer_cursor < (self.buffer.buffer_size / 2)
    }

    fn alloc_lsn(&mut self) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        lsn
    }
}

pub(crate) struct WriteAheadLog {
    inner: Mutex<WriteAheadLogInner>,
    flushed_cond: Condvar,    // for workers that waiting for flush
    need_flush_cond: Condvar, // for background job
    background_job_running: AtomicBool,
    config: Arc<WalConfig>,
}

impl WriteAheadLog {
    /// Create a new wal instance, and start a background thread to flush wal buffer.
    pub(crate) fn new(config: Arc<WalConfig>) -> Arc<Self> {
        let vfs = make_vfs(&config.storage_backend, &config.file_path);
        let wal = WriteAheadLog {
            inner: Mutex::new(WriteAheadLogInner {
                buffer: RawBuffer::new(config.segment_size),
                file_handle: vfs,
                buffer_cursor: 0,
                file_offset: 0,
                next_lsn: 0,
                flushed_lsn: 0,
                need_flush: false,
            }),
            flushed_cond: Condvar::new(),
            need_flush_cond: Condvar::new(),
            background_job_running: AtomicBool::new(true),
            config,
        };

        let wal = Arc::new(wal);
        WriteAheadLog::start_flush_job(wal.clone());
        wal
    }

    fn start_flush_job(wal: Arc<Self>) {
        let h = crate::sync::thread::spawn(move || wal.background_flush_job());
        drop(h); // detach the thread
    }

    pub(crate) fn stop_background_job(&self) {
        self.background_job_running
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.need_flush_cond.notify_all();
    }

    pub(crate) fn background_flush_job(&self) {
        let mut inner = self.inner.lock().unwrap();

        let flush_interval = self.config.flush_interval;
        let mut last_flush = std::time::Instant::now();
        loop {
            let v = self
                .need_flush_cond
                .wait_timeout(inner, flush_interval)
                // wait for a notification or a interval, whichever happens first.
                .unwrap();

            inner = v.0;

            if !self
                .background_job_running
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                // stop the background job, gracefully shutdown.
                break;
            }

            if inner.need_flush || last_flush.elapsed() > flush_interval {
                inner.flush();
                last_flush = std::time::Instant::now();
                self.flushed_cond.notify_all();
            }
        }
    }

    #[must_use = "The returned flushed lsn must be write to page meta"]
    pub(crate) fn append_and_wait<'a>(
        &self,
        log_entry: &impl LogEntryImpl<'a>,
        page_offset: u64,
    ) -> u64 {
        let mut inner = self.inner.lock().unwrap();

        // log header + wal size
        let required_bytes = std::mem::size_of::<LogHeader>() + log_entry.log_size();
        let remaining = inner.buffer.buffer_size - inner.buffer_cursor;
        if required_bytes > remaining {
            // need to flush buffer
            inner.need_flush = true;
            self.need_flush_cond.notify_all();
            inner = self
                .flushed_cond
                .wait_while(inner, |inner| !inner.need_flush)
                .unwrap();
            // we need to retry here because by the time we wake up, the buffer maybe already full again.
            drop(inner);
            return self.append_and_wait(log_entry, page_offset);
        }

        let lsn = inner.alloc_lsn();
        let header = LogHeader::new(lsn, page_offset, required_bytes);
        let buffer = unsafe { inner.alloc_buffer(required_bytes) };
        buffer[0..LogHeader::size()].copy_from_slice(header.as_slice());
        log_entry.write_to_buffer(&mut buffer[LogHeader::size()..]);

        while inner.flushed_lsn < lsn {
            inner = self.flushed_cond.wait(inner).unwrap();
        }
        lsn
    }
}

/// Read the write-ahead-log file produced by Bf-Tree.
///
/// Allows users to iterate over the log entries in the file and decide what to do with them.
///
///
/// Example
/// ```ignore
/// let reader = WalReader::new(&file, 4096);
/// for segment in reader.segment_iter() {
///     let seg_iter = segment.iter();
///     for (header, buffer) in seg_iter {
///         ...
///     }
/// }
/// ```
pub struct WalReader {
    log_file: std::fs::File,
    segment_size: usize,
    file_size: usize,
}

impl WalReader {
    /// Create a new WalReader instance.
    ///
    /// The `segment_size`` should be the same as the one used to create the WriteAheadLog instance.
    ///
    /// Todo: we should include segment_size as a field in the wal file, so that we don't need to pass it in.
    pub fn new(path: impl AsRef<Path>, segment_size: usize) -> Self {
        let log_file = std::fs::OpenOptions::new().read(true).open(path).unwrap();
        let file_size = log_file.metadata().unwrap().len() as usize;
        WalReader {
            log_file,
            segment_size,
            file_size,
        }
    }

    /// Iterate through all the segments in the wal file.
    ///
    /// Each segment contains multiple log entries,
    /// you can iterate through the log entries in each segment using the `iter` method on `WalSegment`.
    pub fn segment_iter(&self) -> WalSegmentIter<'_> {
        WalSegmentIter {
            reader: self,
            cursor: 0,
        }
    }
}

pub struct WalSegmentIter<'a> {
    reader: &'a WalReader,
    cursor: u64,
}

impl Iterator for WalSegmentIter<'_> {
    type Item = WalSegment;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor as usize >= self.reader.file_size {
            return None;
        }

        let mut buffer = vec![0u8; self.reader.segment_size];
        let page_offset = self.cursor;

        #[cfg(unix)]
        {
            self.reader
                .log_file
                .read_exact_at(&mut buffer, page_offset)
                .unwrap();
        }
        #[cfg(windows)]
        {
            let bytes_to_read = buffer.len();
            let bytes_read = self
                .reader
                .log_file
                .seek_read(&mut buffer, page_offset)
                .unwrap();
            assert_eq!(bytes_to_read, bytes_read);
        }

        self.cursor += self.reader.segment_size as u64;

        Some(WalSegment { data: buffer })
    }
}

pub struct WalSegment {
    data: Vec<u8>,
}

impl WalSegment {
    /// Iterate through all the log entries in the segment.
    pub fn entry_iter(&self) -> WalEntryIter<'_> {
        WalEntryIter {
            segment: self,
            cur_offset: 0,
        }
    }
}

pub struct WalEntryIter<'a> {
    segment: &'a WalSegment,
    cur_offset: u64,
}

impl<'a> Iterator for WalEntryIter<'a> {
    type Item = (LogHeader, &'a [u8]);
    fn next(&mut self) -> Option<Self::Item> {
        if (self.cur_offset as usize + LogHeader::size()) >= self.segment.data.len() {
            return None;
        }

        let header = LogHeader::from_slice(&self.segment.data[self.cur_offset as usize..]);

        if header.log_len == 0 {
            return None;
        }

        let data_start = self.cur_offset as usize + LogHeader::size();
        let data_end = data_start + header.log_len - LogHeader::size();
        let data = &self.segment.data[data_start..data_end];
        self.cur_offset += header.log_len as u64;
        Some((header, data))
    }
}

/// The header of a log entry in the wal file.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LogHeader {
    pub log_len: usize,
    pub lsn: u64,
    pub page_offset: u64,
}

impl LogHeader {
    fn new(lsn: u64, page_offset: u64, log_len: usize) -> Self {
        LogHeader {
            log_len,
            lsn,
            page_offset,
        }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const u8, std::mem::size_of::<Self>())
        }
    }

    fn from_slice(buffer: &[u8]) -> Self {
        let log_len = usize::from_le_bytes(buffer[0..8].try_into().unwrap());
        let lsn = u64::from_le_bytes(buffer[8..16].try_into().unwrap());
        let page_offset = u64::from_le_bytes(buffer[16..24].try_into().unwrap());
        Self::new(lsn, page_offset, log_len)
    }

    const fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

const _: () = assert!(LogHeader::size() == 24);

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::utils;

    use super::*;

    struct TestLogEntry {
        val: usize,
    }

    impl TestLogEntry {
        fn new(val: usize) -> Self {
            TestLogEntry { val }
        }
    }

    impl LogEntryImpl<'_> for TestLogEntry {
        fn log_size(&self) -> usize {
            8
        }

        fn write_to_buffer(&self, buffer: &mut [u8]) {
            buffer.copy_from_slice(&self.val.to_le_bytes());
        }

        fn read_from_buffer(buffer: &[u8]) -> Self {
            let val = usize::from_le_bytes(buffer.try_into().unwrap());
            TestLogEntry { val }
        }
    }

    fn make_test_wal(name: &str, segment_size: usize) -> Arc<WriteAheadLog> {
        let tmp_dir = std::env::temp_dir();
        let tmp_file = tmp_dir.join(name);
        let mut wal_config = WalConfig::new(&tmp_file);
        wal_config.segment_size(segment_size);
        wal_config.flush_interval(Duration::from_micros(1));
        WriteAheadLog::new(Arc::new(wal_config))
    }

    #[test]
    fn simple_wal() {
        const TEST_SEGMENT_SIZE: usize = 4096;
        let wal = make_test_wal("wal_simple_test.log", TEST_SEGMENT_SIZE);
        let tmp_file = wal.config.file_path.clone();

        let log_entry_cnt = 4096;

        for i in 0..log_entry_cnt {
            let log = TestLogEntry::new(i);
            let lsn = wal.append_and_wait(&log, log.val as u64);
            assert_eq!(lsn, i as u64);
        }

        wal.stop_background_job();
        drop(wal);

        let reader = WalReader::new(&tmp_file, TEST_SEGMENT_SIZE);
        let mut cnt = 0;
        for segment in reader.segment_iter() {
            let seg_iter = segment.entry_iter();
            for (header, data) in seg_iter {
                let val = TestLogEntry::read_from_buffer(data);
                assert_eq!(
                    header.log_len,
                    TestLogEntry::new(0).log_size() + LogHeader::size()
                );
                assert_eq!(header.lsn, cnt as u64);
                assert_eq!(header.page_offset, cnt as u64);
                assert_eq!(val.val, cnt);
                cnt += 1;
            }
        }
        assert_eq!(cnt, log_entry_cnt);
        std::fs::remove_file(tmp_file).unwrap();
    }

    #[test]
    fn multi_thread_wal() {
        const TEST_SEGMENT_SIZE: usize = 4096;
        let pid = std::process::id();
        let tid = utils::thread_id_to_u64(std::thread::current().id());
        let wal = make_test_wal(
            &format!("wal_multi_thread_test_{}_{}.log", pid, tid),
            TEST_SEGMENT_SIZE,
        );
        let tmp_file = wal.config.file_path.clone();

        let log_entry_cnt = 4096;
        let thread_cnt = 4;

        let join_handles = (0..thread_cnt)
            .map(|_| {
                let wal_t = wal.clone();
                crate::sync::thread::spawn(move || {
                    for i in 0..log_entry_cnt {
                        let log = TestLogEntry::new(i);
                        let _lsn = wal_t.append_and_wait(&log, log.val as u64);
                    }
                })
            })
            .collect::<Vec<_>>();

        for h in join_handles.into_iter() {
            h.join().unwrap();
        }

        wal.stop_background_job();
        drop(wal);

        let reader = WalReader::new(&tmp_file, TEST_SEGMENT_SIZE);
        let mut cnt = 0;
        for segment in reader.segment_iter() {
            let seg_iter = segment.entry_iter();
            for (header, data) in seg_iter {
                let val = TestLogEntry::read_from_buffer(data);
                assert_eq!(
                    header.log_len,
                    TestLogEntry::new(0).log_size() + LogHeader::size()
                );
                assert_eq!(val.val, header.page_offset as usize);
                cnt += 1;
            }
        }
        assert_eq!(cnt, log_entry_cnt * thread_cnt);
        std::fs::remove_file(tmp_file).unwrap();
    }

    /// As of https://github.com/awslabs/shuttle/issues/74
    /// Shuttle can not properly handle wait_timeout, so we can't really test this with shuttle.
    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_wal_concurrent_op() {
        use std::{path::PathBuf, str::FromStr};

        tracing_subscriber::fmt()
            .with_ansi(true)
            .with_thread_names(false)
            .with_target(false)
            .init();
        let mut config = shuttle::Config::default();
        config.max_steps = shuttle::MaxSteps::None;
        config.failure_persistence =
            shuttle::FailurePersistence::File(Some(PathBuf::from_str("target").unwrap()));

        let mut runner = shuttle::PortfolioRunner::new(true, config);

        let available_cores = std::thread::available_parallelism().unwrap().get().min(4);

        for _i in 0..available_cores {
            runner.add(shuttle::scheduler::PctScheduler::new(10, 4_000));
        }

        runner.run(multi_thread_wal);
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_wal_replay() {
        tracing_subscriber::fmt()
            .with_ansi(true)
            .with_thread_names(false)
            .with_target(false)
            .init();

        shuttle::replay_from_file(multi_thread_wal, "target/schedule003.txt");
    }
}
