// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::path::Path;
use std::sync::Arc;

mod operations;

use crate::config::WalConfig;
use crate::fs::{read_exact_at, VfsImpl};
use crate::storage::make_vfs;
use crate::sync::{atomic::AtomicBool, Condvar, Mutex};

pub(crate) use operations::WriteOp;

const BLOCK_SIZE: usize = 512;

pub(crate) trait LogEntryImpl<'a> {
    fn log_size(&self) -> usize;
    fn write_to_buffer(&self, buffer: &mut [u8]);
    #[cfg(test)]
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
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
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
    flushed_cursor: usize,
    file_offset: usize,
    next_lsn: u64,
    next_flushed_lsn: u64,
    need_flush: bool,
}

impl WriteAheadLogInner {
    fn flush(&mut self, rotate_segment: bool) {
        if self.buffer_cursor == 0 {
            self.need_flush = false;
            return;
        }

        if self.flushed_cursor != self.buffer_cursor {
            self.clear_next_header();
            let write_start = self.flushed_cursor / BLOCK_SIZE * BLOCK_SIZE;
            let write_end = (self.buffer_cursor + std::mem::size_of::<u64>())
                .min(self.buffer.buffer_size)
                .next_multiple_of(BLOCK_SIZE)
                .min(self.buffer.buffer_size);
            self.file_handle.write(
                self.file_offset + write_start,
                &self.buffer.as_slice()[write_start..write_end],
            );
            self.file_handle.flush();
            self.flushed_cursor = self.buffer_cursor;
            self.next_flushed_lsn = self.next_lsn;
        }

        if rotate_segment || !self.should_inplace_flush() {
            self.file_offset += self.buffer.buffer_size;
            self.buffer_cursor = 0;
            self.flushed_cursor = 0;
        }

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
    background_job_stopped: AtomicBool,
    stopped_cond: Condvar,
    config: Arc<WalConfig>,
}

impl WriteAheadLog {
    /// Create a new wal instance, and start a background thread to flush wal buffer.
    pub(crate) fn new(config: Arc<WalConfig>) -> Arc<Self> {
        assert!(
            config.segment_size >= BLOCK_SIZE && config.segment_size.is_multiple_of(BLOCK_SIZE),
            "WAL segment size must be a positive multiple of {BLOCK_SIZE} bytes"
        );
        let (file_offset, next_lsn) = Self::resume_state(&config);
        let vfs = make_vfs(&config.storage_backend, &config.file_path);
        let wal = WriteAheadLog {
            inner: Mutex::new(WriteAheadLogInner {
                buffer: RawBuffer::new(config.segment_size),
                file_handle: vfs,
                buffer_cursor: 0,
                flushed_cursor: 0,
                file_offset,
                next_lsn,
                next_flushed_lsn: next_lsn,
                need_flush: false,
            }),
            flushed_cond: Condvar::new(),
            need_flush_cond: Condvar::new(),
            background_job_running: AtomicBool::new(true),
            background_job_stopped: AtomicBool::new(false),
            stopped_cond: Condvar::new(),
            config,
        };

        let wal = Arc::new(wal);
        WriteAheadLog::start_flush_job(wal.clone());
        wal
    }

    fn resume_state(config: &WalConfig) -> (usize, u64) {
        if config.storage_backend == crate::StorageBackend::Memory || !config.file_path.exists() {
            return (0, 0);
        }

        let reader = WalReader::new(&config.file_path, config.segment_size);
        let (next_lsn, valid_end) = reader
            .last_lsn_and_valid_end()
            .map_or((0, 0), |(lsn, end)| (lsn.saturating_add(1), end));
        // Discard an incomplete tail before extending the file. Otherwise the gap
        // to the next segment would zero-fill a torn record into an apparently
        // complete one on the next recovery.
        if valid_end < reader.file_size {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&config.file_path)
                .unwrap();
            file.set_len(valid_end as u64).unwrap();
        }
        (
            reader.file_size.next_multiple_of(config.segment_size),
            next_lsn,
        )
    }

    fn start_flush_job(wal: Arc<Self>) {
        let h = crate::sync::thread::spawn(move || wal.background_flush_job());
        drop(h); // detach the thread
    }

    pub(crate) fn stop_background_job(&self) {
        if self
            .background_job_stopped
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return;
        }
        self.background_job_running
            .store(false, std::sync::atomic::Ordering::Release);
        self.need_flush_cond.notify_all();

        let mut inner = self.inner.lock().unwrap();
        while !self
            .background_job_stopped
            .load(std::sync::atomic::Ordering::Acquire)
        {
            inner = self.stopped_cond.wait(inner).unwrap();
        }
    }

    pub(crate) fn background_flush_job(&self) {
        let mut inner = self.inner.lock().unwrap();

        let flush_interval = self.config.flush_interval;
        let mut last_flush = std::time::Instant::now();
        loop {
            if !self
                .background_job_running
                .load(std::sync::atomic::Ordering::Acquire)
            {
                inner.flush(false);
                self.flushed_cond.notify_all();
                self.background_job_stopped
                    .store(true, std::sync::atomic::Ordering::Release);
                self.stopped_cond.notify_all();
                break;
            }

            let v = self
                .need_flush_cond
                .wait_timeout(inner, flush_interval)
                // wait for a notification or a interval, whichever happens first.
                .unwrap();

            inner = v.0;

            if !self
                .background_job_running
                .load(std::sync::atomic::Ordering::Acquire)
            {
                inner.flush(false);
                self.flushed_cond.notify_all();
                self.background_job_stopped
                    .store(true, std::sync::atomic::Ordering::Release);
                self.stopped_cond.notify_all();
                break;
            }

            if inner.need_flush || last_flush.elapsed() > flush_interval {
                let rotate_segment = inner.need_flush;
                inner.flush(rotate_segment);
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
        assert!(
            required_bytes <= inner.buffer.buffer_size,
            "WAL entry size {required_bytes} exceeds segment size {}",
            inner.buffer.buffer_size
        );

        while required_bytes > inner.buffer.buffer_size - inner.buffer_cursor {
            inner.need_flush = true;
            self.need_flush_cond.notify_all();
            inner = self
                .flushed_cond
                .wait_while(inner, |inner| inner.need_flush)
                .unwrap();
        }

        let lsn = inner.alloc_lsn();
        let header = LogHeader::new(lsn, page_offset, required_bytes);
        let buffer = unsafe { inner.alloc_buffer(required_bytes) };
        buffer[0..LogHeader::size()].copy_from_slice(header.as_slice());
        log_entry.write_to_buffer(&mut buffer[LogHeader::size()..]);

        while inner.next_flushed_lsn <= lsn {
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
        assert!(
            segment_size >= LogHeader::size(),
            "WAL segment size must fit a log header"
        );
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

    fn read_segment_at(&self, offset: usize) -> Option<WalSegment> {
        if offset >= self.file_size {
            return None;
        }

        let bytes_to_read = self.segment_size.min(self.file_size - offset);
        let mut buffer = vec![0u8; bytes_to_read];
        read_exact_at(&self.log_file, &mut buffer, offset as u64).unwrap();
        Some(WalSegment { data: buffer })
    }

    fn last_lsn_and_valid_end(&self) -> Option<(u64, usize)> {
        if self.file_size == 0 {
            return None;
        }

        let last_segment = (self.file_size - 1) / self.segment_size;
        for index in (0..=last_segment).rev() {
            let segment = self.read_segment_at(index * self.segment_size)?;
            let mut last_lsn = None;
            let mut valid_end = index * self.segment_size;
            for (header, _) in segment.entry_iter() {
                last_lsn = Some(last_lsn.map_or(header.lsn, |lsn: u64| lsn.max(header.lsn)));
                valid_end += header.log_len;
            }
            if let Some(lsn) = last_lsn {
                return Some((lsn, valid_end));
            }
        }
        None
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

        let segment = self.reader.read_segment_at(self.cursor as usize)?;
        self.cursor += self.reader.segment_size as u64;
        Some(segment)
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
        let cur_offset = self.cur_offset as usize;
        let remaining = self.segment.data.len().checked_sub(cur_offset)?;
        if remaining < LogHeader::size() {
            return None;
        }

        let header = LogHeader::from_slice(&self.segment.data[cur_offset..]);

        if header.log_len == 0 {
            return None;
        }

        if header.log_len < LogHeader::size() || header.log_len > remaining {
            self.cur_offset = self.segment.data.len() as u64;
            return None;
        }

        let data_start = cur_offset + LogHeader::size();
        let data_end = cur_offset + header.log_len;
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
    use std::sync::atomic::{AtomicUsize, Ordering};
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

    struct BytesLogEntry {
        bytes: Vec<u8>,
    }

    impl LogEntryImpl<'_> for BytesLogEntry {
        fn log_size(&self) -> usize {
            self.bytes.len()
        }

        fn write_to_buffer(&self, buffer: &mut [u8]) {
            buffer.copy_from_slice(&self.bytes);
        }

        fn read_from_buffer(buffer: &[u8]) -> Self {
            Self {
                bytes: buffer.to_vec(),
            }
        }
    }

    fn make_test_wal(name: &str, segment_size: usize) -> Arc<WriteAheadLog> {
        let tmp_dir = std::env::temp_dir();
        let tmp_file = tmp_dir.join(name);
        _ = std::fs::remove_file(&tmp_file);
        let mut wal_config = WalConfig::new(&tmp_file);
        wal_config.segment_size(segment_size);
        wal_config.flush_interval(Duration::from_micros(1));
        WriteAheadLog::new(Arc::new(wal_config))
    }

    #[derive(Default)]
    struct RecordingVfs {
        writes: std::sync::Mutex<Vec<(usize, Vec<u8>)>>,
        flushes: AtomicUsize,
    }

    impl VfsImpl for RecordingVfs {
        fn read(&self, _offset: usize, _buf: &mut [u8]) {
            unreachable!()
        }

        fn write(&self, offset: usize, buf: &[u8]) {
            self.writes.lock().unwrap().push((offset, buf.to_vec()));
        }

        fn alloc_offset(&self, _size: usize) -> usize {
            unreachable!()
        }

        fn dealloc_offset(&self, _offset: usize) {
            unreachable!()
        }

        fn flush(&self) {
            self.flushes.fetch_add(1, Ordering::Relaxed);
        }

        fn open(_path: impl AsRef<Path>) -> Self {
            unreachable!()
        }
    }

    #[test]
    fn clean_wal_flush_skips_io_but_still_rotates_on_request() {
        let vfs = Arc::new(RecordingVfs::default());
        let mut inner = WriteAheadLogInner {
            buffer: RawBuffer::new(4096),
            file_handle: vfs.clone(),
            buffer_cursor: 0,
            flushed_cursor: 0,
            file_offset: 0,
            next_lsn: 0,
            next_flushed_lsn: 0,
            need_flush: false,
        };
        let append = |inner: &mut WriteAheadLogInner, data: &[u8]| {
            let lsn = inner.alloc_lsn();
            let size = LogHeader::size() + data.len();
            let buffer = unsafe { inner.alloc_buffer(size) };
            buffer[..LogHeader::size()].copy_from_slice(LogHeader::new(lsn, 0, size).as_slice());
            buffer[LogHeader::size()..].copy_from_slice(data);
        };

        inner.flush(false);
        assert_eq!(vfs.flushes.load(Ordering::Relaxed), 0);
        append(&mut inner, &[1; 600]);
        inner.flush(false);
        for _ in 0..10 {
            inner.flush(false);
        }
        assert_eq!(vfs.flushes.load(Ordering::Relaxed), 1);
        assert_eq!(inner.next_flushed_lsn, 1);

        append(&mut inner, &[2; 8]);
        inner.flush(false);
        assert_eq!(vfs.flushes.load(Ordering::Relaxed), 2);
        assert_eq!(inner.next_flushed_lsn, 2);
        inner.need_flush = true;
        inner.flush(true);
        assert_eq!(vfs.flushes.load(Ordering::Relaxed), 2);
        assert!(!inner.need_flush);
        assert_eq!(inner.buffer_cursor, 0);
        assert_eq!(inner.flushed_cursor, 0);
        assert_eq!(inner.file_offset, 4096);

        append(&mut inner, &[3; 8]);
        inner.flush(false);
        assert_eq!(inner.next_flushed_lsn, 3);
        assert_eq!(vfs.flushes.load(Ordering::Relaxed), 3);

        let writes = vfs.writes.lock().unwrap();
        assert_eq!(writes.len(), 3);
        assert_eq!((writes[0].0, writes[0].1.len()), (0, 1024));
        assert_eq!((writes[1].0, writes[1].1.len()), (512, 512));
        assert_eq!((writes[2].0, writes[2].1.len()), (4096, 512));
        let mut data = vec![0; 4096];
        for (offset, bytes) in &writes[..2] {
            data[*offset..*offset + bytes.len()].copy_from_slice(bytes);
        }
        let segment = WalSegment { data };
        let entries: Vec<_> = segment.entry_iter().map(|(_, data)| data).collect();
        assert_eq!(entries, [&[1; 600][..], &[2; 8][..]]);
    }

    #[test]
    fn truncated_wal_records_are_not_zero_filled_or_revived_on_resume() {
        const SEGMENT_SIZE: usize = 512;
        const ENTRY_SIZE: usize = LogHeader::size() + 8;
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut config = WalConfig::new(file.path());
        config.segment_size(SEGMENT_SIZE);

        let make_entry = |lsn, value: usize| {
            let mut entry = LogHeader::new(lsn, 0, ENTRY_SIZE).as_slice().to_vec();
            entry.extend_from_slice(&value.to_le_bytes());
            entry
        };
        // Exercise a torn header and payload both within a segment and after a
        // segment boundary, including files containing no complete record.
        for second_offset in [ENTRY_SIZE, SEGMENT_SIZE] {
            let mut original = vec![0; second_offset];
            original[..ENTRY_SIZE].copy_from_slice(&make_entry(0, 10));
            original.extend_from_slice(&make_entry(1, 20));
            for cut in 0..=original.len() {
                std::fs::write(file.path(), &original[..cut]).unwrap();
                let expected_count =
                    usize::from(cut >= ENTRY_SIZE) + usize::from(cut >= second_offset + ENTRY_SIZE);
                let read_values = || {
                    WalReader::new(file.path(), SEGMENT_SIZE)
                        .segment_iter()
                        .flat_map(|segment| {
                            segment
                                .entry_iter()
                                .map(|(_, data)| TestLogEntry::read_from_buffer(data).val)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(read_values(), [10, 20][..expected_count], "cut {cut}");

                let (offset, next_lsn) = WriteAheadLog::resume_state(&config);
                assert_eq!(next_lsn, expected_count as u64, "cut {cut}");
                crate::fs::write_all_at(file.as_file(), &make_entry(next_lsn, 30), offset as u64)
                    .unwrap();
                let mut expected = [10, 20][..expected_count].to_vec();
                expected.push(30);
                assert_eq!(read_values(), expected, "resumed at cut {cut}");
            }
        }
    }

    #[test]
    fn reopening_wal_appends_with_monotonic_lsn() {
        const TEST_SEGMENT_SIZE: usize = 512;
        let pid = std::process::id();
        let tid = utils::thread_id_to_u64(std::thread::current().id());
        let tmp_file = std::env::temp_dir().join(format!("wal_reopen_test_{pid}_{tid}.log"));
        _ = std::fs::remove_file(&tmp_file);

        let make_wal = || {
            let mut config = WalConfig::new(&tmp_file);
            config
                .segment_size(TEST_SEGMENT_SIZE)
                .flush_interval(Duration::from_micros(1));
            WriteAheadLog::new(Arc::new(config))
        };

        let wal = make_wal();
        assert_eq!(wal.append_and_wait(&TestLogEntry::new(10), 10), 0);
        wal.stop_background_job();
        drop(wal);

        let wal = make_wal();
        assert_eq!(wal.append_and_wait(&TestLogEntry::new(20), 20), 1);
        wal.stop_background_job();
        drop(wal);

        let values = WalReader::new(&tmp_file, TEST_SEGMENT_SIZE)
            .segment_iter()
            .flat_map(|segment| {
                segment
                    .entry_iter()
                    .map(|(_, data)| TestLogEntry::read_from_buffer(data).val)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, [10, 20]);
        std::fs::remove_file(tmp_file).unwrap();
    }

    #[test]
    fn malformed_wal_header_stops_segment_without_panicking() {
        let mut data = vec![0; 512];
        data[0..8].copy_from_slice(&usize::MAX.to_le_bytes());
        let segment = WalSegment { data };
        assert!(segment.entry_iter().next().is_none());
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
    fn first_append_is_durable_before_returning() {
        const TEST_SEGMENT_SIZE: usize = 512;
        let pid = std::process::id();
        let tid = utils::thread_id_to_u64(std::thread::current().id());
        let wal = make_test_wal(
            &format!("wal_first_append_test_{pid}_{tid}.log"),
            TEST_SEGMENT_SIZE,
        );
        let tmp_file = wal.config.file_path.clone();

        let log = TestLogEntry::new(42);
        assert_eq!(wal.append_and_wait(&log, 7), 0);

        let reader = WalReader::new(&tmp_file, TEST_SEGMENT_SIZE);
        let entries = reader
            .segment_iter()
            .flat_map(|segment| {
                segment
                    .entry_iter()
                    .map(|(header, data)| (header, data.to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0.lsn, 0);
        assert_eq!(entries[0].0.page_offset, 7);
        assert_eq!(TestLogEntry::read_from_buffer(&entries[0].1).val, 42);

        wal.stop_background_job();
        drop(wal);
        std::fs::remove_file(tmp_file).unwrap();
    }

    #[test]
    fn append_rotates_when_entry_does_not_fit_remaining_segment() {
        const TEST_SEGMENT_SIZE: usize = 512;
        let pid = std::process::id();
        let tid = utils::thread_id_to_u64(std::thread::current().id());
        let wal = make_test_wal(
            &format!("wal_rotate_test_{pid}_{tid}.log"),
            TEST_SEGMENT_SIZE,
        );
        let tmp_file = wal.config.file_path.clone();

        let first = BytesLogEntry {
            bytes: vec![1; 200],
        };
        let second = BytesLogEntry {
            bytes: vec![2; 300],
        };
        assert_eq!(wal.append_and_wait(&first, 1), 0);
        assert_eq!(wal.append_and_wait(&second, 2), 1);

        let reader = WalReader::new(&tmp_file, TEST_SEGMENT_SIZE);
        let entries = reader
            .segment_iter()
            .flat_map(|segment| {
                segment
                    .entry_iter()
                    .map(|(header, data)| (header, data.to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].1, first.bytes);
        assert_eq!(entries[1].1, second.bytes);

        wal.stop_background_job();
        drop(wal);
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
