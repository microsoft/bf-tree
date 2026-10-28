// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    fs::{File, OpenOptions},
    path::PathBuf,
};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use crate::counter;

use super::{OffsetAlloc, VfsImpl};

pub(crate) struct StdVfs {
    file: File,
    offset_alloc: OffsetAlloc,
    _path: PathBuf,
}

impl VfsImpl for StdVfs {
    fn alloc_offset(&self, size: usize) -> usize {
        self.offset_alloc.alloc(size)
    }

    fn open(path: impl AsRef<std::path::Path>) -> Self
    where
        Self: Sized,
    {
        let path = path.as_ref().to_path_buf();
        let parent = path.parent().unwrap();
        _ = std::fs::create_dir_all(parent);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .unwrap();
        let offset = file.metadata().unwrap().len();
        Self {
            file,
            offset_alloc: OffsetAlloc::new_with(offset as usize),
            _path: path.to_path_buf(),
        }
    }

    fn dealloc_offset(&self, offset: usize) {
        self.offset_alloc.dealloc_offset(offset)
    }

    #[cfg(unix)]
    fn read(&self, offset: usize, buf: &mut [u8]) {
        counter!(IOReadRequest);
        self.file.read_exact_at(buf, offset as u64).unwrap();
    }

    #[cfg(windows)]
    fn read(&self, offset: usize, buf: &mut [u8]) {
        counter!(IOReadRequest);
        let mut done = 0;
        while done < buf.len() {
            match self
                .file
                .seek_read(&mut buf[done..], (offset + done) as u64)
            {
                Ok(0) => panic!("unexpected EOF reading at offset {}", offset + done),
                Ok(n) => done += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => panic!("read failed at offset {}: {e}", offset + done),
            }
        }
    }

    fn flush(&self) {
        self.file.sync_all().unwrap();
    }

    #[cfg(unix)]
    fn write(&self, offset: usize, buf: &[u8]) {
        counter!(IOWriteRequest);
        self.file.write_all_at(buf, offset as u64).unwrap();
    }

    #[cfg(windows)]
    fn write(&self, offset: usize, buf: &[u8]) {
        counter!(IOWriteRequest);
        let mut done = 0;
        while done < buf.len() {
            match self.file.seek_write(&buf[done..], (offset + done) as u64) {
                Ok(0) => panic!("zero-length write at offset {}", offset + done),
                Ok(n) => done += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => panic!("write failed at offset {}: {e}", offset + done),
            }
        }
    }

    fn reset(&self) {
        assert!(self.file.set_len(0).is_ok());
        let offset = self.file.metadata().unwrap().len();
        self.offset_alloc.reset(offset as usize);
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    #[test]
    fn positioned_io_and_short_read() {
        let dir = tempfile::tempdir().unwrap();
        let vfs = StdVfs::open(dir.path().join("positioned.db"));
        let offset = vfs.alloc_offset(512);
        vfs.write(offset, &[0x35; 512]);
        let mut bytes = [0; 512];
        vfs.read(offset, &mut bytes);
        assert_eq!(bytes, [0x35; 512]);
        vfs.file.set_len((offset + 256) as u64).unwrap();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            vfs.read(offset, &mut bytes);
        }))
        .is_err());
    }
}
