// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    ffi::CString,
    fs::File,
    io::Write,
    os::{
        fd::FromRawFd,
        unix::{ffi::OsStrExt, fs::FileExt},
    },
    path::PathBuf,
};

use crate::counter;

use super::{OffsetAlloc, VfsImpl};

pub(crate) struct StdDirectVfs {
    file: File,
    offset_alloc: OffsetAlloc,
    _path: PathBuf,
}

impl VfsImpl for StdDirectVfs {
    fn alloc_offset(&self, size: usize) -> usize {
        self.offset_alloc.alloc(size)
    }

    fn open(path: impl AsRef<std::path::Path>) -> Self
    where
        Self: Sized,
    {
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

        Self {
            file,
            offset_alloc: OffsetAlloc::new_with(offset as usize),
            _path: path.to_path_buf(),
        }
    }

    fn dealloc_offset(&self, offset: usize) {
        self.offset_alloc.dealloc_offset(offset)
    }

    fn read(&self, offset: usize, buf: &mut [u8]) {
        counter!(IOReadRequest);
        self.file.read_at(buf, offset as u64).unwrap();
    }

    fn flush(&self) {
        self.file.sync_all().unwrap();
    }

    fn write(&self, offset: usize, buf: &[u8]) {
        counter!(IOWriteRequest);
        self.file.write_at(buf, offset as u64).unwrap();
    }
}
