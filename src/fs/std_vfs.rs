// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    fs::{File, OpenOptions},
    path::PathBuf,
};

use crate::counter;

use super::{read_exact_at, write_all_at, OffsetAlloc, VfsImpl};

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

    fn read(&self, offset: usize, buf: &mut [u8]) {
        counter!(IOReadRequest);
        read_exact_at(&self.file, buf, offset as u64).unwrap();
    }

    fn flush(&self) {
        self.file.sync_all().unwrap();
    }

    fn write(&self, offset: usize, buf: &[u8]) {
        counter!(IOWriteRequest);
        write_all_at(&self.file, buf, offset as u64).unwrap();
    }

    fn reset(&self) {
        assert!(self.file.set_len(0).is_ok());
        let offset = self.file.metadata().unwrap().len();
        self.offset_alloc.reset(offset as usize);
    }
}
