// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::nodes::LeafNode;

use super::VfsImpl;

pub(crate) struct MemoryVfs {}

impl MemoryVfs {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl VfsImpl for MemoryVfs {
    fn open(_path: impl AsRef<std::path::Path>) -> Self
    where
        Self: Sized,
    {
        Self::new()
    }
    fn read(&self, offset: usize, buf: &mut [u8]) {
        let buf_to_read = unsafe { std::slice::from_raw_parts(offset as *const u8, buf.len()) };
        buf.copy_from_slice(buf_to_read);
    }

    fn write(&self, offset: usize, buf: &[u8]) {
        let buf_to_write = unsafe { std::slice::from_raw_parts_mut(offset as *mut u8, buf.len()) };
        buf_to_write.copy_from_slice(buf);
    }

    fn flush(&self) {
        // Noop
    }

    fn alloc_offset(&self, size: usize) -> usize {
        LeafNode::make_base_page(size) as usize
    }

    fn dealloc_offset(&self, offset: usize) {
        LeafNode::free_base_page(offset as *mut LeafNode);
    }
}
