// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(target_os = "linux")]
use crate::fs::IoUringVfs;

#[cfg(all(target_os = "linux", feature = "spdk"))]
use crate::fs::SpdkVfs;

use crate::{
    circular_buffer::{
        CircularBuffer, CircularBufferError, CircularBufferMetrics, CircularBufferPtr,
        TombstoneHandle,
    },
    counter,
    error::TreeError,
    fs::{MemoryVfs, StdVfs, VfsImpl},
    mini_page_op::{LeafEntrySLocked, LeafEntryXLocked},
    nodes::{LeafNode, PageID},
    utils::{rw_lock::RwLock, MappingTable},
    Config, StorageBackend,
};
use std::{path::Path, sync::Arc};

#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) enum PageLocation {
    Mini(*mut LeafNode),
    Full(*mut LeafNode), // full pages are in memory
    Base(usize),
    Null,
}

impl From<CircularBufferError> for TreeError {
    fn from(value: CircularBufferError) -> Self {
        match value {
            CircularBufferError::WouldBlock => TreeError::Locked,
            CircularBufferError::Full => TreeError::CircularBufferFull,
            CircularBufferError::EmptyAlloc => unreachable!(),
        }
    }
}

pub(crate) struct PageTableIter<'a> {
    table: &'a PageTable,
    cur_id: u64,
    high_id: u64,
}

impl<'a> PageTableIter<'a> {
    fn new(table: &'a PageTable) -> Self {
        let high_id = table.table.peek_next_id();
        Self {
            table,
            cur_id: 0,
            high_id,
        }
    }
}

impl<'a> Iterator for PageTableIter<'a> {
    type Item = (&'a RwLock<PageLocation>, PageID);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_id == self.high_id {
            return None;
        }

        let id = PageID::from_id(self.cur_id);
        self.cur_id += 1;

        Some((self.table.table.get(id.as_id()), id))
    }
}

pub(crate) struct PageTable {
    table: MappingTable<RwLock<PageLocation>>,
    vfs: Arc<dyn VfsImpl>,
    pub(crate) config: Arc<Config>,
}

impl PageTable {
    fn new(file_handle: Arc<dyn VfsImpl>, config: Arc<Config>) -> Self {
        let mapping = MappingTable::default();
        Self {
            table: mapping,
            vfs: file_handle,
            config,
        }
    }

    pub(crate) fn new_from_mapping(
        mapping: impl Iterator<Item = (PageID, PageLocation)>,
        vfs: Arc<dyn VfsImpl>,
        config: Arc<Config>,
    ) -> Self {
        let mapping = mapping.map(|(pid, loc)| (pid.as_id(), RwLock::new(loc)));
        Self {
            table: MappingTable::new_from_iter(mapping),
            vfs,
            config,
        }
    }

    pub(crate) fn get(&self, pid: &PageID) -> LeafEntrySLocked<'_> {
        let id = pid.as_id();
        let v = self.table.get(id);
        let v = v.read();
        LeafEntrySLocked::new(
            v,
            #[cfg(feature = "tracing")]
            *pid,
            self.vfs.as_ref(),
            self.config.leaf_page_size,
        )
    }

    pub(crate) fn get_mut(&self, pid: &PageID) -> LeafEntryXLocked<'_> {
        let id = pid.as_id();
        let v = self.table.get(id);
        let v = v.write();
        LeafEntryXLocked::new(
            v,
            #[cfg(feature = "tracing")]
            *pid,
            self.vfs.as_ref(),
            self.config.leaf_page_size,
        )
    }

    /// Mostly, you need to clean up the offset, i.e., adding the offset to the free list.
    #[must_use = "this function allocates resources, your responsibility to cleanup if not used."]
    pub(crate) fn alloc_base_page_mapping(&self) -> (PageID, LeafEntryXLocked<'_>) {
        counter!(AllocDiskID);
        let loc = PageLocation::Base(self.vfs.alloc_offset(self.config.leaf_page_size)); // Allocate space in disk for a full leaf page
        let entry = RwLock::new(loc);
        let (id, value) = self.table.insert(entry);
        let pid = PageID::from_id(id);
        let lock_guard = value.try_write().unwrap();
        let base_ptr = LeafNode::make_base_page(self.config.leaf_page_size);
        let x_locked = LeafEntryXLocked::with_buffer(
            lock_guard,
            self.vfs.as_ref(),
            #[cfg(feature = "tracing")]
            pid,
            self.config.leaf_page_size,
            base_ptr,
        );
        LeafNode::free_base_page(base_ptr);

        (pid, x_locked)
    }

    pub(crate) fn iter(&self) -> PageTableIter<'_> {
        PageTableIter::new(self)
    }

    /// Add a mini-page into the mapping table
    pub(crate) fn insert_mini_page_mapping(
        &self,
        mini_loc: PageLocation,
    ) -> (PageID, LeafEntryXLocked<'_>) {
        match mini_loc {
            PageLocation::Mini(_) => {}
            _ => {
                panic!("Expecting to insert a new mini-page into mapping table but got a full/base page.");
            }
        }
        let entry = RwLock::new(mini_loc);
        let (id, value) = self.table.insert(entry);
        let pid = PageID::from_id(id);
        let lock_guard = value.try_write().unwrap();

        let x_locked = LeafEntryXLocked::new(
            lock_guard,
            #[cfg(feature = "tracing")]
            pid,
            self.vfs.as_ref(),
            self.config.leaf_page_size,
        );

        (pid, x_locked)
    }
}

pub(crate) struct LeafStorage {
    pub(crate) page_table: PageTable,
    pub(crate) circular_buffer: CircularBuffer,
    pub(crate) vfs: Arc<dyn VfsImpl>,
    pub(crate) config: Arc<Config>,
}

impl Drop for LeafStorage {
    fn drop(&mut self) {}
}

impl LeafStorage {
    pub(crate) fn new(config: Arc<Config>, buffer_ptr: Option<*mut u8>) -> Self {
        let vfs: Arc<dyn VfsImpl> = make_vfs(&config.storage_backend, &config.file_path);
        let page_table = PageTable::new(vfs.clone(), config.clone());
        let circular_buffer = CircularBuffer::new(
            config.cb_size_byte,
            config.cb_copy_on_access_ratio,
            config.cb_min_record_size,
            config.cb_max_record_size,
            config.leaf_page_size,
            config.max_fence_len,
            buffer_ptr,
            config.cache_only,
        );
        Self::new_inner(config, page_table, circular_buffer, vfs)
    }

    pub(crate) fn new_inner(
        config: Arc<Config>,
        page_table: PageTable,
        circular_buffer: CircularBuffer,
        vfs: Arc<dyn VfsImpl>,
    ) -> Self {
        Self {
            page_table,
            circular_buffer,
            config,
            vfs,
        }
    }

    pub(crate) fn mapping_table(&self) -> &PageTable {
        &self.page_table
    }

    pub(crate) fn alloc_disk_offset(&self, size: usize) -> DiskOffsetGuard<'_> {
        DiskOffsetGuard::new(self.vfs.alloc_offset(size), self.vfs.as_ref())
    }

    pub(crate) fn get_buffer_metrics(&self) -> CircularBufferMetrics {
        self.circular_buffer.get_metrics()
    }

    pub(crate) fn alloc_mini_page(&self, size: usize) -> Result<CircularBufferPtr<'_>, TreeError> {
        let v = self.circular_buffer.alloc(size)?;
        Ok(v)
    }

    pub(crate) fn alloc_base_page_and_lock(&self) -> (PageID, LeafEntryXLocked<'_>) {
        let (pid, base_entry) = self.page_table.alloc_base_page_mapping();

        (pid, base_entry)
    }

    /// Returns the actual n
    pub(crate) fn evict_from_buffer(
        &self,
        mut callback: impl FnMut(&TombstoneHandle) -> Result<(), TreeError>,
    ) -> Result<u32, TreeError> {
        let mut callback = |h| -> Result<TombstoneHandle, TombstoneHandle> {
            match callback(&h) {
                Ok(_) => Ok(h),
                Err(_e) => Err(h),
            }
        };
        let advanced = self.circular_buffer.evict_one(&mut callback).unwrap_or(0);

        Ok(advanced)
    }

    /// Returns the dealloc handle.
    /// Errors on contention
    pub(crate) fn begin_dealloc_mini_page(
        &self,
        mini_page: *mut LeafNode,
    ) -> Result<TombstoneHandle, TreeError> {
        match unsafe {
            self.circular_buffer
                .acquire_exclusive_dealloc_handle(mini_page as *mut u8)
        } {
            Ok(h) => Ok(h),
            Err(_) => Err(TreeError::NeedRestart),
        }
    }

    /// Deallocates a mini page, won't acquire lock on mini page store.
    pub(crate) fn finish_dealloc_mini_page(&self, mini_page: TombstoneHandle) {
        #[cfg(debug_assertions)]
        {
            let mini_page_ref = unsafe { &*(mini_page.ptr as *mut LeafNode) };
            let size = mini_page_ref.meta.node_size;
            unsafe {
                std::ptr::write_bytes(mini_page.ptr, 0, size as usize);
            }
        }

        {
            self.circular_buffer.dealloc(mini_page);
        }
    }

    pub(crate) fn mini_page_copy_on_access(&self, mini_page: &LeafNode) -> bool {
        self.circular_buffer
            .ptr_is_copy_on_access(mini_page as *const LeafNode as *mut u8)
    }

    /// Full page can't discard old cache, because every record is a cache when promoted to full page.
    /// TODO: This is probably not desired, we should refactor to this:
    /// Have a page-level dirty flag, indicating whether the page is dirty.
    pub(crate) fn move_full_page_to_tail(
        &self,
        mini_page: TombstoneHandle,
        size: usize,
    ) -> Result<CircularBufferPtr<'_>, TreeError> {
        let new_page = self.circular_buffer.alloc(size)?;

        unsafe {
            std::ptr::copy_nonoverlapping(mini_page.ptr, new_page.as_ptr(), size);
        }
        self.circular_buffer.dealloc(mini_page);
        Ok(new_page)
    }

    pub(crate) fn move_mini_page_to_tail(
        &self,
        mini_page: TombstoneHandle,
        size: usize,
    ) -> Result<CircularBufferPtr<'_>, TreeError> {
        let new_page = self.circular_buffer.alloc(size)?;

        let mini_page_ptr = mini_page.ptr as *mut LeafNode;
        unsafe { &*mini_page_ptr }.copy_initialize_to(
            new_page.as_ptr() as *mut LeafNode,
            size,
            true,
        );

        self.circular_buffer.dealloc(mini_page);
        unsafe {
            debug_assert!(
                (&*(new_page.as_ptr() as *mut LeafNode))
                    .meta
                    .meta_count_with_fence()
                    > 0
            );
        }
        Ok(new_page)
    }
}

pub(crate) struct DiskOffsetGuard<'a> {
    offset: usize,
    vfs: &'a dyn VfsImpl,
}

impl<'a> DiskOffsetGuard<'a> {
    pub(crate) fn new(offset: usize, vfs: &'a dyn VfsImpl) -> Self {
        Self { offset, vfs }
    }

    pub(crate) fn take(self) -> usize {
        let o = self.offset;
        std::mem::forget(self);
        o
    }
}

impl Drop for DiskOffsetGuard<'_> {
    fn drop(&mut self) {
        self.vfs.dealloc_offset(self.offset);
    }
}

pub(crate) fn make_vfs(
    storage_backend: &StorageBackend,
    path: impl AsRef<Path>,
) -> Arc<dyn VfsImpl> {
    match storage_backend {
        StorageBackend::Memory => Arc::new(MemoryVfs::new()),
        StorageBackend::Std => Arc::new(StdVfs::open(path.as_ref())),

        #[cfg(target_os = "linux")]
        StorageBackend::IoUringPolling => Arc::new(IoUringVfs::open(path.as_ref())),

        #[cfg(target_os = "linux")]
        StorageBackend::IoUringBlocking => Arc::new(IoUringVfs::new_blocking(path.as_ref())),

        #[cfg(target_os = "linux")]
        StorageBackend::StdDirect => Arc::new(crate::fs::StdDirectVfs::open(path.as_ref())),

        #[cfg(all(target_os = "linux", feature = "spdk"))]
        StorageBackend::Spdk => Arc::new(SpdkVfs::open(path.as_ref())),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{PageLocation, PageTable};
    use crate::{
        fs::{MemoryVfs, VfsImpl},
        mini_page_op::LeafOperations,
        utils::MappingTable,
        Config,
    };

    #[test]
    fn page_table_de_ser_round_trip() {
        let vfs = Arc::new(MemoryVfs::new());
        let config = Arc::new(Config::default());
        let table = PageTable {
            table: MappingTable::new(1024),
            vfs: vfs.clone(),
            config: config.clone(),
        };

        let mut allocated = Vec::new();
        for _i in 0..2048 {
            let (pid, leaf) = table.alloc_base_page_mapping();
            let loc = leaf.get_page_location();
            allocated.push((pid, loc.clone()));
        }

        let pt_iter = table.iter();

        for (a, b) in allocated.iter().zip(pt_iter) {
            assert_eq!(a.0, b.1);
        }

        let pt_iter = table.iter().map(|(lock, pid)| {
            let loc = lock.try_read().unwrap();
            (pid, loc.clone())
        });

        let recovered = PageTable::new_from_mapping(pt_iter, vfs.clone(), config.clone());

        for (a, b) in allocated.iter().zip(recovered.iter()) {
            assert_eq!(a.0, b.1);
        }

        for a in allocated.into_iter() {
            match a.1 {
                PageLocation::Base(offset) => vfs.dealloc_offset(offset),
                _ => unreachable!(),
            }
        }
    }
}
