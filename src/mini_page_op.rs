// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(feature = "tracing")]
use crate::nodes::PageID;
use crate::{
    circular_buffer::TombstoneHandle,
    counter,
    error::TreeError,
    fs::{buffer_alloc, buffer_dealloc, VfsImpl},
    histogram, info,
    nodes::{
        leaf_node::{
            GetScanRecordByPosResult, LeafNode, LeafReadResult as LeafResult, MiniPageNextLevel,
            OpType,
        },
        DISK_PAGE_SIZE,
    },
    range_scan::{ScanError, ScanPosition, ScanReturnField},
    storage::{LeafStorage, PageLocation},
    tree::key_value_physical_size,
    utils::stats::LeafStats,
    utils::{
        inner_lock::ReadGuard,
        rw_lock::{RwLockReadGuard, RwLockWriteGuard},
    },
};
use core::panic;
use std::{
    alloc::Layout,
    ops::{Deref, DerefMut},
};

struct TmpBuffer {
    is_dirty: bool,
    size: usize, // Equals to leaf base page size
    ptr: *mut u8,
}

impl TmpBuffer {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, DISK_PAGE_SIZE).unwrap();
        let buffer = buffer_alloc(layout);
        Self {
            is_dirty: false,
            size,
            ptr: buffer,
        }
    }

    fn from_leaf_node(leaf: *mut LeafNode, size: usize) -> Self {
        let mut buffer = Self::new(size);
        assert!(unsafe { &*leaf }.meta.node_size as usize == size);

        unsafe {
            std::ptr::copy_nonoverlapping(leaf as *const u8, buffer.ptr, size);
        }
        buffer.is_dirty = true;

        buffer
    }

    fn as_u8_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    fn as_u8_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    fn as_leaf_node(&self) -> &LeafNode {
        unsafe { &*(self.ptr as *const LeafNode) }
    }

    fn as_leaf_node_mut(&mut self) -> &mut LeafNode {
        self.is_dirty = true;
        unsafe { &mut *(self.ptr as *mut LeafNode) }
    }
}

impl Drop for TmpBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, DISK_PAGE_SIZE).unwrap();
        buffer_dealloc(self.ptr, layout);
    }
}

pub(crate) trait LeafOperations {
    /// Panic if the page is not in the base page.
    fn load_base_page_from_buffer(&self) -> &LeafNode;

    fn get_page_location(&self) -> &PageLocation;

    fn load_base_page(&mut self, offset: usize) -> &LeafNode;

    fn load_cache_page(&self, ptr: *mut LeafNode) -> &LeafNode {
        unsafe { &*ptr }
    }

    fn scan_record_by_pos_with_bound(
        &self,
        pos: &ScanPosition,
        out_buffer: &mut [u8],
        return_field: ScanReturnField,
        end_key: &Option<Vec<u8>>,
    ) -> GetScanRecordByPosResult {
        match pos {
            ScanPosition::Base(pos) => {
                let base = self.load_base_page_from_buffer();
                base.get_record_by_pos_with_bound(*pos, out_buffer, return_field, end_key)
            }
            ScanPosition::Full(pos) => {
                let page_ptr = match self.get_page_location() {
                    PageLocation::Base(_) | PageLocation::Mini(_) => unreachable!(),
                    PageLocation::Full(ptr) => *ptr,
                    PageLocation::Null => panic!("scan_value_by_pos on Null page"),
                };
                let full = self.load_cache_page(page_ptr);
                full.get_record_by_pos_with_bound(*pos, out_buffer, return_field, end_key)
            }
        }
    }

    fn get_right_sibling(&mut self) -> Vec<u8> {
        let page_loc = self.get_page_location();
        match page_loc {
            PageLocation::Base(offset) => {
                let base_ref = self.load_base_page(*offset);
                base_ref.get_high_fence_key()
            }
            PageLocation::Full(ptr) => {
                let page_ref = self.load_cache_page(*ptr);
                page_ref.get_high_fence_key()
            }
            PageLocation::Mini(ptr) => {
                let mini = self.load_cache_page(*ptr);
                let offset = mini.next_level.as_offset();
                let base_ref = self.load_base_page(offset);
                base_ref.get_high_fence_key()
            }
            PageLocation::Null => panic!("get_right_sibling on Null page"),
        }
    }

    fn cache_page_about_to_evict(&self, storage: &LeafStorage) -> bool {
        match self.get_page_location() {
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let mini_page = self.load_cache_page(*ptr);
                storage.mini_page_copy_on_access(mini_page)
            }
            PageLocation::Base(_) => false,
            PageLocation::Null => panic!("cache_page_about_to_evict on Null page"),
        }
    }

    fn read(
        &mut self,
        key: &[u8],
        out_buffer: &mut [u8],
        mini_page_binary_search: bool,
        cache_only: bool,
    ) -> ReadResult {
        let page_loc = self.get_page_location();

        let next_level = match page_loc {
            PageLocation::Base(offset) => {
                let base_ref = self.load_base_page(*offset);
                let out = base_ref.read_by_key(key, out_buffer);
                return ReadResult::Base(out);
            }
            PageLocation::Full(ptr) => {
                counter!(FullPageRead);
                let base_ref = self.load_cache_page(*ptr);
                let out = base_ref.read_by_key(key, out_buffer);
                histogram!(HitMiniPage, base_ref.meta.node_size as u64);
                return ReadResult::Full(out);
            }
            PageLocation::Mini(ptr) => {
                counter!(MiniPageRead);
                let mini_page = self.load_cache_page(*ptr);
                let out = mini_page.read_by_key_inner(key, out_buffer, mini_page_binary_search);
                histogram!(HitMiniPage, mini_page.meta.node_size as u64);
                match out {
                    LeafResult::Found(_) | LeafResult::Deleted | LeafResult::InvalidKey => {
                        return ReadResult::Mini(out);
                    }
                    LeafResult::NotFound => {
                        // In cache only mode, we return not found
                        if cache_only {
                            return ReadResult::None;
                        }

                        // fall through
                        counter!(MiniPageReadMiss);
                        mini_page.next_level
                    }
                }
            }
            PageLocation::Null => {
                return ReadResult::None;
            }
        };

        let base_ref = self.load_base_page(next_level.as_offset());

        let out = base_ref.read_by_key(key, out_buffer);
        ReadResult::Base(out)
    }

    /// Returns the pos of the key when scanning, i.e., the keys[pos] >= key.
    ///
    /// If we hit a page with mini page, we need to merge it, consolidate it, then scan.
    fn get_scan_position(&mut self, key: &[u8]) -> Result<ScanPosition, ScanError> {
        let page_loc = self.get_page_location();

        match page_loc {
            PageLocation::Base(offset) => {
                let base_ref = self.load_base_page(*offset);
                let pos = base_ref.lower_bound(key);
                Ok(ScanPosition::Base(pos as u32))
            }
            PageLocation::Full(ptr) => {
                let page_ref = self.load_cache_page(*ptr);
                let pos = page_ref.lower_bound(key);
                Ok(ScanPosition::Full(pos as u32))
            }
            PageLocation::Mini(_ptr) => Err(ScanError::NeedMergeMiniPage),
            PageLocation::Null => panic!("get_scan_position on Null page"),
        }
    }
}

/// Page table read locked entry.
pub(crate) struct LeafEntrySLocked<'a> {
    raw_guard: RwLockReadGuard<'a, PageLocation>,
    #[cfg(feature = "tracing")]
    pid: PageID,
    file_handle: &'a dyn VfsImpl,
    tmp_buffer_size: usize,
    tmp_buffer: Option<TmpBuffer>,
}

impl LeafOperations for LeafEntrySLocked<'_> {
    fn load_base_page_from_buffer(&self) -> &LeafNode {
        let l = self.tmp_buffer.as_ref().unwrap().as_leaf_node();
        debug_assert!(l.is_base_page());
        l
    }

    fn get_page_location(&self) -> &PageLocation {
        self.raw_guard.deref()
    }

    fn load_base_page(&mut self, offset: usize) -> &LeafNode {
        match self.tmp_buffer {
            Some(ref buffer) => buffer.as_leaf_node(),
            None => {
                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                let slice = buffer.as_u8_slice_mut();
                self.file_handle.read(offset, slice);

                self.tmp_buffer = Some(buffer);
                self.load_base_page(offset)
            }
        }
    }
}

impl<'a> LeafEntrySLocked<'a> {
    pub(crate) fn new(
        guard: RwLockReadGuard<'a, PageLocation>,
        #[cfg(feature = "tracing")] pid: PageID,
        file_handle: &'a dyn VfsImpl,
        leaf_page_size: usize,
    ) -> Self {
        Self {
            raw_guard: guard,
            #[cfg(feature = "tracing")]
            pid,
            file_handle,
            tmp_buffer_size: leaf_page_size,
            tmp_buffer: None,
        }
    }

    /// Upgrade the reader lock to writer lock.
    pub(crate) fn try_upgrade(mut self) -> Result<LeafEntryXLocked<'a>, LeafEntrySLocked<'a>> {
        match self.raw_guard.try_upgrade() {
            Ok(g) => {
                let x = LeafEntryXLocked {
                    raw_guard: g,
                    #[cfg(feature = "tracing")]
                    pid: self.pid,
                    file_handle: self.file_handle,
                    tmp_buffer_size: self.tmp_buffer_size,
                    tmp_buffer: self.tmp_buffer.take(),
                };
                // here we don't need to explicitly forget self.
                return Ok(x);
            }
            Err(e) => {
                self.raw_guard = e;
            }
        };
        Err(self)
    }
}

pub(crate) enum ReadResult {
    Mini(LeafResult),
    Full(LeafResult),
    Base(LeafResult),
    None,
}

/// Page table write locked entry.
pub(crate) struct LeafEntryXLocked<'a> {
    raw_guard: RwLockWriteGuard<'a, PageLocation>,
    #[cfg(feature = "tracing")]
    pid: PageID,
    file_handle: &'a dyn VfsImpl,
    tmp_buffer_size: usize,
    tmp_buffer: Option<TmpBuffer>,
}

impl Drop for LeafEntryXLocked<'_> {
    fn drop(&mut self) {
        if let Some(ref b) = self.tmp_buffer {
            if b.is_dirty {
                let offset = match self.raw_guard.deref() {
                    PageLocation::Base(offset) => *offset,
                    PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                        let mini_page = self.load_cache_page_mut(*ptr);
                        mini_page.next_level.as_offset()
                    }
                    PageLocation::Null => panic!("Dropping a tmp buffer of a Null page"),
                };
                let slice = b.as_u8_slice();
                self.file_handle.write(offset, slice);
            }
        }
    }
}

impl LeafOperations for LeafEntryXLocked<'_> {
    fn load_base_page_from_buffer(&self) -> &LeafNode {
        let l = self.tmp_buffer.as_ref().unwrap().as_leaf_node();
        debug_assert!(l.is_base_page());
        l
    }

    fn get_page_location(&self) -> &PageLocation {
        self.raw_guard.deref()
    }

    fn load_base_page(&mut self, offset: usize) -> &LeafNode {
        match self.tmp_buffer {
            Some(ref mut buffer) => return buffer.as_leaf_node(),
            None => {
                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                let slice = buffer.as_u8_slice_mut();
                self.file_handle.read(offset, slice);

                self.tmp_buffer = Some(buffer);

                if let Some(ref b) = self.tmp_buffer {
                    return b.as_leaf_node();
                }
            }
        };

        unreachable!();
    }
}

impl<'a> LeafEntryXLocked<'a> {
    pub(crate) fn new(
        raw_guard: RwLockWriteGuard<'a, PageLocation>,
        #[cfg(feature = "tracing")] page_id: PageID,
        file_handle: &'a dyn VfsImpl,
        tmp_buffer_size: usize,
    ) -> Self {
        Self {
            raw_guard,
            file_handle,
            #[cfg(feature = "tracing")]
            pid: page_id,
            tmp_buffer_size,
            tmp_buffer: None,
        }
    }

    pub(crate) fn with_buffer(
        raw_guard: RwLockWriteGuard<'a, PageLocation>,
        file_handle: &'a dyn VfsImpl,
        #[cfg(feature = "tracing")] page_id: PageID,
        tmp_buffer_size: usize,
        leaf_buffer: *mut LeafNode,
    ) -> Self {
        Self {
            raw_guard,
            file_handle,
            #[cfg(feature = "tracing")]
            pid: page_id,
            tmp_buffer_size,
            tmp_buffer: Some(TmpBuffer::from_leaf_node(leaf_buffer, tmp_buffer_size)),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn downgrade(self) -> LeafEntrySLocked<'a> {
        todo!("downgrade is very challenging with current implementation!")
    }

    pub(crate) fn dealloc_self(&mut self, storage: &LeafStorage, cache_only: bool) {
        let page_loc = self.raw_guard.deref().clone();
        match page_loc {
            PageLocation::Base(offset) => {
                assert!(self.load_base_page(offset).next_level.is_null());
                self.file_handle.dealloc_offset(offset);
            }
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let leaf_node = self.load_cache_page_mut(ptr);
                let h = storage.begin_dealloc_mini_page(leaf_node).unwrap();
                let base_page = leaf_node.next_level;
                storage.finish_dealloc_mini_page(h);

                // In cache-only mode, there is no base page allocated
                if !cache_only {
                    self.file_handle.dealloc_offset(base_page.as_offset());
                }
            }
            PageLocation::Null => {}
        }
        // we need to drop buffer here, o.w., the buffer will be write to offset, which is no longer accessible.
        let buffer = self.tmp_buffer.take();
        std::mem::drop(buffer);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert(
        &mut self,
        key: &[u8],
        value: &[u8],
        parent: Option<ReadGuard<'_>>,
        op_type: OpType,
        storage: &LeafStorage,
        write_load_full_page: &bool,
        cache_only: &bool,
        mini_page_size_classes: &[usize],
    ) -> Result<(), TreeError> {
        let page_loc = self.raw_guard.deref().clone();
        match page_loc {
            PageLocation::Base(offset) => {
                if *cache_only {
                    panic!("Insertion to a base page detected in cache-only mode");
                }

                // Root leaf node does not have a corresponding mini-page
                if parent.is_none() {
                    let success = self.load_base_page_mut().insert(
                        key,
                        value,
                        op_type,
                        storage.config.max_fence_len,
                    );
                    if !success {
                        self.load_base_page_mut().set_split_flag();
                        return Err(TreeError::NeedRestart);
                    } else {
                        return Ok(());
                    }
                }

                let mini_page_size = LeafNode::get_chain_size_hint(
                    key.len(),
                    value.len(),
                    mini_page_size_classes,
                    *cache_only,
                );
                let mini_page_guard = storage.alloc_mini_page(mini_page_size)?;

                LeafNode::initialize_mini_page(
                    &mini_page_guard,
                    mini_page_size,
                    MiniPageNextLevel::new(offset),
                    false,
                );

                let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                let mini_loc = PageLocation::Mini(new_mini_ptr);
                self.create_cache_page_loc(mini_loc);

                let mini_page_ref = self.load_cache_page_mut(new_mini_ptr);
                let insert_success = mini_page_ref.insert(key, value, op_type, 0);
                assert!(insert_success);
                counter!(InsertCreatedMiniPage);

                info!(
                    base_id = self.pid.raw(),
                    "finished creating mini page for base"
                );
                Ok(())
            }
            PageLocation::Full(ptr) => {
                if *cache_only {
                    panic!("Insertion to a full page detected in cache-only mode");
                }

                histogram!(HitMiniPage, storage.config.leaf_page_size as u64);
                let mini_page = unsafe { &mut *ptr };
                let insert_success =
                    mini_page.insert(key, value, op_type, storage.config.max_fence_len);
                if insert_success {
                    return Ok(());
                }

                self.merge_full_page_and_dealloc(mini_page, storage)?;

                counter!(InsertMergeFullPage);

                Err(TreeError::NeedRestart)
            }

            PageLocation::Mini(ptr) => {
                // Root leaf node does not have a corresponding mini-page
                // In cache-only mode, root leaf page is a mini-page
                // Unlike other leaf nodes, root leaf node's split is done through
                // detecting split_flag upon tree traversal
                if *cache_only && parent.is_none() {
                    let root_page = self.load_cache_page_mut(ptr);
                    let mut success = root_page.insert(key, value, op_type, 0);

                    // Consolidate first, and then try insert again
                    // We cannot split a root mini-page with a single key
                    if !success {
                        root_page.consolidate_inner(OpType::Insert, None, true, *cache_only, None);
                        success = root_page.insert(key, value, op_type, 0);
                        if !success {
                            // There are at least two distinctive INSERT keys in the root, safe for splitting.
                            root_page.set_split_flag();
                            return Err(TreeError::NeedRestart);
                        } else {
                            return Ok(());
                        }
                    }
                }

                let mini_page = self.load_cache_page_mut(ptr);
                if !(*cache_only) {
                    debug_assert!(!mini_page.next_level.is_null());
                } else {
                    assert!(mini_page.next_level.is_null());
                }
                histogram!(HitMiniPage, mini_page.meta.node_size as u64);
                let insert_success = mini_page.insert(key, value, op_type, 0);

                if insert_success {
                    counter!(InsertToMiniPageSuccess);
                    return Ok(());
                }

                let kv_size = key_value_physical_size(key, value);
                let new_size =
                    mini_page.new_size_if_upgrade(kv_size, mini_page_size_classes, *cache_only);

                match new_size {
                    Some(s) => {
                        info!(
                            from = mini_page.meta.node_size,
                            to = s,
                            "upgrading mini page size"
                        );

                        let h = storage.begin_dealloc_mini_page(mini_page)?;

                        let mini_page_guard = storage.alloc_mini_page(s)?;
                        LeafNode::initialize_mini_page(
                            &mini_page_guard,
                            s,
                            mini_page.next_level,
                            *cache_only,
                        );
                        mini_page.copy_initialize_to(
                            mini_page_guard.as_ptr() as *mut LeafNode,
                            s,
                            true,
                        );

                        let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                        let new_mini_loc = PageLocation::Mini(new_mini_ptr);
                        self.create_cache_page_loc(new_mini_loc);

                        let rt = self
                            .load_cache_page_mut(new_mini_ptr)
                            .insert(key, value, op_type, 0);
                        drop(mini_page_guard);

                        storage.finish_dealloc_mini_page(h);

                        counter!(InsertMiniPageUpgraded);
                        assert!(rt);
                        Ok(())
                    }
                    None => {
                        info!(
                            to_insert = kv_size,
                            pid = self.pid.raw(),
                            "mini page merge because too large"
                        );

                        // In cache-only mode, when a mini-page grows out of the leaf page size it is split into two mini-pages directly.
                        // To avoid creating empty mini-pages during consolidation after split, we do consolidation first to see if we could
                        // avoid page splitting.
                        if *cache_only {
                            let cur_mini_page = self.load_cache_page_mut(ptr);

                            // Only split when the current mini-page has reached the max leaf page size
                            assert!(
                                cur_mini_page.meta.node_size as usize
                                    == storage.config.leaf_page_size
                            );

                            // Consolidate the existing node first while skipping the record with the same key as k, if exists
                            cur_mini_page.consolidate_skip_key(key);

                            // Insert the k/v pair
                            let insert_success = cur_mini_page.insert(key, value, op_type, 0);

                            if insert_success {
                                counter!(InsertToMiniPageSuccess);
                                return Ok(());
                            }

                            // This is guaranteed to be true because:
                            // Assume it is not true, then there is only one key in the current mini-page.
                            // Since the required new size is greater than the leaf page size, the current leaf
                            // page size must be greater than or equal to half of the leaf page size as the max record size
                            // is less than half of the leaf page size. However, if that's true then the above insert must
                            // have succeeded which is a contradiction.
                            assert!(cur_mini_page.meta.meta_count_without_fence() > 0);

                            // Upon reaching here, it is guaranteed that all records are INSERT and there
                            // are at two records with distinctive keys (including the new k/v pair) s.t. upon split and consolidation,
                            // there is at least one record per page.
                            let record_size = (key.len() + value.len()) as u16;
                            let insert_split_key =
                                cur_mini_page.get_cache_only_insert_split_key(key, &record_size);

                            // Obtain version lock on self's parent
                            let self_parent = parent.expect("Non-root leaf node has no parent !");
                            self_parent.check_version()?;
                            let mut x_parent = self_parent.upgrade().map_err(|(_, e)| e)?;

                            if x_parent.as_ref().have_space_for(&insert_split_key) {
                                // Create a new mini-page of the same size as the current root node
                                let mini_page_guard =
                                    storage.alloc_mini_page(storage.config.leaf_page_size)?;
                                LeafNode::initialize_mini_page(
                                    &mini_page_guard,
                                    storage.config.leaf_page_size,
                                    MiniPageNextLevel::new_null(),
                                    *cache_only,
                                );
                                let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                                let mini_loc = PageLocation::Mini(new_mini_ptr);

                                // Insert the new page into mapping table
                                let (sibling_id, _mini_lock) =
                                    storage.mapping_table().insert_mini_page_mapping(mini_loc);

                                // Split current page using the splitting key
                                let cur_page_loc = self.get_page_location().clone();
                                match cur_page_loc {
                                    PageLocation::Mini(_) => {
                                        let sibling_page = unsafe { &mut *new_mini_ptr };
                                        cur_mini_page.split_with_key(
                                            sibling_page,
                                            &insert_split_key,
                                            true,
                                        );
                                        x_parent.as_mut().insert(&insert_split_key, sibling_id);
                                        {
                                            // We are safe to drop parent here because both sibling nodes are already locked.
                                            // Holding parent lock slows down other operations.
                                            drop(x_parent);
                                        }

                                        // Directly insert the new record into its correponding mini-page
                                        let cmp = key.cmp(&insert_split_key);
                                        match cmp {
                                            std::cmp::Ordering::Greater
                                            | std::cmp::Ordering::Equal => {
                                                let ok =
                                                    sibling_page.insert(key, value, op_type, 0);
                                                assert!(ok);
                                            }
                                            std::cmp::Ordering::Less => {
                                                let ok =
                                                    cur_mini_page.insert(key, value, op_type, 0);
                                                assert!(ok);
                                            }
                                        }

                                        debug_assert!(
                                            cur_mini_page.meta.meta_count_with_fence() > 0
                                        );

                                        debug_assert!(
                                            sibling_page.meta.meta_count_with_fence() > 0
                                        );
                                        return Ok(());
                                    }
                                    _ => {
                                        panic!("A non mini-page is found in cache-only mode")
                                    }
                                }
                            } else {
                                x_parent.as_mut().meta.set_split_flag();
                                counter!(MergeFailedDueToParentFull); // [TODO] a different counter
                                return Err(TreeError::NeedRestart);
                            }
                        } else {
                            // we are already too large, we need to do whole page cache.
                            // Whole page cache is different from mini page because it is a gap cache, rather than record cache.
                            // it caches the entire gap.
                            let base_offset = mini_page.next_level;
                            self.merge_mini_page_and_dealloc(
                                mini_page,
                                storage,
                                parent.expect("parent must exists here"),
                            )?;

                            info!(pid = self.pid.raw(), "old mini page deallocated");
                            if *write_load_full_page {
                                // now we merged the mini page, we need to create a full page cache because it seems that this page is very hot.
                                let base_page_ref = self.load_base_page(base_offset.as_offset());

                                // Avoid bringing back empty full page
                                if base_page_ref.meta.meta_count_without_fence() != 0 {
                                    let full_page_loc =
                                        upgrade_to_full_page(storage, base_page_ref, base_offset)?;

                                    self.create_cache_page_loc(full_page_loc);

                                    info!(
                                        pid = self.pid.raw(),
                                        "created a whole page cache mini page"
                                    );

                                    counter!(UpgradeToFullPage);
                                }
                            }
                        }
                        Err(TreeError::NeedRestart)
                    }
                }
            }
            PageLocation::Null => panic!("mini_page_op insert into Null Page"),
        }
    }

    pub(crate) fn load_cache_page_mut<'b>(&self, ptr: *mut LeafNode) -> &'b mut LeafNode {
        unsafe { &mut *ptr }
    }

    pub(crate) fn get_split_flag(&mut self) -> bool {
        let page_loc = self.raw_guard.deref();
        match page_loc {
            PageLocation::Base(offset) => {
                let base_ref = self.load_base_page(*offset);
                base_ref.get_split_flag()
            }
            PageLocation::Full(ptr) | PageLocation::Mini(ptr) => {
                let base_ref = self.load_cache_page_mut(*ptr);
                base_ref.get_split_flag()
            }
            PageLocation::Null => false, // This happens in the rare case in cache-only mode where the leaf node to insert a page in is evicted.
        }
    }

    /// Calling this function will set base page to be dirty
    pub(crate) fn load_base_page_mut(&mut self) -> &mut LeafNode {
        let page_loc = self.raw_guard.deref().clone();

        match self.tmp_buffer {
            Some(ref mut buffer) => return buffer.as_leaf_node_mut(),
            None => {
                let offset = match page_loc {
                    PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                        let page = self.load_cache_page_mut(ptr);
                        page.next_level.as_offset()
                    }
                    PageLocation::Base(offset) => offset,
                    PageLocation::Null => panic!("load_base_page_mut on Null page"),
                };

                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                let slice = buffer.as_u8_slice_mut();
                self.file_handle.read(offset, slice);

                self.tmp_buffer = Some(buffer);

                if let Some(ref mut b) = self.tmp_buffer {
                    return b.as_leaf_node_mut();
                }
            }
        };
        unreachable!()
    }

    pub(crate) fn merge_mini_page_and_dealloc(
        &mut self,
        mini_page: &mut LeafNode,
        storage: &LeafStorage,
        parent: ReadGuard<'_>,
    ) -> Result<(), TreeError> {
        let h = storage.begin_dealloc_mini_page(mini_page)?;
        self.try_merge_mini_page(&h, parent, storage)?;
        self.change_to_base_loc();
        storage.finish_dealloc_mini_page(h);
        Ok(())
    }

    pub(crate) fn merge_full_page_and_dealloc(
        &mut self,
        mini_page: &mut LeafNode,
        storage: &LeafStorage,
    ) -> Result<(), TreeError> {
        let h = storage.begin_dealloc_mini_page(mini_page)?;
        self.merge_full_page(&h);
        storage.finish_dealloc_mini_page(h);
        Ok(())
    }

    /// TODO: we want to shrink the cache page size if possible.
    pub(crate) fn move_cache_page_to_tail(
        &mut self,
        storage: &LeafStorage,
    ) -> Result<(), TreeError> {
        let page_loc = self.raw_guard.deref();
        let new_loc = match page_loc {
            PageLocation::Mini(ptr) => {
                let mini_page = self.load_cache_page_mut(*ptr);
                let h: TombstoneHandle = storage.begin_dealloc_mini_page(mini_page)?;
                let new_page =
                    storage.move_mini_page_to_tail(h, mini_page.meta.node_size as usize)?;

                counter!(MoveMiniPageToTail);
                PageLocation::Mini(new_page.as_ptr() as *mut LeafNode)
            }
            PageLocation::Full(ptr) => {
                let mini_page = self.load_cache_page_mut(*ptr);
                assert!(mini_page.meta.node_size as usize == storage.config.leaf_page_size);
                let h = storage.begin_dealloc_mini_page(mini_page)?;
                let new_page =
                    storage.move_full_page_to_tail(h, mini_page.meta.node_size as usize)?;

                counter!(MoveFullPageToTail);
                PageLocation::Full(new_page.as_ptr() as *mut LeafNode)
            }
            PageLocation::Base(_) => unreachable!(),
            PageLocation::Null => panic!("move_cache_page_to_tail on Null page"),
        };
        *self.raw_guard.deref_mut() = new_loc;
        Ok(())
    }

    /// Flush a full page into its corresponding base page
    pub(crate) fn merge_full_page(&mut self, mini_page_handle: &TombstoneHandle) {
        let mini_page = self.load_cache_page_mut(mini_page_handle.as_ptr() as *mut LeafNode);
        assert!(mini_page.meta.node_size as usize == self.tmp_buffer_size);

        if !mini_page.need_actually_merge_to_disk() {
            self.change_to_base_loc();
            return;
        }

        mini_page.convert_cache_records_to_insert();

        self.change_to_base_loc();
        let buffer = self.tmp_buffer.take();
        match buffer {
            Some(mut b) => {
                assert!(b.is_dirty);

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mini_page_handle.as_ptr(),
                        b.ptr,
                        mini_page.meta.node_size as usize,
                    );
                }
                let base = b.as_leaf_node_mut();
                base.next_level = MiniPageNextLevel::new_null();

                self.tmp_buffer = Some(b);
            }
            None => {
                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mini_page_handle.as_ptr(),
                        buffer.ptr,
                        mini_page.meta.node_size as usize,
                    );
                }

                let base = buffer.as_leaf_node_mut();
                base.next_level = MiniPageNextLevel::new_null();
                self.tmp_buffer = Some(buffer);
            }
        }
    }

    /// On success, returns the old mini page pointer.
    /// It's caller's responsibility to free the old mini page.
    /// It's caller's responsibility to acquire tombstone handle.
    /// It's caller's responsibility to update the mapping table to point to the new mini page, if any.
    pub(crate) fn try_merge_mini_page(
        &mut self,
        mini_page_handle: &TombstoneHandle,
        parent: ReadGuard,
        storage: &LeafStorage,
    ) -> Result<MergeResult, TreeError> {
        parent.check_version()?;
        let mini_page = self.load_cache_page_mut(mini_page_handle.as_ptr() as *mut LeafNode);

        if !mini_page.need_actually_merge_to_disk() {
            return Ok(MergeResult::NoSplit);
        }

        let base_ref = self.load_base_page_mut();

        // If base page has only one record, consolidate it first
        if base_ref.meta.meta_count_without_fence() == 1 {
            base_ref.consolidate_inner(OpType::Insert, None, true, false, None);
        }

        if base_ref.merge_mini_page(mini_page, storage.config.max_fence_len) {
            return Ok(MergeResult::NoSplit);
        }

        info!("mini page merge causing base page to split");
        counter!(MergeTriggerSplit);

        let mut x_parent = parent.upgrade().map_err(|(_, e)| e)?;

        // If there is only one distinctive key, then the merge should have succeeded before.
        // Choose a splitting key based on records in both mini-page and the correponding base page
        let merge_split_key = base_ref.get_merge_split_key(mini_page);

        if x_parent.as_ref().have_space_for(&merge_split_key) {
            let (sibling_node_id, mut sibling_node) = storage.alloc_base_page_and_lock();
            let sibling_node_ref = sibling_node.load_base_page_mut();
            base_ref.split_with_key(sibling_node_ref, &merge_split_key, false);
            x_parent.as_mut().insert(&merge_split_key, sibling_node_id);
            {
                // We are safe to drop parent here because both sibling nodes are already locked.
                // Holding parent lock slows down other operations.
                drop(x_parent);
            }

            // TODO: we have a design problem here:
            // what if the splitted pages are still too large that it need to be split again to absorb the mini page?
            // Right now, we keep splitting the base page in multiple iterations until the new record fits.
            // However, this could incur unnecessary mini page merges compared to doing the multiple page splitting in one run
            for kv_meta in mini_page.meta_iter() {
                let op_type = kv_meta.op_type();
                if !op_type.is_dirty() {
                    continue;
                }

                let key_to_merge = mini_page.get_full_key(kv_meta);
                let value = mini_page.get_value(kv_meta);

                let cmp = key_to_merge.cmp(&merge_split_key);
                match cmp {
                    std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => {
                        let ok = sibling_node_ref.insert(
                            &key_to_merge,
                            value,
                            op_type,
                            storage.config.max_fence_len,
                        );

                        if !ok {
                            let mini_record_num = mini_page.meta.meta_count_without_fence();
                            let base_record_num = base_ref.meta.meta_count_without_fence();
                            let sibling_record_num =
                                sibling_node_ref.meta.meta_count_without_fence();

                            panic!(
                                "{}, {}, {}",
                                mini_record_num, base_record_num, sibling_record_num
                            ); // Debug
                        }

                        assert!(ok);
                    }
                    std::cmp::Ordering::Less => {
                        let ok = base_ref.insert(
                            &key_to_merge,
                            value,
                            op_type,
                            storage.config.max_fence_len,
                        );
                        assert!(ok);
                    }
                }
            }

            Ok(MergeResult::MergeAndSplit)
        } else {
            x_parent.as_mut().meta.set_split_flag();
            counter!(MergeFailedDueToParentFull);
            Err(TreeError::NeedRestart)
        }
    }

    pub(crate) fn get_stats(&mut self) -> LeafStats {
        let page_loc = self.raw_guard.deref();
        match page_loc {
            PageLocation::Mini(ptr) => {
                let mini_page = self.load_cache_page_mut(*ptr);
                let mut mini_stats = mini_page.get_stats();
                let next_level = mini_page.next_level;

                let base_ref = self.load_base_page(next_level.as_offset());
                let stats = base_ref.get_stats();
                mini_stats.base_node = Some(Box::new(stats));
                mini_stats
            }
            PageLocation::Full(ptr) => {
                let base_ref = self.load_cache_page_mut(*ptr);
                base_ref.get_stats()
            }
            PageLocation::Base(offset) => {
                let mut buffer = TmpBuffer::new(self.tmp_buffer_size);
                let slice = buffer.as_u8_slice_mut();
                self.file_handle.read(*offset, slice);
                let base_ref = buffer.as_leaf_node();
                base_ref.get_stats()
            }
            PageLocation::Null => panic!("get_stats on Null page"),
        }
    }

    pub(crate) fn create_cache_page_loc(&mut self, cache_page_loc: PageLocation) {
        // (1) we don't need to manually flush buffer here, it will auto evict when the lock drops.
        // (2) It's possible that we are already mini page, due to mini page grow.
        let _old_loc = std::mem::replace(self.raw_guard.deref_mut(), cache_page_loc);
    }

    pub(crate) fn change_to_base_loc(&mut self) {
        let old_loc = self.raw_guard.deref().clone();
        match old_loc {
            PageLocation::Base(_) => {
                panic!("the page is already base page!");
            }
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let mini_page = self.load_cache_page_mut(ptr);
                let offset = mini_page.next_level.as_offset();
                let base_loc = PageLocation::Base(offset);

                let _old_loc = std::mem::replace(self.raw_guard.deref_mut(), base_loc);
                // we don't need to manually flush buffer here, it will auto evict when the lock drops.
            }
            PageLocation::Null => panic!("change_to_base_loc on Null page"),
        }
    }

    // Change the mapping entry to null
    pub(crate) fn change_to_null_loc(&mut self) {
        let _old_loc = std::mem::replace(self.raw_guard.deref_mut(), PageLocation::Null);
    }

    pub(crate) fn get_disk_offset(&self) -> u64 {
        let page_loc = self.raw_guard.deref();
        match page_loc {
            PageLocation::Base(offset) => *offset as u64,
            PageLocation::Mini(ptr) | PageLocation::Full(ptr) => {
                let mini_page = self.load_cache_page(*ptr);
                mini_page.next_level.as_offset() as u64
            }
            PageLocation::Null => panic!("get_disk_offset on Null page"),
        }
    }

    pub(crate) fn update_lsn(&mut self, lsn: u64) {
        let page_loc = self.raw_guard.deref();
        match page_loc {
            PageLocation::Base(_offset) => {
                let base_ref = self.load_base_page_mut();
                base_ref.lsn = lsn;
            }
            PageLocation::Full(ptr) | PageLocation::Mini(ptr) => {
                let page_ref = self.load_cache_page_mut(*ptr);
                page_ref.lsn = lsn;
            }
            PageLocation::Null => panic!("update_lsn on Null page"),
        }
    }
}

pub(crate) enum MergeResult {
    NoSplit,
    MergeAndSplit,
}

pub(crate) fn upgrade_to_full_page(
    storage: &LeafStorage,
    base_page: &LeafNode,
    base_page_offset: MiniPageNextLevel,
) -> Result<PageLocation, TreeError> {
    let full_page = storage.alloc_mini_page(storage.config.leaf_page_size)?;

    assert_eq!(
        base_page.meta.node_size as usize,
        storage.config.leaf_page_size
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            base_page as *const LeafNode as *const u8,
            full_page.as_ptr(),
            storage.config.leaf_page_size,
        );
    }
    let full_page_ptr = full_page.as_ptr() as *mut LeafNode;
    let full_page_ref = unsafe { &mut *full_page_ptr };
    full_page_ref.covert_insert_records_to_cache();
    full_page_ref.next_level = base_page_offset;

    Ok(PageLocation::Full(full_page.as_ptr() as *mut LeafNode))
}
