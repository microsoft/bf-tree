// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::{storage::DiskOffsetGuard, sync::atomic::AtomicU64};
use std::cmp::Ordering;

use super::{node_meta::NodeMeta, PageID, INNER_NODE_SIZE};
use crate::utils::stats::InnerStats;

pub(crate) const INVALID_DISK_OFFSET: usize = usize::MAX;
const INNER_NODE_ROOT_FLAG: u64 = 0x8000_0000_0000_0000;
const INNER_NODE_SNAPSHOT_VERSION_MASK: u64 = 0x7FFF_FFFF_FFFF_FFFF;

#[repr(C)]
pub(crate) struct InnerKVMeta {
    pub offset: u16,
    pub key_len: u16, // internal node has fixed value length of 8.
    pub key_prefix: [u8; InnerKVMeta::KEY_LOOK_AHEAD_SIZE],
}

const _: () = assert!(std::mem::size_of::<InnerKVMeta>() == 8);

impl InnerKVMeta {
    pub const KEY_LOOK_AHEAD_SIZE: usize = 4;

    pub(crate) fn make_prefixed_meta(key: &[u8], offset: u16) -> Self {
        let mut meta = Self {
            offset,
            key_len: key.len() as u16,
            key_prefix: [0; Self::KEY_LOOK_AHEAD_SIZE],
        };

        let min_len = std::cmp::min(key.len(), Self::KEY_LOOK_AHEAD_SIZE);
        meta.key_prefix[0..min_len].copy_from_slice(&key[0..min_len]);

        meta
    }
}

#[derive(Debug)]
#[repr(C)]
pub(crate) struct InnerNodeStorage<const DATA_SIZE: usize> {
    pub(crate) meta: NodeMeta,
    pub(crate) version_lock: AtomicU64,
    pub(crate) disk_offset: u64,
    snapshot_version: u64,
    data: [u8; DATA_SIZE],
}

const INNER_NODE_HEADER_SIZE: usize = std::mem::size_of::<InnerNodeStorage<0>>();
const INNER_NODE_DATA_SIZE: usize = INNER_NODE_SIZE - INNER_NODE_HEADER_SIZE;

// Give references to an inner node the bounds of the entire allocation. A
// zero-length trailing array cannot lend valid references to its payload.
pub(crate) type InnerNode = InnerNodeStorage<INNER_NODE_DATA_SIZE>;

const _: () = assert!(std::mem::size_of::<InnerNode>() == INNER_NODE_SIZE);
const _: () = assert!(std::mem::offset_of!(InnerNode, data) == INNER_NODE_HEADER_SIZE);

struct InnerPtrGuard {
    ptr: *mut InnerNode,
}

impl InnerPtrGuard {
    fn make() -> Self {
        let layout = std::alloc::Layout::from_size_align(INNER_NODE_SIZE, INNER_NODE_SIZE).unwrap();
        // The entire page is serialized, including its unused space and header padding.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut InnerNode;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // All other fields have valid all-zero representations. Initialize the atomic
        // through a raw pointer before forming a reference, including for Shuttle.
        unsafe {
            std::ptr::addr_of_mut!((*ptr).version_lock).write(AtomicU64::new(0));
        }
        Self { ptr }
    }

    fn take(self) -> *mut InnerNode {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for InnerPtrGuard {
    fn drop(&mut self) {
        // When the builder is dropped, i.e., it is not eventually used to build a node,
        // we should free the memory allocated for the node.
        InnerNode::free_node(self.ptr);
    }
}

pub(crate) struct InnerNodeBuilder<'a> {
    left_most_page_id: Option<PageID>,
    children_is_leaf: Option<bool>,
    disk_offset: Option<DiskOffsetGuard<'a>>,
    records: Vec<(Vec<u8>, PageID)>,
    raw_ptr: InnerPtrGuard,
}

impl<'a> InnerNodeBuilder<'a> {
    pub(crate) fn new() -> Self {
        Self {
            left_most_page_id: None,
            children_is_leaf: None,
            disk_offset: None,
            records: Vec::with_capacity(64),
            raw_ptr: InnerPtrGuard::make(),
        }
    }

    pub(crate) fn set_left_most_page_id(&mut self, page_id: PageID) -> &mut Self {
        self.left_most_page_id = Some(page_id);
        self
    }

    pub(crate) fn set_children_is_leaf(&mut self, is_leaf: bool) -> &mut Self {
        self.children_is_leaf = Some(is_leaf);
        self
    }

    pub(crate) fn set_disk_offset(&mut self, offset: DiskOffsetGuard<'a>) -> &mut Self {
        self.disk_offset = Some(offset);
        self
    }

    pub(crate) fn add_record(&mut self, key: Vec<u8>, value: PageID) -> &mut Self {
        self.records.push((key, value));
        self
    }

    pub(crate) fn get_page_id(&self) -> PageID {
        PageID::from_pointer(self.raw_ptr.ptr)
    }

    pub(crate) fn build(self, snapshot_version: u64) -> *mut InnerNode {
        let node = unsafe { &mut *self.raw_ptr.ptr };
        let offset = match self.disk_offset {
            Some(x) => x.take(),
            None => INVALID_DISK_OFFSET,
        };

        node.reinitialize(
            self.left_most_page_id.unwrap(),
            self.children_is_leaf.unwrap(),
            offset as u64,
        );
        for (key, value) in self.records {
            let rt = node.insert(&key, value);
            assert!(rt);
        }

        // Note that, when snapshot is not enabled the version of a leaf node would be the INVALID_SNAPSHOT_VERSION
        node.set_snapshot_version(snapshot_version);

        self.raw_ptr.take()
    }

    pub(crate) fn build_from_slice(self, slice: &[u8]) -> *mut InnerNode {
        assert!(
            slice.len() >= INNER_NODE_SIZE,
            "inner node image is too short"
        );
        let ptr = self.raw_ptr.take();
        // The lock belongs to this allocation, not to the serialized image. In
        // particular, Shuttle's atomic contains owned state that must not be copied.
        let lock_offset = std::mem::offset_of!(InnerNode, version_lock);
        let lock_end = lock_offset + std::mem::size_of::<AtomicU64>();
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr.cast::<u8>(), lock_offset);
            std::ptr::copy_nonoverlapping(
                slice.as_ptr().add(lock_end),
                ptr.cast::<u8>().add(lock_end),
                INNER_NODE_SIZE - lock_end,
            );
        }
        ptr
    }
}

impl InnerNode {
    pub(crate) fn free_node(ptr: *mut InnerNode) {
        let layout = std::alloc::Layout::from_size_align(INNER_NODE_SIZE, INNER_NODE_SIZE).unwrap();
        unsafe {
            // A standard atomic has no destructor; Shuttle's atomic owns its clock.
            std::ptr::drop_in_place(std::ptr::addr_of_mut!((*ptr).version_lock));
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }

    pub(crate) fn set_disk_offset(&mut self, offset: u64) {
        self.disk_offset = offset;
    }

    /// Initialize the node with the left most page id and whether the children are leaf nodes.
    fn reinitialize(
        &mut self,
        left_most_page_id: PageID,
        children_is_leaf: bool,
        disk_offset: u64,
    ) {
        self.meta = NodeMeta::new(
            InnerNode::max_data_size() as u16,
            children_is_leaf,
            false,
            INNER_NODE_SIZE as u16,
            false,
        );
        self.disk_offset = disk_offset;

        let offset = self.current_lowest_offset() - std::mem::size_of::<PageID>() as u16;
        let new_meta = InnerKVMeta {
            offset,
            key_len: 0,
            key_prefix: [0; InnerKVMeta::KEY_LOOK_AHEAD_SIZE],
        };

        let pos = 0;
        unsafe {
            let ptr = self.data.as_mut_ptr();
            *(ptr.add(pos * std::mem::size_of::<InnerKVMeta>()) as *mut InnerKVMeta) = new_meta;

            let pair_ptr = ptr.add(offset as usize);
            std::ptr::write_unaligned(pair_ptr as *mut PageID, left_most_page_id);
        }

        self.meta.remaining_size -=
            (std::mem::size_of::<PageID>() + std::mem::size_of::<InnerKVMeta>()) as u16;
        self.meta.increment_value_count();
    }

    pub(crate) fn max_data_size() -> usize {
        INNER_NODE_DATA_SIZE
    }

    pub(crate) fn get_kv_meta(&self, index: u16) -> &InnerKVMeta {
        let ptr = self.data.as_ptr();
        unsafe {
            &*(ptr.add((index as usize) * std::mem::size_of::<InnerKVMeta>()) as *const InnerKVMeta)
        }
    }

    pub(crate) fn current_lowest_offset(&self) -> u16 {
        let value_count = self.meta.meta_count_with_fence();
        let rt =
            (value_count * std::mem::size_of::<InnerKVMeta>() as u16) + self.meta.remaining_size;

        // Sanity check
        #[cfg(debug_assertions)]
        {
            let mut min_offset = InnerNode::max_data_size() as u16;
            for i in 0..value_count {
                let kv_meta = self.get_kv_meta(i);
                min_offset = std::cmp::min(min_offset, kv_meta.offset);
            }
            assert!(min_offset == rt);
        }
        rt
    }

    pub(crate) fn get_clean_snapshot_version(&self) -> u64 {
        self.snapshot_version & INNER_NODE_SNAPSHOT_VERSION_MASK
    }

    pub(crate) fn set_snapshot_version(&mut self, snapshot_version: u64) {
        assert_eq!(snapshot_version & INNER_NODE_ROOT_FLAG, 0);
        let root_flag = self.snapshot_version & INNER_NODE_ROOT_FLAG;
        self.snapshot_version = (snapshot_version & INNER_NODE_SNAPSHOT_VERSION_MASK) | root_flag;
    }

    pub(crate) fn is_root(&self) -> bool {
        (self.snapshot_version & INNER_NODE_ROOT_FLAG) != 0
    }

    pub(crate) fn set_root(&mut self, is_root: bool) {
        if is_root {
            self.snapshot_version |= INNER_NODE_ROOT_FLAG;
        } else {
            self.snapshot_version &= INNER_NODE_SNAPSHOT_VERSION_MASK;
        }
    }

    pub(crate) fn get_full_key(&self, meta: &InnerKVMeta) -> Vec<u8> {
        let post_key_len = (meta.key_len as usize).saturating_sub(InnerKVMeta::KEY_LOOK_AHEAD_SIZE);
        let post_key_span = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr().add(meta.offset as usize), post_key_len)
        };
        let prefix_span = &meta.key_prefix
            [0..std::cmp::min(InnerKVMeta::KEY_LOOK_AHEAD_SIZE, meta.key_len as usize)];

        let mut key = Vec::with_capacity(meta.key_len as usize);
        key.extend_from_slice(prefix_span);
        key.extend_from_slice(post_key_span);
        key
    }

    pub(crate) fn get_post_key_ref(&self, meta: &InnerKVMeta) -> &[u8] {
        let len = (meta.key_len as usize).saturating_sub(InnerKVMeta::KEY_LOOK_AHEAD_SIZE);
        unsafe {
            let start_ptr = self.data.as_ptr().add(meta.offset as usize);
            std::slice::from_raw_parts(start_ptr, len)
        }
    }

    /// We can not have a reference to the PageID here,
    /// because the &PageID is unaligned and it is ub to read from it.
    /// It creates quite a lot of issues, so we have to return a copy of the PageID.
    pub(crate) fn get_value(&self, meta: &InnerKVMeta) -> PageID {
        unsafe {
            let start_ptr = self.data.as_ptr().add(
                meta.offset as usize
                    + (meta.key_len as usize).saturating_sub(InnerKVMeta::KEY_LOOK_AHEAD_SIZE),
            );
            let value = std::ptr::read_unaligned(start_ptr as *const u64);
            PageID::from_raw(value)
        }
    }

    fn key_compare(&self, key: &[u8], meta: &InnerKVMeta) -> Ordering {
        let search_key_prefix = &key[0..std::cmp::min(InnerKVMeta::KEY_LOOK_AHEAD_SIZE, key.len())];
        let search_key_postfix = &key[std::cmp::min(InnerKVMeta::KEY_LOOK_AHEAD_SIZE, key.len())..];
        let prefix_key = &meta.key_prefix
            [0..std::cmp::min(InnerKVMeta::KEY_LOOK_AHEAD_SIZE, meta.key_len as usize)];

        let mut cmp = prefix_key.cmp(search_key_prefix);
        if cmp != Ordering::Equal {
            return cmp;
        }

        let rest_key = self.get_post_key_ref(meta);
        cmp = rest_key.cmp(search_key_postfix);
        cmp
    }

    pub(crate) fn lower_bound(&self, key: &[u8]) -> u64 {
        let mut lower: u16 = 1; // Note: the first key is dummy, we don't use it.
        let mut upper: u16 = self.meta.meta_count_with_fence();

        let search_key_prefix = &key[0..std::cmp::min(InnerKVMeta::KEY_LOOK_AHEAD_SIZE, key.len())];
        let search_key_postfix = &key[std::cmp::min(InnerKVMeta::KEY_LOOK_AHEAD_SIZE, key.len())..];

        while lower < upper {
            let mid = lower + (upper - lower) / 2;
            let key_meta = self.get_kv_meta(mid);

            let prefix_key = &key_meta.key_prefix
                [0..std::cmp::min(InnerKVMeta::KEY_LOOK_AHEAD_SIZE, key_meta.key_len as usize)];

            let mut cmp = prefix_key.cmp(search_key_prefix);

            // If prefix compare is the same, we need to compare the full key.
            if cmp == Ordering::Equal
                && ((key_meta.key_len > InnerKVMeta::KEY_LOOK_AHEAD_SIZE as u16)
                    || !search_key_postfix.is_empty())
            {
                let rest_key = self.get_post_key_ref(key_meta);
                cmp = rest_key.cmp(search_key_postfix);
            }

            match cmp {
                Ordering::Greater => {
                    upper = mid;
                }
                Ordering::Equal => {
                    return mid as u64;
                }
                Ordering::Less => {
                    lower = mid + 1;
                }
            }
        }
        (lower - 1) as u64
    }

    pub(crate) fn insert(&mut self, key: &[u8], child: PageID) -> bool {
        let value_count = self.meta.meta_count_with_fence();
        let pos = self.lower_bound(key);
        if pos > 0 && pos < value_count as u64 {
            let kv_meta = self.get_kv_meta(pos as u16);
            if self.key_compare(key, kv_meta) == Ordering::Equal {
                // Replacing a child does not consume space, even when the node is full.
                self.update_at_pos(pos as usize, child);
                return true;
            }
        }

        let post_key_len = key.len().saturating_sub(InnerKVMeta::KEY_LOOK_AHEAD_SIZE);
        let kv_len = post_key_len + std::mem::size_of::<PageID>();
        let required_len = kv_len + std::mem::size_of::<InnerKVMeta>();
        // Check before narrowing sizes or performing pointer arithmetic. Oversized
        // keys must not wrap a u16 and pass the capacity check.
        if (self.meta.remaining_size as usize) < required_len {
            return false;
        }

        let offset = self.current_lowest_offset() - kv_len as u16;
        let new_meta = InnerKVMeta::make_prefixed_meta(key, offset);
        let pos = pos as usize + 1; // The search returns the predecessor's key position.

        let metas_size = std::mem::size_of::<InnerKVMeta>() * (value_count as usize - pos);

        unsafe {
            // Derive all destinations from one borrow of the data array. A later
            // as_mut_ptr() borrow would invalidate pointers already derived here.
            let data_ptr = self.data.as_mut_ptr();
            std::ptr::copy(
                data_ptr.add(pos * std::mem::size_of::<InnerKVMeta>()),
                data_ptr.add((pos + 1) * std::mem::size_of::<InnerKVMeta>()),
                metas_size,
            );

            let pair_ptr = data_ptr.add(new_meta.offset as usize);
            data_ptr.cast::<InnerKVMeta>().add(pos).write(new_meta);

            if post_key_len != 0 {
                std::ptr::copy_nonoverlapping(
                    key.as_ptr().add(InnerKVMeta::KEY_LOOK_AHEAD_SIZE),
                    pair_ptr,
                    post_key_len,
                );
            }
            std::ptr::write_unaligned(pair_ptr.add(post_key_len) as *mut PageID, child);
        }

        self.meta.remaining_size -= required_len as u16;
        self.meta.increment_value_count();
        true
    }

    pub(crate) fn consolidate(&mut self, snapshot_version: u64) {
        let value_count = self.meta.meta_count_with_fence();
        let data_size = Self::max_data_size();
        let mut packed = [0u8; INNER_NODE_SIZE];
        let mut offset = data_size;
        // Preserve the sorted prefix metadata and compact only the retained payloads.
        // Staging them first prevents writes from overwriting a later source record.
        for i in 0..value_count {
            let meta = self.get_kv_meta(i);
            let postfix = self.get_post_key_ref(meta);
            let value = self.get_value(meta).raw().to_ne_bytes();
            offset -= postfix.len() + value.len();
            let value_offset = offset + postfix.len();
            packed[offset..value_offset].copy_from_slice(postfix);
            packed[value_offset..value_offset + value.len()].copy_from_slice(&value);
            // Metadata does not overlap the existing payloads. Only this record's
            // offset changes; the original bytes remain intact until all are staged.
            unsafe {
                let meta_ptr = self.data.as_mut_ptr().cast::<InnerKVMeta>().add(i as usize);
                (*meta_ptr).offset = offset as u16;
            }
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                packed.as_ptr().add(offset),
                self.data.as_mut_ptr().add(offset),
                data_size - offset,
            );
        }
        let remaining_size = offset - value_count as usize * std::mem::size_of::<InnerKVMeta>();
        self.meta = NodeMeta::new(
            remaining_size as u16,
            self.meta.children_is_leaf(),
            false,
            INNER_NODE_SIZE as u16,
            false,
        );
        self.meta.set_value_count(value_count);

        // Note that, when snapshot is not enabled the version of a leaf node would be the INVALID_SNAPSHOT_VERSION
        self.set_snapshot_version(snapshot_version);
    }

    pub(crate) fn get_split_key(&self) -> Vec<u8> {
        let pos = self.meta.meta_count_with_fence() - self.meta.meta_count_with_fence() / 2;
        let split_meta = self.get_kv_meta(pos);
        self.get_full_key(split_meta)
    }

    pub(crate) fn split(
        &mut self,
        new_node: &mut InnerNodeBuilder,
        snapshot_version: u64,
    ) -> Vec<u8> {
        let current_count = self.meta.meta_count_with_fence();
        let sibling_node_count = current_count / 2;
        let new_node_count = current_count - sibling_node_count;

        let rt_meta = self.get_kv_meta(new_node_count);
        let split_key = self.get_full_key(rt_meta);
        let split_value = self.get_value(rt_meta);

        new_node
            .set_children_is_leaf(self.meta.children_is_leaf())
            .set_left_most_page_id(split_value);

        // Now we have to do two things:
        // Copy the second half of the key-value pairs to the new node, setting the correct offsets.
        // Consolidate the key-value pairs in the current node, setting the correct offsets.
        for i in 1..sibling_node_count {
            let kv_meta = self.get_kv_meta(new_node_count + i);
            let key = self.get_full_key(kv_meta);
            let value = self.get_value(kv_meta);

            new_node.add_record(key, value);
        }

        self.meta.set_value_count(new_node_count);
        self.consolidate(snapshot_version);

        split_key
    }

    pub(crate) fn have_space_for(&self, key: &[u8]) -> bool {
        let post_key_len = key.len().saturating_sub(InnerKVMeta::KEY_LOOK_AHEAD_SIZE);
        let kv_len = post_key_len + std::mem::size_of::<PageID>();
        let required_len = kv_len + std::mem::size_of::<InnerKVMeta>();

        self.meta.remaining_size as usize >= required_len
    }

    pub(crate) fn update_at_pos(&mut self, pos: usize, new_id: PageID) {
        let kv_meta = self.get_kv_meta(pos as u16);
        let post_key_len =
            (kv_meta.key_len as usize).saturating_sub(InnerKVMeta::KEY_LOOK_AHEAD_SIZE);
        let offset = kv_meta.offset as usize + post_key_len;

        unsafe {
            let start_ptr = self.data.as_mut_ptr().add(offset).cast::<PageID>();
            std::ptr::write_unaligned(start_ptr, new_id);
        }
    }

    /// Used when merging the delta chains.
    #[allow(dead_code)]
    pub(crate) fn update(&mut self, key: &[u8], new_id: PageID) {
        assert!(!new_id.is_inner_node_pointer()); // Assuming is_heap_pointer is a method on PageID
        let pos = self.lower_bound(key) as usize;
        self.update_at_pos(pos, new_id);
    }

    pub(crate) fn get_stats(&self) -> InnerStats {
        let keys: Vec<Vec<u8>> = KeyIter {
            node: self,
            cur_idx: 0,
        }
        .collect();
        let child_id = self.get_child_iter().collect::<Vec<PageID>>();

        InnerStats {
            child_keys: keys,
            child_id,
            child_is_leaf: self.meta.children_is_leaf(),
        }
    }

    pub(crate) fn get_child_iter(&self) -> ChildIter<'_> {
        ChildIter {
            node: self,
            cur_idx: 0,
        }
    }

    /// Returns the entire code as a u8 slice,
    /// Used when we serialize the node to disk.
    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const InnerNode as *const u8, INNER_NODE_SIZE)
        }
    }

    /// The disk offset is invalid in cache-only mode
    pub(crate) fn is_valid_disk_offset(&self) -> bool {
        if self.disk_offset != INVALID_DISK_OFFSET as u64 {
            return true;
        }
        false
    }
}

struct KeyIter<'a> {
    node: &'a InnerNode,
    cur_idx: usize,
}

impl Iterator for KeyIter<'_> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_idx >= self.node.meta.meta_count_with_fence() as usize {
            return None;
        }

        let meta = self.node.get_kv_meta(self.cur_idx as u16);
        let key = self.node.get_full_key(meta);
        self.cur_idx += 1;
        Some(key)
    }
}

pub(crate) struct ChildIter<'a> {
    node: &'a InnerNode,
    cur_idx: usize,
}

impl Iterator for ChildIter<'_> {
    type Item = PageID;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_idx >= self.node.meta.meta_count_with_fence() as usize {
            return None;
        }

        let meta = self.node.get_kv_meta(self.cur_idx as u16);
        let value = self.node.get_value(meta);
        self.cur_idx += 1;
        Some(value)
    }
}
