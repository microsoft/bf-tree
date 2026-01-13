// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::{storage::DiskOffsetGuard, sync::atomic::AtomicU16};
use std::cmp::Ordering;

use super::{node_meta::NodeMeta, PageID, INNER_NODE_SIZE};
use crate::utils::stats::InnerStats;

const INVALID_DISK_OFFSET: usize = usize::MAX;

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
pub(crate) struct InnerNode {
    pub(crate) meta: NodeMeta,
    pub(crate) version_lock: AtomicU16, // Is 16 bits enough?
    pub(crate) disk_offset: u64,
    data: [u8; 0],
}

#[cfg(not(feature = "shuttle"))]
const _: () = assert!(std::mem::size_of::<InnerNode>() == 16);

struct InnerPtrGuard {
    ptr: *mut InnerNode,
}

impl InnerPtrGuard {
    fn make() -> Self {
        let layout = std::alloc::Layout::from_size_align(INNER_NODE_SIZE, INNER_NODE_SIZE).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut InnerNode;
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

    pub(crate) fn build(self) -> *mut InnerNode {
        let node = unsafe { &mut *self.raw_ptr.ptr };
        let offset = match self.disk_offset {
            Some(x) => x.take(),
            None => INVALID_DISK_OFFSET,
        };

        unsafe {
            std::ptr::write(&mut node.version_lock, AtomicU16::new(0));
        }
        node.reinitialize(
            self.left_most_page_id.unwrap(),
            self.children_is_leaf.unwrap(),
            offset as u64,
        );
        for (key, value) in self.records {
            let rt = node.insert(&key, value);
            assert!(rt);
        }

        self.raw_ptr.take()
    }

    pub(crate) fn build_from_slice(self, slice: &[u8]) -> *mut InnerNode {
        let ptr = self.raw_ptr.take();
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr as *mut u8, INNER_NODE_SIZE);
        }
        ptr
    }
}

impl InnerNode {
    pub(crate) fn free_node(ptr: *mut InnerNode) {
        let layout = std::alloc::Layout::from_size_align(INNER_NODE_SIZE, INNER_NODE_SIZE).unwrap();
        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) }
    }

    /// Initialize the node with the left most page id and whether the children are leaf nodes.
    ///
    /// Note that the self is not initialized, we use ptr::write to avoid calling drop on the old values.
    /// TODO: maybe change the interface to InnerNode::initialize(ptr: *mut InnerNode, ...)?
    fn reinitialize(
        &mut self,
        left_most_page_id: PageID,
        children_is_leaf: bool,
        disk_offset: u64,
    ) {
        unsafe {
            std::ptr::write(
                &mut self.meta,
                NodeMeta::new(
                    InnerNode::max_data_size() as u16,
                    children_is_leaf,
                    false,
                    INNER_NODE_SIZE as u16,
                    false,
                ),
            );
        }
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
        INNER_NODE_SIZE - std::mem::size_of::<InnerNode>()
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
                && (key_meta.key_len > InnerKVMeta::KEY_LOOK_AHEAD_SIZE as u16)
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
        let post_key_len = std::cmp::max(key.len(), InnerKVMeta::KEY_LOOK_AHEAD_SIZE)
            - InnerKVMeta::KEY_LOOK_AHEAD_SIZE;
        let kv_len = post_key_len + std::mem::size_of::<PageID>();
        let required_len = kv_len + std::mem::size_of::<InnerKVMeta>();

        let remaining = self.meta.remaining_size;
        if remaining < required_len as u16 {
            return false;
        }

        let value_count = self.meta.meta_count_with_fence();

        let offset = self.current_lowest_offset() - kv_len as u16;
        let new_meta = InnerKVMeta::make_prefixed_meta(key, offset);

        let pos = self.lower_bound(key);
        if pos > 0 && pos < value_count as u64 {
            let kv_meta = self.get_kv_meta(pos as u16);
            let cmp = self.key_compare(key, kv_meta);
            if cmp == Ordering::Equal {
                unsafe {
                    let start_ptr = self
                        .data
                        .as_ptr()
                        .add(kv_meta.offset as usize + post_key_len)
                        as *mut PageID;
                    std::ptr::write_unaligned(start_ptr, child);
                }
                return true;
            }
        }

        let pos = self.lower_bound(key) as usize + 1; // the pos returned is the key position, different from the value position (remember that inner node has one less key than value).

        let metas_size = std::mem::size_of::<InnerKVMeta>() * (value_count as usize - pos);

        unsafe {
            std::ptr::copy(
                self.data
                    .as_mut_ptr()
                    .add(pos * std::mem::size_of::<InnerKVMeta>()),
                self.data
                    .as_mut_ptr()
                    .add((pos + 1) * std::mem::size_of::<InnerKVMeta>()),
                metas_size,
            );

            let pair_ptr = self.data.as_mut_ptr().add(new_meta.offset as usize);

            *(self
                .data
                .as_mut_ptr()
                .add(pos * std::mem::size_of::<InnerKVMeta>()) as *mut InnerKVMeta) = new_meta;

            std::ptr::copy_nonoverlapping(
                key.as_ptr().add(InnerKVMeta::KEY_LOOK_AHEAD_SIZE),
                pair_ptr,
                post_key_len,
            );
            std::ptr::write_unaligned(pair_ptr.add(post_key_len) as *mut PageID, child);
        }

        self.meta.remaining_size -= required_len as u16;
        self.meta.increment_value_count();
        true
    }

    pub(crate) fn consolidate(&mut self) {
        let mut pairs: Vec<(Vec<u8>, PageID)> = Vec::new();
        for i in 1..self.meta.meta_count_with_fence() {
            let meta = self.get_kv_meta(i);
            pairs.push((self.get_full_key(meta), self.get_value(meta)));
        }

        let left_most_page_id = self.get_value(self.get_kv_meta(0));
        let children_is_leaf = self.meta.children_is_leaf();
        self.reinitialize(left_most_page_id, children_is_leaf, self.disk_offset);

        for (key, id) in pairs {
            let tmp_k = key;
            self.insert(&tmp_k, id);
        }
    }

    pub(crate) fn get_split_key(&self) -> Vec<u8> {
        let pos = self.meta.meta_count_with_fence() - self.meta.meta_count_with_fence() / 2;
        let split_meta = self.get_kv_meta(pos);
        self.get_full_key(split_meta)
    }

    pub(crate) fn split(&mut self, new_node: &mut InnerNodeBuilder) -> Vec<u8> {
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
        self.consolidate();

        split_key
    }

    pub(crate) fn have_space_for(&self, key: &[u8]) -> bool {
        let post_key_len = std::cmp::max(key.len(), InnerKVMeta::KEY_LOOK_AHEAD_SIZE)
            - InnerKVMeta::KEY_LOOK_AHEAD_SIZE;
        let kv_len = post_key_len + std::mem::size_of::<PageID>();
        let required_len = kv_len + std::mem::size_of::<InnerKVMeta>();

        let remaining = self.meta.remaining_size;
        remaining >= required_len as u16
    }

    pub(crate) fn update_at_pos(&mut self, pos: usize, new_id: PageID) {
        let kv_meta = self.get_kv_meta(pos as u16);
        let post_key_len =
            std::cmp::max(kv_meta.key_len as usize, InnerKVMeta::KEY_LOOK_AHEAD_SIZE)
                - InnerKVMeta::KEY_LOOK_AHEAD_SIZE;

        unsafe {
            let start_ptr =
                self.data
                    .as_ptr()
                    .add(kv_meta.offset as usize + post_key_len) as *mut PageID;
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
