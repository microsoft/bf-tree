// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use core::panic;
use std::{alloc::Layout, cmp::Ordering, sync::atomic};

use crate::{
    circular_buffer::CircularBufferPtr, counter, error::TreeError, range_scan::ScanReturnField,
    utils::stats::LeafStats,
};

use super::{node_meta::NodeMeta, FENCE_KEY_CNT};

/// Invariant: leaf page can only have Insert and Delete type.
///            mini page can have all.
#[repr(u8)]
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub(crate) enum OpType {
    Insert = 0,
    Delete = 1,
    Cache = 2,   // the clean version of insert.
    Phantom = 3, // the clean version of delete.
}

impl OpType {
    pub(crate) fn is_dirty(&self) -> bool {
        match self {
            OpType::Insert | OpType::Delete => true,
            OpType::Cache | OpType::Phantom => false,
        }
    }

    fn is_absent(&self) -> bool {
        match self {
            OpType::Insert | OpType::Cache => false,
            OpType::Phantom | OpType::Delete => true,
        }
    }

    fn is_cache(&self) -> bool {
        match self {
            OpType::Cache | OpType::Phantom => true,
            OpType::Insert | OpType::Delete => false,
        }
    }
}

const PREVIEW_SIZE: usize = 2;

const LOW_FENCE_IDX: usize = 0;
const HIGH_FENCE_IDX: usize = 1;

const OP_TYPE_SHIFT: u16 = 14;
const KEY_LEN_MASK: u16 = 0x3F_FF; // lower 14 bits on the key_len

// highest bit on the value_len;
const REF_BIT_MASK: u16 = 0x80_00;
// third highest bit on the value_len;
const VALUE_LEN_MASK: u16 = 0x7F_FF; // lower 15 bits on the value_len;

pub(crate) fn common_prefix_len(low_fence: &[u8], high_fence: &[u8]) -> u16 {
    let mut prefix_len = 0;
    for i in 0..std::cmp::min(low_fence.len(), high_fence.len()) {
        if low_fence[i] == high_fence[i] {
            prefix_len += 1;
        } else {
            break;
        }
    }
    prefix_len
}

#[repr(C)]
pub(crate) struct LeafKVMeta {
    offset: u16,
    op_type_key_len_in_byte: u16, // Highest 2 bits: op_type, Lower 14 bits: Key length in bytes.
    ref_value_len_in_byte: std::sync::atomic::AtomicU16, // Highest bit: ref, Lower 15 bits: value length in bytes. here we don't use shuttle's AtomicU16, because this is not a real sync point.
    preview_bytes: [u8; PREVIEW_SIZE],
}

impl LeafKVMeta {
    pub(crate) fn make_prefixed_meta(
        offset: u16,
        value_len: u16,
        key: &[u8],
        prefix_len: u16,
        op_type: OpType,
    ) -> Self {
        let mut meta = Self {
            offset,
            op_type_key_len_in_byte: key.len() as u16 | ((op_type as u16) << OP_TYPE_SHIFT),
            ref_value_len_in_byte: std::sync::atomic::AtomicU16::new(value_len),
            preview_bytes: [0; PREVIEW_SIZE],
        };

        for i in 0..std::cmp::min(key.len().saturating_sub(prefix_len as usize), PREVIEW_SIZE) {
            meta.preview_bytes[i] = key[i + prefix_len as usize];
        }

        // The initial value is not referenced, this is important because during the Garbage reclaim of the delta chain,
        // we will call Insert to reset the states, which calls this function.
        meta.clear_ref();
        meta
    }

    pub fn make_infinite_high_fence_key() -> Self {
        assert_eq!(std::mem::size_of::<Self>(), super::KV_META_SIZE);

        Self {
            offset: u16::MAX,
            op_type_key_len_in_byte: 0,
            ref_value_len_in_byte: std::sync::atomic::AtomicU16::new(0),
            preview_bytes: [0; PREVIEW_SIZE],
        }
    }

    pub fn make_infinite_low_fence_key() -> Self {
        Self {
            offset: u16::MAX - 1,
            op_type_key_len_in_byte: 0,
            ref_value_len_in_byte: std::sync::atomic::AtomicU16::new(0),
            preview_bytes: [0; PREVIEW_SIZE],
        }
    }

    pub fn is_infinite_low_fence_key(&self) -> bool {
        self.offset == u16::MAX - 1
    }

    pub fn is_infinite_high_fence_key(&self) -> bool {
        self.offset == u16::MAX
    }

    pub fn value_len(&self) -> u16 {
        self.ref_value_len_in_byte.load(atomic::Ordering::Relaxed) & VALUE_LEN_MASK
    }

    pub fn set_value_len(&mut self, value: u16) {
        let v = self.ref_value_len_in_byte.load(atomic::Ordering::Relaxed);
        self.ref_value_len_in_byte.store(
            (v & !VALUE_LEN_MASK) | (value & VALUE_LEN_MASK),
            atomic::Ordering::Relaxed,
        );
    }

    pub fn set_op_type(&mut self, op_type: OpType) {
        self.op_type_key_len_in_byte = self.get_key_len() | ((op_type as u16) << OP_TYPE_SHIFT);
    }

    pub fn op_type(&self) -> OpType {
        let l = self.op_type_key_len_in_byte;
        let b = l >> OP_TYPE_SHIFT;
        unsafe { std::mem::transmute(b as u8) }
    }

    pub fn mark_as_ref(&self) {
        self.ref_value_len_in_byte
            .fetch_or(REF_BIT_MASK, atomic::Ordering::Relaxed);
    }

    pub fn clear_ref(&self) {
        self.ref_value_len_in_byte
            .fetch_and(!REF_BIT_MASK, atomic::Ordering::Relaxed);
    }

    pub fn is_referenced(&self) -> bool {
        self.ref_value_len_in_byte.load(atomic::Ordering::Relaxed) & REF_BIT_MASK != 0
    }

    pub fn mark_as_deleted(&mut self) {
        self.set_op_type(OpType::Delete);
    }

    #[allow(dead_code)]
    pub fn is_deleted(&self) -> bool {
        self.op_type() == OpType::Delete
    }

    pub(crate) fn get_offset(&self) -> u16 {
        self.offset
    }

    pub(crate) fn get_key_len(&self) -> u16 {
        self.op_type_key_len_in_byte & KEY_LEN_MASK
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MiniPageNextLevel {
    val: usize,
}

impl MiniPageNextLevel {
    pub(crate) fn new(val: usize) -> Self {
        Self { val }
    }

    pub(crate) fn as_offset(&self) -> usize {
        assert!(!self.is_null());
        self.val
    }

    pub(crate) fn new_null() -> Self {
        Self { val: usize::MAX }
    }

    pub(crate) fn is_null(&self) -> bool {
        self.val == usize::MAX
    }
}

#[repr(C)]
pub(crate) struct LeafNode {
    pub(crate) meta: NodeMeta,
    prefix_len: u16,
    pub(crate) next_level: MiniPageNextLevel,
    pub(crate) lsn: u64,
    data: [u8; 0],
}

const _: () = assert!(std::mem::size_of::<LeafNode>() == 24);

impl LeafNode {
    fn max_data_size(node_size: usize) -> usize {
        node_size - std::mem::size_of::<LeafNode>()
    }

    pub(crate) fn initialize_mini_page(
        ptr: &CircularBufferPtr,
        node_size: usize,
        next_level: MiniPageNextLevel,
        cache_only: bool,
    ) {
        unsafe {
            Self::init_node_with_fence(
                ptr.as_ptr(),
                &[],
                &[],
                node_size,
                next_level,
                false,
                cache_only,
            )
        }
    }

    pub(crate) fn make_base_page(node_size: usize) -> *mut Self {
        let layout = Layout::from_size_align(node_size, std::mem::align_of::<LeafNode>()).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        unsafe {
            Self::init_node_with_fence(
                ptr,
                &[],
                &[],
                node_size,
                MiniPageNextLevel::new_null(),
                true,
                false,
            );
        }
        ptr as *mut Self
    }

    /// After promoting a base page to full page cache, we need to mark every records as cache.
    pub(crate) fn covert_insert_records_to_cache(&mut self) {
        let mut cur = self.first_meta_pos_after_fence();

        while cur < self.meta.meta_count_with_fence() {
            let meta = self.get_kv_meta_mut(cur as usize);
            let op_type = meta.op_type();

            match op_type {
                OpType::Insert => {
                    meta.set_op_type(OpType::Cache);
                }
                OpType::Delete => {
                    meta.set_op_type(OpType::Phantom);
                }
                OpType::Cache | OpType::Phantom => {
                    unreachable!("Base page should not have op type: {:?}", op_type);
                }
            };

            cur += 1;
        }
    }

    /// When merge full page cache to base page, we need to set clean records to dirty records.
    pub(crate) fn convert_cache_records_to_insert(&mut self) {
        let mut cur = self.first_meta_pos_after_fence();

        while cur < self.meta.meta_count_with_fence() {
            let meta = self.get_kv_meta_mut(cur as usize);
            let op_type = meta.op_type();

            match op_type {
                OpType::Cache => {
                    meta.set_op_type(OpType::Insert);
                }
                OpType::Phantom => {
                    meta.set_op_type(OpType::Delete);
                }
                OpType::Insert | OpType::Delete => {
                    // other types might present.
                }
            };

            cur += 1;
        }
    }

    unsafe fn init_node_with_fence(
        ptr: *mut u8,
        low_fence: &[u8],
        high_fence: &[u8],
        node_size: usize,
        next_level: MiniPageNextLevel,
        has_fence: bool,
        cache_only: bool,
    ) {
        let ptr = ptr as *mut Self;

        { &mut *ptr }.initialize(
            low_fence, high_fence, node_size, next_level, has_fence, cache_only,
        );
    }

    pub(crate) fn get_kv_meta(&self, index: usize) -> &LeafKVMeta {
        debug_assert!(index < self.meta.meta_count_with_fence() as usize);
        let meta_ptr = self.data.as_ptr() as *const LeafKVMeta;
        unsafe { &*meta_ptr.add(index) }
    }

    pub(crate) fn get_full_key(&self, meta: &LeafKVMeta) -> Vec<u8> {
        let prefix = self.get_prefix();
        let remaining = self.get_remaining_key(meta);
        [prefix, remaining].concat()
    }

    /// Get the full key for low fence which is not prefix compressed
    pub(crate) fn get_low_fence_full_key(&self) -> Vec<u8> {
        debug_assert!(LOW_FENCE_IDX < self.meta.meta_count_with_fence() as usize);
        let meta_ptr = self.data.as_ptr() as *const LeafKVMeta;
        let meta = unsafe { &*meta_ptr.add(LOW_FENCE_IDX) };

        let key_offset = meta.get_offset();
        let key_ptr = unsafe { self.data.as_ptr().add(key_offset as usize) };
        unsafe {
            let key = std::slice::from_raw_parts(key_ptr, (meta.get_key_len()) as usize);
            [key].concat()
        }
    }

    pub(crate) fn get_kv_meta_mut(&mut self, index: usize) -> &mut LeafKVMeta {
        debug_assert!(index < self.meta.meta_count_with_fence() as usize);
        let meta_ptr = self.data.as_mut_ptr() as *mut LeafKVMeta;
        unsafe { meta_ptr.add(index).as_mut().unwrap() }
    }

    pub(crate) fn write_initial_kv_meta(&mut self, index: usize, meta: LeafKVMeta) {
        let meta_ptr = self.data.as_mut_ptr() as *mut LeafKVMeta;
        unsafe { meta_ptr.add(index).write(meta) };
    }

    pub(crate) fn get_remaining_key(&self, meta: &LeafKVMeta) -> &[u8] {
        let key_offset = meta.get_offset();
        let key_ptr = unsafe { self.data.as_ptr().add(key_offset as usize) };
        unsafe {
            std::slice::from_raw_parts(key_ptr, (meta.get_key_len() - self.prefix_len) as usize)
        }
    }

    pub(crate) fn get_prefix(&self) -> &[u8] {
        let m = self.get_kv_meta(LOW_FENCE_IDX);
        let key_offset = m.get_offset();
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().add(key_offset as usize),
                self.prefix_len as usize,
            )
        }
    }

    pub(crate) fn install_fence_key(&mut self, key: &[u8], is_high_fence: bool) {
        let loc: u16 = self.meta.meta_count_with_fence();
        if !is_high_fence {
            debug_assert!(self.meta.meta_count_with_fence() == LOW_FENCE_IDX as u16);
        } else {
            debug_assert!(self.meta.meta_count_with_fence() == HIGH_FENCE_IDX as u16);
        }

        let cur_low_offset = self.current_lowest_offset();

        self.meta.increment_value_count();
        self.meta.remaining_size -= std::mem::size_of::<LeafKVMeta>() as u16;

        if key.is_empty() {
            let fence = if is_high_fence {
                LeafKVMeta::make_infinite_high_fence_key()
            } else {
                LeafKVMeta::make_infinite_low_fence_key()
            };
            self.write_initial_kv_meta(loc as usize, fence);
        } else {
            let prefix_len = if is_high_fence { self.prefix_len } else { 0 }; // we don't compress low fence.

            let post_fix_len = key.len() as u16 - prefix_len;
            let val_len = 0u16;
            let kv_len = post_fix_len + val_len;

            let remaining = self.meta.remaining_size;

            debug_assert!(kv_len <= remaining);

            let offset = cur_low_offset - kv_len;

            let new_meta =
                LeafKVMeta::make_prefixed_meta(offset, 0, key, prefix_len, OpType::Insert);

            self.write_initial_kv_meta(loc as usize, new_meta);

            unsafe {
                let start_ptr = self.data.as_mut_ptr().add(offset as usize);

                let key_slice = std::slice::from_raw_parts(
                    key.as_ptr().add(prefix_len as usize),
                    post_fix_len as usize,
                );
                std::ptr::copy_nonoverlapping(key_slice.as_ptr(), start_ptr, post_fix_len as usize);
            }

            self.meta.remaining_size -= kv_len;
        }
    }

    pub(crate) fn current_lowest_offset(&self) -> u16 {
        let value_count = self.meta.meta_count_with_fence();
        let rt =
            (value_count * std::mem::size_of::<LeafKVMeta>() as u16) + self.meta.remaining_size;

        // Sanity check
        #[cfg(debug_assertions)]
        {
            let mut min_offset = LeafNode::max_data_size(self.meta.node_size as usize) as u16;
            for i in 0..value_count {
                let kv_meta = self.get_kv_meta(i as usize);
                if kv_meta.is_infinite_low_fence_key() || kv_meta.is_infinite_high_fence_key() {
                    continue;
                }
                min_offset = std::cmp::min(min_offset, kv_meta.offset);
            }
            assert!(min_offset == rt);
        }
        rt
    }

    pub(crate) fn get_value(&self, meta: &LeafKVMeta) -> &[u8] {
        let val_offset = meta.get_offset() + meta.get_key_len() - self.prefix_len;
        let val_ptr = unsafe { self.data.as_ptr().add(val_offset as usize) };
        unsafe { std::slice::from_raw_parts(val_ptr, meta.value_len() as usize) }
    }

    /// A good split key has two properties:
    /// (1). it split the nodes into two roughly equal sizes.
    /// (2). it should be short, so that parent node search can be faster.
    ///
    /// Get the split key is tricky:
    /// 1. We can't just use the middle key, because the key/value are variable length. We want the key that split the node in size, not in key count.
    /// 2. We don't try to find small split key here.
    ///
    /// Returns the key that split the node into two roughly equal size, and the new node count.
    /// This is only invoked once for splitting the root node.
    pub fn get_split_key(&self, cache_only: bool) -> (Vec<u8>, u16) {
        let mut data_size = 0;

        for meta in self.meta_iter() {
            let key_len = meta.get_key_len();
            let value_len = meta.value_len();

            data_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;
        }

        let split_target_size = data_size / 2;
        let mut cur_size = 0;

        for (i, meta) in self.meta_iter().enumerate() {
            let key_len = meta.get_key_len();
            let value_len = meta.value_len();

            cur_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;

            // Here we guarantee in cache-only mode, the root node has at least two keys
            if cache_only && i == 0 {
                continue;
            }

            if cur_size >= split_target_size {
                return (self.get_full_key(meta), i as u16);
            }
        }
        unreachable!();
    }

    /// Get # of keys strictly smaller than a merge_split_key
    #[allow(clippy::unused_enumerate_index)]
    pub fn get_kv_num_below_key(&self, merge_split_key: &Vec<u8>) -> u16 {
        // Linear search
        let mut cnt: u16 = 0;
        for (_, meta) in self.meta_iter().enumerate() {
            let key = self.get_full_key(meta);
            let cmp = key.cmp(merge_split_key);
            // Pick all records from the base page whose key is smaller
            // than merge_split_key
            if cmp == std::cmp::Ordering::Less {
                cnt += 1;
            } else {
                break;
            }
        }
        cnt
    }

    /// [Cache-only mode]: Find the splitting key that evenly splits all the records in the
    /// current node and the to-be-inserted record in half. The caller needs to guarantee that
    /// there are at least two records with different such that it won't result in an empty page
    /// after split
    pub fn get_cache_only_insert_split_key(&self, key: &[u8], new_record_size: &u16) -> Vec<u8> {
        let mut merge_split_key_1: Option<Vec<u8>> = None;
        let mut merge_split_key_2: Option<Vec<u8>> = None;
        let mut diff_1: i16 = i16::MAX;
        let mut diff_2: i16 = i16::MAX;

        // The total size of all records including the new record to insert
        let mut total_merged_size: u16 = 0;

        for meta in self.meta_iter() {
            let key_len = meta.get_key_len();
            let value_len = meta.value_len();

            total_merged_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;
        }

        total_merged_size += new_record_size + std::mem::size_of::<LeafKVMeta>() as u16;
        let split_target_size = total_merged_size / 2;

        // Search for the splitting key
        let mut merged_size: u16 = 0;
        let mut self_meta_iter = self.meta_iter();
        let mut self_meta_option = self_meta_iter.next();

        // Go through all records in the base page whose keys are smaller than the new record's key
        if self_meta_option.is_some() {
            let mut cur_base_meta = self_meta_option.unwrap();
            let mut cur_base_key = self.get_full_key(cur_base_meta);
            let mut cmp = cur_base_key.as_slice().cmp(key);

            while cmp == std::cmp::Ordering::Less {
                let key_len = cur_base_meta.get_key_len();
                let value_len = cur_base_meta.value_len();
                merged_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;
                if merged_size >= split_target_size {
                    // Two split key candidates are already found
                    // Stop
                    if merge_split_key_2.is_some() {
                        break;
                    }

                    let left_side: u16 = merged_size
                        - key_len
                        - value_len
                        - std::mem::size_of::<LeafKVMeta>() as u16;
                    let right_side = total_merged_size - left_side;

                    if merge_split_key_1.is_none() {
                        merge_split_key_1 = Some(cur_base_key);
                        diff_1 = (left_side as i16 - right_side as i16).abs();
                    } else {
                        merge_split_key_2 = Some(cur_base_key);
                        diff_2 = (left_side as i16 - right_side as i16).abs();
                    }
                }

                self_meta_option = self_meta_iter.next();
                if let Some(meta) = self_meta_option {
                    cur_base_meta = meta;
                    cur_base_key = self.get_full_key(cur_base_meta);
                    cmp = cur_base_key.as_slice().cmp(key);
                } else {
                    break;
                }
            }
        }

        // Count the new key
        if merge_split_key_2.is_none() {
            merged_size += new_record_size + std::mem::size_of::<LeafKVMeta>() as u16;

            if merged_size >= split_target_size {
                // Two split key candidates are already found
                // Stop

                let left_side: u16 =
                    merged_size - new_record_size - std::mem::size_of::<LeafKVMeta>() as u16;
                let right_side = total_merged_size - left_side;

                if merge_split_key_1.is_none() {
                    merge_split_key_1 = Some(key.to_vec());
                    diff_1 = (left_side as i16 - right_side as i16).abs();
                } else {
                    merge_split_key_2 = Some(key.to_vec());
                    diff_2 = (left_side as i16 - right_side as i16).abs();
                }
            }
        }

        // Go through the rest of records in the base page
        while self_meta_option.is_some() {
            let base_meta = self_meta_option.unwrap();
            let key_len = base_meta.get_key_len();
            let value_len = base_meta.value_len();
            merged_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;

            if merged_size >= split_target_size {
                // Return the splitting key
                let cur_base_key = self.get_full_key(base_meta);
                if merge_split_key_2.is_some() {
                    break;
                }

                let left_side: u16 =
                    merged_size - key_len - value_len - std::mem::size_of::<LeafKVMeta>() as u16;
                let right_side = total_merged_size - left_side;

                if merge_split_key_1.is_none() {
                    merge_split_key_1 = Some(cur_base_key);
                    diff_1 = (left_side as i16 - right_side as i16).abs();
                } else {
                    merge_split_key_2 = Some(cur_base_key);
                    diff_2 = (left_side as i16 - right_side as i16).abs();
                }
            }
            self_meta_option = self_meta_iter.next();
        }

        // Pick the splitting that achieves the smallest size difference between
        // the two halves
        if merge_split_key_1.is_none() {
            panic!(
                "Fail to find a splitting key for merging mini and base page.{}, {}",
                merged_size, split_target_size
            );
        }

        if merge_split_key_2.is_none() || diff_1 < diff_2 {
            return merge_split_key_1.unwrap();
        }

        merge_split_key_2.unwrap()
    }

    /// Find the splitting key that divides the merged records of
    /// the mini-page and its correponding base page evenly in two
    /// groups (base pages). Special considerations go to the split-key
    /// bearing (k,v) pair.
    /// Caller needs to ensure there are at least two distinct keys among the
    /// mini page and self.
    #[allow(clippy::unnecessary_unwrap)]
    pub(crate) fn get_merge_split_key(&mut self, mini_page: &LeafNode) -> Vec<u8> {
        let mut merge_split_key_1: Option<Vec<u8>> = None;
        let mut merge_split_key_2: Option<Vec<u8>> = None;
        let mut diff_1: i16 = i16::MAX;
        let mut diff_2: i16 = i16::MAX;

        let mut total_merged_size: u16 = 0;
        let mut base_meta_iter = self.meta_iter();
        let mut cur_pos = base_meta_iter.cur;
        let mut cur_base_meta_option = base_meta_iter.next();
        let mut duplicate_positions = vec![];

        // Calculate the size of all distinctively merged records through merge-sort
        for mini_meta in mini_page.meta_iter() {
            let c_type = mini_meta.op_type();
            // Skip cached or phantom records as they will not be merged into base pages
            if !c_type.is_dirty() {
                continue;
            }

            let cur_mini_key = mini_page.get_full_key(mini_meta);
            if cur_base_meta_option.is_some() {
                let mut cur_base_meta = cur_base_meta_option.unwrap();
                let mut cur_base_key = self.get_full_key(cur_base_meta);
                let mut cmp = cur_base_key.cmp(&cur_mini_key);

                // Go through all records from the base page whose key is strictly smaller
                // than the current record from the mini page
                while cmp == std::cmp::Ordering::Less {
                    let key_len = cur_base_meta.get_key_len();
                    let value_len = cur_base_meta.value_len();
                    total_merged_size +=
                        key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;

                    cur_pos = base_meta_iter.cur;
                    cur_base_meta_option = base_meta_iter.next();
                    if cur_base_meta_option.is_some() {
                        cur_base_meta = cur_base_meta_option.unwrap();
                        cur_base_key = self.get_full_key(cur_base_meta);
                        cmp = cur_base_key.cmp(&cur_mini_key);
                    } else {
                        break;
                    }
                }

                // If the comparison is equal then advance the base page iterator to avoid counting it
                // twice
                if cmp == std::cmp::Ordering::Equal && cur_base_meta_option.is_some() {
                    // Mark the duplicate entry for deletion
                    duplicate_positions.push(cur_pos);
                    cur_pos = base_meta_iter.cur;
                    cur_base_meta_option = base_meta_iter.next();
                }
            }

            let key_len = mini_meta.get_key_len();
            let value_len = mini_meta.value_len();
            total_merged_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;
        }

        // Mini-page records are exhuasted, go through the rest of
        // records from the base page, if any
        while cur_base_meta_option.is_some() {
            let base_meta = cur_base_meta_option.unwrap();
            let key_len = base_meta.get_key_len();
            let value_len = base_meta.value_len();
            total_merged_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;

            cur_base_meta_option = base_meta_iter.next();
        }

        // The target split size is half of all the merged records
        let split_target_size = total_merged_size / 2;

        // Merge sort the distinct records from the mini page and the base page
        // until the size of the sorted records reaches the target split size
        let mut merged_size: u16 = 0;
        base_meta_iter = self.meta_iter();
        cur_base_meta_option = base_meta_iter.next();

        for mini_meta in mini_page.meta_iter() {
            // Skip cached or phantom records as they will
            // not take additional space after merging
            let c_type = mini_meta.op_type();
            if !c_type.is_dirty() {
                continue;
            }

            let cur_mini_key = mini_page.get_full_key(mini_meta);
            if cur_base_meta_option.is_some() {
                let mut cur_base_meta = cur_base_meta_option.unwrap();
                let mut cur_base_key = self.get_full_key(cur_base_meta);
                let mut cmp = cur_base_key.cmp(&cur_mini_key);
                // Go through all records from the base page whose key is strictly smaller
                // than the current record from the mini page
                while cmp == std::cmp::Ordering::Less {
                    let key_len = cur_base_meta.get_key_len();
                    let value_len = cur_base_meta.value_len();
                    merged_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;
                    if merged_size >= split_target_size {
                        // Two split key candidates are already found
                        // Stop
                        if merge_split_key_2.is_some() {
                            break;
                        }

                        let left_side: u16 = merged_size
                            - key_len
                            - value_len
                            - std::mem::size_of::<LeafKVMeta>() as u16;
                        let right_side = total_merged_size - left_side;

                        if merge_split_key_1.is_none() {
                            merge_split_key_1 = Some(cur_base_key);
                            diff_1 = (left_side as i16 - right_side as i16).abs();
                        } else {
                            merge_split_key_2 = Some(cur_base_key);
                            diff_2 = (left_side as i16 - right_side as i16).abs();
                        }
                    }

                    cur_base_meta_option = base_meta_iter.next();
                    if cur_base_meta_option.is_some() {
                        cur_base_meta = cur_base_meta_option.unwrap();
                        cur_base_key = self.get_full_key(cur_base_meta);
                        cmp = cur_base_key.cmp(&cur_mini_key);
                    } else {
                        break;
                    }
                }

                // If the comparison is equal then advance the base page iterator to avoid counting it
                // twice
                if cmp == std::cmp::Ordering::Equal && cur_base_meta_option.is_some() {
                    cur_base_meta_option = base_meta_iter.next();
                }
            }

            let key_len = mini_meta.get_key_len();
            let value_len = mini_meta.value_len();
            merged_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;

            if merged_size >= split_target_size {
                // Two split key candidates are already found
                // Stop
                if merge_split_key_2.is_some() {
                    break;
                }

                let left_side: u16 =
                    merged_size - key_len - value_len - std::mem::size_of::<LeafKVMeta>() as u16;
                let right_side = total_merged_size - left_side;

                if merge_split_key_1.is_none() {
                    merge_split_key_1 = Some(cur_mini_key);
                    diff_1 = (left_side as i16 - right_side as i16).abs();
                } else {
                    merge_split_key_2 = Some(cur_mini_key);
                    diff_2 = (left_side as i16 - right_side as i16).abs();
                }
            }
        }

        // Mini-page records are exhuasted, go through the rest of
        // records from the base page, if any
        while cur_base_meta_option.is_some() {
            let base_meta = cur_base_meta_option.unwrap();
            let key_len = base_meta.get_key_len();
            let value_len = base_meta.value_len();
            merged_size += key_len + value_len + std::mem::size_of::<LeafKVMeta>() as u16;

            if merged_size >= split_target_size {
                // Return the splitting key
                let cur_base_key = self.get_full_key(base_meta);
                if merge_split_key_2.is_some() {
                    break;
                }

                let left_side: u16 =
                    merged_size - key_len - value_len - std::mem::size_of::<LeafKVMeta>() as u16;
                let right_side = total_merged_size - left_side;

                if merge_split_key_1.is_none() {
                    merge_split_key_1 = Some(cur_base_key);
                    diff_1 = (left_side as i16 - right_side as i16).abs();
                } else {
                    merge_split_key_2 = Some(cur_base_key);
                    diff_2 = (left_side as i16 - right_side as i16).abs();
                }
            }
            cur_base_meta_option = base_meta_iter.next();
        }

        // Delete all duplicate entries from the base page
        for pos in duplicate_positions {
            let pos_meta = self.get_kv_meta_mut(pos);
            pos_meta.mark_as_deleted();
        }

        if merge_split_key_1.is_none() {
            unreachable!(
                "Fail to find a splitting key for merging mini and base page.{}, {}",
                merged_size, split_target_size
            );
        }

        // The two split keys must be different
        if merge_split_key_2.is_some() {
            let cmp = merge_split_key_1
                .as_ref()
                .unwrap()
                .cmp(merge_split_key_2.as_ref().unwrap());
            assert_ne!(cmp, std::cmp::Ordering::Equal);
        }

        let mut splitting_key = if merge_split_key_2.is_none() || diff_1 < diff_2 {
            merge_split_key_1.as_ref().unwrap()
        } else {
            merge_split_key_2.as_ref().unwrap()
        };

        // The splitting key cannot be the low fence as it leads to invalid page
        // and duplicate entries in inner node
        let mut fence_meta = self.get_kv_meta(LOW_FENCE_IDX);
        if !fence_meta.is_infinite_low_fence_key() {
            let low_fence_key = self.get_low_fence_key();
            let cmp = splitting_key.cmp(&low_fence_key);
            if cmp == std::cmp::Ordering::Equal {
                splitting_key = merge_split_key_2.as_ref().unwrap();
            }
        }

        fence_meta = self.get_kv_meta(HIGH_FENCE_IDX);
        if !fence_meta.is_infinite_high_fence_key() {
            let high_fence_key = self.get_high_fence_key();
            let cmp = splitting_key.cmp(&high_fence_key);
            assert_ne!(cmp, std::cmp::Ordering::Equal);
        }

        splitting_key.clone()
    }

    pub fn get_key_to_reach_this_node(&self) -> Vec<u8> {
        let meta = self.get_kv_meta(self.first_meta_pos_after_fence() as usize);
        self.get_full_key(meta)
    }

    /// In cache-only mode, transient state could be observed when one thread is in the middle of
    /// consolidating the leaf node, while the evicting callback thread is attempting a
    /// unprotected read. As such, this function guarantees no panic happens during the key read
    pub fn try_get_key_to_reach_this_node(&self) -> Result<Vec<u8>, TreeError> {
        assert!(self.meta.is_cache_only_leaf());

        // Get the meta of the first key
        let index = 0;
        if index >= self.meta.meta_count_with_fence() as usize {
            return Err(TreeError::NeedRestart);
        }

        let meta_ptr = self.data.as_ptr() as *const LeafKVMeta;
        let meta = unsafe { &*meta_ptr.add(index) };

        let key_offset = meta.get_offset();
        let key_len = meta.get_key_len();

        if key_offset + key_len > LeafNode::max_data_size(self.meta.node_size as usize) as u16 {
            return Err(TreeError::NeedRestart);
        }

        let key_ptr = unsafe { self.data.as_ptr().add(key_offset as usize) };
        let key_slice =
            unsafe { std::slice::from_raw_parts(key_ptr, (key_len - self.prefix_len) as usize) };
        Ok(key_slice.to_vec())
    }

    pub fn get_low_fence_key(&self) -> Vec<u8> {
        assert!(self.has_fence());
        let fence_meta = self.get_kv_meta(LOW_FENCE_IDX);
        if fence_meta.is_infinite_low_fence_key() {
            vec![]
        } else {
            unsafe {
                let start_ptr = self.data.as_ptr().add(fence_meta.offset as usize);
                let key_slice =
                    std::slice::from_raw_parts(start_ptr, fence_meta.get_key_len() as usize);
                key_slice.to_vec()
            }
        }
    }

    pub fn get_high_fence_key(&self) -> Vec<u8> {
        assert!(self.has_fence());
        let fence_meta = self.get_kv_meta(HIGH_FENCE_IDX);
        if fence_meta.is_infinite_high_fence_key() {
            vec![]
        } else {
            self.get_full_key(fence_meta)
        }
    }

    /// Compares the input key with the existing key specified by `meta`
    /// By convention, key_cmp(meta, key) returns the ordering matching the expression meta <operator> key if true.
    #[inline]
    pub(crate) fn key_cmp(&self, meta: &LeafKVMeta, key: &[u8]) -> Ordering {
        let search_key_prefix = &key[(self.prefix_len as usize)..]
            [..std::cmp::min(key.len() - self.prefix_len as usize, PREVIEW_SIZE)];

        let prefix_key = &meta.preview_bytes[..std::cmp::min(
            PREVIEW_SIZE,
            meta.get_key_len() as usize - self.prefix_len as usize,
        )];
        let mut cmp = prefix_key.cmp(search_key_prefix);

        // If the prefix matches, compare the full key
        if cmp == Ordering::Equal {
            let full_key = self.get_remaining_key(meta);
            let search_key_postfix = &key[self.prefix_len as usize..];
            cmp = full_key.cmp(search_key_postfix);
        }
        cmp
    }

    pub(crate) fn linear_lower_bound(&self, key: &[u8]) -> u16 {
        debug_assert!(key.len() >= self.prefix_len as usize);

        let mut index = self.first_meta_pos_after_fence();

        while index < self.meta.meta_count_with_fence() {
            let key_meta = self.get_kv_meta(index as usize);

            #[cfg(target_arch = "x86_64")]
            unsafe {
                // For bw-tree-like linear search, we use clflush to simulate pointer chasing.
                core::arch::x86_64::_mm_clflush(key_meta as *const LeafKVMeta as *const u8);
            }
            let cmp = self.key_cmp(key_meta, key);

            if cmp != Ordering::Less {
                return index;
            }

            index += 1;
        }

        index
    }

    pub(crate) fn lower_bound(&self, key: &[u8]) -> u16 {
        let mut lower = self.first_meta_pos_after_fence();
        let mut upper = self.meta.meta_count_with_fence();

        let search_key_prefix = &key[(self.prefix_len as usize)..]
            [..std::cmp::min(key.len() - self.prefix_len as usize, PREVIEW_SIZE)];

        debug_assert!(key.len() >= self.prefix_len as usize);

        while lower < upper {
            let mid = lower + (upper - lower) / 2;
            let key_meta = self.get_kv_meta(mid as usize);

            let prefix_key = &key_meta.preview_bytes[..std::cmp::min(
                PREVIEW_SIZE,
                key_meta.get_key_len() as usize - self.prefix_len as usize,
            )];
            let mut cmp = prefix_key.cmp(search_key_prefix);

            // If the prefix matches, compare the full key
            if cmp == Ordering::Equal {
                let remaining_key = self.get_remaining_key(key_meta);
                let search_key_postfix = &key[self.prefix_len as usize..];
                cmp = remaining_key.cmp(search_key_postfix);
            }

            match cmp {
                Ordering::Greater => {
                    upper = mid;
                }
                Ordering::Equal => {
                    return mid;
                }
                Ordering::Less => {
                    lower = mid + 1;
                }
            }
        }
        lower
    }

    /// Take a deep breath before you read/change this function.
    ///
    /// Leaf node insert is different from inner node that:
    /// 1. Leaf node value is variable length
    /// 2. Leaf node need to handle duplicate key, in which case it need to remove the old value and insert the new value.
    ///
    /// Returns true if the insert is successful, false if out of space.
    pub(crate) fn insert(
        &mut self,
        key: &[u8],
        value: &[u8],
        op_type: OpType,
        max_fence_len: usize,
    ) -> bool {
        debug_assert!(key.len() as u16 >= self.prefix_len);
        match op_type {
            OpType::Insert | OpType::Cache => {
                debug_assert!(!value.is_empty());
            }
            OpType::Delete | OpType::Phantom => {}
        }

        let post_fix_len = key.len() as u16 - self.prefix_len;
        let val_len = value.len() as u16;
        let kv_len = post_fix_len + val_len;

        let value_count_with_fence = self.meta.meta_count_with_fence();

        let pos = self.lower_bound(key) as usize;

        if pos < value_count_with_fence as usize {
            let prefix_len = self.prefix_len as usize;
            let pos_meta = self.get_kv_meta(pos);
            let pos_key = self.get_remaining_key(pos_meta);
            let search_key_postfix = &key[prefix_len..];
            if pos_key.cmp(search_key_postfix) == Ordering::Equal {
                // The key already exists.
                counter!(LeafInsertDuplicate);
                if op_type == OpType::Delete {
                    let pos_meta = self.get_kv_meta_mut(pos);
                    pos_meta.mark_as_deleted();
                    return true;
                }

                let pos_value = self.get_value(pos_meta);
                let pos_value_len = pos_value.len() as u16;

                if pos_value_len >= val_len {
                    // we are lucky, old value is larger than new value. We just overwrite the old value.
                    unsafe {
                        let pair_ptr = self.data.as_ptr().add(pos_meta.offset as usize) as *mut u8;
                        std::ptr::copy_nonoverlapping(
                            value.as_ptr(),
                            pair_ptr.add(post_fix_len as usize),
                            val_len as usize,
                        );
                    }
                    let pos_meta = self.get_kv_meta_mut(pos);
                    pos_meta.set_value_len(val_len);
                    pos_meta.set_op_type(op_type);
                    return true;
                }

                if self.meta.remaining_size < kv_len {
                    return false;
                }
                assert!(op_type != OpType::Cache);
                let offset = self.current_lowest_offset() - kv_len;
                unsafe {
                    let pair_ptr = self.data.as_ptr().add(offset as usize) as *mut u8;

                    let pos_meta = self.get_kv_meta_mut(pos);
                    pos_meta.set_value_len(val_len);
                    pos_meta.set_op_type(op_type);
                    pos_meta.offset = offset;
                    std::ptr::copy_nonoverlapping(
                        key[self.prefix_len as usize..].as_ptr(),
                        pair_ptr,
                        post_fix_len as usize,
                    );
                    std::ptr::copy_nonoverlapping(
                        value.as_ptr(),
                        pair_ptr.add(post_fix_len as usize),
                        val_len as usize,
                    );
                }
                self.meta.remaining_size -= kv_len;
                return true;
            }
        }

        // The key is not already in the node.
        // For any in-memory page, skipping delete records here could lead to empty page
        // which becomes un-evictable.
        if op_type == OpType::Delete && (self.is_base_page()) {
            // we are deleting a key that is not in the page.
            return true;
        }

        //Check if the node has capacity for the new record with or without fences
        if self.full_with_fences(
            kv_len + std::mem::size_of::<LeafKVMeta>() as u16,
            max_fence_len,
        ) {
            return false;
        }

        counter!(LeafInsertNew);

        let offset = self.current_lowest_offset() - kv_len;
        let new_meta =
            LeafKVMeta::make_prefixed_meta(offset, val_len, key, self.prefix_len, op_type);

        unsafe {
            let metas_size = std::mem::size_of::<LeafKVMeta>()
                * (self.meta.meta_count_with_fence() - pos as u16) as usize;
            let src_ptr = self
                .data
                .as_ptr()
                .add(pos * std::mem::size_of::<LeafKVMeta>());
            let dest_ptr = self
                .data
                .as_mut_ptr()
                .add((pos + 1) * std::mem::size_of::<LeafKVMeta>());
            std::ptr::copy(src_ptr, dest_ptr, metas_size);

            self.write_initial_kv_meta(pos, new_meta);

            let pair_ptr = self.data.as_mut_ptr().add(offset as usize);
            std::ptr::copy_nonoverlapping(
                key[self.prefix_len as usize..].as_ptr(),
                pair_ptr,
                post_fix_len as usize,
            );
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                pair_ptr.add(post_fix_len as usize),
                val_len as usize,
            );
        }

        self.meta.remaining_size -= kv_len + std::mem::size_of::<LeafKVMeta>() as u16;
        self.meta.increment_value_count();
        true
    }

    /// Base pages are leaf pages without next pointer in non cache-only mode
    pub(crate) fn is_base_page(&self) -> bool {
        (!self.meta.is_cache_only_leaf()) && self.next_level.is_null()
    }

    /// Returns the splitting key after splitting self
    /// The new high fence key of self (left-side node) and low fence
    /// of the new base page (right-side node) equal to the splitting key
    pub fn split(&mut self, sibling: &mut LeafNode, cache_only: bool) -> Vec<u8> {
        if cache_only {
            assert!(self.meta.is_cache_only_leaf() && !self.has_fence());
        } else {
            assert!(self.is_base_page() && self.has_fence());
        }

        let current_count = self.meta.meta_count_without_fence();

        // [new_cur_count..] are moved to the sibling node
        // [..new_cur_count - 1] are kept in self
        // where splitting_key is the key at new_cur_count
        // Also, new_cur_count is in [0..(#non_fence_meta - 1)]
        let (splitting_key, new_cur_count) = self.get_split_key(cache_only);
        let sibling_cnt = current_count - new_cur_count;

        // Now we have to do two things:
        // Copy the second half of the key-value pairs to the new node, setting the correct offsets.
        // Consolidate the key-value pairs in the current node, setting the correct offsets
        if !cache_only {
            let low_fence_for_right = self.get_kv_meta(FENCE_KEY_CNT + new_cur_count as usize);
            assert!(!low_fence_for_right.is_infinite_low_fence_key());
            let high_fence_for_right = self.get_kv_meta(HIGH_FENCE_IDX);

            let low_fence_key = self.get_full_key(low_fence_for_right);
            let high_fence_key = if high_fence_for_right.is_infinite_high_fence_key() {
                vec![]
            } else {
                self.get_full_key(high_fence_for_right)
            };

            sibling.initialize(
                &low_fence_key,
                &high_fence_key,
                sibling.meta.node_size as usize,
                MiniPageNextLevel::new_null(),
                true,
                cache_only,
            );
        } else {
            sibling.initialize(
                &[],
                &[],
                sibling.meta.node_size as usize,
                MiniPageNextLevel::new_null(),
                false,
                cache_only,
            );
        }

        let starting_kv_idx = if cache_only { 0 } else { FENCE_KEY_CNT };

        for i in 0..sibling_cnt {
            let kv_meta = self.get_kv_meta((new_cur_count + i) as usize + starting_kv_idx);
            if kv_meta.op_type() == OpType::Delete {
                // skip deleted records.
                continue;
            }
            let key = self.get_full_key(kv_meta);
            let value = self.get_value(kv_meta);
            let insert_rt = sibling.insert(&key, value, OpType::Insert, 0);
            assert!(insert_rt);
        }

        if !cache_only {
            self.meta
                .set_value_count(new_cur_count + FENCE_KEY_CNT as u16);
        } else {
            self.meta.set_value_count(new_cur_count);
        }

        self.consolidate_after_split(&splitting_key);

        splitting_key
    }

    /// Use the passed in `merge_split_key` as the splitting key
    /// to divide the base page in two. Also, install it as the high
    /// fence key of the left-side node and the low fence key of the
    /// right-side node
    pub fn split_with_key(
        &mut self,
        sibling: &mut LeafNode,
        merge_split_key: &Vec<u8>,
        cache_only: bool,
    ) {
        if cache_only {
            assert!(self.meta.is_cache_only_leaf() && !self.has_fence());
        } else {
            assert!(self.is_base_page() && self.has_fence());
        }

        let current_count = self.meta.meta_count_without_fence();
        // [new_cur_count..] are moved to the sibling node
        // [..new_cur_count - 1] are kept in self
        // where splitting_key is the key at new_cur_count
        // Also, new_cur_count is in [0..#non_fence_meta]
        let mut new_cur_count = self.get_kv_num_below_key(merge_split_key);
        let sibling_cnt = current_count - new_cur_count;

        // Now we have to do two things:
        // Copy the second half of the key-value pairs to the new node, setting the correct offsets.
        // Consolidate the key-value pairs in the current node, setting the correct offsets.
        if !cache_only {
            let high_fence_for_right = self.get_kv_meta(HIGH_FENCE_IDX);

            let low_fence_key = merge_split_key.clone();
            let high_fence_key = if high_fence_for_right.is_infinite_high_fence_key() {
                vec![]
            } else {
                self.get_full_key(high_fence_for_right)
            };

            sibling.initialize(
                &low_fence_key,
                &high_fence_key,
                sibling.meta.node_size as usize,
                MiniPageNextLevel::new_null(),
                true,
                cache_only,
            );
        } else {
            sibling.initialize(
                &[],
                &[],
                sibling.meta.node_size as usize,
                MiniPageNextLevel::new_null(),
                false,
                cache_only,
            );
        }

        let starting_kv_index = if cache_only { 0 } else { FENCE_KEY_CNT };

        for i in 0..sibling_cnt {
            let kv_meta = self.get_kv_meta((new_cur_count + i) as usize + starting_kv_index);
            if kv_meta.op_type() == OpType::Delete {
                // skip deleted records
                continue;
            }
            let key = self.get_full_key(kv_meta);
            let value = self.get_value(kv_meta);
            let insert_rt = sibling.insert(&key, value, OpType::Insert, 0);
            assert!(insert_rt);
        }

        if !cache_only {
            new_cur_count += FENCE_KEY_CNT as u16;
        }

        self.meta.set_value_count(new_cur_count);
        self.consolidate_after_split(merge_split_key);

        if !cache_only {
            // Assert the high fence of the left side node and the low fence of the right side node
            // are both set to the merge split key
            let left_high_fence = self.get_kv_meta(HIGH_FENCE_IDX);
            let left_high_fence_key = self.get_full_key(left_high_fence);
            let right_low_fence_key = sibling.get_low_fence_full_key();

            assert_eq!(left_high_fence_key, *merge_split_key); // The high fence of the left side node == splitting key
            assert_eq!(right_low_fence_key, *merge_split_key);
        }
    }

    pub(crate) fn consolidate_inner(
        &mut self,
        new_optype: OpType,
        new_high_fence: Option<&[u8]>,
        skip_tombstone: bool,
        cache_only: bool,
        skip_key: Option<&[u8]>,
    ) {
        let mut pairs = Vec::new();

        for meta in self.meta_iter() {
            if skip_tombstone && meta.op_type() == OpType::Delete {
                // Skip tombstone values.
                continue;
            }

            match skip_key {
                // Skip the record with the skip_key
                Some(s_k) => {
                    let k = self.get_full_key(meta);
                    let cmp = s_k.cmp(&k);

                    if cmp != Ordering::Equal {
                        pairs.push((k, self.get_value(meta).to_owned(), meta.op_type()));
                    }
                }
                None => {
                    pairs.push((
                        self.get_full_key(meta),
                        self.get_value(meta).to_owned(),
                        meta.op_type(),
                    ));
                }
            }
        }

        let has_fence = self.has_fence();
        let (low_fence, high_fence) = if has_fence {
            let high_fence_key = match new_high_fence {
                None => self.get_high_fence_key(),
                Some(key) => key.to_vec(),
            };
            (self.get_low_fence_key(), high_fence_key)
        } else {
            (vec![], vec![])
        };

        let node_size = self.meta.node_size;

        self.initialize(
            &low_fence,
            &high_fence,
            node_size as usize,
            self.next_level,
            has_fence,
            cache_only,
        );

        for (key, value, op_type) in pairs {
            let rt = if op_type == OpType::Delete {
                self.insert(&key, &value, OpType::Delete, 0)
            } else {
                self.insert(&key, &value, new_optype, 0)
            };
            assert!(rt);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn consolidate(&mut self) {
        self.consolidate_inner(
            OpType::Insert,
            None,
            true,
            self.meta.is_cache_only_leaf(),
            None,
        );
    }

    /// For mini page only.
    /// After consolidation, every tombstone records are removed, every insert records become cache records.
    #[allow(dead_code)]
    pub(crate) fn consolidate_after_merge(&mut self) {
        assert!(!self.is_base_page());
        self.consolidate_inner(
            OpType::Cache,
            None,
            false,
            self.meta.is_cache_only_leaf(),
            None,
        );
    }

    /// For base page in non cache-only mode or mini page in cache-only mode.
    /// Note that this operation could empty the page so the caller needs to ensure it won't lead to an empty mini page in memory.
    /// Re-organize the key-value pairs to remove the holes in the data.
    /// A delete/split operation will leave holes in the data field, which is not good for space efficiency.
    /// The easiest (but not most efficient) way to do this is to first populate every key value out, then reset the node state, then insert them back.
    fn consolidate_after_split(&mut self, high_fence: &[u8]) {
        assert!(self.meta.is_cache_only_leaf() || self.is_base_page());
        self.consolidate_inner(
            OpType::Insert,
            Some(high_fence),
            true,
            self.meta.is_cache_only_leaf(),
            None,
        );
    }

    /// For mini page only in cache-only mode only.
    /// Note that this operation could empty the page so the caller needs to ensure it won't lead to an empty mini page in memory.
    /// Same as consolidate(), except that any record with the given key is dropped
    pub(crate) fn consolidate_skip_key(&mut self, key: &[u8]) {
        assert!(!self.is_base_page());
        assert!(self.meta.is_cache_only_leaf());
        self.consolidate_inner(
            OpType::Insert,
            None,
            true,
            self.meta.is_cache_only_leaf(),
            Some(key),
        );
    }

    /// Copy a mini-page to a new memory location
    pub(crate) fn copy_initialize_to(
        &self,
        dst_node: *mut LeafNode,
        dst_size: usize,
        discard_cold_cache: bool,
    ) {
        assert!(!self.is_base_page());
        assert!(self.meta.node_size as usize <= dst_size);
        let dst_ref = unsafe { &mut *dst_node };
        let empty = vec![];

        dst_ref.initialize(
            &empty,
            &empty,
            dst_size,
            self.next_level,
            false, // Mini-page only, thus no fence
            self.meta.is_cache_only_leaf(),
        );

        for meta in self.meta_iter() {
            let op = meta.op_type();

            // Skip the cold values
            // This won't happen in cache-only mode as all records are dirty
            // In non cache-only mode, this won't lead to empty mini-page as
            // this is invoked after a read-hot or write-hot record.
            if discard_cold_cache && op.is_cache() && !meta.is_referenced() {
                continue;
            }

            let value = self.get_value(meta);
            let rt = dst_ref.insert(&self.get_full_key(meta), value, op, 0);
            assert!(rt);
        }
    }

    /// Initialize a LeafNode
    pub(crate) fn initialize(
        &mut self,
        low_fence: &[u8],
        high_fence: &[u8],
        node_size: usize,
        next_level: MiniPageNextLevel,
        has_fence: bool,
        cache_only: bool,
    ) {
        if !has_fence {
            assert!(low_fence.is_empty());
            assert!(high_fence.is_empty());

            self.meta = NodeMeta::new(
                LeafNode::max_data_size(node_size) as u16,
                false,
                false,
                node_size as u16,
                cache_only,
            );
            self.prefix_len = 0;
            self.next_level = next_level;
        } else {
            self.meta = NodeMeta::new(
                LeafNode::max_data_size(node_size) as u16, // - max_fence_len as u16, // Reserve space for
                false,
                true,
                node_size as u16,
                cache_only,
            );

            self.prefix_len = common_prefix_len(low_fence, high_fence);
            self.next_level = next_level;
            self.install_fence_key(low_fence, false);
            self.install_fence_key(high_fence, true);
        }
    }

    pub(crate) fn set_split_flag(&mut self) {
        self.meta.set_split_flag();
    }

    pub(crate) fn get_split_flag(&self) -> bool {
        self.meta.get_split_flag()
    }

    pub(crate) fn read_by_key(&self, search_key: &[u8], out_buffer: &mut [u8]) -> LeafReadResult {
        self.read_by_key_inner(search_key, out_buffer, true)
    }

    /// Read by key.
    #[must_use]
    pub(crate) fn read_by_key_inner(
        &self,
        search_key: &[u8],
        out_buffer: &mut [u8],
        binary_search: bool,
    ) -> LeafReadResult {
        let val_count = self.meta.meta_count_with_fence();
        let pos = if binary_search {
            self.lower_bound(search_key)
        } else {
            self.linear_lower_bound(search_key)
        };

        if pos >= val_count {
            counter!(LeafNotFoundDueToRange);
            return LeafReadResult::NotFound;
        }

        let kv_meta = self.get_kv_meta(pos as usize);
        let target_key = self.get_remaining_key(kv_meta);

        // If the key is not already referenced, we need to mark it as referenced.
        if !kv_meta.is_referenced() {
            kv_meta.mark_as_ref();
        }

        let input_post_key = &search_key[self.prefix_len as usize..];
        let cmp = target_key.cmp(input_post_key);

        if cmp != Ordering::Equal {
            counter!(LeafNotFoundDueToKey);
            LeafReadResult::NotFound
        } else {
            if kv_meta.op_type().is_absent() {
                return LeafReadResult::Deleted;
            }
            let val_len = kv_meta.value_len();
            let val_ref = self.get_value(kv_meta);
            debug_assert_eq!(val_len as usize, val_ref.len());
            out_buffer[..val_len as usize].copy_from_slice(val_ref);
            LeafReadResult::Found(val_len as u32)
        }
    }

    /// Pick the smallest mini-page size that the record fits in without filling it full
    /// For Bf-Tree with backend storage, the size is strictly less than the leaf page size (full)
    /// For cache-only Bf-Tree, we allow the mini-page size to reach leaf page size
    /// Assuming page_classes is in ascending order
    pub(crate) fn get_chain_size_hint(
        key_len: usize,
        value_len: usize,
        page_classes: &[usize],
        cache_only: bool,
    ) -> usize {
        let mut initial_record_size = key_len + value_len + std::mem::size_of::<LeafKVMeta>();
        initial_record_size += std::mem::size_of::<LeafNode>();

        if let Some(s) = page_classes[0..(page_classes.len() - 1)]
            .iter()
            .position(|x| initial_record_size < *x)
        {
            return page_classes[s];
        } else if cache_only && initial_record_size <= page_classes[page_classes.len() - 1] {
            return page_classes[page_classes.len() - 1];
        }

        panic!(
            "Record size {} plus metadata exceeds the max mini-page size {:?}",
            initial_record_size, page_classes
        );
    }

    /// A mini-page is upgraded to the next size up where the record fits in without filling it full.
    /// For Bf-Tree with backend storage, the new size must be smaller than the leaf page size (full)
    /// For cache-only Bf-Tree, we allow the mini-page size to reach leaf page size
    /// If not possible, return None
    /// Assuming page_classes is in ascending order
    pub(crate) fn new_size_if_upgrade(
        &self,
        incoming_size: usize,
        page_classes: &[usize],
        cache_only: bool,
    ) -> Option<usize> {
        let cur_size = self.meta.node_size as usize;
        let request_size = cur_size + incoming_size;

        if let Some(s) = page_classes[0..(page_classes.len() - 1)]
            .iter()
            .position(|x| request_size < *x)
        {
            if s == 0 {
                panic!("Should not be here");
            }
            return Some(page_classes[s]);
        } else if cache_only && request_size <= page_classes[page_classes.len() - 1] {
            return Some(page_classes[page_classes.len() - 1]);
        }

        None
    }

    /// Currently free node can only be called with base node. Mini page should be freed differently.
    pub(crate) fn free_base_page(node: *mut LeafNode) {
        assert!(unsafe { &*node }.is_base_page());
        let node_size = unsafe { &*node }.meta.node_size as usize;
        let layout = Layout::from_size_align(node_size, std::mem::align_of::<LeafNode>()).unwrap();
        unsafe {
            std::alloc::dealloc(node as *mut u8, layout);
        }
    }

    pub(crate) fn need_actually_merge_to_disk(&self) -> bool {
        for meta in self.meta_iter() {
            if meta.op_type().is_dirty() {
                return true;
            }
        }

        false
    }

    fn estimate_merge_size(&self) -> usize {
        let mut required_size = 0;

        for meta in self.meta_iter() {
            if meta.op_type() == OpType::Insert {
                required_size += (meta.get_key_len() + meta.value_len()) as usize
                    + std::mem::size_of::<LeafKVMeta>();
            }
        }

        required_size
    }

    /// Determine if the leaf node has space of the requested_size given the full fence length
    /// This is required as fences could be added to a base page during consolidation
    pub(crate) fn full_with_fences(&self, requested_size: u16, max_fence_len: usize) -> bool {
        if self.meta.remaining_size < requested_size {
            return true;
        }

        if self.meta.has_fence() {
            let mut empty_data_size = self.meta.remaining_size; // Counting fence keys

            let low_key_meta = self.get_kv_meta(LOW_FENCE_IDX);
            if !low_key_meta.is_infinite_low_fence_key() {
                empty_data_size += low_key_meta.get_key_len();
            }

            let high_key_meta = self.get_kv_meta(HIGH_FENCE_IDX);
            if !high_key_meta.is_infinite_high_fence_key() {
                empty_data_size += high_key_meta.get_key_len();
            }

            if empty_data_size >= requested_size + max_fence_len as u16 {
                return false;
            }

            return true;
        }

        false
    }

    pub(crate) fn merge_mini_page(&mut self, mini_page: &LeafNode, max_fence_len: usize) -> bool {
        let size_required = mini_page.estimate_merge_size();

        if self.full_with_fences(size_required as u16, max_fence_len) {
            return false;
        }

        for meta in mini_page.meta_iter() {
            let c_key = mini_page.get_full_key(meta);
            let c_type = meta.op_type();
            if !c_type.is_dirty() {
                // This is important. We don't want to merge cache records, not only for performance but also correctness.
                // A cached record might be inaccessible (thus invalid).
                // Consider the case where scan operation merged the mini page, which triggers split,
                // The scan operation want to keep the mini page (why not), but the records are all in cache mode.
                continue;
            }
            let c_value = mini_page.get_value(meta);
            let rt = self.insert(&c_key, c_value, c_type, 0);
            assert!(rt);
        }

        true
    }

    pub(crate) fn get_stats(&self) -> LeafStats {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut op_types = Vec::new();
        let node_size = self.meta.node_size as usize;
        let prefix = self.get_prefix().to_owned();

        for meta in self.meta_iter() {
            let key = self.get_full_key(meta);
            let value = self.get_value(meta);
            keys.push(key);
            values.push(value.to_owned());
            op_types.push(meta.op_type());
        }

        LeafStats {
            keys,
            values,
            op_types,
            prefix,
            base_node: None,
            next_level: self.next_level,
            node_size,
        }
    }

    pub(crate) fn has_fence(&self) -> bool {
        self.meta.has_fence()
    }

    /// Returns an iterator that iterates over all key-value pairs, skipping the fence keys.
    pub(crate) fn meta_iter(&self) -> LeafMetaIter<'_> {
        LeafMetaIter::new(self)
    }

    fn first_meta_pos_after_fence(&self) -> u16 {
        if self.has_fence() {
            FENCE_KEY_CNT as u16
        } else {
            0
        }
    }

    pub(crate) fn get_record_by_pos_with_bound(
        &self,
        pos: u32,
        out_buffer: &mut [u8],
        return_field: ScanReturnField,
        bound_key: &Option<Vec<u8>>,
    ) -> GetScanRecordByPosResult {
        if pos >= self.meta.meta_count_with_fence() as u32 {
            return GetScanRecordByPosResult::EndOfLeaf;
        }

        let meta = self.get_kv_meta(pos as usize);

        if meta.op_type().is_absent() {
            return GetScanRecordByPosResult::Deleted;
        }

        match return_field {
            ScanReturnField::Value => {
                if let Some(bk) = bound_key {
                    let cmp = self.get_full_key(meta).as_slice().cmp(bk);
                    if cmp == Ordering::Greater {
                        return GetScanRecordByPosResult::BoundKeyExceeded;
                    }
                }

                let value = self.get_value(meta);
                let value_len = meta.value_len() as usize;
                out_buffer[..value_len].copy_from_slice(value);
                GetScanRecordByPosResult::Found(0, value_len as u32)
            }
            ScanReturnField::Key => {
                let full_key = self.get_full_key(meta);

                if let Some(bk) = bound_key {
                    let cmp = full_key.as_slice().cmp(bk);
                    if cmp == Ordering::Greater {
                        return GetScanRecordByPosResult::BoundKeyExceeded;
                    }
                }

                let key_len = full_key.len();
                out_buffer[..key_len].copy_from_slice(&full_key);
                GetScanRecordByPosResult::Found(key_len as u32, 0)
            }
            ScanReturnField::KeyAndValue => {
                let full_key = self.get_full_key(meta);

                if let Some(bk) = bound_key {
                    let cmp = full_key.as_slice().cmp(bk);
                    if cmp == Ordering::Greater {
                        return GetScanRecordByPosResult::BoundKeyExceeded;
                    }
                }

                let key_len = full_key.len();
                let value = self.get_value(meta);
                let value_len = meta.value_len() as usize;

                out_buffer[..key_len].copy_from_slice(&full_key);
                out_buffer[key_len..key_len + value_len].copy_from_slice(value);

                GetScanRecordByPosResult::Found(key_len as u32, value_len as u32)
            }
        }
    }
}

pub(crate) struct LeafMetaIter<'a> {
    node: &'a LeafNode,
    cur: usize,
}

impl LeafMetaIter<'_> {
    fn new(leaf: &LeafNode) -> LeafMetaIter<'_> {
        let cur = leaf.first_meta_pos_after_fence();
        LeafMetaIter {
            node: leaf,
            cur: cur as usize,
        }
    }
}

impl<'a> Iterator for LeafMetaIter<'a> {
    type Item = &'a LeafKVMeta;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur < self.node.meta.meta_count_with_fence() as usize {
            let rt = self.node.get_kv_meta(self.cur);
            self.cur += 1;
            Some(rt)
        } else {
            None
        }
    }
}

pub(crate) enum GetScanRecordByPosResult {
    EndOfLeaf,
    Deleted,
    Found(u32, u32),  // length of returned key and value
    BoundKeyExceeded, // The key at the pos exceeds a given bound key
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LeafReadResult {
    Deleted,
    NotFound,
    Found(u32),
    InvalidKey,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::cast_slice;
    use rstest::rstest;

    #[test]
    fn test_op_type_enum() {
        assert_eq!(OpType::Insert as u8, 0);
        assert_eq!(OpType::Delete as u8, 1);
        assert_eq!(OpType::Cache as u8, 2);
    }

    #[test]
    fn test_make_prefixed_meta() {
        let key = [1, 2, 3, 4, 5];
        let prefix_len = 2;
        let meta = LeafKVMeta::make_prefixed_meta(10, 20, &key, prefix_len, OpType::Insert);

        assert_eq!(meta.offset, 10);
        assert_eq!(meta.get_key_len(), 5);
        assert_eq!(meta.value_len(), 20);
        assert_eq!(meta.preview_bytes, [3, 4]);
        assert_eq!(meta.op_type(), OpType::Insert);
        assert!(!meta.is_referenced());
    }

    #[test]
    fn test_make_infinite_high_fence_key() {
        let meta = LeafKVMeta::make_infinite_high_fence_key();
        assert_eq!(meta.offset, u16::MAX);
        assert_eq!(meta.get_key_len(), 0);
        assert_eq!(meta.value_len(), 0);
        assert!(meta.is_infinite_high_fence_key());
    }

    #[test]
    fn test_make_infinite_low_fence_key() {
        let meta = LeafKVMeta::make_infinite_low_fence_key();
        assert_eq!(meta.offset, u16::MAX - 1);
        assert_eq!(meta.get_key_len(), 0);
        assert_eq!(meta.value_len(), 0);
        assert!(meta.is_infinite_low_fence_key());
    }

    #[test]
    fn test_value_len() {
        let mut meta = LeafKVMeta::make_prefixed_meta(10, 20, &[1, 2, 3], 0, OpType::Insert);
        assert_eq!(meta.value_len(), 20);

        meta.set_value_len(30);
        assert_eq!(meta.value_len(), 30);
    }

    #[test]
    fn test_op_type() {
        let meta = LeafKVMeta::make_prefixed_meta(10, 20, &[1, 2, 3], 0, OpType::Delete);
        assert_eq!(meta.op_type(), OpType::Delete);
    }

    #[test]
    fn test_mark_as_ref_and_clear_ref() {
        let meta = LeafKVMeta::make_prefixed_meta(10, 20, &[1, 2, 3], 0, OpType::Insert);
        assert!(!meta.is_referenced());

        meta.mark_as_ref();
        assert!(meta.is_referenced());

        meta.clear_ref();
        assert!(!meta.is_referenced());
    }

    #[test]
    fn test_mark_as_deleted_and_is_deleted() {
        let mut meta = LeafKVMeta::make_prefixed_meta(10, 20, &[1, 2, 3], 0, OpType::Insert);
        assert!(!meta.is_deleted());

        meta.mark_as_deleted();
        assert!(meta.is_deleted());
    }

    /// This test verifies that the merge split key divides
    /// the combined records of the base page (self) and its
    /// to-be-merged mini-page in half
    #[rstest]
    #[case(vec![1], vec![2], 2)]
    #[case(vec![2], vec![1], 2)]
    fn test_get_merge_split_key(
        #[case] base_page_values: Vec<usize>,
        #[case] mini_page_values: Vec<usize>,
        #[case] splitting_key: usize,
    ) {
        let base = unsafe { &mut *LeafNode::make_base_page(4096) };
        let mini = unsafe { &mut *LeafNode::make_base_page(4096) }; // Using base page as substitute

        // Insert values to base page and mini page accordingly
        for i in 0..base_page_values.len() {
            let n = &base_page_values[i];
            let n_slice = unsafe { std::slice::from_raw_parts(n as *const usize, 1) };
            let key = cast_slice::<usize, u8>(n_slice);
            let value = cast_slice::<usize, u8>(n_slice);

            let rt = base.insert(key, value, OpType::Insert, 2);
            assert!(rt);
        }

        for i in 0..mini_page_values.len() {
            let n = &mini_page_values[i];
            let n_slice = unsafe { std::slice::from_raw_parts(n as *const usize, 1) };
            let key = cast_slice::<usize, u8>(n_slice);
            let value = cast_slice::<usize, u8>(n_slice);

            let rt = mini.insert(key, value, OpType::Insert, 2);
            assert!(rt);
        }

        // Find the splitting key
        let merge_split_key_byte = base.get_merge_split_key(mini);
        let merge_splitting_key = cast_slice::<u8, usize>(&merge_split_key_byte);

        assert_eq!(merge_splitting_key[0], splitting_key);

        LeafNode::free_base_page(base);
        LeafNode::free_base_page(mini);
    }

    /// This test verifies that a base page is correctly split
    /// up based on a splitting key with the left side node (self)
    /// containing keys less than the key and right side node (new)
    /// containing keys greater than or equal to the splitting key
    #[rstest]
    #[case(vec![1, 2, 3, 4], 3)]
    #[case(vec![1], 2)]
    #[case(vec![2], 2)]
    fn test_split_with_key(#[case] base_page_values: Vec<usize>, #[case] splitting_key: usize) {
        let base = unsafe { &mut *LeafNode::make_base_page(4096) };
        let sibling = unsafe { &mut *LeafNode::make_base_page(4096) };

        // Insert values to base page
        for i in 0..base_page_values.len() {
            let n = &base_page_values[i];
            let n_slice = unsafe { std::slice::from_raw_parts(n as *const usize, 1) };
            let key = cast_slice::<usize, u8>(n_slice);
            let value = cast_slice::<usize, u8>(n_slice);

            let rt = base.insert(key, value, OpType::Insert, 2);
            assert!(rt);
        }

        let splitting_key_ptr = &splitting_key;
        let splitting_key_slice =
            unsafe { std::slice::from_raw_parts(splitting_key_ptr as *const usize, 1) };
        let splitting_key_byte_arrary = cast_slice::<usize, u8>(splitting_key_slice).to_vec();

        // Split the base page using the splitting key
        base.split_with_key(sibling, &splitting_key_byte_arrary, false);

        // All values less than splitting key are in the left side node (self)
        // while those greater than or equal to the key are in the sibling node
        let mut out_buffer = vec![0u8; 1024];
        for i in 0..base_page_values.len() {
            let mut page = &mut *base;
            if base_page_values[i] >= splitting_key {
                page = &mut *sibling;
            }

            let n = &base_page_values[i];
            let n_slice = unsafe { std::slice::from_raw_parts(n as *const usize, 1) };
            let key = cast_slice::<usize, u8>(n_slice);

            let rt = page.read_by_key(key, &mut out_buffer);

            assert_eq!(rt, LeafReadResult::Found(key.len() as u32)); // key and value were set the same
            assert_eq!(&out_buffer[0..key.len()], key);
        }

        LeafNode::free_base_page(base);
        LeafNode::free_base_page(sibling);
    }
}
