// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::{
    check_parent, counter,
    error::TreeError,
    mini_page_op::{
        upgrade_to_full_page, LeafEntrySLocked, LeafEntryXLocked, LeafOperations, MergeResult,
    },
    nodes::leaf_node::{GetScanRecordByPosResult, MiniPageNextLevel},
    storage::PageLocation,
    utils::{inner_lock::ReadGuard, Backoff},
    BfTree,
};

pub(crate) enum ScanError {
    NeedMergeMiniPage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanReturnField {
    Key,
    Value,
    KeyAndValue,
}

pub(crate) enum ScanPosition {
    Base(u32),
    Full(u32),
    Mini(u32), // cache-only mode
    Null,      // cache-only mode, evicted mini page
}

impl ScanPosition {
    fn move_to_next(&mut self) {
        match self {
            ScanPosition::Base(offset) => *offset += 1,
            ScanPosition::Full(offset) => *offset += 1,
            ScanPosition::Mini(offset) => *offset += 1,
            ScanPosition::Null => panic!("move_to_next on Null page"),
        }
    }
}

// I think we only need s-lock. but we do x-lock because we can't downgrade a x-lock to s-lock yet.
// implementing the downgrade is more challenging than I thought.
// we currently keep both, but for performance we shouldn't hold the x-lock for too long.
enum ScanLock<'b> {
    S(LeafEntrySLocked<'b>),
    X(LeafEntryXLocked<'b>),
}

impl ScanLock<'_> {
    fn get_record_by_pos_with_bound(
        &self,
        pos: &ScanPosition,
        out_buffer: &mut [u8],
        return_field: ScanReturnField,
        end_key: &Option<Vec<u8>>,
    ) -> GetScanRecordByPosResult {
        match self {
            ScanLock::S(leaf) => {
                leaf.scan_record_by_pos_with_bound(pos, out_buffer, return_field, end_key)
            }
            ScanLock::X(leaf) => {
                leaf.scan_record_by_pos_with_bound(pos, out_buffer, return_field, end_key)
            }
        }
    }

    fn get_right_sibling(&mut self) -> Vec<u8> {
        match self {
            ScanLock::S(leaf) => leaf.get_right_sibling(),
            ScanLock::X(leaf) => leaf.get_right_sibling(),
        }
    }
}

pub struct ScanIterMut<'a, 'b: 'a> {
    tree: &'b BfTree,
    scan_cnt: usize,

    scan_position: ScanPosition,

    leaf_lock: LeafEntryXLocked<'a>,

    return_field: ScanReturnField,

    end_key: Option<Vec<u8>>,
}

impl<'b> ScanIterMut<'_, 'b> {
    pub fn new_with_scan_count(
        tree: &'b BfTree,
        start_key: &'b [u8],
        scan_cnt: usize,
        return_field: ScanReturnField,
    ) -> Self {
        let backoff = Backoff::new();
        let mut aggressive_split = false;

        loop {
            let (scan_pos, lock) = match move_cursor_to_leaf_mut(tree, start_key, aggressive_split)
            {
                Ok((pos, lock)) => (pos, lock),
                Err(TreeError::Locked) => {
                    backoff.spin();
                    continue;
                }
                Err(TreeError::NeedRestart) => {
                    aggressive_split = true;
                    backoff.spin();
                    continue;
                }
                Err(TreeError::CircularBufferFull) => {
                    _ = tree.evict_from_circular_buffer();
                    aggressive_split = true;
                    continue;
                }
            };

            return Self {
                tree,
                scan_cnt,
                scan_position: scan_pos,
                leaf_lock: lock,
                return_field,
                end_key: None,
            };
        }
    }

    pub fn new_with_end_key(
        tree: &'b BfTree,
        start_key: &'b [u8],
        end_key: &[u8],
        return_field: ScanReturnField,
    ) -> Self {
        let mut si = Self::new_with_scan_count(tree, start_key, usize::MAX, return_field);
        si.end_key = Some(end_key.to_vec());
        si
    }

    pub fn next(&mut self, out_buffer: &mut [u8]) -> Option<(usize, usize)> {
        loop {
            if self.scan_cnt == 0 && self.end_key.is_none() {
                return None;
            }

            match self.leaf_lock.scan_record_by_pos_with_bound(
                &self.scan_position,
                out_buffer,
                self.return_field,
                &self.end_key,
            ) {
                GetScanRecordByPosResult::Deleted => {
                    self.scan_position.move_to_next();
                    continue;
                }
                GetScanRecordByPosResult::Found(key_len, value_len) => {
                    self.scan_position.move_to_next();
                    self.scan_cnt -= 1;

                    // since we are mut, we need to mark as dirty.
                    match self.leaf_lock.get_page_location() {
                        PageLocation::Base(_offset) => {
                            self.leaf_lock.load_base_page_mut();
                        }
                        PageLocation::Full(_) => {
                            // do nothing.
                        }
                        PageLocation::Mini(_) => {
                            unreachable!()
                        }
                        PageLocation::Null => panic!("range_scan next on Null page"),
                    }
                    return Some((key_len as usize, value_len as usize));
                }
                GetScanRecordByPosResult::EndOfLeaf => {
                    // we need to load next leaf.
                    let right_sibling = self.leaf_lock.get_right_sibling();

                    if right_sibling.is_empty() {
                        self.scan_cnt = 0;
                        return None;
                    }

                    if self
                        .end_key
                        .as_ref()
                        .is_some_and(|end| &right_sibling > end)
                    {
                        self.scan_cnt = 0;
                        return None;
                    }

                    let backoff = Backoff::new();

                    let mut aggressive_split = false;
                    loop {
                        let (pos, lock) = match move_cursor_to_leaf_mut(
                            self.tree,
                            &right_sibling,
                            aggressive_split,
                        ) {
                            Ok((pos, lock)) => (pos, lock),
                            Err(TreeError::Locked) => {
                                backoff.spin();
                                continue;
                            }
                            Err(TreeError::CircularBufferFull) => {
                                // We can't call eviction here because we are holding a lock, which may happened to be evicted!
                                // It is safe bc circular buffer full is caused by promoting to full page, which is a performance concern not correctness.
                                //
                                aggressive_split = true;
                                continue;
                            }
                            Err(TreeError::NeedRestart) => {
                                aggressive_split = true;
                                backoff.spin();
                                continue;
                            }
                        };
                        self.scan_position = pos;
                        self.leaf_lock = lock;
                        break;
                    }
                    continue;
                }
                GetScanRecordByPosResult::BoundKeyExceeded => {
                    self.scan_cnt = 0;
                    return None;
                }
            }
        }
    }
}

/// The scan iterator obtained from [BfTree::scan].
pub struct ScanIter<'a, 'b: 'a> {
    tree: &'b BfTree,
    scan_cnt: usize,

    scan_position: ScanPosition,

    leaf_lock: ScanLock<'a>,

    return_field: ScanReturnField,

    end_key: Option<Vec<u8>>,

    next_key: Option<Vec<u8>>,
}

impl<'b> ScanIter<'_, 'b> {
    pub fn new_with_scan_count(
        tree: &'b BfTree,
        start_key: &[u8],
        scan_cnt: usize,
        return_field: ScanReturnField,
    ) -> Self {
        let backoff = Backoff::new();
        let mut aggressive_split = false;

        let mut next_key = if tree.cache_only {
            Some(Vec::new())
        } else {
            None
        };

        loop {
            let (scan_pos, lock) =
                match move_cursor_to_leaf(tree, start_key, aggressive_split, next_key.as_mut()) {
                    Ok((pos, lock)) => (pos, lock),
                    Err(TreeError::Locked) => {
                        backoff.spin();
                        continue;
                    }
                    Err(TreeError::NeedRestart) => {
                        aggressive_split = true;
                        backoff.spin();
                        continue;
                    }
                    Err(TreeError::CircularBufferFull) => {
                        _ = tree.evict_from_circular_buffer();
                        aggressive_split = true;
                        continue;
                    }
                };

            return Self {
                tree,
                scan_cnt,
                scan_position: scan_pos,
                leaf_lock: lock,
                return_field,
                end_key: None,
                next_key,
            };
        }
    }

    pub fn new_with_end_key(
        tree: &'b BfTree,
        start_key: &[u8],
        end_key: &[u8],
        return_field: ScanReturnField,
    ) -> Self {
        let mut si = Self::new_with_scan_count(tree, start_key, usize::MAX, return_field);
        si.end_key = Some(end_key.to_vec());
        si
    }

    // Here we need to busy loop? Is that safe?
    /// Scan next value into `out_buffer`.
    /// next() terminates if 1) reached the last key. 2) scanned `scan_cnt` records, if set. 3) reached end_key, if set.
    /// Returns the length of the record fields copied into `out_buffer` or None if there is no more value.
    pub fn next(&mut self, out_buffer: &mut [u8]) -> Option<(usize, usize)> {
        loop {
            if self.scan_cnt == 0 && self.end_key.is_none() {
                return None;
            }

            match self.leaf_lock.get_record_by_pos_with_bound(
                &self.scan_position,
                out_buffer,
                self.return_field,
                &self.end_key,
            ) {
                GetScanRecordByPosResult::Deleted => {
                    self.scan_position.move_to_next();
                    continue;
                }
                GetScanRecordByPosResult::Found(key_len, value_len) => {
                    self.scan_position.move_to_next();
                    self.scan_cnt -= 1;
                    return Some((key_len as usize, value_len as usize));
                }
                GetScanRecordByPosResult::EndOfLeaf => {
                    // we need to load next leaf.
                    counter!(ScanGoNextLeaf);
                    let right_sibling = if !self.tree.cache_only {
                        self.leaf_lock.get_right_sibling()
                    } else {
                        if let Some(key) = self.next_key.as_ref() {
                            key.clone()
                        } else {
                            panic!("next_key is None in cache_only mode");
                        }
                    };

                    if right_sibling.is_empty() {
                        self.scan_cnt = 0;
                        return None;
                    }

                    if self
                        .end_key
                        .as_ref()
                        .is_some_and(|end| &right_sibling > end)
                    {
                        self.scan_cnt = 0;
                        return None;
                    }

                    let backoff = Backoff::new();

                    let mut aggressive_split = false;
                    loop {
                        let (pos, lock) = match move_cursor_to_leaf(
                            self.tree,
                            &right_sibling,
                            aggressive_split,
                            self.next_key.as_mut(),
                        ) {
                            Ok((pos, lock)) => (pos, lock),
                            Err(TreeError::Locked) => {
                                backoff.spin();
                                continue;
                            }
                            Err(TreeError::CircularBufferFull) => {
                                // We can't call eviction here because we are holding a lock, which may happened to be evicted!
                                // It is safe bc circular buffer full is caused by promoting to full page, which is a performance concern not correctness.
                                //
                                // Should we consider making the below function to be unsafe? To arise the awareness?
                                // _ = self.tree.evict_from_circular_buffer();
                                aggressive_split = true;
                                continue;
                            }
                            Err(TreeError::NeedRestart) => {
                                aggressive_split = true;
                                backoff.spin();
                                continue;
                            }
                        };
                        self.scan_position = pos;
                        self.leaf_lock = lock;
                        break;
                    }
                    continue;
                }
                GetScanRecordByPosResult::BoundKeyExceeded => {
                    self.scan_cnt = 0;
                    return None;
                }
            }
        }
    }
}

fn promote_or_merge_mini_page<'a>(
    tree: &'a BfTree,
    key: &[u8],
    leaf: &mut LeafEntryXLocked<'a>,
    parent: ReadGuard<'a>,
) -> Result<ScanPosition, TreeError> {
    let page_loc = leaf.get_page_location();
    match page_loc {
        PageLocation::Full(_) => {
            unreachable!()
        }
        PageLocation::Base(offset) => {
            counter!(ScanPromoteBaseToFull);
            // upgrade this page to full page.
            let next_level = MiniPageNextLevel::new(*offset);
            let base_page_ref = leaf.load_base_page_from_buffer();
            let pos = base_page_ref.lower_bound(key);

            // Upgrade only if not empty
            if base_page_ref.meta.meta_count_without_fence() > 0 {
                let full_page_loc =
                    match upgrade_to_full_page(&tree.storage, base_page_ref, next_level) {
                        Ok(location) => location,
                        Err(TreeError::CircularBufferFull) => {
                            return Ok(ScanPosition::Base(pos as u32))
                        }
                        Err(error) => return Err(error),
                    };

                leaf.create_cache_page_loc(full_page_loc);

                Ok(ScanPosition::Full(pos as u32))
            } else {
                Ok(ScanPosition::Base(pos as u32))
            }
        }
        PageLocation::Mini(ptr) => {
            counter!(ScanMergeMiniPage);
            let mini_page = leaf.load_cache_page_mut(*ptr);
            // acquire the handle so that the eviction process with not contend with us.
            let h = tree.storage.begin_dealloc_mini_page(mini_page)?;
            let merge_result = leaf.try_merge_mini_page(&h, parent, &tree.storage)?;

            match merge_result {
                MergeResult::NoSplit => {
                    // A clean mini-page can skip merging without loading the base.
                    leaf.load_base_page(mini_page.next_level.as_offset());
                    // if no split, we face two choices:
                    // (1) keep using the merged base page
                    // (2) upgrade this page to full page, so that future scans don't need to load base page.
                    //      this is done with probability to avoid polluting the cache.
                    if tree.should_promote_scan_page() {
                        // upgrade to full page
                        let base_offset = mini_page.next_level;
                        leaf.change_to_base_loc();
                        tree.storage.finish_dealloc_mini_page(h);

                        let base_page_ref = leaf.load_base_page_from_buffer();
                        let pos = base_page_ref.lower_bound(key);
                        if base_page_ref.meta.meta_count_without_fence() > 0 {
                            let full_page_loc = match upgrade_to_full_page(
                                &tree.storage,
                                base_page_ref,
                                base_offset,
                            ) {
                                Ok(location) => location,
                                Err(TreeError::CircularBufferFull) => {
                                    return Ok(ScanPosition::Base(pos as u32))
                                }
                                Err(error) => return Err(error),
                            };

                            leaf.create_cache_page_loc(full_page_loc);
                            Ok(ScanPosition::Full(pos as u32))
                        } else {
                            Ok(ScanPosition::Base(pos as u32))
                        }
                    } else {
                        leaf.change_to_base_loc();
                        tree.storage.finish_dealloc_mini_page(h);
                        let base_ref = leaf.load_base_page_from_buffer();
                        let pos = base_ref.lower_bound(key);
                        Ok(ScanPosition::Base(pos as u32))
                    }
                }
                MergeResult::MergeAndSplit => {
                    // if split happens, the mini page contains records that does not belong to us, we need to drop it.
                    leaf.change_to_base_loc();
                    tree.storage.finish_dealloc_mini_page(h);

                    // we need to restart traverse to leaf, because merging splitted the base,
                    // which may cause us to land on the wrong leaf.
                    // retry on this might cause unnecessary IO (dropped the base), but it's rare.
                    Err(TreeError::NeedRestart)
                }
            }
        }
        PageLocation::Null => panic!("promote_or_merge_mini_page on Null page"),
    }
}

fn move_cursor_to_leaf_mut<'a>(
    tree: &'a BfTree,
    key: &[u8],
    aggressive_split: bool,
) -> Result<(ScanPosition, LeafEntryXLocked<'a>), TreeError> {
    let (pid, parent) = tree.traverse_to_leaf(key, aggressive_split, false, None)?;

    let mut leaf = tree.mapping_table().get_mut(&pid);

    check_parent!(tree, pid, parent);

    if let Ok(pos) = leaf.get_scan_position(key, false) {
        match pos {
            ScanPosition::Base(_) => {
                if parent.is_none() || !tree.should_promote_scan_page() {
                    return Ok((pos, leaf));
                }
                // o.w. fall through and upgrade to full page.
            }
            ScanPosition::Full(_) => {
                return Ok((pos, leaf));
            }
            _ => {
                panic!("Scanning mini or null page in ScanIterMut");
            }
        }
    }

    // we need to merge mini page.

    let v = promote_or_merge_mini_page(tree, key, &mut leaf, parent.unwrap())?;
    Ok((v, leaf))
}

fn move_cursor_to_leaf<'a>(
    tree: &'a BfTree,
    key: &[u8],
    aggressive_split: bool,
    mut next_key: Option<&mut Vec<u8>>,
) -> Result<(ScanPosition, ScanLock<'a>), TreeError> {
    let (pid, parent) = if tree.cache_only {
        if let Some(key) = next_key.as_deref_mut() {
            key.clear();
        } else {
            panic!("next_key is None in cache_only mode");
        }
        tree.traverse_to_leaf(key, aggressive_split, true, next_key)?
    } else {
        tree.traverse_to_leaf(key, aggressive_split, false, None)?
    };

    let mut leaf = tree.mapping_table().get(&pid);

    check_parent!(tree, pid, parent);

    let mut base_position = None;
    if let Ok(pos) = leaf.get_scan_position(key, tree.cache_only) {
        match pos {
            ScanPosition::Base(_) => {
                counter!(ScanBasePage);
                if parent.is_none() || !tree.should_promote_scan_page() {
                    return Ok((pos, ScanLock::S(leaf)));
                }
                base_position = Some(pos);
            }
            ScanPosition::Full(_) => {
                counter!(ScanFullPage);
                return Ok((pos, ScanLock::S(leaf)));
            }
            ScanPosition::Mini(_) => {
                assert!(tree.cache_only);
                counter!(ScanMiniPage);
                return Ok((pos, ScanLock::S(leaf)));
            }
            ScanPosition::Null => {
                counter!(ScanNullPage);
                return Ok((pos, ScanLock::S(leaf)));
            }
        }
    }

    // we need to merge mini page.
    let mut x_leaf = match leaf.try_upgrade(tree.snapshot_mgr.clone(), pid) {
        Ok(leaf) => leaf,
        Err(leaf) => {
            // Optional promotion must not force competing readers to reload the page.
            if let Some(pos) = base_position {
                return Ok((pos, ScanLock::S(leaf)));
            }
            // Queue for a required merge rather than repeatedly racing other readers.
            drop(leaf);
            return move_cursor_to_leaf_mut(tree, key, aggressive_split)
                .map(|(pos, leaf)| (pos, ScanLock::X(leaf)));
        }
    };

    let v = promote_or_merge_mini_page(tree, key, &mut x_leaf, parent.unwrap())?;
    Ok((v, ScanLock::X(x_leaf)))
}

#[cfg(test)]
mod tests {
    use crate::utils::test_util::install_value_to_buffer;
    use crate::{BfTree, Config};
    use crate::{LeafInsertResult, ScanReturnField};
    use std::mem::size_of;

    #[cfg(not(feature = "shuttle"))]
    mod regressions {
        use super::*;
        use crate::mini_page_op::LeafOperations;
        use crate::range_scan::{move_cursor_to_leaf, ScanLock, ScanPosition};
        use crate::storage::PageLocation;

        fn fixture() -> BfTree {
            let mut config = Config::default();
            config
                .cb_size_byte(1024 * 1024)
                .cb_min_record_size(16)
                .cb_max_record_size(256)
                .cb_max_key_len(16)
                .scan_promotion_rate(100)
                .read_promotion_rate(100);
            let tree = BfTree::with_config(config, None).unwrap();
            for id in 0..300u64 {
                assert_eq!(
                    tree.insert(&id.to_be_bytes(), &[1; 16]),
                    LeafInsertResult::Success
                );
            }
            tree.storage.circular_buffer.drain(|handle| {
                if crate::tree::eviction_callback(&handle, &tree).is_ok() {
                    Ok(handle)
                } else {
                    Err(handle)
                }
            });
            tree
        }

        #[test]
        fn mutable_root_scan_with_promotion() {
            let mut config = Config::default();
            config.scan_promotion_rate(100);
            let tree = BfTree::with_config(config, None).unwrap();
            let key = 0u64.to_be_bytes();
            assert_eq!(tree.insert(&key, &[7; 64]), LeafInsertResult::Success);
            let mut scan = tree
                .scan_mut_with_count(&key, 2, ScanReturnField::Value)
                .unwrap();
            let mut out = [0; 64];
            assert_eq!(scan.next(&mut out), Some((0, 64)));
            assert_eq!(out, [7; 64]);
            assert_eq!(scan.next(&mut out), None);
        }

        #[test]
        fn shared_base_scan_skips_optional_promotion() {
            let tree = fixture();
            let key = 150u64.to_be_bytes();
            let (pid, parent) = tree.traverse_to_leaf(&key, false, false, None).unwrap();
            assert!(parent.is_some());
            let held = tree.mapping_table().get(&pid);
            assert!(matches!(held.get_page_location(), PageLocation::Base(_)));
            assert!(matches!(
                move_cursor_to_leaf(&tree, &key, false, None),
                Ok((ScanPosition::Base(_), ScanLock::S(_)))
            ));
            drop(held);
            assert!(matches!(
                move_cursor_to_leaf(&tree, &key, false, None),
                Ok((ScanPosition::Full(_), ScanLock::X(_)))
            ));
        }

        #[test]
        fn scan_merge_shared_mini_waits_for_exclusive_lock() {
            use crate::sync::thread;
            use std::time::{Duration, Instant};

            let tree = fixture();
            let key = 150u64.to_be_bytes();
            assert_eq!(tree.insert(&key, &[2; 16]), LeafInsertResult::Success);
            let (pid, _) = tree.traverse_to_leaf(&key, false, false, None).unwrap();
            let lock = tree
                .mapping_table()
                .iter()
                .find(|(_, id)| *id == pid)
                .unwrap()
                .0;
            let held = tree.mapping_table().get(&pid);
            assert!(matches!(held.get_page_location(), PageLocation::Mini(_)));
            let (queued, merged) = thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    matches!(
                        move_cursor_to_leaf(&tree, &key, false, None),
                        Ok((ScanPosition::Full(_), ScanLock::X(_)))
                    )
                });
                let deadline = Instant::now() + Duration::from_secs(10);
                while !lock.has_writer_intent()
                    && !worker.is_finished()
                    && Instant::now() < deadline
                {
                    thread::yield_now();
                }
                let queued = lock.has_writer_intent();
                drop(held);
                (queued, worker.join().unwrap())
            });
            assert!(queued, "required merge must queue for exclusive access");
            assert!(merged);
            let mut value = [0; 16];
            assert_eq!(
                tree.read(&key, &mut value),
                crate::LeafReadResult::Found(16)
            );
            assert_eq!(value, [2; 16]);
        }

        #[test]
        fn scan_promotion_empty_base_reclaims_mini() {
            use crate::nodes::leaf_node::OpType;
            let tree = fixture();
            let key = 150u64.to_be_bytes();
            assert_eq!(
                tree.read(&key, &mut [0; 16]),
                crate::LeafReadResult::Found(16)
            );
            let (pid, parent) = tree.traverse_to_leaf(&key, false, false, None).unwrap();
            let mut entry = tree.mapping_table().get_mut(&pid);
            let PageLocation::Mini(ptr) = *entry.get_page_location() else {
                panic!("read promotion must create a mini-page");
            };
            let base_offset = entry.load_cache_page(ptr).next_level.as_offset();
            let size = entry.load_cache_page(ptr).meta.node_size as usize;
            let base = entry.load_base_page(base_offset);
            let keys: Vec<_> = base
                .meta_iter()
                .map(|meta| base.get_full_key(meta))
                .collect();
            let base = entry.load_base_page_mut();
            for key in &keys {
                assert!(base.insert(key, &[], OpType::Delete, tree.config.max_fence_len));
            }
            base.consolidate_inner(
                OpType::Insert,
                None,
                true,
                false,
                None,
                base.get_clean_snapshot_version(),
            );
            assert_eq!(base.meta.meta_count_without_fence(), 0);
            assert!(matches!(
                super::super::promote_or_merge_mini_page(&tree, &key, &mut entry, parent.unwrap()),
                Ok(ScanPosition::Base(_))
            ));
            assert!(matches!(entry.get_page_location(), PageLocation::Base(_)));
            drop(entry);
            let reused = tree.storage.circular_buffer.alloc(size).unwrap();
            assert_eq!(reused.as_ptr(), ptr.cast());
            let ptr = reused.as_ptr();
            drop(reused);
            tree.storage.circular_buffer.dealloc(unsafe {
                tree.storage
                    .circular_buffer
                    .acquire_exclusive_dealloc_handle(ptr)
                    .unwrap()
            });
            assert!(matches!(
                tree.read(&key, &mut [0; 16]),
                crate::LeafReadResult::NotFound | crate::LeafReadResult::Deleted
            ));
        }

        #[rstest::rstest]
        fn bounded_scan_does_not_lock_a_successor_outside_the_range(
            #[values(false, true)] mutable: bool,
        ) {
            use crate::sync::{mpsc, thread};
            use std::time::Duration;
            let tree = fixture();
            let start = 0u64.to_be_bytes();
            let (first, _) = tree.traverse_to_leaf(&start, false, false, None).unwrap();
            let (end, successor, count) = {
                let mut leaf = tree.mapping_table().get(&first);
                let PageLocation::Base(offset) = *leaf.get_page_location() else {
                    panic!("expected drained base")
                };
                let base = leaf.load_base_page(offset);
                (
                    base.get_full_key(base.meta_iter().last().unwrap()),
                    base.get_high_fence_key(),
                    base.meta.meta_count_without_fence() as usize,
                )
            };
            let (next, _) = tree
                .traverse_to_leaf(&successor, false, false, None)
                .unwrap();
            assert_ne!(first, next);
            let held = tree.mapping_table().get_mut(&next);
            let (sent, received) = mpsc::channel();
            let finished_before_unlock = thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    let mut out = [0; 8];
                    let mut seen = 0;
                    if mutable {
                        let mut iter = tree
                            .scan_mut_with_end_key(&start, &end, ScanReturnField::Key)
                            .unwrap();
                        while iter.next(&mut out).is_some() {
                            seen += 1;
                        }
                    } else {
                        let mut iter = tree
                            .scan_with_end_key(&start, &end, ScanReturnField::Key)
                            .unwrap();
                        while iter.next(&mut out).is_some() {
                            seen += 1;
                        }
                    }
                    assert_eq!(seen, count);
                    sent.send(()).unwrap();
                });
                let finished = received.recv_timeout(Duration::from_secs(5)).is_ok();
                drop(held);
                worker.join().unwrap();
                finished
            });
            assert!(
                finished_before_unlock,
                "bounded scan waited on an out-of-range leaf"
            );
        }

        #[rstest::rstest]
        fn scan_with_full_buffer(#[values(false, true)] clean: bool) {
            let tree = fixture();
            let key = 150u64.to_be_bytes();
            if clean {
                assert_eq!(
                    tree.read(&key, &mut [0; 16]),
                    crate::LeafReadResult::Found(16)
                );
            } else {
                assert_eq!(tree.insert(&key, &[2; 16]), LeafInsertResult::Success);
            }
            let (pid, _) = tree.traverse_to_leaf(&key, false, false, None).unwrap();
            assert!(matches!(
                tree.mapping_table().get(&pid).get_page_location(),
                PageLocation::Mini(_)
            ));

            let buffer = &tree.storage.circular_buffer;
            let mut held = Vec::new();
            while let Ok(frame) = buffer.alloc(tree.config.leaf_page_size) {
                held.push(frame.as_ptr());
                drop(frame);
            }
            assert!(!held.is_empty());
            // Check the fallible helper before the iterator, so a regression fails rather than loops.
            assert!(matches!(
                move_cursor_to_leaf(&tree, &key, false, None),
                Ok((ScanPosition::Base(_), ScanLock::X(_)))
            ));
            let start = 0u64.to_be_bytes();
            fn verify(clean: bool, mut next: impl FnMut(&mut [u8]) -> Option<(usize, usize)>) {
                let mut out = [0; 24];
                for id in 0..300u64 {
                    assert_eq!(next(&mut out), Some((8, 16)));
                    assert_eq!(&out[..8], &id.to_be_bytes());
                    assert_eq!(&out[8..], &[if id == 150 && !clean { 2 } else { 1 }; 16]);
                }
                assert_eq!(next(&mut out), None);
            }
            {
                let mut scan = tree
                    .scan_with_count(&start, 301, ScanReturnField::KeyAndValue)
                    .unwrap();
                verify(clean, |out| scan.next(out));
            }
            {
                let mut scan = tree
                    .scan_mut_with_count(&start, 301, ScanReturnField::KeyAndValue)
                    .unwrap();
                verify(clean, |out| scan.next(out));
            }
            for ptr in held {
                buffer.dealloc(unsafe { buffer.acquire_exclusive_dealloc_handle(ptr).unwrap() });
            }
            assert!(matches!(
                move_cursor_to_leaf(&tree, &key, false, None),
                Ok((ScanPosition::Full(_), ScanLock::X(_)))
            ));
        }

        #[test]
        fn scans_skip_long_deleted_prefix() {
            let mut config = Config::default();
            config.cb_min_record_size(16).scan_promotion_rate(0);
            let tree = BfTree::with_config(config, None).unwrap();
            for id in 0..20_000u64 {
                assert_eq!(
                    tree.insert(&id.to_be_bytes(), &[1; 64]),
                    LeafInsertResult::Success
                );
            }
            for id in 0..19_995u64 {
                tree.delete(&id.to_be_bytes());
            }
            let start = 0u64.to_be_bytes();
            let mut out = [0; 8];
            {
                let mut scan = tree
                    .scan_with_count(&start, 20_000, ScanReturnField::Key)
                    .unwrap();
                for id in 19_995..20_000u64 {
                    assert_eq!(scan.next(&mut out), Some((8, 0)));
                    assert_eq!(out, id.to_be_bytes());
                }
                assert_eq!(scan.next(&mut out), None);
            }
            let mut scan = tree
                .scan_mut_with_count(&start, 20_000, ScanReturnField::Key)
                .unwrap();
            for id in 19_995..20_000u64 {
                assert_eq!(scan.next(&mut out), Some((8, 0)));
                assert_eq!(out, id.to_be_bytes());
            }
            assert_eq!(scan.next(&mut out), None);
        }
    }

    #[test]
    fn test_scan_with_count() {
        let tree = BfTree::default();

        // Insert 1000 consecutive keys
        let key_len: usize = tree.config.max_fence_len / 2;
        let mut key_buffer = vec![0; key_len / size_of::<usize>()];

        let value_len: usize = 1024; // 1KB long values
        let mut value_buffer = vec![0; value_len / size_of::<usize>()];

        for i in 0..1_000 {
            let key = install_value_to_buffer(&mut key_buffer, i);
            let value = install_value_to_buffer(&mut value_buffer, i);
            if tree.insert(key, value) != LeafInsertResult::Success {
                panic!("Insert failed");
            }
        }

        // Scan with invalid count
        let mut start_key = install_value_to_buffer(&mut key_buffer, 0);
        let r = tree.scan_with_count(start_key, 0, ScanReturnField::Value);
        match r {
            Err(e) => {
                assert_eq!(e, crate::ScanIterError::InvalidCount);
            }
            _ => {
                panic!("Should not succeed");
            }
        }

        // Scan 100 at a time for 9 times
        let mut output_buffer = vec![0u8; key_len + value_len];
        let mut prev_key = vec![0u8; key_len];
        for _ in 0..9 {
            start_key = &prev_key.as_slice().as_ref();
            let mut scan_iter = tree
                .scan_with_count(start_key, 101, ScanReturnField::KeyAndValue)
                .expect("Scan failed");

            let mut cnt = 0;
            while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
                let scanned_key = &output_buffer[0..kl];
                assert!(kl == key_len);

                if cnt != 0 {
                    let cmp_res = scanned_key.cmp(&prev_key);
                    if cmp_res == std::cmp::Ordering::Less {
                        panic!("Keys are not in order");
                    }
                    assert_eq!(cmp_res, std::cmp::Ordering::Greater);
                }

                prev_key[..kl].copy_from_slice(scanned_key);

                assert!(vl == value_len);
                cnt += 1;
            }
            assert!(cnt == 101);
        }

        // Scan 120 for the last 100 keys
        start_key = &prev_key.as_slice().as_ref();
        let mut scan_iter = tree
            .scan_with_count(start_key, 120, ScanReturnField::Key)
            .expect("Scan failed");
        let mut cnt = 0;

        while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
            let scanned_key = &output_buffer[0..kl];
            assert!(kl == key_len);

            if cnt != 0 {
                let cmp_res = scanned_key.cmp(&prev_key);
                assert_eq!(cmp_res, std::cmp::Ordering::Greater);
            }
            prev_key[..kl].copy_from_slice(scanned_key);
            assert!(vl == 0);
            cnt += 1;
        }
        assert!(cnt == 100);
    }

    #[test]
    fn test_scan_with_end_key() {
        let tree = BfTree::default();

        // Insert 1000 consecutive keys
        let key_len: usize = tree.config.max_fence_len / 2;
        let mut key_buffer = vec![0; key_len / size_of::<usize>()];

        let value_len: usize = 1024; // 1KB long values
        let mut value_buffer = vec![0; value_len / size_of::<usize>()];

        for i in 0..1_000 {
            let key = install_value_to_buffer(&mut key_buffer, i);
            let value = install_value_to_buffer(&mut value_buffer, i);
            if tree.insert(key, value) != LeafInsertResult::Success {
                panic!("Insert failed");
            }
        }

        // Scan with invalid keys
        let mut start_key = install_value_to_buffer(&mut key_buffer, 1);
        let mut invalid_key_buffer: Vec<usize> = vec![0; key_len / size_of::<usize>() + 1];
        let mut invalid_key = install_value_to_buffer(&mut invalid_key_buffer, 1);

        let mut r = tree.scan_with_end_key(start_key, invalid_key, ScanReturnField::Value);
        match r {
            Err(e) => {
                assert_eq!(e, crate::ScanIterError::InvalidEndKey);
            }
            _ => {
                panic!("Should not succeed");
            }
        }

        invalid_key = install_value_to_buffer(&mut invalid_key_buffer, 0);

        r = tree.scan_with_end_key(invalid_key, start_key, ScanReturnField::Value);
        match r {
            Err(e) => {
                assert_eq!(e, crate::ScanIterError::InvalidStartKey);
            }
            _ => {
                panic!("Should not succeed");
            }
        }

        let mut end_key_buffer = vec![0; key_len / size_of::<usize>()];
        let mut end_key = install_value_to_buffer(&mut end_key_buffer, 0);

        r = tree.scan_with_end_key(start_key, end_key, ScanReturnField::Value);
        match r {
            Err(e) => {
                assert_eq!(e, crate::ScanIterError::InvalidKeyRange);
            }
            _ => {
                panic!("Should not succeed");
            }
        }

        start_key = install_value_to_buffer(&mut key_buffer, 0);
        end_key = install_value_to_buffer(&mut end_key_buffer, 777);

        let mut scan_iter = tree
            .scan_with_end_key(start_key, end_key, ScanReturnField::Key)
            .expect("Scan failed");
        let mut output_buffer = vec![0u8; key_len];
        let mut prev_key = vec![0u8; key_len];
        let mut cnt = 0;

        while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
            let scanned_key = &output_buffer[0..kl];
            assert!(kl == key_len);

            if cnt != 0 {
                let cmp_res = scanned_key.cmp(&prev_key);
                assert_eq!(cmp_res, std::cmp::Ordering::Greater);
            }
            prev_key[..kl].copy_from_slice(scanned_key);

            let cmp_res = scanned_key.cmp(end_key);
            assert!(cmp_res == std::cmp::Ordering::Less || cmp_res == std::cmp::Ordering::Equal);

            assert!(vl == 0);
            cnt += 1;
        }

        let cmp_res = prev_key.as_slice().cmp(end_key);
        assert!(cmp_res == std::cmp::Ordering::Equal);
        assert!(cnt == 40);
    }

    fn cache_only_tree(cb_size_byte: Option<usize>) -> BfTree {
        let mut config = Config::default();
        config.cache_only(true);
        if let Some(size) = cb_size_byte {
            config.cb_size_byte(size);
        }
        BfTree::with_config(config, None).expect("Failed to create cache-only BfTree")
    }

    #[test]
    fn test_scan_with_count_cache_only() {
        let tree = cache_only_tree(None);
        assert!(tree.cache_only);

        // Insert 1000 consecutive keys
        let key_len: usize = tree.config.max_fence_len / 2;
        let mut key_buffer = vec![0; key_len / size_of::<usize>()];

        let value_len: usize = 1024; // 1KB long values
        let mut value_buffer = vec![0; value_len / size_of::<usize>()];

        for i in 0..1_000 {
            let key = install_value_to_buffer(&mut key_buffer, i);
            let value = install_value_to_buffer(&mut value_buffer, i);
            if tree.insert(key, value) != LeafInsertResult::Success {
                panic!("Insert failed");
            }
        }

        // Scan with invalid count
        let mut start_key = install_value_to_buffer(&mut key_buffer, 0);
        let r = tree.scan_with_count(start_key, 0, ScanReturnField::Value);
        match r {
            Err(e) => {
                assert_eq!(e, crate::ScanIterError::InvalidCount);
            }
            _ => {
                panic!("Should not succeed");
            }
        }

        // Scan 100 at a time for 9 times
        let mut output_buffer = vec![0u8; key_len + value_len];
        let mut prev_key = vec![0u8; key_len];
        for _ in 0..9 {
            start_key = &prev_key.as_slice().as_ref();
            let mut scan_iter = tree
                .scan_with_count(start_key, 101, ScanReturnField::KeyAndValue)
                .expect("Scan failed");

            let mut cnt = 0;
            while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
                let scanned_key = &output_buffer[0..kl];
                assert!(kl == key_len);

                if cnt != 0 {
                    let cmp_res = scanned_key.cmp(&prev_key);
                    if cmp_res == std::cmp::Ordering::Less {
                        panic!("Keys are not in order");
                    }
                    assert_eq!(cmp_res, std::cmp::Ordering::Greater);
                }

                prev_key[..kl].copy_from_slice(scanned_key);

                assert!(vl == value_len);
                cnt += 1;
            }
            assert!(cnt == 101);
        }

        // Scan 120 for the last 100 keys
        start_key = &prev_key.as_slice().as_ref();
        let mut scan_iter = tree
            .scan_with_count(start_key, 120, ScanReturnField::Key)
            .expect("Scan failed");
        let mut cnt = 0;

        while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
            let scanned_key = &output_buffer[0..kl];
            assert!(kl == key_len);

            if cnt != 0 {
                let cmp_res = scanned_key.cmp(&prev_key);
                assert_eq!(cmp_res, std::cmp::Ordering::Greater);
            }
            prev_key[..kl].copy_from_slice(scanned_key);
            assert!(vl == 0);
            cnt += 1;
        }
        assert!(cnt == 100);
    }

    #[test]
    fn test_scan_with_end_key_cache_only() {
        let tree = cache_only_tree(None);
        assert!(tree.cache_only);

        // Insert 1000 consecutive keys
        let key_len: usize = tree.config.max_fence_len / 2;
        let mut key_buffer = vec![0; key_len / size_of::<usize>()];

        let value_len: usize = 1024; // 1KB long values
        let mut value_buffer = vec![0; value_len / size_of::<usize>()];

        for i in 0..1_000 {
            let key = install_value_to_buffer(&mut key_buffer, i);
            let value = install_value_to_buffer(&mut value_buffer, i);
            if tree.insert(key, value) != LeafInsertResult::Success {
                panic!("Insert failed");
            }
        }

        // Scan with invalid keys
        let start_key = install_value_to_buffer(&mut key_buffer, 1);
        let mut invalid_key_buffer: Vec<usize> = vec![0; key_len / size_of::<usize>() + 1];
        let mut invalid_key = install_value_to_buffer(&mut invalid_key_buffer, 1);

        let mut r = tree.scan_with_end_key(start_key, invalid_key, ScanReturnField::Value);
        match r {
            Err(e) => {
                assert_eq!(e, crate::ScanIterError::InvalidEndKey);
            }
            _ => {
                panic!("Should not succeed");
            }
        }

        invalid_key = install_value_to_buffer(&mut invalid_key_buffer, 0);

        r = tree.scan_with_end_key(invalid_key, start_key, ScanReturnField::Value);
        match r {
            Err(e) => {
                assert_eq!(e, crate::ScanIterError::InvalidStartKey);
            }
            _ => {
                panic!("Should not succeed");
            }
        }

        let mut end_key_buffer = vec![0; key_len / size_of::<usize>()];
        let mut end_key = install_value_to_buffer(&mut end_key_buffer, 0);

        r = tree.scan_with_end_key(start_key, end_key, ScanReturnField::Value);
        match r {
            Err(e) => {
                assert_eq!(e, crate::ScanIterError::InvalidKeyRange);
            }
            _ => {
                panic!("Should not succeed");
            }
        }

        let start_key = install_value_to_buffer(&mut key_buffer, 0);
        end_key = install_value_to_buffer(&mut end_key_buffer, 777);

        let mut scan_iter = tree
            .scan_with_end_key(start_key, end_key, ScanReturnField::Key)
            .expect("Scan failed");
        let mut output_buffer = vec![0u8; key_len];
        let mut prev_key = vec![0u8; key_len];
        let mut cnt = 0;

        while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
            let scanned_key = &output_buffer[0..kl];
            assert!(kl == key_len);

            if cnt != 0 {
                let cmp_res = scanned_key.cmp(&prev_key);
                assert_eq!(cmp_res, std::cmp::Ordering::Greater);
            }
            prev_key[..kl].copy_from_slice(scanned_key);

            let cmp_res = scanned_key.cmp(end_key);
            assert!(cmp_res == std::cmp::Ordering::Less || cmp_res == std::cmp::Ordering::Equal);

            assert!(vl == 0);
            cnt += 1;
        }

        let cmp_res = prev_key.as_slice().cmp(end_key);
        assert!(cmp_res == std::cmp::Ordering::Equal);
        assert!(cnt == 40);
    }

    #[test]
    fn test_scan_mut_disallowed_in_cache_only() {
        let tree = cache_only_tree(None);
        assert!(tree.cache_only);

        let key_len: usize = tree.config.max_fence_len / 2;
        let mut key_buffer = vec![0; key_len / size_of::<usize>()];
        let start_key = install_value_to_buffer(&mut key_buffer, 0);

        let mut end_key_buffer = vec![0; key_len / size_of::<usize>()];
        let end_key = install_value_to_buffer(&mut end_key_buffer, 1);

        let r = tree.scan_mut_with_count(start_key, 10, ScanReturnField::Value);
        match r {
            Err(e) => assert_eq!(e, crate::ScanIterError::CacheOnlyMode),
            _ => panic!("scan_mut_with_count should be disallowed in cache-only mode"),
        }

        let r = tree.scan_mut_with_end_key(start_key, end_key, ScanReturnField::Value);
        match r {
            Err(e) => assert_eq!(e, crate::ScanIterError::CacheOnlyMode),
            _ => panic!("scan_mut_with_end_key should be disallowed in cache-only mode"),
        }
    }

    /// A small circular buffer (32 KiB cache, 4 KiB leaf pages -> only a
    /// handful of leaves fit) so that inserting the test workload forces
    /// evictions (Null pages) during scan.
    const SMALL_CB_SIZE: usize = 32 * 1024;

    #[test]
    fn test_scan_with_count_cache_only_lossy() {
        let tree = cache_only_tree(Some(SMALL_CB_SIZE));
        assert!(tree.cache_only);

        let key_len: usize = tree.config.max_fence_len / 2;
        let mut key_buffer = vec![0; key_len / size_of::<usize>()];

        // Use ~1 KiB values so the workload (10K records ~ 10 MiB) is
        // dramatically larger than the 32 KiB circular buffer; evictions
        // are guaranteed.
        let value_len: usize = 1024;
        let mut value_buffer = vec![0; value_len / size_of::<usize>()];

        let total: usize = 10_000;
        for i in 0..total {
            let key = install_value_to_buffer(&mut key_buffer, i);
            let value = install_value_to_buffer(&mut value_buffer, i);
            if tree.insert(key, value) != LeafInsertResult::Success {
                panic!("Insert failed");
            }
        }

        // Scan everything from the smallest key.
        let start_key = install_value_to_buffer(&mut key_buffer, 0);
        let mut scan_iter = tree
            .scan_with_count(start_key, usize::MAX, ScanReturnField::Key)
            .expect("Scan failed");

        let mut output_buffer = vec![0u8; key_len + value_len];
        let mut prev_key = vec![0u8; key_len];
        let mut cnt: usize = 0;
        while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
            assert_eq!(kl, key_len);
            assert_eq!(vl, 0); // ScanReturnField::Key

            let scanned_key = &output_buffer[0..kl];
            if cnt != 0 {
                let cmp_res = scanned_key.cmp(&prev_key);
                assert_eq!(
                    cmp_res,
                    std::cmp::Ordering::Greater,
                    "scan must produce strictly increasing keys"
                );
            }
            prev_key[..kl].copy_from_slice(scanned_key);
            cnt += 1;
        }

        // Some pages were evicted, so we should get strictly fewer than total.
        assert!(
            cnt < total,
            "expected evictions to drop some keys, got cnt={} total={}",
            cnt,
            total
        );
        // But the remaining cached data should still produce a non-trivial scan.
        assert!(cnt > 0, "scan returned no records");
    }

    #[test]
    fn test_scan_with_end_key_cache_only_lossy() {
        let tree = cache_only_tree(Some(SMALL_CB_SIZE));
        assert!(tree.cache_only);

        let key_len: usize = tree.config.max_fence_len / 2;
        let mut key_buffer = vec![0; key_len / size_of::<usize>()];

        let value_len: usize = 1024;
        let mut value_buffer = vec![0; value_len / size_of::<usize>()];

        let total: usize = 5_000;
        for i in 0..total {
            let key = install_value_to_buffer(&mut key_buffer, i);
            let value = install_value_to_buffer(&mut value_buffer, i);
            if tree.insert(key, value) != LeafInsertResult::Success {
                panic!("Insert failed");
            }
        }

        // Bounded range scan; bound is well within the inserted key space.
        let start_key = install_value_to_buffer(&mut key_buffer, 0);
        let mut end_key_buffer = vec![0; key_len / size_of::<usize>()];
        let end_key_idx: usize = 2_000;
        let end_key = install_value_to_buffer(&mut end_key_buffer, end_key_idx);

        let mut scan_iter = tree
            .scan_with_end_key(start_key, end_key, ScanReturnField::Key)
            .expect("Scan failed");

        let mut output_buffer = vec![0u8; key_len];
        let mut prev_key = vec![0u8; key_len];
        let mut cnt: usize = 0;
        while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
            assert_eq!(kl, key_len);
            assert_eq!(vl, 0);

            let scanned_key = &output_buffer[0..kl];
            if cnt != 0 {
                let cmp_res = scanned_key.cmp(&prev_key);
                assert_eq!(cmp_res, std::cmp::Ordering::Greater);
            }
            prev_key[..kl].copy_from_slice(scanned_key);

            // Every scanned key must be <= end_key.
            let cmp_res = scanned_key.cmp(end_key);
            assert!(cmp_res == std::cmp::Ordering::Less || cmp_res == std::cmp::Ordering::Equal);

            cnt += 1;
        }

        // Even with eviction we expect at least some keys, and we should
        // never see more than the bound implies.
        assert!(
            cnt <= end_key_idx + 1,
            "scan returned more keys ({}) than the upper bound ({})",
            cnt,
            end_key_idx + 1
        );
    }

    #[test]
    fn test_scan_with_value_cache_only_lossy() {
        // Verifies that scanned (key, value) pairs are self-consistent (the
        // returned value matches the encoding of the returned key) even when
        // pages are being evicted, and that scan tolerates Null pages.
        let tree = cache_only_tree(Some(SMALL_CB_SIZE));
        assert!(tree.cache_only);

        let key_len: usize = tree.config.max_fence_len / 2;
        let mut key_buffer = vec![0; key_len / size_of::<usize>()];

        let value_len: usize = 1024;
        let mut value_buffer = vec![0; value_len / size_of::<usize>()];

        let total: usize = 8_000;
        for i in 0..total {
            let key = install_value_to_buffer(&mut key_buffer, i);
            let value = install_value_to_buffer(&mut value_buffer, i);
            if tree.insert(key, value) != LeafInsertResult::Success {
                panic!("Insert failed");
            }
        }

        let start_key = install_value_to_buffer(&mut key_buffer, 0);
        let mut scan_iter = tree
            .scan_with_count(start_key, usize::MAX, ScanReturnField::KeyAndValue)
            .expect("Scan failed");

        let mut output_buffer = vec![0u8; key_len + value_len];
        let mut prev_key = vec![0u8; key_len];
        let mut cnt: usize = 0;

        while let Some((kl, vl)) = scan_iter.next(&mut output_buffer) {
            assert_eq!(kl, key_len);
            assert_eq!(vl, value_len);

            let scanned_key = &output_buffer[0..kl];
            let scanned_value = &output_buffer[kl..kl + vl];

            if cnt != 0 {
                assert_eq!(scanned_key.cmp(&prev_key), std::cmp::Ordering::Greater);
            }
            prev_key[..kl].copy_from_slice(scanned_key);

            // `install_value_to_buffer` fills the entire buffer by repeating
            // `murmur64(id)`; for the same id, the first `key_len` bytes of
            // the value buffer equal the entire key buffer. Use this to
            // verify the (key, value) pair is self-consistent.
            assert_eq!(
                &scanned_value[..key_len],
                scanned_key,
                "scanned value prefix does not match key (key/value out of sync)"
            );
            // The value is also a repetition of the same word: verify every
            // chunk of `key_len` bytes within the value matches the key.
            for chunk in scanned_value.chunks_exact(key_len) {
                assert_eq!(chunk, scanned_key, "value is not a repetition of key");
            }

            cnt += 1;
        }

        assert!(cnt > 0, "scan returned no records");
        assert!(
            cnt < total,
            "expected evictions to drop some keys, got cnt={} total={}",
            cnt,
            total
        );
    }
}
