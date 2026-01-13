use crate::{
    check_parent, counter,
    error::TreeError,
    mini_page_op::{
        upgrade_to_full_page, LeafEntrySLocked, LeafEntryXLocked, LeafOperations, MergeResult,
    },
    nodes::leaf_node::{GetValueByPosResult, MiniPageNextLevel},
    storage::PageLocation,
    utils::{inner_lock::ReadGuard, Backoff},
    BfTree,
};

pub(crate) enum ScanError {
    NeedMergeMiniPage,
}

pub(crate) enum ScanPosition {
    Base(u32),
    Full(u32),
    // can't scan on mini page.
}

impl ScanPosition {
    fn move_to_next(&mut self) {
        match self {
            ScanPosition::Base(offset) => *offset += 1,
            ScanPosition::Full(offset) => *offset += 1,
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
    fn get_value_by_pos(&self, pos: &ScanPosition, out_buffer: &mut [u8]) -> GetValueByPosResult {
        match self {
            ScanLock::S(leaf) => leaf.scan_value_by_pos(pos, out_buffer),
            ScanLock::X(leaf) => leaf.scan_value_by_pos(pos, out_buffer),
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
}

impl<'b> ScanIterMut<'_, 'b> {
    pub(crate) fn new(tree: &'b BfTree, start_key: &'b [u8], scan_cnt: usize) -> Self {
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
            };
        }
    }

    pub fn next(&mut self, out_buffer: &mut [u8]) -> Option<usize> {
        if self.scan_cnt == 0 {
            return None;
        }

        match self
            .leaf_lock
            .scan_value_by_pos(&self.scan_position, out_buffer)
        {
            GetValueByPosResult::Deleted => {
                self.scan_position.move_to_next();
                self.next(out_buffer)
            }
            GetValueByPosResult::Found(len) => {
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
                Some(len as usize)
            }
            GetValueByPosResult::EndOfLeaf => {
                // we need to load next leaf.
                let right_sibling = self.leaf_lock.get_right_sibling();

                if right_sibling.is_empty() {
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
                self.next(out_buffer)
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
}

impl<'b> ScanIter<'_, 'b> {
    pub(crate) fn new(tree: &'b BfTree, start_key: &'b [u8], scan_cnt: usize) -> Self {
        let backoff = Backoff::new();
        let mut aggressive_split = false;

        loop {
            let (scan_pos, lock) = match move_cursor_to_leaf(tree, start_key, aggressive_split) {
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
            };
        }
    }

    // Here we need to busy loop? Is that safe?
    /// Scan next value into `out_buffer`.
    /// Returns the length of the value or None if there is no more value.
    pub fn next(&mut self, out_buffer: &mut [u8]) -> Option<usize> {
        if self.scan_cnt == 0 {
            return None;
        }

        match self
            .leaf_lock
            .get_value_by_pos(&self.scan_position, out_buffer)
        {
            GetValueByPosResult::Deleted => {
                self.scan_position.move_to_next();
                self.next(out_buffer)
            }
            GetValueByPosResult::Found(len) => {
                self.scan_position.move_to_next();
                self.scan_cnt -= 1;
                Some(len as usize)
            }
            GetValueByPosResult::EndOfLeaf => {
                // we need to load next leaf.
                counter!(ScanGoNextLeaf);
                let right_sibling = self.leaf_lock.get_right_sibling();

                if right_sibling.is_empty() {
                    self.scan_cnt = 0;
                    return None;
                }

                let backoff = Backoff::new();

                let mut aggressive_split = false;
                loop {
                    let (pos, lock) =
                        match move_cursor_to_leaf(self.tree, &right_sibling, aggressive_split) {
                            Ok((pos, lock)) => (pos, lock),
                            Err(TreeError::Locked) => {
                                backoff.spin();
                                continue;
                            }
                            Err(TreeError::CircularBufferFull) => {
                                // We can't call eviction here becuase we are holding a lock, which may happened to be evicted!
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
                self.next(out_buffer)
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

            let full_page_loc = upgrade_to_full_page(&tree.storage, base_page_ref, next_level)?;

            leaf.create_cache_page_loc(full_page_loc);

            Ok(ScanPosition::Base(pos as u32))
        }
        PageLocation::Mini(ptr) => {
            counter!(ScanMergeMiniPage);
            let mini_page = leaf.load_cache_page_mut(*ptr);
            // acquire the handle so that the eviction process with not contend with us.
            let h = tree.storage.begin_dealloc_mini_page(mini_page)?;
            let merge_result = leaf.try_merge_mini_page(&h, parent, &tree.storage)?;

            match merge_result {
                MergeResult::NoSplit => {
                    // if no split, we face two choices:
                    // (1) keep using this mini-page
                    // (2) upgrade this page to full page, so that future scans don't need to load base page.
                    //      this is done with probability to avoid polluting the cache.
                    if tree.should_promote_scan_page() {
                        // upgrade to full page
                        let base_offset = mini_page.next_level;
                        leaf.change_to_base_loc();
                        tree.storage.finish_dealloc_mini_page(h);

                        let base_page_ref = leaf.load_base_page_from_buffer();
                        let full_page_loc =
                            upgrade_to_full_page(&tree.storage, base_page_ref, base_offset)?;

                        leaf.create_cache_page_loc(full_page_loc);
                        Err(TreeError::NeedRestart)
                    } else {
                        mini_page.consolidate_after_merge();
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
    let (pid, parent) = tree.traverse_to_leaf(key, aggressive_split)?;

    let mut leaf = tree.mapping_table().get_mut(&pid);

    check_parent!(tree, pid, parent);

    if let Ok(pos) = leaf.get_scan_position(key) {
        match pos {
            ScanPosition::Base(_) => {
                if !tree.should_promote_scan_page() {
                    return Ok((pos, leaf));
                }
                // o.w. fall through and upgrade to full page.
            }
            ScanPosition::Full(_) => {
                return Ok((pos, leaf));
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
) -> Result<(ScanPosition, ScanLock<'a>), TreeError> {
    let (pid, parent) = tree.traverse_to_leaf(key, aggressive_split)?;

    let mut leaf = tree.mapping_table().get(&pid);

    check_parent!(tree, pid, parent);

    if let Ok(pos) = leaf.get_scan_position(key) {
        match pos {
            ScanPosition::Base(_) => {
                counter!(ScanBasePage);
                if parent.is_none() || !tree.should_promote_scan_page() {
                    return Ok((pos, ScanLock::S(leaf)));
                }
                // o.w. fall through and upgrade to full page.
            }
            ScanPosition::Full(_) => {
                counter!(ScanFullPage);
                return Ok((pos, ScanLock::S(leaf)));
            }
        }
    }

    // we need to merge mini page.
    let mut x_leaf = leaf.try_upgrade().map_err(|_e| TreeError::Locked)?;

    let v = promote_or_merge_mini_page(tree, key, &mut x_leaf, parent.unwrap())?;
    Ok((v, ScanLock::X(x_leaf)))
}
