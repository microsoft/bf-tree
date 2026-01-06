// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::nodes::FENCE_KEY_CNT;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct NodeMeta {
    encoded: u16,
    // if highest bit, it is a leaf node, otherwise inner node.
    // if second highest bit set, the node needs to be split.
    // remaining bits are value count.
    pub(crate) node_size: u16,
    pub(crate) remaining_size: u16,
}

const HAS_FENCE_MASK: u16 = 0x8000;
const SHOULD_SPLIT_MASK: u16 = 0x4000;
const CHILDREN_IS_LEAF_MASK: u16 = 0x2000;
const CACHE_LEAF_MASK: u16 = 0x1000;

// Only applies to inner node.
// If children are leaves, their values are page IDs,
// otherwise their values are pointers to the next level inner nodes.
const VALUE_COUNT_MASK: u16 = 0x0FFF;

impl NodeMeta {
    pub(crate) fn new(
        remaining_size: u16,
        children_is_leaf: bool,
        has_fence: bool,
        node_size: u16,
        cache_only: bool,
    ) -> Self {
        let mut encoded = 0;

        if children_is_leaf {
            encoded |= CHILDREN_IS_LEAF_MASK;
        }

        if has_fence {
            // fence key is only for leaf node
            // assert_eq!(node_size as usize, DEFAULT_LEAF_NODE_SIZE);
            assert!(!children_is_leaf);
            encoded |= HAS_FENCE_MASK;
        }

        if cache_only {
            encoded |= CACHE_LEAF_MASK;
        }

        Self {
            encoded,
            node_size,
            remaining_size,
        }
    }

    pub(crate) fn meta_count_with_fence(&self) -> u16 {
        self.encoded & VALUE_COUNT_MASK
    }

    #[allow(dead_code)]
    pub(crate) fn value_count_inner(&self) -> u16 {
        self.meta_count_with_fence() - 1
    }

    /// TODO: this is undefined for inner nodes.
    pub(crate) fn meta_count_without_fence(&self) -> u16 {
        if self.has_fence() {
            self.meta_count_with_fence() - FENCE_KEY_CNT as u16
        } else {
            self.encoded & VALUE_COUNT_MASK
        }
    }

    pub(crate) fn has_fence(&self) -> bool {
        (self.encoded & HAS_FENCE_MASK) != 0
    }

    pub(crate) fn set_value_count(&mut self, count: u16) {
        self.encoded = (self.encoded & !VALUE_COUNT_MASK) | count;
    }

    pub(crate) fn children_is_leaf(&self) -> bool {
        (self.encoded & CHILDREN_IS_LEAF_MASK) != 0
    }

    pub(crate) fn is_cache_only_leaf(&self) -> bool {
        (self.encoded & CACHE_LEAF_MASK) != 0
    }

    #[allow(dead_code)]
    fn set_children_is_leaf(&mut self, is_leaf: bool) {
        if is_leaf {
            self.encoded |= CHILDREN_IS_LEAF_MASK;
        } else {
            self.encoded &= !CHILDREN_IS_LEAF_MASK;
        }
    }

    pub(crate) fn increment_value_count(&mut self) {
        self.encoded += 1;
    }

    #[allow(dead_code)]
    fn decrement_value_count(&mut self) {
        self.encoded -= 1;
    }

    pub(crate) fn get_split_flag(&self) -> bool {
        (self.encoded & SHOULD_SPLIT_MASK) != 0
    }

    pub(crate) fn set_split_flag(&mut self) {
        self.encoded |= SHOULD_SPLIT_MASK;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node_meta() {
        let meta = NodeMeta::new(100, false, false, 200, false);
        assert_eq!(meta.encoded & CHILDREN_IS_LEAF_MASK, 0);
        assert_eq!(meta.node_size, 200);
    }

    #[test]
    fn test_meta_count() {
        let mut meta = NodeMeta::new(100, false, false, 200, false);
        meta.set_value_count(5);
        assert_eq!(meta.meta_count_with_fence(), 5);
    }

    #[test]
    fn test_set_value_count() {
        let mut meta = NodeMeta::new(100, false, false, 200, false);
        meta.set_value_count(10);
        assert_eq!(meta.meta_count_with_fence(), 10);
    }

    #[test]
    fn test_children_is_leaf() {
        let meta = NodeMeta::new(100, true, false, 200, false);
        assert!(meta.children_is_leaf());
    }

    #[test]
    fn test_set_children_is_leaf() {
        let mut meta = NodeMeta::new(100, false, false, 200, false);
        meta.set_children_is_leaf(true);
        assert!(meta.children_is_leaf());
    }

    #[test]
    fn test_increment_and_decrement_value_count() {
        let mut meta = NodeMeta::new(100, false, false, 200, false);
        meta.set_value_count(5);
        meta.increment_value_count();
        assert_eq!(meta.meta_count_with_fence(), 6);
        meta.decrement_value_count();
        assert_eq!(meta.meta_count_with_fence(), 5);
    }

    #[test]
    fn test_get_and_set_split_flag() {
        let mut meta = NodeMeta::new(100, false, true, 4096, false);
        assert!(!meta.get_split_flag());
        meta.set_split_flag();
        assert!(meta.get_split_flag());
    }
}
