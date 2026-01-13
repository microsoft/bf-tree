// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::nodes::InnerNode;

/// A page ID can be either a heap pointer (pointing to inner nodes)
/// or a ID number used to reference mini page or leaf page.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub(crate) struct PageID {
    value: u64,
}

/// If it is a ID, it is not heap pointer.
const ID_MASK: u64 = 0x4000_0000_0000_0000;

impl PageID {
    pub(crate) fn new(value: u64) -> Self {
        assert_eq!(std::mem::size_of::<Self>(), 8);
        Self { value }
    }

    /// Only used when you load a page ID from somewhere.
    pub(crate) unsafe fn from_raw(value: u64) -> Self {
        Self::new(value)
    }

    pub(crate) fn from_pointer(ptr: *const InnerNode) -> Self {
        Self::new(ptr as u64)
    }

    pub(crate) fn from_id(id: u64) -> Self {
        Self::new(id | ID_MASK)
    }

    pub(crate) fn is_id(&self) -> bool {
        (self.value & ID_MASK) != 0
    }

    pub(crate) fn as_id(&self) -> u64 {
        assert!(self.is_id());
        self.value & !ID_MASK
    }

    pub(crate) fn is_inner_node_pointer(&self) -> bool {
        (self.value & ID_MASK) == 0
    }

    pub(crate) fn raw(&self) -> u64 {
        self.value
    }

    pub(crate) fn as_inner_node(&self) -> *const InnerNode {
        assert!(self.is_inner_node_pointer());
        self.value as *const InnerNode
    }
}
