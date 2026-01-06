// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

mod inner_node;
pub(crate) mod leaf_node;
mod node_meta;
mod page_id;

pub(crate) use inner_node::{InnerNode, InnerNodeBuilder};
pub(crate) use leaf_node::LeafNode;
pub(crate) use page_id::PageID;

pub(crate) const INNER_NODE_SIZE: usize = 4096;
pub(crate) const DISK_PAGE_SIZE: usize = 4096; // The size of a disk page.
pub(crate) const MAX_LEAF_PAGE_SIZE: usize = 32768; // 32KB
pub(crate) const MAX_KEY_LEN: usize = 2020; // 2KB
pub(crate) const MAX_VALUE_LEN: usize = 16332; // 16KB
pub(crate) const CACHE_LINE_SIZE: usize = 64;
pub(crate) const FENCE_KEY_CNT: usize = 2;
pub(crate) const KV_META_SIZE: usize = 8;
