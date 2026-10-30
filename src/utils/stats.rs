// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#![allow(unused)]

use crate::nodes::{leaf_node::MiniPageNextLevel, PageID};
use crate::utils::{inner_lock::ReadGuard, BfsVisitor, NodeInfo};
use crate::{nodes::leaf_node::OpType, BfTree};

pub(crate) enum NodeLevelStats {
    Inner(InnerStats),
    Leaf(LeafStats),
}

pub(crate) struct InnerStats {
    pub child_keys: Vec<Vec<u8>>,
    pub child_id: Vec<PageID>,
    pub child_is_leaf: bool,
}

pub(crate) struct LeafStats {
    pub keys: Vec<Vec<u8>>,
    pub values: Vec<Vec<u8>>,
    pub op_types: Vec<OpType>,
    pub prefix: Vec<u8>,
    pub base_node: Option<Box<LeafStats>>,
    pub next_level: MiniPageNextLevel,
    pub node_size: usize,
}

pub(crate) struct PerNodeStats {
    pub level: usize,
    pub value_cnt: usize,
    pub stats: NodeLevelStats,
}

impl BfTree {
    #[allow(dead_code)]
    pub(crate) fn get_stats(&self) -> (Vec<PerNodeStats>, usize) {
        let mut total_cnt = 0;
        let mut nodes = Vec::<PerNodeStats>::new();

        let visitor = BfsVisitor::new_all_nodes(self);

        for node_info in visitor {
            match node_info {
                NodeInfo::Leaf { level, page_id } => {
                    let mut leaf = self.mapping_table().get_mut(&page_id);
                    let stats = leaf.get_stats();
                    let node_cnt = stats.keys.len();
                    total_cnt += node_cnt;
                    nodes.push(PerNodeStats {
                        level,
                        value_cnt: node_cnt,
                        stats: NodeLevelStats::Leaf(stats),
                    });
                }
                NodeInfo::Inner { level, ptr } => {
                    let backoff = crate::utils::Backoff::new();
                    let stats = loop {
                        match ReadGuard::try_read(ptr).and_then(|inner| inner.get_stats()) {
                            Ok(stats) => break stats,
                            Err(_) => backoff.spin(),
                        }
                    };

                    nodes.push(PerNodeStats {
                        level,
                        value_cnt: stats.child_keys.len(),
                        stats: NodeLevelStats::Inner(stats),
                    });
                }
            }
        }

        (nodes, total_cnt)
    }
}
