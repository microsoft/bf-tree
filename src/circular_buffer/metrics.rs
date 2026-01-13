// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::HashMap;

use super::States;

#[derive(serde::Serialize)]
pub struct CircularBufferMetrics {
    pub capacity: usize,
    pub head_addr: usize,
    pub tail_addr: usize,
    pub evicting_addr: usize,
    pub head_tail_distance: usize,
    pub allocated_cnt: usize,
    pub not_ready_cnt: usize,
    pub ready_cnt: usize,
    pub tombstone_cnt: usize,
    pub tombstone_size: usize,
    pub begin_tombstone_cnt: usize,
    pub free_listed_cnt: usize,
    pub evicted_cnt: usize,
    pub size_cnt: HashMap<usize, usize>,
}

impl CircularBufferMetrics {
    pub(super) fn new(cap: usize, states: &States) -> Self {
        let head_tail_distance = states.tail_addr() - states.head_addr();
        Self {
            capacity: cap,
            head_addr: states.head_addr(),
            tail_addr: states.tail_addr(),
            evicting_addr: states.evicting_addr,
            head_tail_distance,
            allocated_cnt: 0,
            not_ready_cnt: 0,
            ready_cnt: 0,
            tombstone_cnt: 0,
            tombstone_size: 0,
            begin_tombstone_cnt: 0,
            free_listed_cnt: 0,
            evicted_cnt: 0,
            size_cnt: HashMap::new(),
        }
    }
}
