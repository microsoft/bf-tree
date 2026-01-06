// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

pub(crate) mod atomic_wait;
pub(crate) mod inner_lock;
mod mapping_table;
pub(crate) mod rw_lock;
pub(crate) mod stats;
pub(crate) mod test_util;

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::{cell::Cell, fmt};

pub(crate) use mapping_table::MappingTable;

const SPIN_LIMIT: u32 = 6;
const YIELD_LIMIT: u32 = 10;

/// Backoff implementation from the Crossbeam, added shuttle instrumentation
pub(crate) struct Backoff {
    step: Cell<u32>,
}

impl Backoff {
    #[inline]
    pub(crate) fn new() -> Self {
        Backoff { step: Cell::new(0) }
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn reset(&self) {
        self.step.set(0);
    }

    #[inline]
    pub(crate) fn spin(&self) {
        for _ in 0..1 << self.step.get().min(SPIN_LIMIT) {
            std::hint::spin_loop();
        }

        if self.step.get() <= SPIN_LIMIT {
            self.step.set(self.step.get() + 1);
        }

        #[cfg(all(feature = "shuttle", test))]
        shuttle::thread::yield_now();
    }

    #[inline]
    pub(crate) fn snooze(&self) {
        if self.step.get() <= SPIN_LIMIT {
            for _ in 0..1 << self.step.get() {
                std::hint::spin_loop();
            }
        } else {
            #[cfg(all(feature = "shuttle", test))]
            shuttle::thread::yield_now();

            #[cfg(not(all(feature = "shuttle", test)))]
            ::std::thread::yield_now();
        }

        if self.step.get() <= YIELD_LIMIT {
            self.step.set(self.step.get() + 1);
        }
    }

    #[inline]
    pub(crate) fn is_completed(&self) -> bool {
        self.step.get() > YIELD_LIMIT
    }
}

impl fmt::Debug for Backoff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Backoff")
            .field("step", &self.step)
            .field("is_completed", &self.is_completed())
            .finish()
    }
}

impl Default for Backoff {
    fn default() -> Backoff {
        Backoff::new()
    }
}

use std::rc::Rc;

use rand::rngs::SmallRng;
use rand::{Error, RngCore, SeedableRng};

use crate::nodes::PageID;
use crate::BfTree;
use inner_lock::ReadGuard;

pub struct SmallThreadRng {
    rng: Rc<UnsafeCell<rand::rngs::SmallRng>>,
}

#[cfg(not(all(feature = "shuttle", test)))]
pub(crate) fn get_rng() -> SmallThreadRng {
    let rng = THREAD_RNG_KEY.with(|t| t.clone());
    SmallThreadRng { rng }
}

pub(crate) fn thread_id_to_u64(tid: std::thread::ThreadId) -> u64 {
    let mut hasher = DefaultHasher::new();
    tid.hash(&mut hasher);
    hasher.finish()
}

thread_local! {
    // Initialize a thread-local SmallRng
    static THREAD_RNG_KEY: Rc<UnsafeCell<SmallRng>> = Rc::new( UnsafeCell::new( SmallRng::seed_from_u64(
        thread_id_to_u64(std::thread::current().id())
    )));
}

impl RngCore for SmallThreadRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.try_fill_bytes(dest)
    }
}

#[cfg(all(feature = "shuttle", test))]
pub(crate) fn get_rng() -> shuttle::rand::rngs::ThreadRng {
    shuttle::rand::thread_rng()
}

pub(crate) enum NodeInfo {
    Leaf {
        level: usize,
        page_id: PageID,
    },
    Inner {
        level: usize,
        ptr: *const crate::nodes::InnerNode,
    },
}

impl NodeInfo {
    fn new(level: usize, page_id: PageID, is_leaf: bool) -> Self {
        if is_leaf {
            NodeInfo::Leaf { level, page_id }
        } else {
            NodeInfo::Inner {
                level,
                ptr: page_id.as_inner_node(),
            }
        }
    }
}

/// Visit the BfTree nodes in a BFS manner.
pub(crate) struct BfsVisitor<'a> {
    _tree: &'a BfTree,
    node_queue: VecDeque<NodeInfo>,
    inner_nodes_only: bool,
}

impl<'a> BfsVisitor<'a> {
    /// Visit all nodes in this BfTree.
    pub(crate) fn new_all_nodes(tree: &'a BfTree) -> BfsVisitor<'a> {
        Self::new(tree, false)
    }

    /// Visit only inner nodes in this BfTree.
    pub(crate) fn new_inner_only(tree: &'a BfTree) -> BfsVisitor<'a> {
        Self::new(tree, true)
    }

    fn new(tree: &'a BfTree, inner_only: bool) -> Self {
        let mut node_queue = VecDeque::new();
        let root = tree.get_root_page();
        let node_info = NodeInfo::new(0, root.0, root.1);

        node_queue.push_back(node_info);

        BfsVisitor {
            _tree: tree,
            node_queue,
            inner_nodes_only: inner_only,
        }
    }
}

impl Iterator for BfsVisitor<'_> {
    type Item = NodeInfo;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node_info) = self.node_queue.pop_front() {
            match node_info {
                NodeInfo::Leaf { .. } => Some(node_info),
                NodeInfo::Inner { level, ptr } => {
                    let inner = ReadGuard::try_read(ptr).unwrap();
                    let child_is_leaf = inner.as_ref().meta.children_is_leaf();
                    if self.inner_nodes_only && child_is_leaf {
                        return Some(node_info);
                    }

                    let children = inner.as_ref().get_child_iter();
                    for child in children {
                        let node_info = NodeInfo::new(level + 1, child, child_is_leaf);
                        self.node_queue.push_back(node_info)
                    }

                    Some(node_info)
                }
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
pub(crate) struct TestVfs {}

#[cfg(test)]
impl TestVfs {
    pub fn new() -> Self {
        TestVfs {}
    }
}

#[cfg(test)]
impl crate::fs::VfsImpl for TestVfs {
    fn read(&self, _offset: usize, _buf: &mut [u8]) {}

    fn write(&self, _offset: usize, _buf: &[u8]) {}

    fn alloc_offset(&self, _size: usize) -> usize {
        0
    }

    fn dealloc_offset(&self, _offset: usize) {}

    fn flush(&self) {}

    fn open(_path: impl AsRef<std::path::Path>) -> Self
    where
        Self: Sized,
    {
        TestVfs {}
    }
}
