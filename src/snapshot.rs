// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::cell::UnsafeCell;
use std::collections::{HashMap, VecDeque};
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::panic;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(not(all(feature = "shuttle", test)))]
use rand::Rng;
#[cfg(all(feature = "shuttle", test))]
use shuttle::rand::Rng;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

#[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
use thread_local::ThreadLocal;

use crate::{
    circular_buffer::CircularBuffer,
    error::ConfigError,
    fs::VfsImpl,
    mini_page_op::LeafOperations,
    nodes::{leaf_node::MiniPageNextLevel, LeafNode, INVALID_DISK_OFFSET},
    nodes::{InnerNode, InnerNodeBuilder, PageID, DISK_PAGE_SIZE, INNER_NODE_SIZE},
    storage::{make_vfs, LeafStorage, PageLocation, PageTable},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    sync::thread,
    sync::RwLock,
    utils::{get_rng, inner_lock::ReadGuard, BfsVisitor, NodeInfo},
    wal::{LogEntry, LogEntryImpl, WriteAheadLog},
    BfTree, Config, StorageBackend, WalConfig, WalReader,
};

const BF_TREE_MAGIC_BEGIN: &[u8; 16] = b"BF-TREE-V0-BEGIN";
const BF_TREE_MAGIC_END: &[u8; 14] = b"BF-TREE-V0-END";
const META_DATA_PAGE_OFFSET: usize = 0;

const INVALID_SNAPSHOT_THREAD_ID: usize = usize::MAX; // Invalid thread slot id
const NULL_PAGE_LOCATION_OFFSET: usize = usize::MAX; // Special page loc offset for a Null page
const INVALID_SNAPSHOT_STATE: u64 = u64::MAX; // Invalid snapshot state
pub const INVALID_SNAPSHOT_VERSION: u64 = u64::MAX >> 1; // Invalid snapshot version
const DEFAULT_MAX_SNAPSHOT_THREAD_NUM: usize = 64; // Maximum numbers of concurrent threads in Bf-tree, if snapshot is enabled.
const SNAPSHOT_STATE_PHASE_ID_SHIFT: usize = 61; // Number of bits to shift for phase id
const SNAPSHOT_STATE_PHASE_NUM: u64 = 4; // There are 4 snapshot phases
const SNAPSHOT_STATE_PHASE_ID_MASK: u64 = 0b111 << SNAPSHOT_STATE_PHASE_ID_SHIFT; // Most significant 3 bits for phase id (allowing up to 8 phases)
const SNAPSHOT_STATE_VERSION_MASK: u64 = (1 << SNAPSHOT_STATE_PHASE_ID_SHIFT) - 1; // Least significant 61 bits for version number

/// Snapshot phase identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseId {
    Rest,
    Prepare,
    InProgress,
    Sweep,
}

impl PhaseId {
    fn from_raw(value: u64) -> Self {
        match value {
            0 => PhaseId::Rest,
            1 => PhaseId::Prepare,
            2 => PhaseId::InProgress,
            3 => PhaseId::Sweep,
            _ => panic!("Invalid phase id: {}", value),
        }
    }

    fn as_raw(self) -> u64 {
        match self {
            PhaseId::Rest => 0,
            PhaseId::Prepare => 1,
            PhaseId::InProgress => 2,
            PhaseId::Sweep => 3,
        }
    }
}

/// A simplified CPR snapshot of a Bf-Tree.
/// For details, see the original CPR paper.
/// When compared to the original CPR paper, the following simplifications were made:
/// 1. No epoch framework, global state machine only.
/// 2. No 2PL and partial execution of read/upsert/delete/ in one version allowed
///    E.g., a bf-tree upsert operation requires a series of mini-transcational modifications m_0..m_n.
///    We do not lock all m(s) beforehand. Also, we allow the operation to restart if at m_i CPR detects
///    a version inconsistency (PREPARE/v thread seeing a (v+1) record). This is OK for bf-tree
///    because all m(s) are independent even though a m_i could involve multiple page modifications.
pub struct CPRSnapShotMgr {
    // Snapshot state
    // | 3 bits   |     61 bits    |
    // | phase id | version number |
    global_state: AtomicU64,
    // False means the thread slot is vacant while True means occupied
    thread_slots: [AtomicBool; DEFAULT_MAX_SNAPSHOT_THREAD_NUM],
    // Thread-level snapshot state
    thread_local_states: [AtomicU64; DEFAULT_MAX_SNAPSHOT_THREAD_NUM],
    // Mappings of base/mini/inner nodes to their corresponding snapshot file offsets.
    // Each user thread updates its own mappings asyncrhonously and get merged in the end.
    thread_local_inner_mappings:
        UnsafeCell<[Vec<(*const InnerNode, usize)>; DEFAULT_MAX_SNAPSHOT_THREAD_NUM]>,
    thread_local_base_mappings: UnsafeCell<[Vec<(PageID, usize)>; DEFAULT_MAX_SNAPSHOT_THREAD_NUM]>,
    thread_local_mini_mappings: UnsafeCell<[Vec<(PageID, usize)>; DEFAULT_MAX_SNAPSHOT_THREAD_NUM]>,
    thread_local_mini_size_mappings:
        UnsafeCell<[Vec<(PageID, usize)>; DEFAULT_MAX_SNAPSHOT_THREAD_NUM]>,
    root_id: AtomicU64, // Page ID of the root node.
    pause_snapshot: AtomicBool,
    // The physical file of snapshot. Wrapped in RwLock so the underlying
    // vfs can be swapped when taking a new snapshot.
    // Arc<Box<dyn VfsImpl>> is another option.
    vfs: RwLock<Arc<dyn VfsImpl>>,
    // Ensuring only one snapshot is in progress at a time.
    snapshot_in_progress: AtomicBool,
}

unsafe impl Sync for CPRSnapShotMgr {}

unsafe impl Send for CPRSnapShotMgr {}

/// Within the life time of this guard, bf-tree transactions are protected by CPR logic.
pub struct CPRSnapshotGuard {
    snapshot_mgr: Option<Arc<CPRSnapShotMgr>>,
    thread_slot_id: usize,
    snapshot_version: u64,
    phase_id: PhaseId,
}

impl CPRSnapshotGuard {
    pub fn new(snapshot_mgr: Option<Arc<CPRSnapShotMgr>>) -> Result<Self, ()> {
        match snapshot_mgr {
            None => Ok(Self {
                snapshot_mgr: None,
                thread_slot_id: INVALID_SNAPSHOT_THREAD_ID,
                snapshot_version: INVALID_SNAPSHOT_VERSION,
                phase_id: PhaseId::Rest,
            }),
            Some(ref mgr) => {
                let (thread_slot_id, snapshot_version, phase_id) = mgr.reserve_thread_slot()?;

                Ok(Self {
                    snapshot_mgr: Some(mgr.clone()),
                    thread_slot_id,
                    snapshot_version,
                    phase_id,
                })
            }
        }
    }

    /// Returns the snapshot version for this thread's transaction.
    pub fn snapshot_version(&self) -> u64 {
        self.snapshot_version
    }

    /// Returns the phase id at the time the guard was acquired.
    pub fn get_local_phase_id(&self) -> PhaseId {
        self.phase_id
    }

    /// Returns true if the snapshot guard has a valid thread slot id.
    pub fn is_protected(&self) -> bool {
        self.thread_slot_id != INVALID_SNAPSHOT_THREAD_ID
    }

    pub fn snapshot_base_page(&self, id: PageID, ptr: &[u8], size: usize) {
        self.snapshot_mgr
            .as_ref()
            .unwrap()
            .snapshot_base_page(id, ptr, size, self.thread_slot_id);
    }

    pub fn snapshot_mini_page(&self, id: PageID, ptr: &[u8], size: usize) {
        self.snapshot_mgr
            .as_ref()
            .unwrap()
            .snapshot_mini_page(id, ptr, size, self.thread_slot_id);
    }

    pub fn snapshot_inner_node(&self, ptr: *const InnerNode) {
        self.snapshot_mgr
            .as_ref()
            .unwrap()
            .snapshot_inner_node(ptr, self.thread_slot_id);
    }

    pub fn snapshot_root_page(&self, root_id: PageID) {
        self.snapshot_mgr
            .as_ref()
            .unwrap()
            .snapshot_root_page(root_id);
    }
}

impl Drop for CPRSnapshotGuard {
    fn drop(&mut self) {
        if let Some(ref mgr) = self.snapshot_mgr {
            mgr.release_thread_slot(self.thread_slot_id);
        }
    }
}

impl CPRSnapShotMgr {
    pub fn are_all_threads_in_next_version(&self) -> bool {
        if !self.snapshot_in_progress.load(Ordering::Acquire) {
            false
        } else {
            let global_phase_id = self.get_global_phase_id();
            global_phase_id == PhaseId::Sweep || global_phase_id == PhaseId::Rest
        }
    }

    /// Initialize a new snapshot instance.
    pub fn new(version: u64) -> Self {
        let vfs: Arc<dyn VfsImpl> = make_vfs(&StorageBackend::Memory, ":memory:");
        Self {
            global_state: AtomicU64::new(Self::new_snapshot_state(0, version)), // Initial state: (REST, version)
            thread_slots: [const { AtomicBool::new(false) }; DEFAULT_MAX_SNAPSHOT_THREAD_NUM],
            thread_local_states: [const { AtomicU64::new(INVALID_SNAPSHOT_STATE) };
                DEFAULT_MAX_SNAPSHOT_THREAD_NUM],
            thread_local_inner_mappings: UnsafeCell::new(
                [const { Vec::new() }; DEFAULT_MAX_SNAPSHOT_THREAD_NUM],
            ),
            thread_local_base_mappings: UnsafeCell::new(
                [const { Vec::new() }; DEFAULT_MAX_SNAPSHOT_THREAD_NUM],
            ),
            thread_local_mini_mappings: UnsafeCell::new(
                [const { Vec::new() }; DEFAULT_MAX_SNAPSHOT_THREAD_NUM],
            ),
            thread_local_mini_size_mappings: UnsafeCell::new(
                [const { Vec::new() }; DEFAULT_MAX_SNAPSHOT_THREAD_NUM],
            ),
            root_id: AtomicU64::new(0),
            pause_snapshot: AtomicBool::new(false),
            vfs: RwLock::new(vfs),
            snapshot_in_progress: AtomicBool::new(false),
        }
    }

    /// Reset the snapshot
    fn reset(&self) {
        let local_inner_mappings = unsafe { &mut *self.thread_local_inner_mappings.get() };
        let local_mini_mappings = unsafe { &mut *self.thread_local_mini_mappings.get() };
        let local_base_mappings = unsafe { &mut *self.thread_local_base_mappings.get() };
        let local_mini_size_mappings = unsafe { &mut *self.thread_local_mini_size_mappings.get() };
        // De-duplicate mappings
        for thread_slot_id in 0..DEFAULT_MAX_SNAPSHOT_THREAD_NUM {
            local_inner_mappings[thread_slot_id] = Vec::new();
            local_mini_mappings[thread_slot_id] = Vec::new();
            local_base_mappings[thread_slot_id] = Vec::new();
            local_mini_size_mappings[thread_slot_id] = Vec::new();
        }

        self.root_id.store(0, Ordering::Release);
    }

    pub fn new_snapshot_state(phase_id: u64, version: u64) -> u64 {
        assert!(
            phase_id < SNAPSHOT_STATE_PHASE_NUM,
            "Phase id must be less than {}",
            SNAPSHOT_STATE_PHASE_NUM
        );
        assert!(
            version < (1 << SNAPSHOT_STATE_PHASE_ID_SHIFT),
            "Version must be less than 2^61"
        );
        (phase_id << SNAPSHOT_STATE_PHASE_ID_SHIFT) | version
    }

    fn get_global_version(&self) -> u64 {
        self.global_state.load(Ordering::Acquire) & SNAPSHOT_STATE_VERSION_MASK
    }

    fn get_global_phase_id(&self) -> PhaseId {
        PhaseId::from_raw(
            (self.global_state.load(Ordering::Acquire) & SNAPSHOT_STATE_PHASE_ID_MASK)
                >> SNAPSHOT_STATE_PHASE_ID_SHIFT,
        )
    }

    /// Retrieve the local state of a thread specified by its slot id.
    fn get_local_state(&self, thread_slot_id: &usize) -> u64 {
        self.thread_local_states[*thread_slot_id].load(Ordering::Acquire)
    }

    fn set_local_state(&self, thread_slot_id: &usize, state: u64) {
        // Don't dirty the existing state if it's the same.
        let current_state = self.get_local_state(thread_slot_id);
        if current_state == state {
            return;
        }

        self.thread_local_states[*thread_slot_id].store(state, Ordering::Release);
    }

    /// Move the global snapshot stable to the next one.
    fn advance_global_state(&self) -> u64 {
        let phase_id = self.get_global_phase_id();
        let version = self.get_global_version();

        match phase_id {
            PhaseId::Rest => {
                // (REST, v) -> (PREPARE, v)
                let new_state = Self::new_snapshot_state(PhaseId::Prepare.as_raw(), version);
                self.global_state.store(new_state, Ordering::Release);
                new_state
            }
            PhaseId::Prepare => {
                // (PREPARE, v) -> (IN_PROGRESS, v + 1)
                let new_state = Self::new_snapshot_state(PhaseId::InProgress.as_raw(), version + 1);
                self.global_state.store(new_state, Ordering::Release);
                new_state
            }
            PhaseId::InProgress => {
                // (IN_PROGRESS, v + 1) -> (SWEEPING, v + 1)
                let new_state = Self::new_snapshot_state(PhaseId::Sweep.as_raw(), version);
                self.global_state.store(new_state, Ordering::Release);
                new_state
            }
            PhaseId::Sweep => {
                // (SWEEPING, v + 1) -> (REST, v + 1)
                let new_state = Self::new_snapshot_state(PhaseId::Rest.as_raw(), version);
                self.global_state.store(new_state, Ordering::Release);
                new_state
            }
        }
    }

    /// If all thread local states are either invalid or equal to the target state, return true.
    /// This can only be invoked after the global state has advanced to the target_state.
    fn check_if_phase_completed(&self, target_state: u64) -> bool {
        // Checking all thread local states is sufficient because of the guarantee in `reserve_thread_slot`
        for thread_slot_id in 0..DEFAULT_MAX_SNAPSHOT_THREAD_NUM {
            let local_state = self.thread_local_states[thread_slot_id].load(Ordering::Acquire);
            if local_state != INVALID_SNAPSHOT_STATE && local_state != target_state {
                return false;
            }
        }
        true
    }

    /// Obtain an unique thread slot id for the caller thread.
    /// Guarantee that any local state assigned after the global state advances to the next one,
    /// will either be reversed without further action or in the new state.
    pub fn reserve_thread_slot(&self) -> Result<(usize, u64, PhaseId), ()> {
        if self.pause_snapshot.load(Ordering::Acquire) {
            return Err(());
        }

        let start = get_rng().random_range(0..DEFAULT_MAX_SNAPSHOT_THREAD_NUM);
        let end = 2 * DEFAULT_MAX_SNAPSHOT_THREAD_NUM;

        for i in start..end {
            let tid = i % DEFAULT_MAX_SNAPSHOT_THREAD_NUM;
            if self.thread_slots[tid]
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Set the caller thread's snapshot local state to the global state
                let global_state = self.global_state.load(Ordering::Acquire);
                self.set_local_state(&tid, global_state);

                // If the global state has changed or a pause is requested after setting the local state, reset
                // This is to guarantee that as soon as global state rolls to the next one,
                // all new local states will either be reversed without further action or in the new state.
                // Otherwise something bad could happen as described below:
                // T1: state = global phase 'x'
                // Mgr: global phase <- 'x + 1'
                // Mrgr: all threads in phase 'x + 1' or invalid -> Execute 'x + 1' action
                // T1: local state = state <- Inconsistency with global state
                // Similar case for the pause_snapshot flag.
                if self.get_local_state(&tid) != self.global_state.load(Ordering::Acquire)
                    || self.pause_snapshot.load(Ordering::Acquire)
                {
                    self.set_local_state(&tid, INVALID_SNAPSHOT_STATE);
                    assert!(self.thread_slots[tid]
                        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok());
                    return Err(());
                } else {
                    let version = global_state & SNAPSHOT_STATE_VERSION_MASK;
                    let phase_id = PhaseId::from_raw(
                        (global_state & SNAPSHOT_STATE_PHASE_ID_MASK)
                            >> SNAPSHOT_STATE_PHASE_ID_SHIFT,
                    );
                    return Ok((tid, version, phase_id));
                }
            }
        }
        Err(())
    }

    /// Free up the thread slot specified by the given thread slot id.
    pub fn release_thread_slot(&self, thread_slot_id: usize) {
        self.set_local_state(&thread_slot_id, INVALID_SNAPSHOT_STATE);
        self.thread_slots[thread_slot_id].store(false, Ordering::Release);
    }

    pub fn get_snapshot_guard(
        snapshot_mgr: Option<Arc<CPRSnapShotMgr>>,
    ) -> Result<CPRSnapshotGuard, ()> {
        CPRSnapshotGuard::new(snapshot_mgr)
    }

    /// Snapshot a page to the current snapshot file and return its offset in the file.
    /// The invoker needs to guarantee that the page to be copied is xlocked
    /// throughout the lifetime of this function.
    /// Also the invoker needs to guarantee proper alignment of ptr which could be required
    /// by the underlying vfs. (E.g., io_uring_vfs requires 512B alignment)
    pub fn snapshot_page(&self, ptr: &[u8], size: usize) -> usize {
        // Allocate space in the snapshot file
        let vfs = self.vfs.read().unwrap().clone();
        let offset = vfs.alloc_offset(size);

        // Copy the page (ptr) to the new space
        vfs.write(offset, ptr);

        // Return the offset
        offset
    }

    pub fn snapshot_inner_node(&self, ptr: *const InnerNode, thread_slot_id: usize) {
        let offset = unsafe { self.snapshot_page((&*ptr).as_slice(), INNER_NODE_SIZE) };
        let inner_mappings = unsafe { &mut *self.thread_local_inner_mappings.get() };

        inner_mappings[thread_slot_id].push((ptr, offset));
    }

    pub fn snapshot_mini_page(&self, id: PageID, ptr: &[u8], size: usize, thread_slot_id: usize) {
        let offset = if size != 0 {
            self.snapshot_page(ptr, size)
        } else {
            // cache-only mode, NULL page
            NULL_PAGE_LOCATION_OFFSET
        };

        let mini_mappings = unsafe { &mut *self.thread_local_mini_mappings.get() };
        mini_mappings[thread_slot_id].push((id, offset));

        let mini_size_mappings = unsafe { &mut *self.thread_local_mini_size_mappings.get() };
        mini_size_mappings[thread_slot_id].push((id, size));

        assert!(mini_mappings[thread_slot_id].len() == mini_size_mappings[thread_slot_id].len());
    }

    pub fn snapshot_base_page(&self, id: PageID, ptr: &[u8], size: usize, thread_slot_id: usize) {
        let offset = self.snapshot_page(ptr, size);

        let base_mappings = unsafe { &mut *self.thread_local_base_mappings.get() };
        base_mappings[thread_slot_id].push((id, offset));
    }

    pub fn snapshot_root_page(&self, root_id: PageID) {
        let cur_root_id = self.root_id.load(Ordering::Acquire);
        if cur_root_id != 0 {
            assert_eq!(cur_root_id, root_id.raw());
        }

        self.root_id.store(root_id.raw(), Ordering::Release);
    }

    /// Sweep through all data pages and take snapshots of those
    /// whose version is less than the passed-in snapshot version.
    fn sweep(
        &self,
        tree: &BfTree,
        version: u64,
        inner_mapping: &mut Vec<(*const InnerNode, usize)>,
        mini_mapping: &mut Vec<(PageID, usize)>,
        mini_size_mapping: &mut Vec<(PageID, usize)>,
        base_mapping: &mut Vec<(PageID, usize)>,
    ) -> usize {
        // There is no page table for inner nodes including the root, and each inner node's information is only
        // saved in the tree structure itself. As a result, we need to traverse the tree to find all inner nodes.
        // However, without blocking inner node splitting, the tree structure could change concurrently while we
        // are BFS/DFSing the tree, making it difficult to enumerate all inner nodes to build inner node mappings.
        // As such we freeze the tree structure temporarily using a 3-phase approach.
        // Phase 1: Block all snapshot id reservation, drain the thread table.
        // Phase 2: Traverse the tree and take snapshots of inner nodes whose version is < 'version'.
        // Phase 3: Unblock snapshot id reservation.
        // Given that there are usually a very limited number of inner nodes, user workload should be affected minimally.
        // TODO: A complete block-free approach to scan all inner nodes.
        self.pause_snapshot.store(true, Ordering::Release);
        loop {
            if self.check_if_phase_completed(INVALID_SNAPSHOT_STATE) {
                // Upon reaching here, no user threads can obtain a snapshot id nor making changes to the tree structure.
                // We sweep the root node first
                loop {
                    let root_id = tree.get_root_page();
                    let rid = root_id.0;
                    if root_id.1 {
                        // Leaf
                        let mut leaf = tree.mapping_table().get(&rid);
                        let page_loc = leaf.get_page_location();

                        match page_loc {
                            PageLocation::Base(offset) => {
                                let base_ref = leaf.load_base_page(*offset);
                                if base_ref.get_clean_snapshot_version() < version {
                                    let base_ptr = unsafe {
                                        std::slice::from_raw_parts(
                                            base_ref as *const LeafNode as *const u8,
                                            base_ref.meta.node_size as usize,
                                        )
                                    };
                                    let offset = self
                                        .snapshot_page(base_ptr, base_ref.meta.node_size as usize);
                                    base_mapping.push((rid, offset));
                                    self.snapshot_root_page(rid);
                                }
                            }
                            PageLocation::Mini(ptr) => {
                                // Root page is a mini page only in cache-only mode.
                                assert!(tree.cache_only);

                                let mini_ref = leaf.load_cache_page(*ptr);
                                if mini_ref.get_clean_snapshot_version() < version {
                                    let mini_ptr = unsafe {
                                        std::slice::from_raw_parts(
                                            mini_ref as *const LeafNode as *const u8,
                                            mini_ref.meta.node_size as usize,
                                        )
                                    };
                                    let offset = self
                                        .snapshot_page(mini_ptr, mini_ref.meta.node_size as usize);
                                    mini_mapping.push((rid, offset));
                                    mini_size_mapping.push((rid, mini_ref.meta.node_size as usize));
                                    self.snapshot_root_page(rid);
                                }
                            }
                            _ => {
                                panic!("Unexpected page location for root page: {:?}", page_loc);
                            }
                        }

                        break;
                    } else {
                        // Inner
                        let ptr = rid.as_inner_node();
                        // No need for WriteGuard as the tree structured is frozen and there are no active writers.
                        let inner = match ReadGuard::try_read(ptr) {
                            Ok(inner) => inner,
                            Err(_) => continue,
                        };

                        if inner.as_ref().get_clean_snapshot_version() < version {
                            let offset =
                                unsafe { self.snapshot_page((&*ptr).as_slice(), INNER_NODE_SIZE) };
                            inner_mapping.push((ptr, offset));
                            self.snapshot_root_page(rid);
                        }
                        break;
                    }
                }

                // Inner nodes
                let visitor = BfsVisitor::new_inner_only(tree);
                for node in visitor {
                    loop {
                        match node {
                            NodeInfo::Inner { ptr, .. } => {
                                // No need for WriteGuard as the tree structured is frozen and there are no active writers.
                                let inner = match ReadGuard::try_read(ptr) {
                                    Ok(inner) => inner,
                                    Err(_) => continue,
                                };

                                if inner.as_ref().get_clean_snapshot_version() < version {
                                    let offset = unsafe {
                                        self.snapshot_page((&*ptr).as_slice(), INNER_NODE_SIZE)
                                    };
                                    inner_mapping.push((ptr, offset));
                                }

                                break;
                            }
                            NodeInfo::Leaf { level, .. } => {
                                // This should have been captured by the case when root node is a leaf
                                assert_eq!(level, 0);
                                break;
                            }
                        }
                    }
                }

                break;
            }
            // At most wasting 1 second per state transition.
            thread::sleep(std::time::Duration::from_secs(1));

            #[cfg(all(feature = "shuttle", test))]
            shuttle::thread::yield_now();
        }

        // Resume workload
        self.pause_snapshot.store(false, Ordering::Release);

        // In SWEEPING phase, there will be no new disk pages with v < 'version' being created. As such,
        // a sequential sweep of the page table is sufficient to capture all data pages with v < 'version'.
        let page_table_iter = tree.storage.page_table.iter();
        let mut enumerate_leaf_count = 0;

        for (_, pid) in page_table_iter {
            assert!(pid.is_id());

            // A reader lock is enough
            let mut leaf = tree.mapping_table().get(&pid);
            let page_loc = leaf.get_page_location().clone();
            enumerate_leaf_count += 1;

            match page_loc {
                PageLocation::Base(offset) => {
                    let base_ref = leaf.load_base_page(offset);
                    if base_ref.get_clean_snapshot_version() < version {
                        let base_ptr = unsafe {
                            std::slice::from_raw_parts(
                                base_ref as *const LeafNode as *const u8,
                                base_ref.meta.node_size as usize,
                            )
                        };
                        let new_offset =
                            self.snapshot_page(base_ptr, base_ref.meta.node_size as usize);
                        base_mapping.push((pid, new_offset));
                    }
                }
                PageLocation::Full(ptr) => {
                    // We snapshot Full page as a disk page to reduce some complexity as they are equivalent.
                    let full_ref = leaf.load_cache_page(ptr);
                    if full_ref.get_clean_snapshot_version() < version {
                        // Temporarily change the next level to null for snapshotting.
                        // and reverse afterwards.
                        let next_level = full_ref.next_level;
                        let full_page = unsafe { &mut *ptr };
                        full_page.next_level = MiniPageNextLevel::new_null();
                        let full_ptr = unsafe {
                            std::slice::from_raw_parts(
                                full_ref as *const LeafNode as *const u8,
                                full_ref.meta.node_size as usize,
                            )
                        };
                        let offset = self.snapshot_page(full_ptr, full_ref.meta.node_size as usize);
                        full_page.next_level = next_level;
                        base_mapping.push((pid, offset));
                    }
                }
                PageLocation::Mini(ptr) => {
                    let mini_ref = leaf.load_cache_page(ptr);
                    if mini_ref.get_clean_snapshot_version() < version {
                        let mini_ptr = unsafe {
                            std::slice::from_raw_parts(
                                mini_ref as *const LeafNode as *const u8,
                                mini_ref.meta.node_size as usize,
                            )
                        };
                        let offset = self.snapshot_page(mini_ptr, mini_ref.meta.node_size as usize);
                        mini_mapping.push((pid, offset));
                        mini_size_mapping.push((pid, mini_ref.meta.node_size as usize));

                        if !tree.cache_only {
                            // In disk-mode, the base page of mini-page is part of the snapshot as well.
                            let base_ref = leaf.load_base_page(mini_ref.next_level.as_offset());
                            assert!(base_ref.get_clean_snapshot_version() < version); // disk page's version should never be greater than its mini-page's.

                            let base_ptr = unsafe {
                                std::slice::from_raw_parts(
                                    base_ref as *const LeafNode as *const u8,
                                    base_ref.meta.node_size as usize,
                                )
                            };
                            let offset =
                                self.snapshot_page(base_ptr, base_ref.meta.node_size as usize);
                            base_mapping.push((pid, offset));
                        }
                    }
                }
                PageLocation::Null => {
                    assert!(tree.cache_only);
                    // In cache-only mode, an entry in page table could be Null when the corresponding mini-page is evicted.
                    // The Null page is also snapshotted with a special marker.
                    // Note that, the underlying assumption here is that Null page is always of an older version which may not be true.
                    // To reconcille with the CPR semantics, we say that any data page in cache-only mode could be either Null or a valid page.
                    // TODO: version the Null pages.
                    mini_mapping.push((pid, NULL_PAGE_LOCATION_OFFSET)); // Special marker
                    mini_size_mapping.push((pid, 0));
                }
            }
        }

        enumerate_leaf_count
    }

    /// Finalize the snapshot file and reset the snapshotmgr's internal data.
    #[allow(clippy::too_many_arguments)]
    fn finalize(
        &self,
        snapshot_version: u64,
        inner_mapping: &mut [(*const InnerNode, usize)],
        mini_mapping: &mut [(PageID, usize)],
        mini_size_mapping: &mut [(PageID, usize)],
        base_mapping: &mut [(PageID, usize)],
        leaf_count_upper: usize,
        config: Arc<Config>,
    ) {
        // There could be duplicate leaf/inner node mappings and we use a hash map to de-duplicate them.
        let mut inner_mapping_unique: HashMap<*const InnerNode, usize> = HashMap::new();
        let mut mini_mapping_unique: HashMap<PageID, usize> = HashMap::new();
        let mut mini_size_mapping_unique: HashMap<PageID, usize> = HashMap::new();
        let mut base_mapping_unique: HashMap<PageID, usize> = HashMap::new();

        let local_inner_mappings = unsafe { &mut *self.thread_local_inner_mappings.get() };
        let local_mini_mappings = unsafe { &mut *self.thread_local_mini_mappings.get() };
        let local_mini_size_mappings = unsafe { &mut *self.thread_local_mini_size_mappings.get() };
        let local_base_mappings = unsafe { &mut *self.thread_local_base_mappings.get() };

        for thread_slot_id in 0..DEFAULT_MAX_SNAPSHOT_THREAD_NUM {
            let entry_num = local_mini_mappings[thread_slot_id].len();
            assert!(
                local_mini_mappings[thread_slot_id].len()
                    == local_mini_size_mappings[thread_slot_id].len()
            );
            for i in 0..entry_num {
                assert!(
                    local_mini_mappings[thread_slot_id][i].0
                        == local_mini_size_mappings[thread_slot_id][i].0
                );

                if local_mini_mappings[thread_slot_id][i].1 == NULL_PAGE_LOCATION_OFFSET {
                    assert_eq!(local_mini_size_mappings[thread_slot_id][i].1, 0);
                } else {
                    assert!(local_mini_size_mappings[thread_slot_id][i].1 > 0);
                }

                if let std::collections::hash_map::Entry::Vacant(e) =
                    mini_mapping_unique.entry(local_mini_mappings[thread_slot_id][i].0)
                {
                    e.insert(local_mini_mappings[thread_slot_id][i].1);
                    mini_size_mapping_unique.insert(
                        local_mini_size_mappings[thread_slot_id][i].0,
                        local_mini_size_mappings[thread_slot_id][i].1,
                    );
                    assert!(mini_mapping_unique.len() == mini_size_mapping_unique.len());
                }
            }
            assert_eq!(local_mini_mappings[thread_slot_id].len(), entry_num);
            assert_eq!(local_mini_size_mappings[thread_slot_id].len(), entry_num);
        }

        // De-duplicate mappings among snapshot threads
        for thread_slot_id in 0..DEFAULT_MAX_SNAPSHOT_THREAD_NUM {
            for m in local_inner_mappings[thread_slot_id].iter() {
                inner_mapping_unique.entry(m.0).or_insert(m.1);
            }

            for m in local_base_mappings[thread_slot_id].iter() {
                base_mapping_unique.entry(m.0).or_insert(m.1);
            }
        }

        // Sanity checks
        assert!(mini_mapping_unique.len() == mini_size_mapping_unique.len());
        for (k, v) in mini_mapping_unique.iter() {
            assert!(mini_size_mapping_unique.contains_key(k));
            if *v == NULL_PAGE_LOCATION_OFFSET {
                assert_eq!(mini_size_mapping_unique.get(k).copied().unwrap(), 0);
            } else {
                assert!(mini_size_mapping_unique.get(k).copied().unwrap() > 0);
            }
        }

        // De-duplicate with the sweep mappings
        for (k, v) in inner_mapping.iter() {
            if !inner_mapping_unique.contains_key(k) {
                inner_mapping_unique.insert(*k, *v);
            }
        }
        for (k, v) in mini_mapping.iter() {
            if !mini_mapping_unique.contains_key(k) {
                mini_mapping_unique.insert(*k, *v);
            }
        }
        for (k, v) in mini_size_mapping.iter() {
            if !mini_size_mapping_unique.contains_key(k) {
                mini_size_mapping_unique.insert(*k, *v);
            } else {
                if !config.cache_only {
                    assert!(*v == mini_size_mapping_unique.get(k).copied().unwrap());
                }
            }
        }
        for (k, v) in base_mapping.iter() {
            if !base_mapping_unique.contains_key(k) {
                base_mapping_unique.insert(*k, *v);
            }
        }

        // Sanity checks
        assert!(mini_mapping_unique.len() == mini_size_mapping_unique.len());
        for (k, v) in mini_mapping_unique.iter() {
            assert!(mini_size_mapping_unique.contains_key(k));
            if *v == NULL_PAGE_LOCATION_OFFSET {
                assert_eq!(mini_size_mapping_unique.get(k).copied().unwrap(), 0);
            } else {
                assert!(mini_size_mapping_unique.get(k).copied().unwrap() > 0);
            }
        }

        if config.cache_only {
            assert!(base_mapping_unique.is_empty());
        } else {
            assert!(mini_mapping_unique.len() <= base_mapping_unique.len());
        }

        // Finalize the inner node mappings of the snapshot
        let mut final_inner_mapping: Vec<(*const InnerNode, usize)> = Vec::new();
        for (k, v) in inner_mapping_unique.into_iter() {
            final_inner_mapping.push((k, v));
        }

        // Finalize the base leaf node mappings of the snapshot, disk-mode only
        // Sort the base leaf node mappings by PageID in ascending order
        // which is required for page table initialization.
        let mut sorted_base_mapping_uninit: Vec<MaybeUninit<(PageID, usize)>> =
            Vec::with_capacity(base_mapping_unique.len());
        unsafe {
            sorted_base_mapping_uninit.set_len(base_mapping_unique.len());
        }
        let mut sorted_base_mapping_init = vec![false; base_mapping_unique.len()];

        for (k, v) in base_mapping_unique.iter() {
            assert!(k.is_id());
            let offset = k.as_id();
            assert!((offset as usize) < sorted_base_mapping_uninit.len());
            sorted_base_mapping_init[offset as usize] = true;
            sorted_base_mapping_uninit[offset as usize].write((*k, *v));
        }
        let final_sorted_base_mapping: Vec<(PageID, usize)> = if !config.cache_only {
            assert_eq!(
                base_mapping_unique.len(),
                sorted_base_mapping_init.iter().filter(|&&b| b).count()
            );
            unsafe {
                std::mem::transmute::<
                    std::vec::Vec<std::mem::MaybeUninit<(PageID, usize)>>,
                    std::vec::Vec<(PageID, usize)>,
                >(sorted_base_mapping_uninit)
            }
        } else {
            Vec::new()
        };

        // Finalize mini-page leaf node mappings of the snapshot
        let mut final_mini_mapping: Vec<(PageID, usize)> = Vec::new();
        for (k, v) in mini_mapping_unique.into_iter() {
            final_mini_mapping.push((k, v));
        }

        let mut final_mini_size_mapping: Vec<(PageID, usize)> = Vec::new();
        for (k, v) in mini_size_mapping_unique.into_iter() {
            final_mini_size_mapping.push((k, v));
        }

        let leaf_page_num = if config.cache_only {
            // For a pagelocation that's NULL (evicted), we don't know its version
            // As such, we assume they are of the snapshot version and should be included.
            // A few slots in page table mnight be wasted due to false positive.
            // TODO: Be precise which requires some more metadata. Ideas:
            // 1) Count number of child leaf nodes in snapshotted inner nodes
            // 2) Put version in Null PageLocation.
            // 3) Compact the snapshot file
            leaf_count_upper
        } else {
            assert!(leaf_count_upper >= final_sorted_base_mapping.len());
            final_sorted_base_mapping.len()
        };

        let mut file_size = std::mem::size_of::<BfTreeMeta>() as u64;

        // Flush various mappings into the snapshot file
        let (inner_offset, inner_size) =
            serialize_vec_to_disk(&final_inner_mapping, &self.vfs.read().unwrap());

        if inner_offset != 0 {
            file_size = (inner_offset + align_to_sector_size(inner_size)) as u64;
        }

        let (mini_offset, mini_size) =
            serialize_vec_to_disk(&final_mini_mapping, &self.vfs.read().unwrap());

        if mini_offset != 0 {
            file_size = (mini_offset + align_to_sector_size(mini_size)) as u64;
        }

        let (mini_size_offset, mini_size_size) =
            serialize_vec_to_disk(&final_mini_size_mapping, &self.vfs.read().unwrap());

        if mini_size_offset != 0 {
            file_size = (mini_size_offset + align_to_sector_size(mini_size_size)) as u64;
        }

        let (base_offset, base_size) =
            serialize_vec_to_disk(&final_sorted_base_mapping, &self.vfs.read().unwrap());

        if base_offset != 0 {
            file_size = (base_offset + align_to_sector_size(base_size)) as u64;
        }

        // Write the header to the first disk page of the snapshot file
        let metadata = BfTreeMeta {
            magic_begin: *BF_TREE_MAGIC_BEGIN,
            root_id: unsafe { PageID::from_raw(self.root_id.load(Ordering::Acquire)) },
            inner_offset,
            inner_size,
            mini_offset,
            mini_size,
            mini_size_offset,
            mini_size_size,
            base_offset,
            base_size,
            file_size,
            leaf_page_num,
            snapshot_version,
            cache_only: config.cache_only,
            cb_size_byte: config.cb_size_byte,
            read_promotion_rate: config.read_promotion_rate.load(Ordering::Relaxed),
            scan_promotion_rate: config.scan_promotion_rate.load(Ordering::Relaxed),
            cb_min_record_size: config.cb_min_record_size,
            cb_max_record_size: config.cb_max_record_size,
            leaf_page_size: config.leaf_page_size,
            cb_max_key_len: config.cb_max_key_len,
            max_fence_len: config.max_fence_len,
            cb_copy_on_access_ratio: config.cb_copy_on_access_ratio,
            read_record_cache: config.read_record_cache,
            max_mini_page_size: config.max_mini_page_size,
            mini_page_binary_search: config.mini_page_binary_search,
            write_load_full_page: config.write_load_full_page,
            magic_end: *BF_TREE_MAGIC_END,
        };

        let vfs = self.vfs.read().unwrap();
        vfs.write(META_DATA_PAGE_OFFSET, metadata.as_slice());
        vfs.flush();

        self.reset();
    }

    /// Take a CPR snapshot of a Bf-Tree
    pub fn snapshot(&self, tree: &BfTree, snapshot_file_path: impl AsRef<Path>) {
        // Allowing only one active snapshot at a time.
        if self
            .snapshot_in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            println!("Another snapshot is in progress, skipping this snapshot request.");
            return;
        }

        // Create a vfs for the snapshot
        let mut vfs_guard = self.vfs.write().unwrap();
        let snapshot_vfs = make_vfs(&tree.config.snapshot_backend, snapshot_file_path);
        let old_vfs = std::mem::replace(&mut *vfs_guard, snapshot_vfs);
        drop(old_vfs);

        // Reset the snapshot vfs before use
        vfs_guard.reset();

        // Drop the guard
        drop(vfs_guard);

        // Initialize a inner node and leaf node mapping for the sweep
        let mut sweep_inner_mapping: Vec<(*const InnerNode, usize)> = Vec::new();
        let mut sweep_mini_mapping: Vec<(PageID, usize)> = Vec::new();
        let mut sweep_mini_size_mapping: Vec<(PageID, usize)> = Vec::new();
        let mut sweep_base_mapping: Vec<(PageID, usize)> = Vec::new();

        // At the beginning, the global phase id must be 0 (REST).
        let mut current_global_phase_id = self.get_global_phase_id();
        assert_eq!(current_global_phase_id, PhaseId::Rest);

        // Version of the ongoing snapshot is set
        let snapshot_version = self.get_global_version();

        // Immediately move the global state to (1 (PREPARE), snapshot_version)
        let mut current_global_state = self.advance_global_state();
        current_global_phase_id = self.get_global_phase_id();
        assert_eq!(current_global_phase_id, PhaseId::Prepare);
        assert_eq!(snapshot_version, self.get_global_version());

        let mut leaf_node_count_upper_bound = 0; // Indicate the total number of leaf nodes in the captured snapshot.

        loop {
            if self.check_if_phase_completed(current_global_state) {
                match current_global_phase_id {
                    PhaseId::Rest => {
                        // Upon reaching here, all user threads are in (REST, snapshot_version + 1).
                        // All snapshots of pages of snapshot_version are done, and no more snapshot operations
                        // neither. As such, we can safely finalize the snapshot by writing out the
                        // metadata and page mappings.
                        self.finalize(
                            snapshot_version,
                            &mut sweep_inner_mapping,
                            &mut sweep_mini_mapping,
                            &mut sweep_mini_size_mapping,
                            &mut sweep_base_mapping,
                            leaf_node_count_upper_bound,
                            tree.config.clone(),
                        );

                        // Close the snapshot vfs
                        let mut vfs_guard = self.vfs.write().unwrap();
                        let snapshot_vfs = make_vfs(&StorageBackend::Memory, ":memory:");
                        let old_vfs = std::mem::replace(&mut *vfs_guard, snapshot_vfs);
                        drop(old_vfs);
                        drop(vfs_guard);

                        // The snapshot is done.
                        break;
                    }
                    PhaseId::Prepare => {
                        current_global_state = self.advance_global_state();
                        current_global_phase_id = self.get_global_phase_id();
                        assert_eq!(current_global_phase_id, PhaseId::InProgress);
                        assert_eq!(snapshot_version + 1, self.get_global_version());
                    }
                    PhaseId::InProgress => {
                        current_global_state = self.advance_global_state();
                        current_global_phase_id = self.get_global_phase_id();
                        assert_eq!(current_global_phase_id, PhaseId::Sweep);
                        assert_eq!(snapshot_version + 1, self.get_global_version());
                    }
                    PhaseId::Sweep => {
                        // Upon reaching here, there are no user threads with snapshot_version anymore.
                        // Sweep through and snapshot all data pages whose version is less than (snapshot_version + 1)
                        // and build inner node and leaf node mapping for those pages.
                        // Upon completion, all pages with version less than (snapshot_version + 1) should have been captured in the snapshot.
                        // Note that, there could be duplicate snapshots during and after the sweep as user threads in 'SWEEPING'
                        // state keeps taking snapshots of pages even if they have been captured by the sweep as the sweep does not
                        // alter page versions. De-duplication is needed when finalizing the snapshot file.
                        leaf_node_count_upper_bound = self.sweep(
                            tree,
                            snapshot_version + 1,
                            &mut sweep_inner_mapping,
                            &mut sweep_mini_mapping,
                            &mut sweep_mini_size_mapping,
                            &mut sweep_base_mapping,
                        );

                        current_global_state = self.advance_global_state();
                        current_global_phase_id = self.get_global_phase_id();
                        assert_eq!(current_global_phase_id, PhaseId::Rest);
                        assert_eq!(snapshot_version + 1, self.get_global_version());
                    }
                }
            }

            // At most wasting 1 second per state transition.
            thread::sleep(std::time::Duration::from_secs(1));

            #[cfg(all(feature = "shuttle", test))]
            shuttle::thread::yield_now();
        }

        assert!(self
            .snapshot_in_progress
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok());
    }

    /// Recover a Bf-tree from an existing snapshot file at recovery_snapshot_file_path.
    ///
    /// Most configuration params are directly retrieved from the snapshot file
    /// with the following ones for the caller to specify:
    /// 1. use_snapshot: whether the new tree will also support snapshot.
    /// 2. new_snapshot_file_path: If use_snapshot, then provide the path of the snapshot file of the newly recovered Bf-tree.
    ///    Note that, this path must be different from the recovery_snapshot_file_path.
    /// 3. buffer_ptr: optional pointer to a pre-allocated buffer for the newly recovered Bf-tree
    /// 4. buffer_size optional override of the buffer size retrieved from the snapshot file.
    ///    Note that, if the newly specified value is smaller than the one retrieved from the snapshot file,
    ///    then failure to restore a tree might happen. This is because during recovery, the memory cache pages
    ///    of the tree at the moment of snapshot are reinstated into memory too.
    /// 5. wal_config: optional write-ahead log configuration for the newly recovered Bf-tree
    pub fn new_from_snapshot(
        recovery_snapshot_file_path: impl AsRef<Path>, //  The snapshot file to recover from
        use_snapshot: bool,
        buffer_ptr: Option<*mut u8>,
        buffer_size: Option<usize>, // buffer size of the newly created Bf-tree
        wal_config: Option<Arc<WalConfig>>,
    ) -> Result<BfTree, ConfigError> {
        // Check the recovery file is valid
        if !recovery_snapshot_file_path.as_ref().exists() {
            // if not already exist, we just create a new empty file at the location.
            return Err(ConfigError::SnapshotFileInvalid(
                "Not found ".to_string() + recovery_snapshot_file_path.as_ref().to_str().unwrap(),
            ));
        }

        // Create WAL, if specified
        let wal = wal_config.as_ref().map(|s| WriteAheadLog::new(s.clone()));

        // Retrieve the header of the snapshot file and construct a valid config for the to-be-recovered Bf-tree
        let reader = std::fs::File::open(recovery_snapshot_file_path.as_ref()).unwrap();
        let mut metadata = SectorAlignedVector::new_zeroed(DISK_PAGE_SIZE); // Metadata is at most one disk page in size
        #[cfg(unix)]
        {
            reader.read_at(&mut metadata, 0).unwrap();
        }
        #[cfg(windows)]
        {
            reader.seek_read(&mut metadata, 0).unwrap();
        }

        let bf_meta = unsafe { (metadata.as_ptr() as *const BfTreeMeta).read() };
        bf_meta.check_magic();
        assert_eq!(reader.metadata().unwrap().len(), bf_meta.file_size);

        let mut bf_tree_config = Config::new_from_snapshot(&bf_meta);

        let recovery_snapshot_file_backend = StorageBackend::Std; // TODO, recover storage backend from snapshot file
        if !bf_tree_config.cache_only {
            bf_tree_config.file_path(recovery_snapshot_file_path.as_ref());
            // The storage backend should use the same vfs system as the recovery snapshot file
            bf_tree_config.storage_backend = recovery_snapshot_file_backend.clone();
        } else {
            bf_tree_config.storage_backend = StorageBackend::Memory;
        }
        bf_tree_config.use_snapshot = use_snapshot;

        bf_tree_config.snapshot_backend = StorageBackend::Std; // TODO, allow user chosen snapshot backend

        let snapshot_mgr = if bf_tree_config.use_snapshot {
            Some(Arc::new(CPRSnapShotMgr::new(
                bf_tree_config.snapshot_version,
            )))
        } else {
            None
        };

        if let Some(size) = buffer_size {
            bf_tree_config.cb_size_byte = size
        }

        let size_classes = BfTree::create_mem_page_size_classes(
            bf_tree_config.cb_min_record_size,
            bf_tree_config.cb_max_record_size,
            bf_tree_config.leaf_page_size,
            bf_tree_config.max_fence_len,
            bf_tree_config.cache_only,
        );

        bf_tree_config.write_ahead_log = wal_config.clone();
        bf_tree_config.validate()?;

        let config = Arc::new(bf_tree_config);

        // Start Bf-Tree re-construction using recovery_snapshot_file_path and the config
        let recovery_snapshot_vfs = make_vfs(
            &recovery_snapshot_file_backend,
            recovery_snapshot_file_path.as_ref(),
        );

        // Step 1: reconstruct inner nodes.
        let mut root_page_id = bf_meta.root_id;
        let mut inner_node_page_buffer = SectorAlignedVector::new_zeroed(INNER_NODE_SIZE);
        if root_page_id.is_inner_node_pointer() {
            let inner_mapping: Vec<(*const InnerNode, usize)> = read_vec_from_offset(
                bf_meta.inner_offset,
                bf_meta.inner_size,
                &recovery_snapshot_vfs,
            );

            // Sanity check on root nodes
            let mut root_cnt = 0;
            for (_ptr, offset) in &inner_mapping {
                recovery_snapshot_vfs.read(*offset, &mut inner_node_page_buffer);
                let inner_node = InnerNodeBuilder::new().build_from_slice(&inner_node_page_buffer);
                if unsafe { (*inner_node).is_root() } {
                    root_cnt += 1;
                }
                InnerNode::free_node(inner_node);
            }
            assert_eq!(root_cnt, 1, "Root count in inner mapping: {}", root_cnt);

            let mut inner_map = HashMap::new();

            for m in inner_mapping {
                inner_map.insert(m.0, m.1);
            }
            let offset = inner_map.get(&root_page_id.as_inner_node()).unwrap();
            recovery_snapshot_vfs.read(*offset, &mut inner_node_page_buffer);
            let root_page = InnerNodeBuilder::new().build_from_slice(&inner_node_page_buffer);

            // No need for disk offset of a inner node.
            unsafe {
                (*root_page).set_disk_offset(INVALID_DISK_OFFSET as u64);
            }
            // Mark the root node
            unsafe {
                (*root_page).set_root(true);
            }
            root_page_id = PageID::from_pointer(root_page);

            let mut inner_resolve_queue = VecDeque::from([root_page]);
            while !inner_resolve_queue.is_empty() {
                let inner_ptr = inner_resolve_queue.pop_front().unwrap();
                let mut inner = ReadGuard::try_read(inner_ptr).unwrap().upgrade().unwrap();
                if inner.as_ref().meta.children_is_leaf() {
                    continue;
                }
                for (idx, c) in inner.as_ref().get_child_iter().enumerate() {
                    let offset = inner_map.get(&c.as_inner_node()).unwrap();
                    recovery_snapshot_vfs.read(*offset, &mut inner_node_page_buffer);
                    let inner_page =
                        InnerNodeBuilder::new().build_from_slice(&inner_node_page_buffer);
                    unsafe {
                        (*inner_page).set_disk_offset(INVALID_DISK_OFFSET as u64);
                    }
                    let inner_id = PageID::from_pointer(inner_page);
                    inner.as_mut().update_at_pos(idx, inner_id);
                    inner_resolve_queue.push_back(inner_page);
                }
            }
        }

        let raw_root_id = if root_page_id.is_id() {
            root_page_id.raw() | BfTree::ROOT_IS_LEAF_MASK
        } else {
            root_page_id.raw()
        };

        // Step 2. Reconstruct the page table and leaf pages.
        // Here we differentiate cache-only from disk-backed Bf-trees as their recovery process differs.
        if !bf_meta.cache_only {
            // For disk backed bf-tree, we reconstruct the page table using base pages.
            let base_mapping: Vec<(PageID, usize)> = if bf_meta.base_size > 0 {
                read_vec_from_offset(
                    bf_meta.base_offset,
                    bf_meta.base_size,
                    &recovery_snapshot_vfs,
                )
            } else {
                Vec::new()
            };

            let base_page_loc_mapping = base_mapping.into_iter().map(|(pid, offset)| {
                let loc = PageLocation::Base(offset);
                (pid, loc)
            });

            // The file system of the newly constructed Bf-tree is the recovery_snapshot_file
            let pt = PageTable::new_from_mapping(
                base_page_loc_mapping,
                recovery_snapshot_vfs.clone(),
                config.clone(),
                snapshot_mgr.clone(),
            );

            let circular_buffer = CircularBuffer::new(
                config.cb_size_byte,
                config.cb_copy_on_access_ratio,
                config.cb_min_record_size,
                config.cb_max_record_size,
                config.leaf_page_size,
                config.max_fence_len,
                buffer_ptr,
                config.cache_only,
            );

            // Next, we restore the mini-pages and wire them to the corresponding base pages and update the page table.
            let mini_mapping: Vec<(PageID, usize)> = if bf_meta.mini_size > 0 {
                read_vec_from_offset(
                    bf_meta.mini_offset,
                    bf_meta.mini_size,
                    &recovery_snapshot_vfs,
                )
            } else {
                Vec::new()
            };

            let mini_size_mapping: Vec<(PageID, usize)> = if bf_meta.mini_size_size > 0 {
                read_vec_from_offset(
                    bf_meta.mini_size_offset,
                    bf_meta.mini_size_size,
                    &recovery_snapshot_vfs,
                )
            } else {
                Vec::new()
            };

            let mut mini_size_mapping_unique: HashMap<PageID, usize> = HashMap::new();
            for (pid, size) in mini_size_mapping {
                mini_size_mapping_unique.insert(pid, size);
            }

            // Create the storage system first before allocating mini-pages.
            let storage =
                LeafStorage::new_inner(config.clone(), pt, circular_buffer, recovery_snapshot_vfs);

            for (pid, offset) in &mini_mapping {
                let mini_size = *mini_size_mapping_unique.get(pid).unwrap();

                // Allocate space in memory for new mini-page
                let mini_page_guard = match storage.alloc_mini_page(mini_size) {
                    Ok(mini_page_ptr) => mini_page_ptr,
                    Err(_) => {
                        return Err(ConfigError::CircularBufferSize("buffer size set too small. Consider increasing it or not specifying at all".to_string()));
                    }
                };

                // Copy mini-page from snapshot file to the newly allocated mini-page
                let mut page_buffer = SectorAlignedVector::new_zeroed(mini_size);
                storage.vfs.read(*offset, &mut page_buffer);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        page_buffer.as_ptr(),
                        mini_page_guard.as_ptr(),
                        mini_size,
                    );
                }

                // Connect the new mini-page to the corresponding base page in page table
                let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                let mini_page = unsafe { &mut *new_mini_ptr };

                let mut base_page = storage.page_table.get_mut(pid);
                let page_loc = base_page.get_page_location().clone();
                match page_loc {
                    PageLocation::Base(off) => {
                        mini_page.next_level = MiniPageNextLevel::new(off);
                    }
                    _ => {
                        panic!("Unexpected page location for base page");
                    }
                }
                let mini_loc = PageLocation::Mini(new_mini_ptr);

                // Replace the base page with the mini-page in the page table
                base_page.create_cache_page_loc(mini_loc);
            }

            Ok(BfTree {
                storage,
                root_page_id: AtomicU64::new(raw_root_id),
                wal,
                write_load_full_page: config.write_load_full_page,
                cache_only: false,
                mini_page_size_classes: size_classes,
                snapshot_mgr,
                config,
                #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
                metrics_recorder: Some(Arc::new(ThreadLocal::new())),
            })
        } else {
            // For cache-only mode, we create a new page table with NULL pages
            let mini_mapping_unallocated: Vec<(PageID, PageLocation)> = (0..bf_meta.leaf_page_num)
                .map(|pid| (PageID::from_id(pid as u64), PageLocation::Null))
                .collect();

            // Then we restore the mini-pages and replace the corresponding Null pages and update the page table.
            let mini_mapping: Vec<(PageID, usize)> = if bf_meta.mini_size > 0 {
                read_vec_from_offset(
                    bf_meta.mini_offset,
                    bf_meta.mini_size,
                    &recovery_snapshot_vfs,
                )
            } else {
                Vec::new()
            };

            let mini_size_mapping: Vec<(PageID, usize)> = if bf_meta.mini_size_size > 0 {
                read_vec_from_offset(
                    bf_meta.mini_size_offset,
                    bf_meta.mini_size_size,
                    &recovery_snapshot_vfs,
                )
            } else {
                Vec::new()
            };

            let mut mini_size_mapping_unique: HashMap<PageID, usize> = HashMap::new();
            for (pid, size) in mini_size_mapping {
                mini_size_mapping_unique.insert(pid, size);
            }

            // For cache-only mode, the file system of the new bf-tree is memory-based
            let storage_vfs = make_vfs(&config.storage_backend, PathBuf::new());

            let pt = PageTable::new_from_mapping(
                mini_mapping_unallocated.into_iter(),
                storage_vfs.clone(),
                config.clone(),
                snapshot_mgr.clone(),
            );

            let circular_buffer = CircularBuffer::new(
                config.cb_size_byte,
                config.cb_copy_on_access_ratio,
                config.cb_min_record_size,
                config.cb_max_record_size,
                config.leaf_page_size,
                config.max_fence_len,
                buffer_ptr,
                config.cache_only,
            );

            // Create a memory-based storage system before allocating mini-pages.
            let storage = LeafStorage::new_inner(config.clone(), pt, circular_buffer, storage_vfs);

            for (pid, offset) in &mini_mapping {
                let mini_size = *mini_size_mapping_unique.get(pid).unwrap();

                // Skip over null pages as the default is Null in the page table already
                if *offset == NULL_PAGE_LOCATION_OFFSET {
                    continue;
                }

                // Allocate memory for a new mini-page in storage
                let mini_page_guard = match storage.alloc_mini_page(mini_size) {
                    Ok(mini_page_ptr) => mini_page_ptr,
                    Err(_) => {
                        panic!("Please increase cb_size_byte in config");
                    }
                };

                // Copy mini-page from snapshot file to the newly allocated space.
                let mut page_buffer = SectorAlignedVector::new_zeroed(mini_size);
                recovery_snapshot_vfs.read(*offset, &mut page_buffer);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        page_buffer.as_ptr(),
                        mini_page_guard.as_ptr(),
                        mini_size,
                    );
                }

                // Set its next level to oblivion
                let mini_page_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                let mini_page = unsafe { &mut *mini_page_ptr };
                mini_page.next_level = MiniPageNextLevel::new_null();

                // Update the corresponding page location
                let mut null_page = storage.page_table.get_mut(pid);
                let page_loc = null_page.get_page_location().clone();
                match page_loc {
                    PageLocation::Null => {
                        let mini_loc = PageLocation::Mini(mini_page_ptr);
                        null_page.create_cache_page_loc(mini_loc);
                    }
                    _ => {
                        panic!("Unexpected page location for null page");
                    }
                }
            }
            Ok(BfTree {
                storage,
                root_page_id: AtomicU64::new(raw_root_id),
                wal,
                write_load_full_page: config.write_load_full_page,
                cache_only: true,
                mini_page_size_classes: size_classes,
                snapshot_mgr,
                config,
                #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
                metrics_recorder: Some(Arc::new(ThreadLocal::new())),
            })
        }
    }
}

impl BfTree {
    /// Recovery a Bf-Tree from a cpr snapshot and WAL files.
    /// Incomplete function, internal use only
    pub fn recovery(
        recovery_snapshot_file_path: PathBuf, //  The snapshot file to recover from
        wal_file: impl AsRef<Path>,
        use_snapshot: bool,
        buffer_ptr: Option<*mut u8>,
        buffer_size: Option<usize>,
        wal: Option<Arc<WalConfig>>,
    ) {
        let bf_tree = BfTree::new_from_cpr_snapshot(
            recovery_snapshot_file_path,
            use_snapshot,
            buffer_ptr,
            buffer_size,
            wal,
        )
        .unwrap();
        let wal_reader = WalReader::new(wal_file, 4096);

        for seg in wal_reader.segment_iter() {
            for entry in seg.entry_iter() {
                let log_entry = LogEntry::read_from_buffer(entry.1);
                match log_entry {
                    LogEntry::Write(op) => {
                        bf_tree.insert(op.key, op.value);
                    }
                    LogEntry::Split(_op) => {
                        todo!("implement split op in wal!")
                    }
                }
            }
        }
    }

    /// Take a new CPR snapshot
    pub fn cpr_snapshot(&self, snapshot_file_path: impl AsRef<Path>) {
        if !self.config.use_snapshot {
            panic!("Snapshots are not enabled in the configuration");
        }

        let snpshot_mgr = self.snapshot_mgr.clone().unwrap();
        snpshot_mgr.snapshot(self, snapshot_file_path);
    }

    /// Recover a BfTree from a CPR snapshot
    pub fn new_from_cpr_snapshot(
        recovery_snapshot_file_path: impl AsRef<Path>, //  The snapshot file to recover from
        use_snapshot: bool,
        buffer_ptr: Option<*mut u8>,
        buffer_size: Option<usize>,
        wal: Option<Arc<WalConfig>>,
    ) -> Result<BfTree, ConfigError> {
        CPRSnapShotMgr::new_from_snapshot(
            recovery_snapshot_file_path,
            use_snapshot,
            buffer_ptr,
            buffer_size,
            wal,
        )
    }

    /// Check if all threads are running in the next version of the current snapshot
    pub fn are_all_threads_in_next_snapshot_version(&self) -> bool {
        if let Some(snapshot_mgr) = &self.snapshot_mgr {
            return snapshot_mgr.are_all_threads_in_next_version();
        }
        false
    }
}

struct SectorAlignedVector {
    inner: ManuallyDrop<Vec<u8>>,
}

impl Drop for SectorAlignedVector {
    fn drop(&mut self) {
        let layout =
            std::alloc::Layout::from_size_align(self.inner.capacity(), SECTOR_SIZE).unwrap();
        let ptr = self.inner.as_mut_ptr();
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
}

impl SectorAlignedVector {
    fn new_zeroed(capacity: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(capacity, SECTOR_SIZE).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };

        let inner = unsafe { Vec::from_raw_parts(ptr, capacity, capacity) };
        Self {
            inner: ManuallyDrop::new(inner),
        }
    }
}

impl Deref for SectorAlignedVector {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for SectorAlignedVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// We use repr(C) for simplicity, maybe flatbuffer or bincode or even repr(Rust) is better.
/// But we don't care about the space here.
/// I don't want to introduce giant dependencies just for this.
#[repr(C, align(512))]
pub(crate) struct BfTreeMeta {
    magic_begin: [u8; 16],
    // Snapshot file metadata
    root_id: PageID,
    inner_offset: usize,
    inner_size: usize,
    mini_offset: usize,
    mini_size: usize,
    mini_size_offset: usize,
    mini_size_size: usize,
    base_offset: usize,
    base_size: usize,
    file_size: u64,
    leaf_page_num: usize,
    // Bf-tree configuration of the snapshot
    pub(crate) cb_size_byte: usize,
    pub(crate) snapshot_version: u64,
    pub(crate) cache_only: bool,
    pub(crate) read_promotion_rate: usize,
    pub(crate) scan_promotion_rate: usize,
    pub(crate) cb_min_record_size: usize,
    pub(crate) cb_max_record_size: usize,
    pub(crate) leaf_page_size: usize,
    pub(crate) cb_max_key_len: usize,
    pub(crate) max_fence_len: usize,
    pub(crate) cb_copy_on_access_ratio: f64,
    pub(crate) read_record_cache: bool,
    pub(crate) max_mini_page_size: usize,
    pub(crate) mini_page_binary_search: bool,
    pub(crate) write_load_full_page: bool,
    magic_end: [u8; 14],
}
const _: () = assert!(std::mem::size_of::<BfTreeMeta>() <= DISK_PAGE_SIZE);

impl BfTreeMeta {
    fn as_slice(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    fn check_magic(&self) {
        assert_eq!(self.magic_begin, *BF_TREE_MAGIC_BEGIN);
        assert_eq!(self.magic_end, *BF_TREE_MAGIC_END);
    }
}

/// Returns starting offset and total size written to disk.
fn serialize_vec_to_disk<T>(v: &[T], vfs: &Arc<dyn VfsImpl>) -> (usize, usize) {
    if v.is_empty() {
        return (0, 0);
    }
    let unaligned_ptr = v.as_ptr() as *const u8;
    let unaligned_size = std::mem::size_of_val(v);

    let aligned_size = align_to_sector_size(unaligned_size);
    let layout = std::alloc::Layout::from_size_align(aligned_size, SECTOR_SIZE).unwrap();
    unsafe {
        let aligned_ptr = std::alloc::alloc_zeroed(layout);
        std::ptr::copy_nonoverlapping(unaligned_ptr, aligned_ptr, unaligned_size);
        let slice = std::slice::from_raw_parts(aligned_ptr, aligned_size);
        let offset = serialize_u8_slice_to_disk(slice, vfs);
        std::alloc::dealloc(aligned_ptr, layout);
        (offset, unaligned_size)
    }
}

fn read_vec_from_offset<T: Clone>(offset: usize, size: usize, vfs: &Arc<dyn VfsImpl>) -> Vec<T> {
    assert!(size > 0);
    let slice = read_u8_slice_from_disk(offset, size, vfs);
    let ptr = slice.as_ptr() as *const T;
    let size = size / std::mem::size_of::<T>();
    let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
    slice.to_vec()
}

fn read_u8_slice_from_disk(offset: usize, size: usize, vfs: &Arc<dyn VfsImpl>) -> Vec<u8> {
    let mut res = Vec::new();
    let mut buffer = vec![0; DISK_PAGE_SIZE];
    for i in (0..size).step_by(DISK_PAGE_SIZE) {
        vfs.read(offset + i, &mut buffer); // Read one disk page at a time
        res.extend_from_slice(&buffer);
    }
    res
}

const SECTOR_SIZE: usize = 512;

fn align_to_sector_size(n: usize) -> usize {
    (n + SECTOR_SIZE - 1) & !(SECTOR_SIZE - 1)
}

/// Write a slice to disk and return the start offset and page count.
/// TODO: we should not just return offset and count, because the offset is not necessarily continuos.
///     We should return a Vec of offsets. But let's keep it simple for fast prototype.
fn serialize_u8_slice_to_disk(slice: &[u8], vfs: &Arc<dyn VfsImpl>) -> usize {
    let mut start_offset = None;
    for chunk in slice.chunks(DISK_PAGE_SIZE) {
        let offset = vfs.alloc_offset(DISK_PAGE_SIZE); // Write one disk page at a time
        if start_offset.is_none() {
            start_offset = Some(offset);
        }
        vfs.write(offset, chunk);
    }
    start_offset.unwrap()
}

#[cfg(test)]
mod tests {
    use crate::{nodes::leaf_node::LeafReadResult, sync::thread, BfTree, Config};
    use std::panic;
    #[cfg(feature = "shuttle")]
    use std::path::PathBuf;
    use std::str::FromStr;
    use std::sync::atomic::Ordering;
    use std::sync::{atomic::AtomicBool, Arc};

    /// Multiple writer threads write to a BfTree in parallel while a separate thread taking multiple snapshots
    /// A new BfTree recovered from the snapshot should contain a prefix of all the inserts from each writer thread.
    /// A snapshot taken later should cover the previous snapshots.
    #[test]
    fn cpr_snapshot_disk() {
        // Install a panic hook that triggers the just-in-time debugger (e.g. VS debugger)
        // so we can inspect the state at the point of failure.
        panic::set_hook(Box::new(|info| {
            eprintln!("PANIC: {info}");
            unsafe { std::arch::asm!("int 3") };
        }));

        let min_record_size: usize = 64;
        let max_record_size: usize = 2408;
        let leaf_page_size: usize = 8192;
        let snapshot_num: usize = 10;
        let num_threads: usize = 4;
        let file_path: String = "target/test_simple.bftree".to_string();
        let snapshot_file_path: String = "target/test_simple_snapshot.bftree".to_string();

        let tmp_file_path = std::path::PathBuf::from_str(&file_path).unwrap();
        let tmp_snapshot_file_path = std::path::PathBuf::from_str(&snapshot_file_path).unwrap();

        let mut config = Config::new(&tmp_file_path, 128 * 1024); // 128KB buffer pool. insert/split/eviction all triggered
        config.storage_backend(crate::StorageBackend::Std);
        config.cb_min_record_size = min_record_size + 2 * std::mem::size_of::<usize>();
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = min_record_size + 2 * std::mem::size_of::<usize>();
        config.use_snapshot(true);

        let bftree = Arc::new(BfTree::with_config(config.clone(), None).unwrap());
        let finish = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let finish_clone = finish.clone();
                let bftree_clone = bftree.clone();

                thread::spawn(move || {
                    let key_len: usize = min_record_size / 2 + std::mem::size_of::<usize>();
                    assert!(key_len * 2 <= max_record_size);
                    let mut key_buffer = vec![0usize; key_len / std::mem::size_of::<usize>()];

                    let mut r: usize = 0;
                    while !finish_clone.load(Ordering::Relaxed) {
                        key_buffer.fill(r);
                        key_buffer[0] = i;

                        match bftree_clone.insert(
                            bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                            bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                        ) {
                            crate::LeafInsertResult::Success => {}
                            _ => {
                                panic!("Insert failed");
                            }
                        }
                        r += 1;
                    }
                    r
                })
            })
            .collect();

        thread::sleep(std::time::Duration::from_secs(5));
        for _ in 0..snapshot_num {
            // take a snapshot
            let _ = std::fs::remove_file(&tmp_snapshot_file_path);
            bftree.cpr_snapshot(&tmp_snapshot_file_path);
            thread::sleep(std::time::Duration::from_secs(5));
        }

        // Stop all writer threads
        let mut rs = vec![0usize; num_threads];
        finish.store(true, Ordering::Relaxed);
        for (i, h) in handles.into_iter().enumerate() {
            let r = h.join().unwrap();
            rs[i] = r;
        }

        verify_snapshot_recovery(
            &tmp_snapshot_file_path,
            num_threads,
            min_record_size,
            &rs,
            true,
        );

        std::fs::remove_file(tmp_file_path).unwrap();
        std::fs::remove_file(tmp_snapshot_file_path).unwrap();
    }

    /// Testing snapshot for cache-only mode with std::thread
    #[test]
    fn cpr_snapshot_cache_only() {
        let min_record_size: usize = 64;
        let max_record_size: usize = 2408;
        let leaf_page_size: usize = 8192;
        let num_threads: usize = 4;

        let snapshot_file_path: String =
            "target/test_simple_cache_only_snapshot.bftree".to_string();
        let tmp_snapshot_file_path = std::path::PathBuf::from_str(&snapshot_file_path).unwrap();

        let mut config = Config::default(); // Creat a CB that can hold 16 full pages
        config.storage_backend(crate::StorageBackend::Memory);
        config.file_path(":memory:");
        config.cache_only = true;
        config.cb_size_byte(1024 * 1024 * 1024);
        config.cb_min_record_size = min_record_size;
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = max_record_size;
        config.use_snapshot(true);

        let bftree = Arc::new(BfTree::with_config(config.clone(), None).unwrap());
        let finish = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let finish_clone = finish.clone();
                let bftree_clone = bftree.clone();

                thread::spawn(move || {
                    let key_len: usize = min_record_size / 2 + std::mem::size_of::<usize>();
                    assert!(key_len * 2 <= max_record_size);
                    let mut key_buffer = vec![0usize; key_len / std::mem::size_of::<usize>()];

                    let mut r: usize = 0;
                    while !finish_clone.load(Ordering::Relaxed) {
                        key_buffer.fill(r);
                        key_buffer[0] = i;

                        match bftree_clone.insert(
                            bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                            bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                        ) {
                            crate::LeafInsertResult::Success => {}
                            _ => {
                                panic!("Insert failed");
                            }
                        }
                        r += 1;
                    }
                    r
                })
            })
            .collect();

        thread::sleep(std::time::Duration::from_secs(5));
        // take a snapshot
        bftree.cpr_snapshot(&tmp_snapshot_file_path);
        thread::sleep(std::time::Duration::from_secs(5));

        // Stop all writer threads
        let mut rs = vec![0usize; num_threads];
        finish.store(true, Ordering::Relaxed);
        for (i, h) in handles.into_iter().enumerate() {
            let r = h.join().unwrap();
            rs[i] = r;
        }

        verify_snapshot_recovery(
            &tmp_snapshot_file_path,
            num_threads,
            min_record_size,
            &rs,
            false,
        );

        std::fs::remove_file(tmp_snapshot_file_path).unwrap();
    }

    fn verify_snapshot_recovery(
        snapshot_file: impl AsRef<std::path::Path>,
        num_threads: usize,
        min_record_size: usize,
        records_num_per_threads: &Vec<usize>,
        check_prefix: bool,
    ) {
        let bftree = BfTree::new_from_cpr_snapshot(snapshot_file, false, None, None, None)
            .expect("fail to recover from snapshot");

        let mut rs_captured = vec![0usize; num_threads];
        for i in 0..num_threads {
            let record_num = records_num_per_threads[i];

            let key_len: usize = min_record_size / 2 + std::mem::size_of::<usize>();
            let mut key_buffer = vec![0usize; key_len / std::mem::size_of::<usize>()];
            let mut res_buffer = vec![0u8; key_len];
            let mut not_included = false;
            let mut first_gap_record: Option<usize> = None;

            for r in 0..record_num {
                key_buffer.fill(r);
                key_buffer[0] = i;

                match bftree.read(
                    bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                    &mut res_buffer,
                ) {
                    LeafReadResult::Found(v) => {
                        if check_prefix && not_included {
                            // Gather diagnostic info: scan forward to find all gaps
                            let mut gaps = Vec::new();
                            let mut found_after = Vec::new();
                            let gap_start = first_gap_record.unwrap();
                            // Collect all gaps in the range [gap_start..r+50]
                            let scan_end = std::cmp::min(r + 50, record_num);
                            for scan_r in gap_start..scan_end {
                                key_buffer.fill(scan_r);
                                key_buffer[0] = i;
                                match bftree.read(
                                    bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                                    &mut res_buffer,
                                ) {
                                    LeafReadResult::Found(_) => {
                                        found_after.push(scan_r);
                                    }
                                    LeafReadResult::NotFound => {
                                        gaps.push(scan_r);
                                    }
                                    _ => {}
                                }
                            }
                            panic!(
                                "PREFIX VIOLATION: thread={}, first_gap_at={}, found_record_after_gap={}, \
                                 total_captured_before_gap={}, total_records={}\n\
                                 Gaps in [{}, {}): {:?}\n\
                                 Found in [{}, {}): {:?}",
                                i, gap_start, r, rs_captured[i], record_num,
                                gap_start, scan_end, &gaps[..std::cmp::min(gaps.len(), 20)],
                                gap_start, scan_end, &found_after[..std::cmp::min(found_after.len(), 20)],
                            );
                        }
                        assert_eq!(v as usize, key_len);
                        assert_eq!(
                            &res_buffer,
                            bytemuck::must_cast_slice::<usize, u8>(&key_buffer)
                        );
                        rs_captured[i] += 1;
                    }
                    LeafReadResult::NotFound => {
                        if !not_included {
                            not_included = true;
                            first_gap_record = Some(r);
                        }
                    }
                    _ => {
                        panic!("Unexpected read result")
                    }
                }
            }

            assert!(rs_captured[i] <= record_num);
            println!("Total inserted records for thread {}: {}", i, record_num);
            println!(
                "Hit ratio for thread {}: {}",
                i,
                rs_captured[i] as f64 / record_num as f64
            );
        }
    }

    /// Inner body for the cache-only CPR snapshot test, parameterized by an
    /// iteration id so that concurrent shuttle replicas (and successive shuttle
    /// iterations) do not collide on the snapshot file path.
    #[cfg(feature = "shuttle")]
    fn shuttle_cpr_snapshot_cache_only_inner(iter: usize) {
        let min_record_size: usize = 64;
        let max_record_size: usize = 2408;
        let leaf_page_size: usize = 8192;
        let num_threads: usize = 4;
        // Bounded number of inserts per writer so shuttle iterations finish quickly.
        let inserts_per_thread: usize = 1_000; // 1K inserts per thread

        let snapshot_file_path: String = format!(
            "target/shuttle_cpr_snapshot_cache_only_{}_{}.bftree",
            std::process::id(),
            iter,
        );
        let tmp_snapshot_file_path = std::path::PathBuf::from_str(&snapshot_file_path).unwrap();

        let mut config = Config::default();
        config.storage_backend(crate::StorageBackend::Memory);
        config.file_path(":memory:");
        config.cache_only = true;
        // Use a buffer sufficient for the test data. 128KB is more than enough
        // for 16 small records and avoids allocating 1GB per shuttle iteration.
        config.cb_size_byte(1024 * 1024 * 1024);
        config.cb_min_record_size = min_record_size;
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = max_record_size;
        config.use_snapshot(true);

        let bftree = Arc::new(BfTree::with_config(config.clone(), None).unwrap());
        let mut rs = vec![0usize; num_threads];

        for j in 0..2 {
            let handles: Vec<_> = (0..num_threads)
                .map(|i| {
                    let bftree_clone = bftree.clone();
                    let start_id = j * inserts_per_thread;
                    let end_id = start_id + inserts_per_thread;
                    thread::spawn(move || {
                        let key_len: usize = min_record_size / 2 + std::mem::size_of::<usize>();
                        assert!(key_len * 2 <= max_record_size);
                        let mut key_buffer = vec![0usize; key_len / std::mem::size_of::<usize>()];

                        for r in start_id..end_id {
                            key_buffer.fill(r);
                            key_buffer[0] = i;

                            match bftree_clone.insert(
                                bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                                bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                            ) {
                                crate::LeafInsertResult::Success => {}
                                _ => {
                                    panic!("Insert failed");
                                }
                            }
                        }
                        inserts_per_thread
                    })
                })
                .collect();

            // Snapshot thread: takes the snapshot concurrently with the writers.
            let bftree_for_snap = bftree.clone();
            let snap_path = tmp_snapshot_file_path.clone();
            let snap_handle = thread::spawn(move || {
                bftree_for_snap.cpr_snapshot(&snap_path);
            });

            for (i, h) in handles.into_iter().enumerate() {
                let r = h.join().unwrap();
                rs[i] += r;
            }

            // Verify the snapshot taken is valid
            snap_handle.join().unwrap();
            let snap_path = tmp_snapshot_file_path.clone();
            verify_snapshot_recovery(&snap_path, num_threads, min_record_size, &rs, false);
        }

        let _ = std::fs::remove_file(tmp_snapshot_file_path);
    }

    /// Testing
    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_cpr_snapshot_cache_only() {
        use std::sync::atomic::AtomicUsize;

        // Unique iteration id so portfolio replicas / successive iterations do
        // not collide on the snapshot file path.
        static ITER: AtomicUsize = AtomicUsize::new(0);

        let mut shuttle_config = shuttle::Config::default();
        //shuttle_config.max_steps = shuttle::MaxSteps::FailAfter(100_000);
        shuttle_config.max_steps = shuttle::MaxSteps::None;
        shuttle_config.stack_size = 1024 * 1024 * 1024; // 1GB — default 32KB overflows with deep tree ops
        shuttle_config.failure_persistence =
            shuttle::FailurePersistence::File(Some(PathBuf::from_str("target").unwrap()));

        let mut runner = shuttle::PortfolioRunner::new(true, shuttle_config);
        let available_cores = std::thread::available_parallelism().unwrap().get().min(4);
        for _ in 0..available_cores {
            runner.add(shuttle::scheduler::PctScheduler::new(10, 1000));
        }

        runner.run(|| {
            let iter = ITER.fetch_add(1, Ordering::Relaxed);
            shuttle_cpr_snapshot_cache_only_inner(iter);
            eprintln!("Completed shuttle iteration {}", iter);
        });
    }

    /// Inner body for the disk CPR snapshot shuttle test, parameterized by an
    /// iteration id so that concurrent shuttle replicas (and successive shuttle
    /// iterations) do not collide on file paths.
    #[cfg(feature = "shuttle")]
    fn shuttle_cpr_snapshot_disk_inner(iter: usize) {
        let min_record_size: usize = 64;
        let max_record_size: usize = 2408;
        let leaf_page_size: usize = 8192;
        let num_threads: usize = 4;
        // 500 inserts/thread × 4 threads = 2000 records/round.
        // With 128KB buffer (~1600 record capacity), this triggers eviction
        // while staying within shuttle coroutine stack limits.
        let inserts_per_thread: usize = 500;

        let file_path: String = format!(
            "target/shuttle_cpr_snapshot_disk_{}_{}.bftree",
            std::process::id(),
            iter,
        );
        let snapshot_file_path: String = format!(
            "target/shuttle_cpr_snapshot_disk_{}_{}_snap.bftree",
            std::process::id(),
            iter,
        );
        let tmp_file_path = std::path::PathBuf::from_str(&file_path).unwrap();
        let tmp_snapshot_file_path = std::path::PathBuf::from_str(&snapshot_file_path).unwrap();

        let mut config = Config::new(&tmp_file_path, 128 * 1024); // 128KB buffer, triggers eviction
        config.storage_backend(crate::StorageBackend::Std);
        config.cb_min_record_size = min_record_size + 2 * std::mem::size_of::<usize>();
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = min_record_size + 2 * std::mem::size_of::<usize>();
        config.use_snapshot(true);

        let bftree = Arc::new(BfTree::with_config(config.clone(), None).unwrap());
        let mut rs = vec![0usize; num_threads];
        for j in 0..3 {
            let handles: Vec<_> = (0..num_threads)
                .map(|i| {
                    let bftree_clone = bftree.clone();
                    // Always use key range 0..inserts_per_thread so shuttle_replay
                    // can reproduce any failing iteration with iter=0.
                    let start_id = j * inserts_per_thread;
                    let end_id = start_id + inserts_per_thread;
                    thread::spawn(move || {
                        let key_len: usize = min_record_size / 2 + std::mem::size_of::<usize>();
                        assert!(key_len * 2 <= max_record_size);
                        let mut key_buffer = vec![0usize; key_len / std::mem::size_of::<usize>()];

                        for r in start_id..end_id {
                            key_buffer.fill(r);
                            key_buffer[0] = i;

                            match bftree_clone.insert(
                                bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                                bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                            ) {
                                crate::LeafInsertResult::Success => {}
                                _ => {
                                    panic!("Insert failed");
                                }
                            }
                        }
                        inserts_per_thread
                    })
                })
                .collect();

            // Snapshot thread: takes the snapshot concurrently with the writers.
            let bftree_for_snap = bftree.clone();
            let snap_path = tmp_snapshot_file_path.clone();
            let snap_handle = thread::spawn(move || {
                bftree_for_snap.cpr_snapshot(&snap_path);
            });

            for (i, h) in handles.into_iter().enumerate() {
                let r = h.join().unwrap();
                rs[i] += r;
            }

            snap_handle.join().unwrap();

            // Recover from the snapshot and verify invariants
            verify_snapshot_recovery(
                &tmp_snapshot_file_path,
                num_threads,
                min_record_size,
                &rs,
                true,
            );
        }
        let _ = std::fs::remove_file(tmp_file_path);
        let _ = std::fs::remove_file(tmp_snapshot_file_path);
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_cpr_snapshot_disk() {
        use std::sync::atomic::AtomicUsize;

        // Unique iteration id so portfolio replicas / successive iterations do
        // not collide on file paths.
        static ITER: AtomicUsize = AtomicUsize::new(0);

        let mut shuttle_config = shuttle::Config::default();
        shuttle_config.max_steps = shuttle::MaxSteps::None;
        // Default shuttle stack is 32KB which is too small for deep B-tree
        // operations with eviction + file I/O. Increase to avoid stack overflow
        // that manifests as STATUS_HEAP_CORRUPTION on Windows.
        shuttle_config.stack_size = 4 * 1024 * 1024; // 4MB
        shuttle_config.failure_persistence =
            shuttle::FailurePersistence::File(Some(PathBuf::from_str("target").unwrap()));

        let mut runner = shuttle::PortfolioRunner::new(true, shuttle_config);
        let available_cores = std::thread::available_parallelism().unwrap().get().min(4);
        for _ in 0..available_cores {
            runner.add(shuttle::scheduler::PctScheduler::new(10, 100));
        }

        runner.run(|| {
            let iter = ITER.fetch_add(1, Ordering::Relaxed);
            shuttle_cpr_snapshot_disk_inner(iter);
        });
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_replay() {
        let schedule_path = "target/schedule000.txt";
        if !std::path::Path::new(schedule_path).exists() {
            eprintln!("No schedule file at {schedule_path}; run shuttle_cpr_snapshot_disk to generate one on failure.");
            return;
        }

        // install global collector configured based on RUST_LOG env var.
        tracing_subscriber::fmt()
            .with_ansi(true)
            .with_thread_names(false)
            .with_target(false)
            .init();

        shuttle::replay_from_file(|| shuttle_cpr_snapshot_disk_inner(0), schedule_path);
    }
}
