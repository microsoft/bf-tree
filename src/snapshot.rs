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
use std::sync::atomic::AtomicUsize;

#[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
use thread_local::ThreadLocal;

use crate::{
    circular_buffer::{CircularBuffer},
    error::ConfigError,
    fs::VfsImpl,
    mini_page_op::LeafOperations,
    nodes::{leaf_node::MiniPageNextLevel, LeafNode, INVALID_DISK_OFFSET},
    nodes::{InnerNode, InnerNodeBuilder, PageID, DISK_PAGE_SIZE, INNER_NODE_SIZE},
    storage::{make_vfs, LeafStorage, PageLocation, PageTable},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    utils::{get_rng, inner_lock::ReadGuard, BfsVisitor, NodeInfo},
    BfTree, Config, StorageBackend,
};

const BF_TREE_MAGIC_BEGIN: &[u8; 16] = b"BF-TREE-V0-BEGIN";
const BF_TREE_MAGIC_END: &[u8; 14] = b"BF-TREE-V0-END";
const META_DATA_PAGE_OFFSET: usize = 0;

const INVALID_SNAPSHOT_THREAD_ID: usize = usize::MAX; // Invalid thread slot id
const NULL_PAGE_LOCATION_OFFSET: usize = usize::MAX;
const INVALID_SNAPSHOT_STATE: u64 = u64::MAX; // Invalid snapshot state
pub const INVALID_SNAPSHOT_VERSION: u64 = u64::MAX; // Invalid snapshot version
const DEFAULT_MAX_SNAPSHOT_THREAD_NUM: usize = 256; // Maximum numbers of concurrent threads in Bf-tree, if snapshot is enabled.
const SNAPSHOT_STATE_PHASE_ID_SHIFT: usize = 61; // Number of bits to shift for phase id
const SNAPSHOT_STATE_PHASE_NUM: u64 = 4; // There are 4 snapshot phases
const SNAPSHOT_STATE_PHASE_ID_MASK: u64 = 0b111 << SNAPSHOT_STATE_PHASE_ID_SHIFT; // Most significant 3 bits for phase id
const SNAPSHOT_STATE_VERSION_MASK: u64 = (1 << SNAPSHOT_STATE_PHASE_ID_SHIFT) - 1; // Least significant 61 bits for version number

/// A simplified CRP snapshot of a Bf-Tree without the epoch framework.
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
    // Each user thread updates its own mappings asyncrhonously.
    thread_local_inner_mappings:
        UnsafeCell<[Vec<(*const InnerNode, usize)>; DEFAULT_MAX_SNAPSHOT_THREAD_NUM]>,
    thread_local_base_mappings: UnsafeCell<[Vec<(PageID, usize)>; DEFAULT_MAX_SNAPSHOT_THREAD_NUM]>,
    thread_local_mini_mappings: UnsafeCell<[Vec<(PageID, usize)>; DEFAULT_MAX_SNAPSHOT_THREAD_NUM]>,
    thread_local_mini_size_mappings:
        UnsafeCell<[Vec<(PageID, usize)>; DEFAULT_MAX_SNAPSHOT_THREAD_NUM]>,
    root_id: AtomicU64, // Page ID of the root node.
    pause_snapshot: AtomicBool,
    // The physical file of snapshot
    vfs: Arc<dyn VfsImpl>,
    // Ensuring only one snapshot is in progress at a time.
    snapshot_in_progress: AtomicBool,
}

pub struct CPRSnapshotGuard {
    snapshot_mgr: Option<Arc<CPRSnapShotMgr>>,
    thread_slot_id: usize,
}

impl CPRSnapshotGuard {
    pub fn new(snapshot_mgr: Option<Arc<CPRSnapShotMgr>>) -> Result<Self, ()> {
        match snapshot_mgr {
            None => {
                return Ok(Self {
                    snapshot_mgr: None,
                    thread_slot_id: INVALID_SNAPSHOT_THREAD_ID,
                });
            }
            Some(ref mgr) => {
                let thread_slot_id = mgr
                    .reserve_thread_slot()
                    .unwrap_or(INVALID_SNAPSHOT_THREAD_ID);
                if thread_slot_id == INVALID_SNAPSHOT_THREAD_ID {
                    return Err(());
                }

                assert_eq!(
                    thread_slot_id,
                    CPRSnapShotMgr::get_snapshot_thread_id().unwrap()
                );
                assert_eq!(
                    CPRSnapShotMgr::get_snapshot_thread_version(),
                    mgr.get_local_version(&thread_slot_id)
                );

                Ok(Self {
                    snapshot_mgr: Some(mgr.clone()),
                    thread_slot_id,
                })
            }
        }
    }

    /// Returns true if the snapshot guard has a valid thread slot id.
    pub fn is_protected(&self) -> bool {
        self.thread_slot_id != INVALID_SNAPSHOT_THREAD_ID
    }

    pub fn get_local_phase_id(&self) -> u64 {
        self.snapshot_mgr
            .as_ref()
            .unwrap()
            .get_local_phase_id(&self.thread_slot_id)
    }

    pub fn snapshot_base_page(&self, id: PageID, ptr: &[u8], size: usize) {
        self.snapshot_mgr
            .as_ref()
            .unwrap()
            .snapshot_base_page(id, ptr, size);
    }

    pub fn snapshot_mini_page(&self, id: PageID, ptr: &[u8], size: usize) {
        self.snapshot_mgr
            .as_ref()
            .unwrap()
            .snapshot_mini_page(id, ptr, size);
    }

    pub fn snapshot_inner_node(&self, ptr: *const InnerNode) {
        self.snapshot_mgr.as_ref().unwrap().snapshot_inner_node(ptr);
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
    //TODO: version id wraps around. Need to handle it

    // Each user thread is assigned an unique snapshot thread id and local snapshot version
    // for each bf-tree transaction (node split/insert/create), if snapshot is enabled.
    thread_local! {
        static SNAPSHOT_THREAD_ID: AtomicUsize = AtomicUsize::new(INVALID_SNAPSHOT_THREAD_ID);
        // For snapshot thread with valid snapshot_thread_id, its version is already contained in thread_local_states.
        // But we keep an extra copy here for easier access.
        static SNAPSHOT_THREAD_VERSION: AtomicU64 = AtomicU64::new(INVALID_SNAPSHOT_VERSION);
    }

    pub fn get_snapshot_thread_id() -> Result<usize, ()> {
        let tid = Self::SNAPSHOT_THREAD_ID.with(|id| id.load(Ordering::Relaxed));
        if tid == INVALID_SNAPSHOT_THREAD_ID {
            Err(())
        } else {
            Ok(tid)
        }
    }

    pub fn set_snapshot_thread_tls(tid: usize, version: u64) {
        Self::SNAPSHOT_THREAD_ID.with(|id| id.store(tid, Ordering::Relaxed));
        Self::SNAPSHOT_THREAD_VERSION.with(|ver| ver.store(version, Ordering::Relaxed));
    }

    pub fn get_snapshot_thread_version() -> u64 {
        Self::SNAPSHOT_THREAD_VERSION.with(|ver| ver.load(Ordering::Relaxed))
    }

    /// Initialize a new snapshot instance.
    pub fn new(storage_backend: &StorageBackend, version: u64, file_path: &PathBuf) -> Self {
        let vfs: Arc<dyn VfsImpl> = make_vfs(storage_backend, file_path);
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
            vfs,
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
            local_inner_mappings[thread_slot_id].clear();
            local_mini_mappings[thread_slot_id].clear();
            local_base_mappings[thread_slot_id].clear();
            local_mini_size_mappings[thread_slot_id].clear();
        }
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

    fn get_global_phase_id(&self) -> u64 {
        (self.global_state.load(Ordering::Acquire) & SNAPSHOT_STATE_PHASE_ID_MASK)
            >> SNAPSHOT_STATE_PHASE_ID_SHIFT
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

    /// Retrieve the local version of a thread specified by its slot id.
    pub fn get_local_version(&self, thread_slot_id: &usize) -> u64 {
        let state = self.get_local_state(thread_slot_id);
        state & SNAPSHOT_STATE_VERSION_MASK
    }

    /// Retrieve the local phase of a thread specified by its slot id.
    pub fn get_local_phase_id(&self, thread_slot_id: &usize) -> u64 {
        let state = self.get_local_state(thread_slot_id);
        (state & SNAPSHOT_STATE_PHASE_ID_MASK) >> SNAPSHOT_STATE_PHASE_ID_SHIFT
    }

    fn advance_global_state(&self) -> u64 {
        let phase_id = self.get_global_phase_id();
        let version = self.get_global_version();

        match phase_id {
            0 => {
                // (REST, v) -> (PREPARE, v)
                let new_state = Self::new_snapshot_state(1, version);
                self.global_state.store(new_state, Ordering::Release);
                new_state
            }
            1 => {
                // (PREPARE, v) -> (IN_PROGRESS, v + 1)
                let new_state = Self::new_snapshot_state(2, version + 1);
                self.global_state.store(new_state, Ordering::Release);
                new_state
            }
            2 => {
                // (IN_PROGRESS, v + 1) -> (SWEEPING, v + 1)
                let new_state = Self::new_snapshot_state(3, version);
                self.global_state.store(new_state, Ordering::Release);
                new_state
            }
            3 => {
                // (SWEEPING, v + 1) -> (FINISHING, v + 1)
                let new_state = Self::new_snapshot_state(0, version);
                self.global_state.store(new_state, Ordering::Release);
                new_state
            }
            _ => {
                panic!("Invalid global phase id: {}", phase_id);
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
    pub fn reserve_thread_slot(&self) -> Result<usize, ()> {
        if self.pause_snapshot.load(Ordering::Acquire) {
            return Err(());
        }

        let start = get_rng().gen_range(0..DEFAULT_MAX_SNAPSHOT_THREAD_NUM);
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

                // If the global state has changed, reset
                // This is to guarantee that as soon as global state rolls to the next one,
                // all new local states will either be reversed without further action or in the new state.
                if self.get_local_state(&tid) != global_state {
                    self.set_local_state(&tid, INVALID_SNAPSHOT_STATE);
                    assert!(self.thread_slots[tid]
                        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok());
                    return Err(());
                } else {
                    // Thread's TLS data must be empty before setting it
                    assert!(Self::get_snapshot_thread_id().is_err());

                    // Set the caller thread's snapshot thread id and version in TLS.
                    Self::set_snapshot_thread_tls(tid, global_state & SNAPSHOT_STATE_VERSION_MASK);
                    return Ok(tid);
                }
            }
        }
        Err(())
    }

    /// Free up the thread slot specified by the given thread slot id.
    /// Reset the thread's TLS
    pub fn release_thread_slot(&self, thread_slot_id: usize) {
        self.set_local_state(&thread_slot_id, INVALID_SNAPSHOT_STATE);
        self.thread_slots[thread_slot_id].store(false, Ordering::Release);

        Self::set_snapshot_thread_tls(
            INVALID_SNAPSHOT_THREAD_ID,
            INVALID_SNAPSHOT_STATE & SNAPSHOT_STATE_VERSION_MASK,
        );
    }

    pub fn get_snapshot_guard(
        snapshot_mgr: Option<Arc<CPRSnapShotMgr>>,
    ) -> Result<CPRSnapshotGuard, ()> {
        CPRSnapshotGuard::new(snapshot_mgr)
    }

    /// Snapshot a page to the current snapshot file and return its offset in the file.
    /// The invoker needs to guarantee that the page to be copied is xlocked
    /// throughput the lifetime of this function.
    /// Also the invoker needs to guarantee proper alignment of ptr which could required
    /// by the underlying vfs. (E.g., io_uring_vfs requires 512B alignment)
    pub fn snapshot_page(&self, ptr: &[u8], size: usize) -> usize {
        // Allocate space in the snapshot file
        let offset = self.vfs.alloc_offset(size);

        // Copy the page (ptr) to the new space
        self.vfs.write(offset, ptr);

        // Return the offset
        offset
    }

    pub fn snapshot_inner_node(&self, ptr: *const InnerNode) {
        let offset = unsafe { self.snapshot_page((&*ptr).as_slice(), INNER_NODE_SIZE) };
        let inner_mappings = unsafe { &mut *self.thread_local_inner_mappings.get() };

        let tid = Self::get_snapshot_thread_id()
            .expect("Snapshot of page triggered by unregistered thread.");
        inner_mappings[tid].push((ptr, offset));
    }

    pub fn snapshot_mini_page(&self, id: PageID, ptr: &[u8], size: usize) {
        let offset = if size != 0 {
            self.snapshot_page(ptr, size)
        } else {
            // cache-only mode, NULL page
            NULL_PAGE_LOCATION_OFFSET
        };

        let mini_mappings = unsafe { &mut *self.thread_local_mini_mappings.get() };
        let tid = Self::get_snapshot_thread_id()
            .expect("Snapshot of page triggered by unregistered thread.");
        mini_mappings[tid].push((id.clone(), offset));

        let mini_size_mappings = unsafe { &mut *self.thread_local_mini_size_mappings.get() };
        mini_size_mappings[tid].push((id.clone(), size));
    }

    pub fn snapshot_base_page(&self, id: PageID, ptr: &[u8], size: usize) {
        let offset = self.snapshot_page(ptr, size);

        let base_mappings = unsafe { &mut *self.thread_local_base_mappings.get() };
        let tid = Self::get_snapshot_thread_id()
            .expect("Snapshot of page triggered by unregistered thread.");
        base_mappings[tid].push((id.clone(), offset));
    }

    pub fn snapshot_root_page(&self, root_id: PageID) {
        self.root_id.store(root_id.raw(), Ordering::Release);
    }

    /// Sweep through all data pages and take snapshots of those
    /// whose version is less than or equal to the current snapshot version.
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
        // TODO: A better approach to scan all inner nodes.
        self.pause_snapshot.store(true, Ordering::Release);
        loop {
            if self.check_if_phase_completed(INVALID_SNAPSHOT_STATE) {
                // Root node
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
                                if base_ref.get_snapshot_version() < version {
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
                                if mini_ref.get_snapshot_version() < version {
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
                        // No need for WriteGuard as the tree structured is frozen and there is no other WriteGuard active.
                        let inner = match ReadGuard::try_read(ptr) {
                            Ok(inner) => inner,
                            Err(_) => continue,
                        };

                        if inner.as_ref().get_snapshot_version() < version {
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
                                // No need for WriteGuard as the tree structured is frozen and there is no other WriteGuard active.
                                let inner = match ReadGuard::try_read(ptr) {
                                    Ok(inner) => inner,
                                    Err(_) => continue,
                                };

                                if inner.as_ref().get_snapshot_version() < version {
                                    let offset = unsafe {
                                        self.snapshot_page((&*ptr).as_slice(), INNER_NODE_SIZE)
                                    };
                                    inner_mapping.push((ptr, offset));
                                }

                                break;
                            }
                            NodeInfo::Leaf { level, .. } => {
                                // corner case: we might still get a leaf node when the root is leaf...
                                //
                                // When ROOT is leaf, it is in `FORCE` mode, meaning that all data are write to disk.
                                // do don't need to do anything here.
                                assert_eq!(level, 0);
                                break;
                            }
                        }
                    }
                }

                break;
            }
            // At most wasting 1 second per state transition.
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        self.pause_snapshot.store(false, Ordering::Release);

        // In SWEEPING phase, there will be no new disk pages with v < 'version' being created. As such,
        // a sequential sweep of the page table is sufficient to capture all data pages with v < 'version'.
        let page_table_iter = tree.storage.page_table.iter();
        let mut enumerate_leaf_count = 0;

        for (_, pid) in page_table_iter {
            assert!(pid.is_id());

            // A reader lock is enough
            let mut leaf = tree.mapping_table().get(&pid);
            let page_loc = leaf.get_page_location();
            enumerate_leaf_count += 1;

            match page_loc {
                PageLocation::Base(offset) => {
                    let base_ref = leaf.load_base_page(*offset);
                    if base_ref.get_snapshot_version() < version {
                        let base_ptr = unsafe {
                            std::slice::from_raw_parts(
                                base_ref as *const LeafNode as *const u8,
                                base_ref.meta.node_size as usize,
                            )
                        };
                        let offset = self.snapshot_page(base_ptr, base_ref.meta.node_size as usize);
                        base_mapping.push((pid, offset));
                    }
                }
                PageLocation::Full(ptr) => {
                    let full_ref = leaf.load_cache_page(*ptr);
                    if full_ref.get_snapshot_version() < version {
                        let full_ptr = unsafe {
                            std::slice::from_raw_parts(
                                full_ref as *const LeafNode as *const u8,
                                full_ref.meta.node_size as usize,
                            )
                        };
                        let offset = self.snapshot_page(full_ptr, full_ref.meta.node_size as usize);
                        base_mapping.push((pid, offset));
                    }
                }
                PageLocation::Mini(ptr) => {
                    let mini_ref = leaf.load_cache_page(*ptr);
                    if mini_ref.get_snapshot_version() < version {
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
                            let base_ref = leaf.load_base_page(mini_ref.next_level.as_offset());
                            assert!(base_ref.get_snapshot_version() < version);

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
                    mini_mapping.push((pid, NULL_PAGE_LOCATION_OFFSET)); // Special marker
                    mini_size_mapping.push((pid, 0));
                }
            }
        }

        enumerate_leaf_count
    }

    // Finalize the snapshot file and reset the snapshotmgr's internal data.s
    fn finalize(
        &self,
        snapshot_version: u64,
        inner_mapping: &mut Vec<(*const InnerNode, usize)>,
        mini_mapping: &mut Vec<(PageID, usize)>,
        mini_size_mapping: &mut Vec<(PageID, usize)>,
        base_mapping: &mut Vec<(PageID, usize)>,
        leaf_count_upper: usize,
        config: Arc<Config>,
    ) {
        let mut inner_mapping_unique: HashMap<*const InnerNode, usize> = HashMap::new();
        let mut mini_mapping_unique: HashMap<PageID, usize> = HashMap::new();
        let mut mini_size_mapping_unique: HashMap<PageID, usize> = HashMap::new();
        let mut base_mapping_unique: HashMap<PageID, usize> = HashMap::new();
        let local_inner_mappings = unsafe { &mut *self.thread_local_inner_mappings.get() };
        let local_mini_mappings = unsafe { &mut *self.thread_local_mini_mappings.get() };
        let local_mini_size_mappings = unsafe { &mut *self.thread_local_mini_size_mappings.get() };
        let local_base_mappings = unsafe { &mut *self.thread_local_base_mappings.get() };

        // De-duplicate mappings among snapshot threads
        for thread_slot_id in 0..DEFAULT_MAX_SNAPSHOT_THREAD_NUM {
            for m in local_inner_mappings[thread_slot_id].iter() {
                if !inner_mapping_unique.contains_key(&m.0) {
                    inner_mapping_unique.insert(m.0, m.1);
                }
            }
            for m in local_mini_mappings[thread_slot_id].iter() {
                if !mini_mapping_unique.contains_key(&m.0) {
                    mini_mapping_unique.insert(m.0, m.1);
                }
            }
            for m in local_mini_size_mappings[thread_slot_id].iter() {
                if !mini_size_mapping_unique.contains_key(&m.0) {
                    mini_size_mapping_unique.insert(m.0, m.1);
                } else {
                    // In cache-only mode, given that NULL page has no snapshot version, a page
                    // could be snapshotted multiple times with different content.
                    // In disk mode, assert that the same page is only never snapshotted with different
                    // content
                    if !config.cache_only {
                        assert!(m.1 == mini_size_mapping_unique.get(&m.0).copied().unwrap());
                    }
                }
            }

            for m in local_base_mappings[thread_slot_id].iter() {
                if !base_mapping_unique.contains_key(&m.0) {
                    base_mapping_unique.insert(m.0, m.1);
                }
            }
        }

        // De-duplicate with the sweep results
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
                assert!(*v == mini_size_mapping_unique.get(k).copied().unwrap());
            }
        }
        for (k, v) in base_mapping.iter() {
            if !base_mapping_unique.contains_key(k) {
                base_mapping_unique.insert(*k, *v);
            }
        }

        // Sanity checks
        assert!(mini_mapping_unique.len() == mini_size_mapping_unique.len());
        for (k, _) in mini_mapping_unique.iter() {
            assert!(mini_size_mapping_unique.contains_key(k));
        }

        if config.cache_only {
            assert!(base_mapping_unique.is_empty());
        } else {
            assert!(mini_mapping_unique.len() <= base_mapping_unique.len());
        }

        // Finally inner node mappings of the snapshot
        let mut final_inner_mapping: Vec<(*const InnerNode, usize)> = Vec::new();
        for (k, v) in inner_mapping_unique.into_iter() {
            final_inner_mapping.push((k, v));
        }

        // Final base leaf node mappings of the snapshot, disk-mode only
        // Sort the base leaf node mappings by PageID in ascending order
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
            unsafe { std::mem::transmute(sorted_base_mapping_uninit) }
        } else {
            Vec::new()
        };

        // Final mini-page leaf node mappings of the snapshot
        let mut final_mini_mapping: Vec<(PageID, usize)> = Vec::new();
        for (k, v) in mini_mapping_unique.into_iter() {
            final_mini_mapping.push((k, v));
        }

        let mut final_mini_size_mapping: Vec<(PageID, usize)> = Vec::new();
        for (k, v) in mini_size_mapping_unique.into_iter() {
            final_mini_size_mapping.push((k, v));
        }

        let mut leaf_page_num = 0;

        if config.cache_only {
            // For a pagelocation that's NULL (evicted), we don't know its version
            // As such, we assume they are of the snapshot version and should be included.
            // A few slots in page table mnight be wasted due to false positive.
            // TODO: Be precise which requires some more metadata. Ideas:
            // 1) Count number of child leaf nodes in snapshotted inner nodes
            // 2) Put version in Null PageLocation.
            //assert!(leaf_count_upper >= final_mini_mapping.len());
            leaf_page_num = leaf_count_upper;
        } else {
            assert!(leaf_count_upper >= final_sorted_base_mapping.len());
            leaf_page_num = final_sorted_base_mapping.len();
        }

        let mut file_size = std::mem::size_of::<BfTreeMeta>() as u64;

        // Flush final mappings into the snapshot file
        let (inner_offset, inner_size) = serialize_vec_to_disk(&final_inner_mapping, &self.vfs);

        if inner_offset != 0 {
            file_size = (inner_offset + align_to_sector_size(inner_size)) as u64;
        }

        let (mini_offset, mini_size) = serialize_vec_to_disk(&final_mini_mapping, &self.vfs);

        if mini_offset != 0 {
            file_size = (mini_offset + align_to_sector_size(mini_size)) as u64;
        }

        let (mini_size_offset, mini_size_size) =
            serialize_vec_to_disk(&final_mini_size_mapping, &self.vfs);

        if mini_size_offset != 0 {
            file_size = (mini_size_offset + align_to_sector_size(mini_size_size)) as u64;
        }

        let (base_offset, base_size) = serialize_vec_to_disk(&final_sorted_base_mapping, &self.vfs);

        if base_offset != 0 {
            file_size = (base_offset + align_to_sector_size(base_size)) as u64;
        }

        // Flush the header to the snapshot file
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

        self.vfs.write(META_DATA_PAGE_OFFSET, metadata.as_slice());
        self.vfs.flush();

        self.reset();
    }

    /// Take a CRP snapshot of Bf-Tree
    pub fn snapshot(&self, tree: &BfTree) {
        // Allowing only one active snapshot at a time.
        if self
            .snapshot_in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        // Clear the snapshot file
        self.vfs.reset();

        // Initialize a local inner node and leaf node mapping
        let mut inner_mapping: Vec<(*const InnerNode, usize)> = Vec::new();
        let mut mini_mapping: Vec<(PageID, usize)> = Vec::new();
        let mut mini_size_mapping: Vec<(PageID, usize)> = Vec::new();
        let mut base_mapping: Vec<(PageID, usize)> = Vec::new();

        // At the beginning, the global phase id must be 0 (REST).
        let mut current_global_phase_id = self.get_global_phase_id();
        assert_eq!(current_global_phase_id, 0);

        let snapshot_version = self.get_global_version();

        // Immediately move the global state to (1 (PREPARE), V)
        let mut current_global_state = self.advance_global_state();
        current_global_phase_id = self.get_global_phase_id();
        assert_eq!(current_global_phase_id, 1);
        assert_eq!(snapshot_version, self.get_global_version());
        let mut leaf_node_count_upper_bound = 0;

        loop {
            if self.check_if_phase_completed(current_global_state) {
                match current_global_phase_id {
                    // REST
                    0 => {
                        // Upon reaching here, all user threads are in (REST, v + 1).
                        // All snapshots of pages of v are done, and no more snapshot operations
                        // neither. As such, we can safely finalize the snapshot by writing out the
                        // metadata and page mappings.
                        self.finalize(
                            snapshot_version,
                            &mut inner_mapping,
                            &mut mini_mapping,
                            &mut mini_size_mapping,
                            &mut base_mapping,
                            leaf_node_count_upper_bound,
                            tree.config.clone(),
                        );

                        // The snapshot is done.
                        break;
                    }
                    // PREPARE
                    1 => {
                        current_global_state = self.advance_global_state();
                        current_global_phase_id = self.get_global_phase_id();
                        assert_eq!(current_global_phase_id, 2);
                        assert_eq!(snapshot_version + 1, self.get_global_version());
                    }
                    // IN_PROGRESS
                    2 => {
                        current_global_state = self.advance_global_state();
                        current_global_phase_id = self.get_global_phase_id();
                        assert_eq!(current_global_phase_id, 3);
                        assert_eq!(snapshot_version + 1, self.get_global_version());
                    }
                    // SWEEPING
                    3 => {
                        // Sweep through and snapshot all data pages whose version is less than (v + 1)
                        // and build inner node and leaf node mapping for those pages.
                        // Upon completion, all pages with version less than (v + 1) should have been captured in the snapshot.
                        // Note that, there could be duplicate snapshots during and after the sweep as user threads in 'SWEEPING'
                        // state keeps taking snapshots of pages even if they have been captured by the sweep as the sweep does not
                        // alter page versions. De-duplication is needed when finalizing the snapshot file.
                        leaf_node_count_upper_bound = self.sweep(
                            tree,
                            snapshot_version + 1,
                            &mut inner_mapping,
                            &mut mini_mapping,
                            &mut mini_size_mapping,
                            &mut base_mapping,
                        );

                        current_global_state = self.advance_global_state();
                        current_global_phase_id = self.get_global_phase_id();
                        assert_eq!(current_global_phase_id, 0);
                        assert_eq!(snapshot_version + 1, self.get_global_version());
                    }
                    _ => {
                        panic!("Invalid global phase id: {}", current_global_phase_id);
                    }
                }
            }

            // At most wasting 1 second per state transition.
            std::thread::sleep(std::time::Duration::from_secs(1));
        }

        assert!(self
            .snapshot_in_progress
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok());
    }

    /// Recover a Bf-tree from an existing snapshot file (recovery_snapshot_file_path)
    /// Configuration of the newly created Bf-tree is retrieved from the snapshot file
    /// and cannot be changed except:
    /// 1. buffer_ptr (optional)
    /// 2. buffer_size (overrides the size retrieved from the snapshot file)
    /// 3. If enabled, provide path of the snapshot file of the newly recovered Bf-tree
    ///    which must be different from the recovery snapshot file path
    pub fn new_from_snapshot(
        recovery_snapshot_file_path: PathBuf, //  The snapshot file to recover from
        use_snapshot: bool,
        new_snapshot_file_path: Option<PathBuf>, // The snapshot file of the newly recovered Bf-tree
        buffer_ptr: Option<*mut u8>,
        buffer_size: Option<usize>, // buffer size of the newly created Bf-tree
    ) -> Result<BfTree, ConfigError> {
        if !recovery_snapshot_file_path.exists() {
            // if not already exist, we just create a new empty file at the location.
            return Err(ConfigError::SnapshotFileInvalid(
                "Not found ".to_string()
                    + &recovery_snapshot_file_path
                        .clone()
                        .into_os_string()
                        .into_string()
                        .unwrap(),
            ));
        }

        // Retrieve the header of the snapshot file and construct a valid Config of the to-be-recovered Bf-tree
        let reader = std::fs::File::open(recovery_snapshot_file_path.clone()).unwrap();
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

        // TODO, recover storage backend from snapshot file
        let recovery_snapshot_file_backend = StorageBackend::Std;
        if !bf_tree_config.cache_only {
            bf_tree_config.file_path(recovery_snapshot_file_path.clone());
            // The storage backend should use the same vfs system as the recovery snapshot file
            bf_tree_config.storage_backend = recovery_snapshot_file_backend.clone();
        } else {
            bf_tree_config.storage_backend = StorageBackend::Memory;
        }
        bf_tree_config.use_snapshot = use_snapshot;
        bf_tree_config.snapshot_backend = StorageBackend::Std; // TODO, allow user chosen snapshot backend

        bf_tree_config.snapshot_file_path = match new_snapshot_file_path {
            Some(path) => path,
            None => PathBuf::new(),
        };

        let snapshot_mgr = if bf_tree_config.use_snapshot {
            Some(Arc::new(CPRSnapShotMgr::new(
                &bf_tree_config.snapshot_backend,
                bf_tree_config.snapshot_version,
                &bf_tree_config.snapshot_file_path,
            )))
        } else {
            None
        };

        match buffer_size {
            Some(size) => bf_tree_config.cb_size_byte = size,
            None => {}
        }

        let size_classes = BfTree::create_mem_page_size_classes(
            bf_tree_config.cb_min_record_size,
            bf_tree_config.cb_max_record_size,
            bf_tree_config.leaf_page_size,
            bf_tree_config.max_fence_len,
            bf_tree_config.cache_only,
        );

        bf_tree_config.write_ahead_log = None;
        bf_tree_config.validate()?;

        let config = Arc::new(bf_tree_config);

        // Creat a vfs over the recovery snapshot file and start re-constructing a new Bf-tree
        let recovery_snapshot_vfs = make_vfs(
            &recovery_snapshot_file_backend,
            recovery_snapshot_file_path.clone(),
        );
        let mut page_buffer = SectorAlignedVector::new_zeroed(INNER_NODE_SIZE);

        // Step 1: reconstruct inner nodes.
        let mut root_page_id = bf_meta.root_id;
        if root_page_id.is_inner_node_pointer() {
            let inner_mapping: Vec<(*const InnerNode, usize)> = read_vec_from_offset(
                bf_meta.inner_offset,
                bf_meta.inner_size,
                &recovery_snapshot_vfs,
            );
            let mut inner_map = HashMap::new();

            for m in inner_mapping {
                inner_map.insert(m.0, m.1);
            }
            let offset = inner_map.get(&root_page_id.as_inner_node()).unwrap();
            recovery_snapshot_vfs.read(*offset, &mut page_buffer);
            let root_page = InnerNodeBuilder::new().build_from_slice(&page_buffer);

            // No need for disk offset of a inner node.
            unsafe {
                (*root_page).set_disk_offset(INVALID_DISK_OFFSET as u64);
            }
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
                    recovery_snapshot_vfs.read(*offset, &mut page_buffer);
                    let inner_page = InnerNodeBuilder::new().build_from_slice(&page_buffer);
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

        if !bf_meta.cache_only {
            // For disk backed bf-tree, we first reconstruct the page table using base pages.
            let base_mapping: Vec<(PageID, usize)> = if bf_meta.base_size > 0 {
                read_vec_from_offset(
                    bf_meta.base_offset,
                    bf_meta.base_size,
                    &recovery_snapshot_vfs,
                )
            } else {
                Vec::new()
            };

            let base_mapping = base_mapping.into_iter().map(|(pid, offset)| {
                let loc = PageLocation::Base(offset);
                (pid, loc)
            });

            // Note: The vfs of the newly constructed Bf-tree is the recovery snapshot vfs
            let pt = PageTable::new_from_mapping(
                base_mapping,
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

            // Second, we restore the mini-pages and wire them to corresponding base pages and update the page table.
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

            // Note: The vfs of the newly constructed Bf-tree is the recovery snapshot vfs
            let storage =
                LeafStorage::new_inner(config.clone(), pt, circular_buffer, recovery_snapshot_vfs);

            for i in 0..mini_mapping.len() {
                let (pid, offset) = mini_mapping[i];
                let mini_size = *mini_size_mapping_unique.get(&pid).unwrap();

                // Allocate memory in storage
                let mini_page_guard = match storage.alloc_mini_page(mini_size) {
                    Ok(mini_page_ptr) => mini_page_ptr,
                    Err(_) => {
                        panic!("Please increase cb_size_byte in config");
                    }
                };

                // Copy mini-page from snapshot to memory
                let mut page_buffer = SectorAlignedVector::new_zeroed(mini_size);
                storage.vfs.read(offset, &mut page_buffer);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        page_buffer.as_ptr(),
                        mini_page_guard.as_ptr(),
                        mini_size,
                    );
                }

                // Connect the mini-page to the corresponding base page in page table
                let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                let mini_page = unsafe { &mut *new_mini_ptr };

                let mut base_page = storage.page_table.get_mut(&pid);
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
                wal: None,                  // TODO: Allow users to configure WAl
                write_load_full_page: true, //TODO: Save this flag in the snapshot file too.
                cache_only: false,
                mini_page_size_classes: size_classes,
                snapshot_mgr: snapshot_mgr,
                config: config,
                #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
                metrics_recorder: Some(Arc::new(ThreadLocal::new())),
            })
        } else {
            // For cache-only mode, we create a new memory based vfs first
            let storage_vfs = make_vfs(&config.storage_backend, PathBuf::new());

            // Then, we create a new page table with only Null pages
            let mini_mapping: Vec<(PageID, usize)> = if bf_meta.mini_size > 0 {
                read_vec_from_offset(
                    bf_meta.mini_offset,
                    bf_meta.mini_size,
                    &recovery_snapshot_vfs,
                )
            } else {
                Vec::new()
            };
            let mini_mapping_unallocated: Vec<(PageID, PageLocation)> = (0..bf_meta.leaf_page_num)
                .map(|pid| (PageID::from_id(pid as u64), PageLocation::Null))
                .collect();

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

            let storage = LeafStorage::new_inner(config.clone(), pt, circular_buffer, storage_vfs);

            for i in 0..mini_mapping.len() {
                let (pid, offset) = mini_mapping[i];
                let mini_size = *mini_size_mapping_unique.get(&pid).unwrap();

                if offset == NULL_PAGE_LOCATION_OFFSET {
                    continue;
                }

                // Allocate memory in storage
                let mini_page_guard = match storage.alloc_mini_page(mini_size) {
                    Ok(mini_page_ptr) => mini_page_ptr,
                    Err(_) => {
                        panic!("Please increase cb_size_byte in config");
                    }
                };

                // Copy mini-page from snapshot to memory
                let mut page_buffer = SectorAlignedVector::new_zeroed(mini_size);
                recovery_snapshot_vfs.read(offset, &mut page_buffer);
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
                let mut null_page = storage.page_table.get_mut(&pid);
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
                wal: None,
                write_load_full_page: true,
                cache_only: true,
                mini_page_size_classes: size_classes,
                snapshot_mgr: snapshot_mgr,
                config: config,
                #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
                metrics_recorder: Some(Arc::new(ThreadLocal::new())),
            })
        }
    }
}

impl BfTree {
    pub fn cpr_snapshot(&self) {
        if !self.config.use_snapshot {
            panic!("Snapshots are not enabled in the configuration");
        }

        let snpshot_mgr = self.snapshot_mgr.clone().unwrap();
        snpshot_mgr.snapshot(self);
    }

    pub fn new_from_cpr_snapshot(
        recovery_snapshot_file_path: PathBuf, //  The snapshot file to recover from
        use_snapshot: bool,
        new_snapshot_file_path: Option<PathBuf>, // The snapshot file of the newly recovered Bf-tree
        buffer_ptr: Option<*mut u8>,
        buffer_size: Option<usize>,
    ) -> Result<BfTree, ConfigError> {
        CPRSnapShotMgr::new_from_snapshot(
            recovery_snapshot_file_path,
            use_snapshot,
            new_snapshot_file_path,
            buffer_ptr,
            buffer_size,
        )
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
    use crate::{nodes::leaf_node::LeafReadResult, BfTree, Config};
    use std::panic;
    use std::str::FromStr;
    use std::sync::atomic::Ordering;
    use std::sync::{atomic::AtomicBool, Arc};
    use std::thread;

    /// Multiple writer threads write to a BfTree in parallel while a separate thread taking multiple snapshots
    /// A new BfTree recovered from the snapshot should contain a prefix of all the inserts from each writer thread.
    /// A snapshot taken later should cover the previous snapshots.
    #[test]
    fn cpr_snapshot_disk() {
        let min_record_size: usize = 64;
        let max_record_size: usize = 2408;
        let leaf_page_size: usize = 8192;
        let snapshot_num: usize = 2;
        let num_threads: usize = 4;
        let file_path: String = "target/test_simple.bftree".to_string();
        let snapshot_file_path: String = "target/test_simple_snapshot.bftree".to_string();
        let new_snapshot_file_path: String = "target/test_simple_new_snapshot.bftree".to_string();

        let tmp_file_path = std::path::PathBuf::from_str(&file_path).unwrap();
        let tmp_snapshot_file_path = std::path::PathBuf::from_str(&snapshot_file_path).unwrap();
        let tmp_new_snapshot_file_path =
            std::path::PathBuf::from_str(&new_snapshot_file_path).unwrap();

        let mut config = Config::new(&tmp_file_path, 128 * 1024); // 128KB buffer pool. insert/split/eviction all triggered
        config.storage_backend(crate::StorageBackend::Std);
        config.cb_min_record_size = min_record_size;
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = max_record_size;
        config.snapshot_file_path = tmp_snapshot_file_path.clone();
        config.use_snapshot(true);

        let bftree = Arc::new(BfTree::with_config(config.clone(), None).unwrap());
        let finish = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let finish_clone = finish.clone();
                let bftree_clone = bftree.clone();

                std::thread::spawn(move || {
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
            bftree.cpr_snapshot();
            thread::sleep(std::time::Duration::from_secs(5));
        }

        // Stop all writer threads
        let mut rs = vec![0usize; num_threads];
        finish.store(true, Ordering::Relaxed);
        for (i, h) in handles.into_iter().enumerate() {
            let r = h.join().unwrap();
            rs[i] = r;
        }

        drop(bftree);

        // Recover from the snapshot and check invariants
        let bftree = BfTree::new_from_cpr_snapshot(
            std::path::PathBuf::from_str(&snapshot_file_path).unwrap(),
            true,
            Some(std::path::PathBuf::from_str(&(new_snapshot_file_path)).unwrap()),
            None,
            None,
        )
        .unwrap();

        let mut rs_captured = vec![0usize; num_threads];
        for i in 0..num_threads {
            let record_num = rs[i];

            let key_len: usize = min_record_size / 2 + std::mem::size_of::<usize>();
            let mut key_buffer = vec![0usize; key_len / std::mem::size_of::<usize>()];
            let mut res_buffer = vec![0u8; key_len];
            let mut not_included = false;

            for r in 0..record_num {
                key_buffer.fill(r);
                key_buffer[0] = i;

                match bftree.read(
                    bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                    &mut res_buffer,
                ) {
                    LeafReadResult::Found(v) => {
                        if not_included {
                            panic!("Found a record after the prefix for writer thread {}", i);
                        }
                        assert_eq!(v as usize, key_len);
                        assert_eq!(
                            &res_buffer,
                            bytemuck::must_cast_slice::<usize, u8>(&key_buffer)
                        );
                    }
                    LeafReadResult::NotFound => {
                        if !not_included {
                            not_included = true;
                            rs_captured[i] = r;
                        }
                    }
                    _ => {
                        panic!("Unexpected read result")
                    }
                }
            }

            if !not_included {
                rs_captured[i] = record_num;
            }

            // There must be some records that were not included in the snapshot
            assert!(rs_captured[i] < record_num)
        }

        std::fs::remove_file(tmp_file_path).unwrap();
        std::fs::remove_file(tmp_new_snapshot_file_path).unwrap();
        std::fs::remove_file(tmp_snapshot_file_path).unwrap();
    }

    #[test]
    fn cpr_snapshot_cache_only() {
        let min_record_size: usize = 64;
        let max_record_size: usize = 2408;
        let leaf_page_size: usize = 8192;
        let num_threads: usize = 4;

        let snapshot_file_path: String = "target/test_simple_snapshot.bftree".to_string();
        let new_snapshot_file_path: String = "target/test_simple_new_snapshot.bftree".to_string();
        let tmp_snapshot_file_path = std::path::PathBuf::from_str(&snapshot_file_path).unwrap();
        let tmp_new_snapshot_file_path =
            std::path::PathBuf::from_str(&new_snapshot_file_path).unwrap();

        let mut config = Config::default(); // Creat a CB that can hold 16 full pages
        config.storage_backend(crate::StorageBackend::Memory);
        config.file_path(":memory:");
        config.cache_only = true;
        config.cb_size_byte(1024 * 1024 * 1024);
        config.cb_min_record_size = min_record_size;
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = max_record_size;
        config.snapshot_file_path = tmp_snapshot_file_path.clone();
        config.use_snapshot(true);

        let bftree = Arc::new(BfTree::with_config(config.clone(), None).unwrap());
        let finish = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let finish_clone = finish.clone();
                let bftree_clone = bftree.clone();

                std::thread::spawn(move || {
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
        bftree.cpr_snapshot();
        thread::sleep(std::time::Duration::from_secs(5));

        // Stop all writer threads
        let mut rs = vec![0usize; num_threads];
        finish.store(true, Ordering::Relaxed);
        for (i, h) in handles.into_iter().enumerate() {
            let r = h.join().unwrap();
            rs[i] = r;
        }

        drop(bftree);

        // Recover from the snapshot and check invariants
        let bftree = BfTree::new_from_cpr_snapshot(
            std::path::PathBuf::from_str(&snapshot_file_path).unwrap(),
            true,
            Some(std::path::PathBuf::from_str(&(new_snapshot_file_path)).unwrap()),
            None,
            None,
        )
        .unwrap();

        let mut rs_captured = vec![0usize; num_threads];
        for i in 0..num_threads {
            let record_num = rs[i];

            let key_len: usize = min_record_size / 2 + std::mem::size_of::<usize>();
            let mut key_buffer = vec![0usize; key_len / std::mem::size_of::<usize>()];
            let mut res_buffer = vec![0u8; key_len];
            let mut not_included = false;

            for r in 0..record_num {
                key_buffer.fill(r);
                key_buffer[0] = i;

                match bftree.read(
                    bytemuck::must_cast_slice::<usize, u8>(&key_buffer),
                    &mut res_buffer,
                ) {
                    LeafReadResult::Found(v) => {
                        if not_included {
                            panic!("Found a record after the prefix for writer thread {}", i);
                        }
                        assert_eq!(v as usize, key_len);
                        assert_eq!(
                            &res_buffer,
                            bytemuck::must_cast_slice::<usize, u8>(&key_buffer)
                        );
                    }
                    LeafReadResult::NotFound => {
                        if !not_included {
                            not_included = true;
                            rs_captured[i] = r;
                        }
                    }
                    _ => {
                        panic!("Unexpected read result")
                    }
                }
            }

            if !not_included {
                rs_captured[i] = record_num;
            }

            // There must be some records that were not included in the snapshot
            assert!(rs_captured[i] < record_num)
        }

        std::fs::remove_file(tmp_snapshot_file_path).unwrap();
        std::fs::remove_file(tmp_new_snapshot_file_path).unwrap();
    }
}
