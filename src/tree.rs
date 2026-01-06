// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(not(all(feature = "shuttle", test)))]
use rand::Rng;

#[cfg(all(feature = "shuttle", test))]
use shuttle::rand::Rng;

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))] {
        use crate::metric::{Timer, TlsRecorder, timer::{TimerRecorder, DebugTimerGuard}};
        use std::cell::UnsafeCell;
        use thread_local::ThreadLocal;
    }
}

use crate::{
    check_parent,
    circular_buffer::{CircularBufferMetrics, TombstoneHandle},
    counter,
    error::{ConfigError, TreeError},
    histogram, info,
    mini_page_op::{upgrade_to_full_page, LeafEntryXLocked, LeafOperations, ReadResult},
    nodes::{
        leaf_node::{LeafKVMeta, LeafReadResult, MiniPageNextLevel, OpType},
        InnerNode, InnerNodeBuilder, LeafNode, PageID, CACHE_LINE_SIZE, DISK_PAGE_SIZE,
        INNER_NODE_SIZE, MAX_KEY_LEN, MAX_LEAF_PAGE_SIZE, MAX_VALUE_LEN,
    },
    range_scan::{ScanIter, ScanIterMut, ScanReturnField},
    storage::{LeafStorage, PageLocation, PageTable},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    utils::{get_rng, inner_lock::ReadGuard, Backoff, BfsVisitor, NodeInfo},
    wal::{WriteAheadLog, WriteOp},
    Config, StorageBackend,
};
use std::path::Path;

/// The bf-tree instance
pub struct BfTree {
    pub(crate) root_page_id: AtomicU64,
    pub(crate) storage: LeafStorage,
    pub(crate) wal: Option<Arc<WriteAheadLog>>,
    pub(crate) config: Arc<Config>,
    pub(crate) write_load_full_page: bool,
    pub(crate) cache_only: bool, // If true, there is no permenant storage layer thus no durability guarantee of any upsert in the tree
    pub(crate) mini_page_size_classes: Vec<usize>, // Size classes of mini pages
    #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
    pub metrics_recorder: Option<Arc<ThreadLocal<UnsafeCell<TlsRecorder>>>>, // Per-tree metrics recorder under "metrics-rt-debug" feature
}

unsafe impl Sync for BfTree {}

unsafe impl Send for BfTree {}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LeafInsertResult {
    Success,
    InvalidKV(String),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ScanIterError {
    CacheOnlyMode,
    InvalidStartKey,
    InvalidEndKey,
    InvalidCount,
    InvalidKeyRange,
}

impl Drop for BfTree {
    fn drop(&mut self) {
        if let Some(ref wal) = self.wal {
            wal.stop_background_job();
        }

        let visitor = BfsVisitor::new_all_nodes(self);
        for node_info in visitor {
            match node_info {
                NodeInfo::Leaf { page_id, .. } => {
                    let mut leaf = self.mapping_table().get_mut(&page_id);
                    leaf.dealloc_self(&self.storage, self.cache_only);
                }
                NodeInfo::Inner { ptr, .. } => {
                    if unsafe { &*ptr }.is_valid_disk_offset() {
                        let disk_offset = unsafe { &*ptr }.disk_offset;
                        if self.config.storage_backend == StorageBackend::Memory {
                            // special case for memory backend, we need to deallocate the memory.
                            self.storage.vfs.dealloc_offset(disk_offset as usize);
                        }
                    }
                    InnerNode::free_node(ptr as *mut InnerNode);
                }
            }
        }
    }
}

impl Default for BfTree {
    fn default() -> Self {
        Self::new(":memory:", 1024 * 1024 * 32).unwrap()
    }
}

impl BfTree {
    pub(crate) const ROOT_IS_LEAF_MASK: u64 = 0x8000_0000_0000_0000; // This is quite error-prone, make sure the mask is not conflicting with the page id definition.

    /// Create the size classes of all memory pages in acending order based on the record size (key + value) and the leaf page size
    /// [s_0, s_1, ..., s_x], ascending order.
    /// Each s_i is of size 2^i * c + size_of(LeafNode)
    /// where c = (min_record_size + LeafKVMeta) aligned on CACHE_LINE_SIZE and 2^i * c <= leaf_page_size and s_x = leaf_page_size
    ///
    /// In non cache-only mode, the largest mini page size is s_(x-1) and full page/leaf base page size is s_x.
    /// Currently, the design assumes a mini-page can always be successfully merged into a leaf page in one pass (including at most one base page split).
    /// As such, the following three conditions are sufficient in preventing merge failures.
    /// C1) x >= 1
    /// C2) s_x > s_{x-1}
    /// C3) max_record_size + SizeOf(KVMeta) <= (s_x - s_{x-1} - 2 * (fence_meta + max_key_len)).
    ///
    /// In cache-only mode, the largest mini page is s_x as there is no full nor base page.
    /// Although there is no merging of mini-pages, the design assumes a new record can always be successfully inserted into a mini-page in one pass
    /// (including at most one mini-page split). As such, the following sufficient condition is required.
    /// C1) max_record_page + Sizeof(KVMeta) <= (s_x - SizeOf(NodeMeta)) / 2
    /// C2) if x >= 1, s_x >= s_{x-1} + max_record_size + SizeOf(KVMeta)
    ///
    /// C2) is necessary for cache-only mode to guarantee that a mini-page can grow to full page size before being split up to two full-sized pages
    pub(crate) fn create_mem_page_size_classes(
        min_record_size_in_byte: usize,
        max_record_size_in_byte: usize,
        leaf_page_size_in_byte: usize,
        max_fence_len_in_byte: usize,
        cache_only: bool,
    ) -> Vec<usize> {
        // Sanity check of the input parameters
        assert!(
            min_record_size_in_byte > 1,
            "cb_min_record_size in config cannot be less than 2"
        );
        assert!(
            min_record_size_in_byte <= max_record_size_in_byte,
            "cb_min_record_size cannot be larger than cb_max_record_size"
        );
        assert!(
            max_fence_len_in_byte > 0,
            "max_fence_len in config cannot be zero"
        );
        assert!(
            max_fence_len_in_byte / 2 < max_record_size_in_byte,
            "max_fence_len/2 cannot be larger than cb_max_record_size"
        );
        assert!(
            leaf_page_size_in_byte <= MAX_LEAF_PAGE_SIZE,
            "leaf_page_size in config cannot be larger than {}",
            MAX_LEAF_PAGE_SIZE
        );
        assert!(
            max_fence_len_in_byte / 2 <= MAX_KEY_LEN,
            "max_key_len in config cannot be larger than {}",
            MAX_KEY_LEN
        );
        assert!(
            leaf_page_size_in_byte / min_record_size_in_byte <= 4096,
            "Maximum number of records per page (leaf_page_size/min_record_size) cannot exceed 2^12.", // This is restricted by #bits for value count in NodeMeta
        );

        // In non cache-only mode, the leaf page should be in the multiple of DISK_PAGE_SIZE.
        // In cache-only mode, the leaf page should be aligned on cache line size.
        if !cache_only {
            assert!(
                leaf_page_size_in_byte.is_multiple_of(DISK_PAGE_SIZE),
                "leaf_page_size in config should be multiple of {}",
                DISK_PAGE_SIZE
            );
        } else {
            assert!(
                leaf_page_size_in_byte.is_multiple_of(CACHE_LINE_SIZE),
                "leaf_page_size in config should be multiple of {}",
                CACHE_LINE_SIZE
            );
        }

        let max_record_size_with_meta = max_record_size_in_byte + std::mem::size_of::<LeafKVMeta>();
        let mut max_mini_page_size: usize;

        if cache_only {
            // Guarantee C1), C2) for cache-only mode
            max_mini_page_size = leaf_page_size_in_byte - max_record_size_with_meta;
            max_mini_page_size = (max_mini_page_size / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

            assert!(
                leaf_page_size_in_byte
                    >= 2 * max_record_size_with_meta + std::mem::size_of::<LeafNode>(),
                "cb_max_record_size of config should be <= {}",
                (leaf_page_size_in_byte - std::mem::size_of::<LeafNode>()) / 2
                    - std::mem::size_of::<LeafKVMeta>()
            );
        } else {
            // Guarantee C1), C2) and C3) for non cache-only mode
            max_mini_page_size = leaf_page_size_in_byte
                - max_record_size_with_meta
                - max_fence_len_in_byte
                - 2 * std::mem::size_of::<LeafKVMeta>();
            max_mini_page_size = (max_mini_page_size / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

            assert!(
                max_mini_page_size >= max_record_size_with_meta + std::mem::size_of::<LeafNode>(),
                "cb_max_record_size of config should be <= {}",
                max_mini_page_size
                    - std::mem::size_of::<LeafNode>()
                    - std::mem::size_of::<LeafKVMeta>()
            );
        }

        // Generate all size classes for mini-pages
        let mut mem_page_size_classes = Vec::new();

        let base: i32 = 2;
        let mut record_num_per_page_exp: u32 = 0;
        let c: usize = min_record_size_in_byte + std::mem::size_of::<LeafKVMeta>();

        // No need to consider fence here as mini-pages have no fences.
        let mut size_class =
            base.pow(record_num_per_page_exp) as usize * c + std::mem::size_of::<LeafNode>();

        // Memory page size is aligned on cache line size
        if !size_class.is_multiple_of(CACHE_LINE_SIZE) {
            size_class = (size_class / CACHE_LINE_SIZE + 1) * CACHE_LINE_SIZE;
        }

        while size_class <= max_mini_page_size {
            if mem_page_size_classes.is_empty()
                || (mem_page_size_classes[mem_page_size_classes.len() - 1] < size_class)
            {
                mem_page_size_classes.push(size_class);
            }

            record_num_per_page_exp += 1;
            size_class =
                base.pow(record_num_per_page_exp) as usize * c + std::mem::size_of::<LeafNode>();

            if !size_class.is_multiple_of(CACHE_LINE_SIZE) {
                size_class = (size_class / CACHE_LINE_SIZE + 1) * CACHE_LINE_SIZE;
            }
        }

        if !cache_only {
            assert!(!mem_page_size_classes.is_empty());
        }

        // Add the largest mini page size if not already added
        if mem_page_size_classes.is_empty()
            || mem_page_size_classes[mem_page_size_classes.len() - 1] < max_mini_page_size
        {
            mem_page_size_classes.push(max_mini_page_size);
        }

        // The largest page size is the full leaf page size
        if mem_page_size_classes.is_empty()
            || mem_page_size_classes[mem_page_size_classes.len() - 1] < leaf_page_size_in_byte
        {
            mem_page_size_classes.push(leaf_page_size_in_byte);
        }

        if !cache_only {
            assert!(mem_page_size_classes.len() >= 2);
        } else {
            assert!(!mem_page_size_classes.is_empty());
        }

        mem_page_size_classes
    }

    /// Create a new bf-tree instance with customized storage backend and
    /// circular buffer size
    ///
    /// For in-memory tree, use `:memory:` as file path.
    /// For cache-only tree, use `:cache:` as file path
    /// For disk tree, file_path is the path to the index file
    ///
    /// Mini page cache must be at least 8192 bytes for practical workloads.
    ///
    /// ```
    /// use bf_tree::BfTree;
    /// let tree = BfTree::new(":memory:", 8192).unwrap();
    /// ```
    pub fn new(file_path: impl AsRef<Path>, cache_size_byte: usize) -> Result<Self, ConfigError> {
        let config = Config::new(file_path, cache_size_byte);
        Self::with_config(config, None)
    }

    /// Create a new bf-tree instance with customized configuration based on
    /// a config file
    pub fn new_with_config_file<P: AsRef<Path>>(config_file_path: P) -> Result<Self, ConfigError> {
        let config = Config::new_with_config_file(config_file_path);
        Self::with_config(config, None)
    }

    /// Initialize the bf-tree with provided config. For advanced user only.
    /// An optional pre-allocated buffer pointer can be provided to use as the buffer pool memory.
    pub fn with_config(config: Config, buffer_ptr: Option<*mut u8>) -> Result<Self, ConfigError> {
        // Validate the config first
        config.validate()?;

        let wal = match config.write_ahead_log.as_ref() {
            Some(wal_config) => {
                let wal = WriteAheadLog::new(wal_config.clone());
                Some(wal)
            }
            None => None,
        };
        let write_load_full = config.write_load_full_page;
        let config = Arc::new(config);

        // In cache-only mode, the initial root page is a full mini-page
        if config.cache_only {
            let leaf_storage = LeafStorage::new(config.clone(), buffer_ptr);

            // Assuming CB can accommodate at least 2 leaf pages at the same time
            let mini_page_guard = (leaf_storage)
                .alloc_mini_page(config.leaf_page_size)
                .expect("Fail to allocate a mini-page as initial root node");
            LeafNode::initialize_mini_page(
                &mini_page_guard,
                config.leaf_page_size,
                MiniPageNextLevel::new_null(),
                true,
            );
            let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
            let mini_loc = PageLocation::Mini(new_mini_ptr);

            let (root_id, root_lock) = leaf_storage
                .mapping_table()
                .insert_mini_page_mapping(mini_loc);
            assert_eq!(root_id.as_id(), 0);

            drop(root_lock);
            drop(mini_page_guard);
            let root_id = root_id.raw() | Self::ROOT_IS_LEAF_MASK;

            return Ok(Self {
                root_page_id: AtomicU64::new(root_id),
                storage: leaf_storage,
                wal,
                cache_only: config.cache_only,
                write_load_full_page: write_load_full,
                mini_page_size_classes: Self::create_mem_page_size_classes(
                    config.cb_min_record_size,
                    config.cb_max_record_size,
                    config.leaf_page_size,
                    config.max_fence_len,
                    config.cache_only,
                ),
                config,
                #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
                metrics_recorder: Some(Arc::new(ThreadLocal::new())),
            });
        }

        let leaf_storage = LeafStorage::new(config.clone(), buffer_ptr);
        let (root_id, root_lock) = leaf_storage.mapping_table().alloc_base_page_mapping();
        drop(root_lock);
        assert_eq!(root_id.as_id(), 0);

        let root_id = root_id.raw() | Self::ROOT_IS_LEAF_MASK;
        Ok(Self {
            root_page_id: AtomicU64::new(root_id),
            storage: leaf_storage,
            wal,
            cache_only: config.cache_only,
            write_load_full_page: write_load_full,
            mini_page_size_classes: Self::create_mem_page_size_classes(
                config.cb_min_record_size,
                config.cb_max_record_size,
                config.leaf_page_size,
                config.max_fence_len,
                config.cache_only,
            ),
            config,
            #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
            metrics_recorder: Some(Arc::new(ThreadLocal::new())),
        })
    }

    /// Get the buffer metrics of the circular buffer.
    /// This is a blocking call, will stop all other buffer operations,
    /// use it wisely.
    pub fn get_buffer_metrics(&self) -> CircularBufferMetrics {
        self.storage.get_buffer_metrics()
    }

    /// returns the root page id and whether it is a leaf node.
    pub(crate) fn get_root_page(&self) -> (PageID, bool) {
        let root_page_id = self.root_page_id.load(Ordering::Acquire);
        let root_is_leaf = (root_page_id & Self::ROOT_IS_LEAF_MASK) != 0;
        let clean = root_page_id & (!Self::ROOT_IS_LEAF_MASK);

        let page_id = if root_is_leaf {
            PageID::from_id(clean)
        } else {
            PageID::from_pointer(clean as *const InnerNode)
        };

        (page_id, root_is_leaf)
    }

    pub(crate) fn mapping_table(&self) -> &PageTable {
        self.storage.mapping_table()
    }

    pub(crate) fn should_promote_read(&self) -> bool {
        get_rng().gen_range(0..100) < self.config.read_promotion_rate.load(Ordering::Relaxed)
    }

    pub(crate) fn should_promote_scan_page(&self) -> bool {
        get_rng().gen_range(0..100) < self.config.scan_promotion_rate.load(Ordering::Relaxed)
    }

    /// Chance% to promote a base read record to mini page.
    pub fn update_read_promotion_rate(&self, new_rate: usize) {
        self.config
            .read_promotion_rate
            .store(new_rate, Ordering::Relaxed);
    }

    fn try_split_leaf(
        &self,
        cur_page_id: PageID,
        parent: &Option<ReadGuard>,
    ) -> Result<bool, TreeError> {
        debug_assert!(cur_page_id.is_id());

        // here we need to acquire x-lock for performance reason:
        // if we acquire s-lock, it's very difficult for us to later upgrade to x-lock, because rwlock favors readers:
        //      consider readers keep coming, we will never be able to upgrade to x-lock.
        let mut cur_page = self.mapping_table().get_mut(&cur_page_id);

        check_parent!(self, cur_page_id, parent);

        let should_split = cur_page.get_split_flag();
        if !should_split {
            return Ok(false);
        }
        match parent {
            Some(_) => {
                unreachable!("Leaf node split should not happen here");
            }
            None => {
                // only for the case of root node split

                // In cache-only mode, the root mini-page node is split into two equal-sized mini-pages
                if self.cache_only {
                    // Create a new mini-page of the same size as the current root node
                    // Assuming CB is at least able to hold two leaf-page sized mini-pages
                    let mini_page_guard = self
                        .storage
                        .alloc_mini_page(self.config.leaf_page_size)
                        .expect("Fail to allocate a mini-page during root split");
                    LeafNode::initialize_mini_page(
                        &mini_page_guard,
                        self.config.leaf_page_size,
                        MiniPageNextLevel::new_null(),
                        true,
                    );
                    let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                    let mini_loc = PageLocation::Mini(new_mini_ptr);

                    // Insert the new page into mapping table
                    let (sibling_id, _mini_lock) = self
                        .storage
                        .mapping_table()
                        .insert_mini_page_mapping(mini_loc);

                    // Split current page with the newly created mini page
                    let cur_page_loc = cur_page.get_page_location().clone();
                    match cur_page_loc {
                        PageLocation::Mini(ptr) => {
                            let cur_mini_page = cur_page.load_cache_page_mut(ptr);
                            let sibling_page = unsafe { &mut *new_mini_ptr };
                            let split_key = cur_mini_page.split(sibling_page, true);

                            let mut new_root_builder = InnerNodeBuilder::new();
                            new_root_builder
                                .set_left_most_page_id(cur_page_id)
                                .set_children_is_leaf(true)
                                .add_record(split_key, sibling_id);

                            let new_root_ptr = new_root_builder.build();

                            self.root_page_id
                                .store(PageID::from_pointer(new_root_ptr).raw(), Ordering::Release);

                            info!(sibling = sibling_id.raw(), "New root node installed!");

                            debug_assert!(cur_mini_page.meta.meta_count_with_fence() > 0);
                            debug_assert!(sibling_page.meta.meta_count_with_fence() > 0);

                            return Ok(true);
                        }
                        _ => {
                            panic!("The root node is not a mini-page in cache-only mode")
                        }
                    }
                }

                let mut x_page = cur_page;

                let (sibling_id, mut sibling_entry) = self.alloc_base_page_and_lock();

                info!(sibling = sibling_id.raw(), "Splitting root node!");

                let sibling = sibling_entry.load_base_page_mut();

                let leaf_node = x_page.load_base_page_mut();
                let split_key = leaf_node.split(sibling, false);

                let mut new_root_builder = InnerNodeBuilder::new();
                new_root_builder
                    .set_disk_offset(self.storage.alloc_disk_offset(INNER_NODE_SIZE))
                    .set_left_most_page_id(cur_page_id)
                    .set_children_is_leaf(true)
                    .add_record(split_key, sibling_id);

                let new_root_ptr = new_root_builder.build();

                self.root_page_id
                    .store(PageID::from_pointer(new_root_ptr).raw(), Ordering::Release);

                info!(sibling = sibling_id.raw(), "New root node installed!");
                Ok(true)
            }
        }
    }

    fn alloc_base_page_and_lock(&self) -> (PageID, LeafEntryXLocked<'_>) {
        let (pid, base_entry) = self.mapping_table().alloc_base_page_mapping();

        (pid, base_entry)
    }

    fn try_split_inner<'a>(
        &self,
        cur_page: PageID,
        parent: Option<ReadGuard<'a>>,
    ) -> Result<(bool, Option<ReadGuard<'a>>), TreeError> {
        let cur_node = ReadGuard::try_read(cur_page.as_inner_node())?;

        check_parent!(self, cur_page, parent);

        let should_split = cur_node.as_ref().meta.get_split_flag();

        if !should_split {
            return Ok((false, parent));
        }

        info!(has_parent = parent.is_some(), "split inner node");

        match parent {
            Some(p) => {
                let mut x_cur = cur_node.upgrade().map_err(|(_x, e)| e)?;
                let mut x_parent = p.upgrade().map_err(|(_x, e)| e)?;

                let split_key = x_cur.as_ref().get_split_key();

                let mut sibling_builder = InnerNodeBuilder::new();
                sibling_builder.set_disk_offset(self.storage.alloc_disk_offset(INNER_NODE_SIZE));

                let success = x_parent
                    .as_mut()
                    .insert(&split_key, sibling_builder.get_page_id());
                if !success {
                    x_parent.as_mut().meta.set_split_flag();
                    return Err(TreeError::NeedRestart);
                }

                x_cur.as_mut().split(&mut sibling_builder);

                sibling_builder.build();

                Ok((true, Some(x_parent.downgrade())))
            }
            None => {
                let mut x_cur = cur_node.upgrade().map_err(|(_x, e)| e)?;

                let mut sibling_builder = InnerNodeBuilder::new();
                sibling_builder.set_disk_offset(self.storage.alloc_disk_offset(INNER_NODE_SIZE));
                let sibling_id = sibling_builder.get_page_id();

                let split_key = x_cur.as_mut().split(&mut sibling_builder);

                let mut new_root_builder = InnerNodeBuilder::new();
                new_root_builder
                    .set_disk_offset(self.storage.alloc_disk_offset(INNER_NODE_SIZE))
                    .set_left_most_page_id(cur_page)
                    .set_children_is_leaf(false)
                    .add_record(split_key, sibling_id);
                sibling_builder.build();
                let new_root_ptr = new_root_builder.build();
                let _x_root = ReadGuard::try_read(new_root_ptr)
                    .unwrap()
                    .upgrade()
                    .unwrap();
                self.root_page_id
                    .store(PageID::from_pointer(new_root_ptr).raw(), Ordering::Release);

                info!(
                    has_parent = parent.is_some(),
                    cur = cur_page.raw(),
                    "finished split inner node"
                );

                Ok((true, parent))
            }
        }
    }

    pub(crate) fn traverse_to_leaf(
        &self,
        key: &[u8],
        aggressive_split: bool,
    ) -> Result<(PageID, Option<ReadGuard<'_>>), TreeError> {
        let (mut cur_page, mut cur_is_leaf) = self.get_root_page();
        let mut parent: Option<ReadGuard> = None;

        loop {
            if aggressive_split {
                if cur_is_leaf
                    && !cur_page.is_inner_node_pointer()
                    && self.try_split_leaf(cur_page, &parent)?
                {
                    return Err(TreeError::NeedRestart);
                } else if !cur_is_leaf {
                    let (split_success, new_parent) = self.try_split_inner(cur_page, parent)?;
                    if split_success {
                        return Err(TreeError::NeedRestart);
                    } else {
                        parent = new_parent;
                    }
                }
            }

            if cur_is_leaf {
                return Ok((cur_page, parent));
            } else {
                let next = ReadGuard::try_read(cur_page.as_inner_node())?;

                check_parent!(self, cur_page, parent);

                let next_node = next.as_ref();
                let next_is_leaf = next_node.meta.children_is_leaf();
                let pos = next_node.lower_bound(key);
                let kv_meta = next_node.get_kv_meta(pos as u16);
                cur_page = next_node.get_value(kv_meta);
                cur_is_leaf = next_is_leaf;
                parent = Some(next);
            }
        }
    }

    fn write_inner(&self, write_op: WriteOp, aggressive_split: bool) -> Result<(), TreeError> {
        let (pid, parent) = self.traverse_to_leaf(write_op.key, aggressive_split)?;

        let mut leaf_entry = self.mapping_table().get_mut(&pid);

        check_parent!(self, pid, parent);

        let page_loc = leaf_entry.get_page_location();
        match page_loc {
            PageLocation::Null => {
                if !self.cache_only {
                    panic!("Found an Null page in non cache-only mode");
                }

                if write_op.op_type == OpType::Delete {
                    return Ok(());
                }

                // Create a new mini-page to replace the null page
                let mini_page_size = LeafNode::get_chain_size_hint(
                    write_op.key.len(),
                    write_op.value.len(),
                    &self.mini_page_size_classes,
                    self.cache_only,
                );
                let mini_page_guard = self.storage.alloc_mini_page(mini_page_size)?;
                LeafNode::initialize_mini_page(
                    &mini_page_guard,
                    mini_page_size,
                    MiniPageNextLevel::new_null(),
                    true,
                );
                let new_mini_ptr = mini_page_guard.as_ptr() as *mut LeafNode;
                let mini_loc = PageLocation::Mini(new_mini_ptr);

                leaf_entry.create_cache_page_loc(mini_loc);

                let mini_page_ref = leaf_entry.load_cache_page_mut(new_mini_ptr);
                let insert_success =
                    mini_page_ref.insert(write_op.key, write_op.value, write_op.op_type, 0);
                assert!(insert_success);

                debug_assert!(mini_page_ref.meta.meta_count_with_fence() > 0);
                counter!(InsertCreatedMiniPage);
            }
            _ => {
                leaf_entry.insert(
                    write_op.key,
                    write_op.value,
                    parent,
                    write_op.op_type,
                    &self.storage,
                    &self.write_load_full_page,
                    &self.cache_only,
                    &self.mini_page_size_classes,
                )?;

                if leaf_entry.cache_page_about_to_evict(&self.storage) {
                    // we don't care about the result here
                    _ = leaf_entry.move_cache_page_to_tail(&self.storage);
                }

                if let Some(wal) = &self.wal {
                    let lsn = wal.append_and_wait(&write_op, leaf_entry.get_disk_offset());
                    leaf_entry.update_lsn(lsn);
                }
            }
        }

        Ok(())
    }

    /// Make sure you're not holding any lock while calling this function.
    pub(crate) fn evict_from_circular_buffer(&self) -> Result<usize, TreeError> {
        // Why we need to evict multiple times?
        // because we don't want each alloc to trigger evict, i.e., we want alloc to fail less often.
        // with default 1024 bytes, one eviction allows us to alloc 1024 bytes (4 256-byte mini pages) without failure.
        const TARGET_EVICT_SIZE: usize = 1024;
        let mut evicted = 0;

        // A corner case: we may not have enough memory to evict (i.e., the buffer might be empty now)
        let mut retry_cnt = 0;

        while evicted < TARGET_EVICT_SIZE && retry_cnt < 10 {
            let n = self
                .storage
                .evict_from_buffer(|mini_page_handle: &TombstoneHandle| {
                    eviction_callback(mini_page_handle, self)
                })?;
            evicted += n as usize;
            retry_cnt += 1;
        }
        info!("stopped evict from circular buffer");
        Ok(evicted)
    }

    /// Insert a key-value pair to the system, overrides existing value if present.
    ///
    /// ```
    /// use bf_tree::BfTree;
    /// use bf_tree::LeafReadResult;
    ///
    /// let mut config = bf_tree::Config::default();
    /// config.cb_min_record_size(4);
    /// let tree = BfTree::with_config(config, None).unwrap();
    /// tree.insert(b"key", b"value");
    /// let mut buffer = [0u8; 1024];
    /// let read_size = tree.read(b"key", &mut buffer);
    ///
    /// assert_eq!(read_size, LeafReadResult::Found(5));
    /// assert_eq!(&buffer[..5], b"value");
    /// ```
    pub fn insert(&self, key: &[u8], value: &[u8]) -> LeafInsertResult {
        // The input key cannot exceed the configured max key length
        if key.len() > self.config.max_fence_len / 2 || key.len() > MAX_KEY_LEN {
            return LeafInsertResult::InvalidKV(format!("Key too large {}", key.len()));
        }

        // The input key has to be one byte at least
        if key.is_empty() {
            return LeafInsertResult::InvalidKV(format!(
                "Key too small {}, at least one byte",
                key.len()
            ));
        }

        // The input key value pair cannot exceed the configured max record size
        if value.len() > MAX_VALUE_LEN || key.len() + value.len() > self.config.cb_max_record_size {
            return LeafInsertResult::InvalidKV(format!(
                "Record too large {}, {}, please adjust cb_max_record_size in config",
                key.len(),
                value.len()
            ));
        }

        // The input key value pair cannot be smaller than the configured min record size
        if key.len() + value.len() < self.config.cb_min_record_size {
            return LeafInsertResult::InvalidKV(format!(
                "Record too small {}, {}, please adjust cb_min_record_size in config",
                key.len(),
                value.len()
            ));
        }

        let backoff = Backoff::new();
        let mut aggressive_split = false;

        counter!(Insert);
        info!(key_len = key.len(), value_len = value.len(), "insert");

        loop {
            let result = self.write_inner(WriteOp::make_insert(key, value), aggressive_split);
            match result {
                Ok(_) => return LeafInsertResult::Success,
                Err(TreeError::NeedRestart) => {
                    #[cfg(all(feature = "shuttle", test))]
                    {
                        shuttle::thread::yield_now();
                    }
                    counter!(InsertNeedRestart);
                    aggressive_split = true;
                }
                Err(TreeError::CircularBufferFull) => {
                    info!("insert failed, started evict from circular buffer");
                    aggressive_split = true;
                    counter!(InsertCircularBufferFull);
                    _ = self.evict_from_circular_buffer();
                    continue;
                }
                Err(TreeError::Locked) => {
                    counter!(InsertLocked);
                    backoff.snooze();
                }
            }
        }
    }

    /// Read a record from the tree.
    /// Returns the number of bytes read.
    ///
    /// TODO: don't panic if the out_buffer is too small, instead returns a error.
    ///
    /// ```
    /// use bf_tree::BfTree;
    /// use bf_tree::LeafReadResult;
    ///
    /// let mut config = bf_tree::Config::default();
    /// config.cb_min_record_size(4);
    ///
    /// let tree = BfTree::with_config(config, None).unwrap();
    /// tree.insert(b"key", b"value");
    /// let mut buffer = [0u8; 1024];
    /// let read_size = tree.read(b"key", &mut buffer);
    /// assert_eq!(read_size, LeafReadResult::Found(5));
    /// assert_eq!(&buffer[..5], b"value");
    /// ```
    pub fn read(&self, key: &[u8], out_buffer: &mut [u8]) -> LeafReadResult {
        // The input key cannot exceed the configured max key length
        if key.len() > self.config.max_fence_len / 2 || key.len() > MAX_KEY_LEN {
            return LeafReadResult::InvalidKey;
        }

        // The input key has to be one byte at least
        if key.is_empty() {
            return LeafReadResult::InvalidKey;
        }

        let backoff = Backoff::new();

        info!(key_len = key.len(), "read");
        counter!(Read);
        let mut aggressive_split = false;

        #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
        let mut debug_timer = DebugTimerGuard::new(Timer::Read, self.metrics_recorder.clone());

        loop {
            let result = self.read_inner(key, out_buffer, aggressive_split);
            match result {
                Ok(v) => {
                    #[cfg(any(
                        feature = "metrics-rt-debug-all",
                        feature = "metrics-rt-debug-timer"
                    ))]
                    debug_timer.end();

                    return v;
                }
                Err(TreeError::CircularBufferFull) => {
                    info!("read promotion failed, started evict from circular buffer");
                    aggressive_split = true;
                    match self.evict_from_circular_buffer() {
                        Ok(_) => continue,
                        Err(_) => continue,
                    };
                }
                Err(_) => {
                    backoff.spin();
                    aggressive_split = true;
                }
            }
        }
    }

    /// Delete a record from the tree.
    ///
    /// ```
    /// use bf_tree::BfTree;
    /// use bf_tree::LeafReadResult;
    ///
    /// let tree = BfTree::default();
    /// tree.insert(b"key", b"value");
    /// tree.delete(b"key");
    /// let mut buffer = [0u8; 1024];
    /// let rt = tree.read(b"key", &mut buffer);
    /// assert_eq!(rt, LeafReadResult::Deleted);
    /// ```
    pub fn delete(&self, key: &[u8]) {
        // The input key cannot exceed the configured max key length
        if key.len() > self.config.max_fence_len / 2 || key.len() > MAX_KEY_LEN {
            return;
        }

        // The input key has to be one byte at least
        if key.is_empty() {
            return;
        }

        let backoff = Backoff::new();

        info!(key_len = key.len(), "delete");

        let mut aggressive_split = false;

        loop {
            let result = self.write_inner(WriteOp::make_delete(key), aggressive_split);
            match result {
                Ok(_) => return,
                Err(TreeError::CircularBufferFull) => {
                    info!("delete failed, started evict from circular buffer");
                    aggressive_split = true;
                    match self.evict_from_circular_buffer() {
                        Ok(_) => continue,
                        Err(_) => continue,
                    };
                }
                Err(_) => {
                    aggressive_split = true;
                    backoff.spin();
                }
            }
        }
    }

    /// Scan records in the tree, with starting key and desired scan count.
    /// Returns a iterator that yields key-value pairs.
    pub fn scan_with_count<'a>(
        &'a self,
        key: &[u8],
        cnt: usize,
        return_field: ScanReturnField,
    ) -> Result<ScanIter<'a, 'a>, ScanIterError> {
        // In cache-only mode, scan is not supported
        if self.cache_only {
            return Err(ScanIterError::CacheOnlyMode);
        }

        // The start key cannot exceed the configured max key length
        if key.len() > self.config.max_fence_len / 2 || key.len() > MAX_KEY_LEN {
            return Err(ScanIterError::InvalidStartKey);
        }

        // The input key has to be one byte at least
        if key.is_empty() {
            return Err(ScanIterError::InvalidStartKey);
        }

        // The count cannot be zero
        if cnt == 0 {
            return Err(ScanIterError::InvalidCount);
        }

        Ok(ScanIter::new_with_scan_count(self, key, cnt, return_field))
    }

    pub fn scan_with_end_key<'a>(
        &'a self,
        start_key: &[u8],
        end_key: &[u8],
        return_field: ScanReturnField,
    ) -> Result<ScanIter<'a, 'a>, ScanIterError> {
        // In cache-only mode, scan is not supported
        if self.cache_only {
            return Err(ScanIterError::CacheOnlyMode);
        }

        // The start key cannot exceed the configured max key length
        if start_key.len() > self.config.max_fence_len / 2 || start_key.len() > MAX_KEY_LEN {
            return Err(ScanIterError::InvalidStartKey);
        }

        // The input key has to be one byte at least
        if start_key.is_empty() {
            return Err(ScanIterError::InvalidStartKey);
        }

        // The end key cannot exceed the configured max key length
        if end_key.len() > self.config.max_fence_len / 2 || end_key.len() > MAX_KEY_LEN {
            return Err(ScanIterError::InvalidEndKey);
        }

        // The input key has to be one byte at least
        if end_key.is_empty() {
            return Err(ScanIterError::InvalidEndKey);
        }

        // The start key cannot be greater than the end key
        let cmp = start_key.cmp(end_key);
        if cmp == std::cmp::Ordering::Greater {
            return Err(ScanIterError::InvalidKeyRange);
        }

        Ok(ScanIter::new_with_end_key(
            self,
            start_key,
            end_key,
            return_field,
        ))
    }

    #[doc(hidden)]
    pub fn scan_mut_with_count<'a>(
        &'a self,
        key: &'a [u8],
        cnt: usize,
        return_field: ScanReturnField,
    ) -> Result<ScanIterMut<'a, 'a>, ScanIterError> {
        // In cache-only mode, scan is not supported
        if self.cache_only {
            return Err(ScanIterError::CacheOnlyMode);
        }

        // The start key cannot exceed the configured max key length
        if key.len() > self.config.max_fence_len / 2 || key.len() > MAX_KEY_LEN {
            return Err(ScanIterError::InvalidStartKey);
        }

        // The count cannot be zero
        if cnt == 0 {
            return Err(ScanIterError::InvalidCount);
        }

        Ok(ScanIterMut::new_with_scan_count(
            self,
            key,
            cnt,
            return_field,
        ))
    }

    #[doc(hidden)]
    pub fn scan_mut_with_end_key<'a>(
        &'a self,
        start_key: &'a [u8],
        end_key: &'a [u8],
        return_field: ScanReturnField,
    ) -> Result<ScanIterMut<'a, 'a>, ScanIterError> {
        // In cache-only mode, scan is not supported
        if self.cache_only {
            return Err(ScanIterError::CacheOnlyMode);
        }

        // The start key cannot exceed the configured max key length
        if start_key.len() > self.config.max_fence_len / 2 || start_key.len() > MAX_KEY_LEN {
            return Err(ScanIterError::InvalidStartKey);
        }

        // The end key cannot exceed the configured max key length
        if end_key.len() > self.config.max_fence_len / 2 || end_key.len() > MAX_KEY_LEN {
            return Err(ScanIterError::InvalidEndKey);
        }

        Ok(ScanIterMut::new_with_end_key(
            self,
            start_key,
            end_key,
            return_field,
        ))
    }

    fn read_inner(
        &self,
        key: &[u8],
        out_buffer: &mut [u8],
        aggressive_split: bool,
    ) -> Result<LeafReadResult, TreeError> {
        let (node, parent) = self.traverse_to_leaf(key, aggressive_split)?;

        let mut leaf = self.mapping_table().get(&node);

        check_parent!(self, node, parent);

        let out = leaf.read(
            key,
            out_buffer,
            self.config.mini_page_binary_search,
            self.cache_only,
        );
        match out {
            ReadResult::Mini(r) | ReadResult::Full(r) => {
                if leaf.cache_page_about_to_evict(&self.storage) {
                    let mut x_leaf = match leaf.try_upgrade() {
                        Ok(v) => v,
                        Err(_) => return Ok(r),
                    };
                    // we don't care about the result here, because we are in read path, we don't want to block.
                    _ = x_leaf.move_cache_page_to_tail(&self.storage);
                }

                Ok(r)
            }

            ReadResult::Base(r) => {
                counter!(BasePageRead);

                // In cache-only mode, no base page should exist
                if self.cache_only {
                    panic!("Attempt to read a base page while in cache-only mode.");
                }

                let v = match r {
                    LeafReadResult::Found(v) => v,
                    _ => return Ok(r),
                };

                if parent.is_none() || !self.should_promote_read() {
                    return Ok(r);
                }

                let mut x_leaf = match leaf.try_upgrade() {
                    Ok(x) => x,
                    Err(_) => {
                        return Ok(r);
                    }
                };

                if self.config.read_record_cache {
                    // we do record cache.
                    // we roll dice to see if we should insert this value to mini page.

                    let out = x_leaf.insert(
                        key,
                        &out_buffer[0..v as usize],
                        parent,
                        OpType::Cache,
                        &self.storage,
                        &self.write_load_full_page,
                        &self.cache_only,
                        &self.mini_page_size_classes,
                    );

                    match out {
                        Ok(_) => {
                            counter!(ReadPromotionOk);
                            Ok(r)
                        }
                        Err(TreeError::Locked) => {
                            // We are doing this very optimistically, if contention happens, we just abort and return.
                            counter!(ReadPromotionFailed);
                            Ok(r)
                        }
                        Err(TreeError::CircularBufferFull) => {
                            counter!(ReadPromotionFailed);
                            Err(TreeError::CircularBufferFull)
                        }
                        Err(TreeError::NeedRestart) => {
                            // If we need restart here, potentially because parent is full.
                            counter!(ReadPromotionFailed);
                            Err(TreeError::NeedRestart)
                        }
                    }
                } else {
                    match self.upgrade_to_full_page(x_leaf, parent.unwrap()) {
                        Ok(_) | Err(TreeError::Locked) => Ok(r),
                        Err(e) => Err(e),
                    }
                }
            }
            ReadResult::None => Ok(LeafReadResult::NotFound),
        }
    }

    fn upgrade_to_full_page<'a>(
        &'a self,
        mut x_leaf: LeafEntryXLocked<'a>,
        parent: ReadGuard<'a>,
    ) -> Result<LeafEntryXLocked<'a>, TreeError> {
        let page_loc = x_leaf.get_page_location().clone();
        match page_loc {
            PageLocation::Mini(ptr) => {
                let mini_page = x_leaf.load_cache_page_mut(ptr);
                let h = self.storage.begin_dealloc_mini_page(mini_page)?;
                let _merge_result = x_leaf.try_merge_mini_page(&h, parent, &self.storage)?;
                let base_offset = mini_page.next_level;
                x_leaf.change_to_base_loc();
                self.storage.finish_dealloc_mini_page(h);

                let base_page_ref = x_leaf.load_base_page_from_buffer();
                let full_page_loc =
                    upgrade_to_full_page(&self.storage, base_page_ref, base_offset)?;
                x_leaf.create_cache_page_loc(full_page_loc);
                Ok(x_leaf)
            }
            PageLocation::Full(_ptr) => Ok(x_leaf),
            PageLocation::Base(offset) => {
                let base_page_ref = x_leaf.load_base_page(offset);
                let next_level = MiniPageNextLevel::new(offset);
                let full_page_loc = upgrade_to_full_page(&self.storage, base_page_ref, next_level)?;
                x_leaf.create_cache_page_loc(full_page_loc);
                Ok(x_leaf)
            }
            PageLocation::Null => panic!("upgrade_to_full_page on Null page"),
        }
    }

    /// Collect all metrics and reset the metric recorder
    /// The caller needs to ensure there are no references to the bf-tree's metrics recorder anymore.
    pub fn get_metrics(&mut self) -> Option<serde_json::Value> {
        #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
        {
            let recorder = self.metrics_recorder.take();
            match recorder {
                Some(r) => {
                    let recorders = Arc::try_unwrap(r).expect("Failed to obtain the recorders of bf-tree, please make sure no other references exist.");
                    let mut timer_accumulated = TimerRecorder::default();

                    // Only collect timer metrics for now
                    for r in recorders {
                        let t = unsafe { &*r.get() };

                        timer_accumulated += t.timers.clone();
                    }

                    let output = serde_json::json!({
                        "Timers": timer_accumulated,
                    });

                    self.metrics_recorder = Some(Arc::new(ThreadLocal::new()));

                    Some(output)
                }
                None => None,
            }
        }
        #[cfg(not(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer")))]
        {
            None
        }
    }
}

pub(crate) fn key_value_physical_size(key: &[u8], value: &[u8]) -> usize {
    let key_size = key.len();
    let value_size = value.len();
    let meta_size = crate::nodes::KV_META_SIZE;
    key_size + value_size + meta_size
}

pub(crate) fn eviction_callback(
    mini_page_handle: &TombstoneHandle,
    tree: &BfTree,
) -> Result<(), TreeError> {
    let mini_page = mini_page_handle.ptr as *mut LeafNode;
    let key_to_this_page = if tree.cache_only {
        unsafe { &*mini_page }.try_get_key_to_reach_this_node()?
    } else {
        unsafe { &*mini_page }.get_key_to_reach_this_node()
    };

    // Here we need to set aggressive split to true, because we would split parent node due to leaf split.
    let (pid, parent) = tree.traverse_to_leaf(&key_to_this_page, true)?;
    info!(
        pid = pid.raw(),
        "starting to merge mini page in eviction call back"
    );

    let mut leaf_entry = tree.mapping_table().get_mut(&pid);

    histogram!(EvictNodeSize, unsafe { &*mini_page }.meta.node_size as u64);

    match leaf_entry.get_page_location() {
        PageLocation::Mini(ptr) => {
            {
                // In order to lock this node, we need to traverse to this node first;
                // but in order to traverse this node, we need to read the keys in this node;
                // in order to read the keys in this node, we need to lock this node.
                //
                // Because we didn't lock the node while reading `key_to_this_page`,
                // we need to recheck if the node is still the same node.
                if *ptr != mini_page {
                    return Err(TreeError::NeedRestart);
                }
            }

            let parent = parent.expect("Mini page must have a parent");
            parent.check_version()?;

            // In the case of cache_only, the correponding mapping table entry of the mini-page
            // is replaced by a non-existant base page
            if tree.cache_only {
                leaf_entry.change_to_null_loc();
            } else {
                leaf_entry.try_merge_mini_page(mini_page_handle, parent, &tree.storage)?;
                leaf_entry.change_to_base_loc();
                // we don't need to dealloc the old_mini_page here because we are in eviction callback.
            }

            Ok(())
        }

        PageLocation::Full(ptr) => {
            if *ptr != mini_page {
                return Err(TreeError::NeedRestart);
            }

            leaf_entry.merge_full_page(mini_page_handle);
            Ok(())
        }

        // This means the key read from the mini page is corrupted and points to a different page
        PageLocation::Base(_offset) => Err(TreeError::NeedRestart),

        // This means the key read from the mini page is corrupted and points to a different page
        PageLocation::Null => Err(TreeError::NeedRestart),
    }
}

#[cfg(test)]
mod tests {
    use crate::error::ConfigError;
    use crate::BfTree;

    #[test]
    fn test_mini_page_size_classes() {
        let mut size_classes = BfTree::create_mem_page_size_classes(48, 1952, 4096, 64, false);
        assert_eq!(
            size_classes,
            vec![128, 192, 256, 512, 960, 1856, 2048, 4096]
        );

        size_classes = BfTree::create_mem_page_size_classes(1548, 1548, 3136, 64, true);
        assert_eq!(size_classes, vec![1536, 3136]);

        size_classes = BfTree::create_mem_page_size_classes(48, 3072, 12288, 64, false);
        assert_eq!(
            size_classes,
            vec![128, 192, 256, 512, 960, 1856, 3648, 7232, 9088, 12288]
        );

        size_classes = BfTree::create_mem_page_size_classes(4, 1952, 4096, 32, false);
        assert_eq!(size_classes, vec![64, 128, 256, 448, 832, 1600, 2048, 4096]);
    }

    #[test]
    fn test_invalid_config_to_build_bf_tree() {
        // Min record too small
        let mut config = crate::Config::default();
        config.cb_min_record_size(4);
        config.leaf_page_size(32 * 1024);

        if let Err(e) = BfTree::with_config(config.clone(), None) {
            match e {
                ConfigError::MinimumRecordSize(_) => {}
                _ => panic!("Expected InvalidMinimumRecordSize error"),
            }
        } else {
            panic!("Expected error but got Ok");
        }

        // Max record too large
        config = crate::Config::default();
        config.cb_max_record_size(64 * 1024);

        if let Err(e) = BfTree::with_config(config.clone(), None) {
            match e {
                ConfigError::MaximumRecordSize(_) => {}
                _ => panic!("Expected InvalidMaximumRecordSize error"),
            }
        } else {
            panic!("Expected error but got Ok");
        }

        // Leaf page size not aligned
        config = crate::Config::default();
        config.leaf_page_size(4050);

        if let Err(e) = BfTree::with_config(config.clone(), None) {
            match e {
                ConfigError::LeafPageSize(_) => {}
                _ => panic!("Expected InvalidLeafPageSize error"),
            }
        } else {
            panic!("Expected error but got Ok");
        }

        // Circular buffer size too small
        config = crate::Config::default();
        config.leaf_page_size(16 * 1024);
        config.cb_size_byte(16 * 1024);

        if let Err(e) = BfTree::with_config(config.clone(), None) {
            match e {
                ConfigError::CircularBufferSize(_) => {}
                _ => panic!("Expected InvalidCircularBufferSize error"),
            }
        } else {
            panic!("Expected error but got Ok");
        }

        // Circular buffer size not power of two
        config = crate::Config::default();
        config.cb_size_byte(20 * 1024);
        if let Err(e) = BfTree::with_config(config.clone(), None) {
            match e {
                ConfigError::CircularBufferSize(_) => {}
                _ => panic!("Expected InvalidCircularBufferSize error"),
            }
        } else {
            panic!("Expected error but got Ok");
        }

        // Cache-only mode specific
        config = crate::Config::default();
        config.cache_only(true);
        config.cb_size_byte(2 * 4096);

        if let Err(e) = BfTree::with_config(config.clone(), None) {
            match e {
                ConfigError::CircularBufferSize(_) => {}
                _ => panic!("Expected InvalidCircularBufferSize error"),
            }
        } else {
            panic!("Expected error but got Ok");
        }
    }
}
