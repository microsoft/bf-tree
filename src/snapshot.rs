// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::{HashMap, VecDeque};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

#[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
use thread_local::ThreadLocal;

use crate::{
    circular_buffer::{CircularBuffer, TombstoneHandle},
    error::ConfigError,
    fs::VfsImpl,
    nodes::{InnerNode, InnerNodeBuilder, PageID, DISK_PAGE_SIZE, INNER_NODE_SIZE},
    storage::{make_vfs, LeafStorage, PageLocation, PageTable},
    sync::atomic::AtomicU64,
    tree::eviction_callback,
    utils::{inner_lock::ReadGuard, BfsVisitor, NodeInfo},
    wal::{LogEntry, LogEntryImpl, WriteAheadLog},
    BfTree, Config, WalReader,
};

const BF_TREE_MAGIC_BEGIN: &[u8; 16] = b"BF-TREE-V0-BEGIN";
const BF_TREE_MAGIC_END: &[u8; 14] = b"BF-TREE-V0-END";
const META_DATA_PAGE_OFFSET: usize = 0;

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

impl BfTree {
    /// Recovery a Bf-Tree from snapshot and WAL files.
    /// Incomplete function, internal use only
    pub fn recovery(
        config_file: impl AsRef<Path>,
        wal_file: impl AsRef<Path>,
        buffer_ptr: Option<*mut u8>,
    ) {
        let bf_tree_config = Config::new_with_config_file(config_file);
        let bf_tree = BfTree::new_from_snapshot(bf_tree_config, buffer_ptr).unwrap();
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

    /// Instead of creating a new Bf-Tree instance,
    /// it loads a Bf-Tree snapshot file and resume from there.
    pub fn new_from_snapshot(
        bf_tree_config: Config,
        buffer_ptr: Option<*mut u8>,
    ) -> Result<Self, ConfigError> {
        if !bf_tree_config.file_path.exists() {
            // if not already exist, we just create a new empty file at the location.
            return BfTree::with_config(bf_tree_config.clone(), buffer_ptr);
        }

        // Validate the config first
        bf_tree_config.validate()?;

        let reader = std::fs::File::open(bf_tree_config.file_path.clone()).unwrap();
        let mut metadata = SectorAlignedVector::new_zeroed(4096);
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

        let config = Arc::new(bf_tree_config);

        let wal = config
            .write_ahead_log
            .as_ref()
            .map(|s| WriteAheadLog::new(s.clone()));

        let vfs = make_vfs(&config.storage_backend, &config.file_path);

        let mut page_buffer = SectorAlignedVector::new_zeroed(INNER_NODE_SIZE);

        // Step 1: reconstruct inner nodes.
        let mut root_page_id = bf_meta.root_id;
        if root_page_id.is_inner_node_pointer() {
            let inner_mapping: Vec<(*const InnerNode, usize)> =
                read_vec_from_offset(bf_meta.inner_offset, bf_meta.inner_size, &vfs);
            let mut inner_map = HashMap::new();

            for m in inner_mapping {
                inner_map.insert(m.0, m.1);
            }
            let offset = inner_map.get(&root_page_id.as_inner_node()).unwrap();
            vfs.read(*offset, &mut page_buffer);
            let root_page = InnerNodeBuilder::new().build_from_slice(&page_buffer);
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
                    vfs.read(*offset, &mut page_buffer);
                    let inner_page = InnerNodeBuilder::new().build_from_slice(&page_buffer);
                    let inner_id = PageID::from_pointer(inner_page);
                    inner.as_mut().update_at_pos(idx, inner_id);
                    inner_resolve_queue.push_back(inner_page);
                }
            }
        }

        // Step 2: reconstruct leaf mappings.
        let leaf_mapping: Vec<(PageID, usize)> =
            read_vec_from_offset(bf_meta.leaf_offset, bf_meta.leaf_size, &vfs);
        let leaf_mapping = leaf_mapping.into_iter().map(|(pid, offset)| {
            let loc = PageLocation::Base(offset);
            (pid, loc)
        });
        let pt = PageTable::new_from_mapping(leaf_mapping, vfs.clone(), config.clone());
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

        let storage = LeafStorage::new_inner(config.clone(), pt, circular_buffer, vfs);

        let raw_root_id = if root_page_id.is_id() {
            root_page_id.raw() | Self::ROOT_IS_LEAF_MASK
        } else {
            root_page_id.raw()
        };

        let size_classes = Self::create_mem_page_size_classes(
            config.cb_min_record_size,
            config.cb_max_record_size,
            config.leaf_page_size,
            config.max_fence_len,
            config.cache_only,
        );
        Ok(BfTree {
            storage,
            root_page_id: AtomicU64::new(raw_root_id),
            wal,
            write_load_full_page: true,
            cache_only: false,
            mini_page_size_classes: size_classes,
            config,
            #[cfg(any(feature = "metrics-rt-debug-all", feature = "metrics-rt-debug-timer"))]
            metrics_recorder: Some(Arc::new(ThreadLocal::new())),
        })
    }

    /// Stop the world and take a snapshot of the current state.
    ///
    /// Returns the snapshot file path
    pub fn snapshot(&self) -> PathBuf {
        let callback = |h| -> Result<TombstoneHandle, TombstoneHandle> {
            match eviction_callback(&h, self) {
                Ok(_) => Ok(h),
                Err(_e) => Err(h),
            }
        };
        self.storage.circular_buffer.drain(callback);

        let root_id = self.get_root_page();
        let mut inner_mapping: Vec<(*const InnerNode, usize)> = Vec::new();
        let visitor = BfsVisitor::new_inner_only(self);
        for node in visitor {
            match node {
                NodeInfo::Inner { ptr, .. } => {
                    let inner = ReadGuard::try_read(ptr).unwrap();
                    if inner.as_ref().is_valid_disk_offset() {
                        let offset = inner.as_ref().disk_offset as usize;
                        self.storage.vfs.write(offset, inner.as_ref().as_slice());
                        inner_mapping.push((ptr, offset));
                    }
                }
                NodeInfo::Leaf { level, .. } => {
                    // corner case: we might still get a leaf node when the root is leaf...
                    //
                    // When ROOT is leaf, it is in `FORCE` mode, meaning that all data are write to disk.
                    // do don't need to do anything here.
                    assert_eq!(level, 0);
                }
            }
        }
        let (inner_offset, inner_size) = serialize_vec_to_disk(&inner_mapping, &self.storage.vfs);

        let mut leaf_mapping = Vec::new();
        let page_table_iter = self.storage.page_table.iter();
        for (entry, pid) in page_table_iter {
            assert!(pid.is_id());
            match entry.try_read().unwrap().as_ref() {
                PageLocation::Base(base) => leaf_mapping.push((pid, *base)),
                PageLocation::Full(_) | PageLocation::Mini(_) => {
                    unreachable!("Circular buffer should already be drained!")
                }
                PageLocation::Null => panic!("Snapshot of Null page"),
            }
        }

        let (leaf_offset, leaf_size) = serialize_vec_to_disk(&leaf_mapping, &self.storage.vfs);

        let file_size = (leaf_offset + align_to_sector_size(leaf_size)) as u64;

        let metadata = BfTreeMeta {
            magic_begin: *BF_TREE_MAGIC_BEGIN,
            root_id: root_id.0,
            inner_offset,
            inner_size,
            leaf_offset,
            leaf_size,
            file_size,
            magic_end: *BF_TREE_MAGIC_END,
        };

        self.storage
            .vfs
            .write(META_DATA_PAGE_OFFSET, metadata.as_slice());
        self.storage.vfs.flush();
        self.config.file_path.clone()
    }
}

/// We use repr(C) for simplicity, maybe flatbuffer or bincode or even repr(Rust) is better.
/// But we don't care about the space here.
/// I don't want to introduce giant dependencies just for this.
#[repr(C, align(512))]
struct BfTreeMeta {
    magic_begin: [u8; 16],
    root_id: PageID,
    inner_offset: usize,
    inner_size: usize,
    leaf_offset: usize,
    leaf_size: usize,
    file_size: u64,
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
    use crate::{
        nodes::leaf_node::LeafReadResult, utils::test_util::install_value_to_buffer, BfTree, Config,
    };
    use rstest::rstest;
    use std::str::FromStr;

    #[rstest]
    #[case(64, 2408, 8192, 500, "target/test_simple_1.bftree")] // 1 leaf page = 2 disk page
    #[case(64, 2048, 16384, 500, "target/test_simple_2.bftree")] // 1 leaf page = 4 disk page
    #[case(3072, 3072, 16384, 500, "target/test_simple_3.bftree")] // 1 leaf page = 1 disk page, uniform record size
    fn persist_roundtrip_simple(
        #[case] min_record_size: usize,
        #[case] max_record_size: usize,
        #[case] leaf_page_size: usize,
        #[case] record_cnt: usize,
        #[case] snapshot_file_path: String,
    ) {
        let tmp_file_path = std::path::PathBuf::from_str(&snapshot_file_path).unwrap();

        let mut config = Config::new(&tmp_file_path, leaf_page_size * 16); // Creat a CB that can hold 16 full pages
        config.storage_backend(crate::StorageBackend::Std);
        config.cb_min_record_size = min_record_size;
        config.cb_max_record_size = max_record_size;
        config.leaf_page_size = leaf_page_size;
        config.max_fence_len = max_record_size;

        let bftree = BfTree::with_config(config.clone(), None).unwrap();

        let key_len: usize = min_record_size / 2;
        let mut key_buffer = vec![0; key_len / 8];

        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            bftree.insert(key, key);
        }
        bftree.snapshot();
        drop(bftree);

        let bftree = BfTree::new_from_snapshot(config.clone(), None).unwrap();
        let mut out_buffer = vec![0; key_len];
        for r in 0..record_cnt {
            let key = install_value_to_buffer(&mut key_buffer, r);
            let bytes_read = bftree.read(key, &mut out_buffer);

            match bytes_read {
                LeafReadResult::Found(v) => {
                    assert_eq!(v as usize, key_len);
                    assert_eq!(&out_buffer, key);
                }
                _ => {
                    panic!("Key not found");
                }
            }
        }

        std::fs::remove_file(tmp_file_path).unwrap();
    }
}
