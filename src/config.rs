// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use crate::{
    error::ConfigError,
    nodes::{
        leaf_node::LeafKVMeta, LeafNode, CACHE_LINE_SIZE, DISK_PAGE_SIZE, MAX_KEY_LEN,
        MAX_LEAF_PAGE_SIZE,
    },
};
use serde::Deserialize;
use std::sync::atomic::AtomicUsize;

const DEFAULT_PROMOTION_RATE_DEBUG: usize = 50;
const DEFAULT_PROMOTION_RATE_RLEASE: usize = 30;
const DEFAULT_MAX_MINI_PAGE_SIZE: usize = 2048; // Deprecated
const DEFAULT_COPY_ON_ACCESS_RATIO: f64 = 0.1;
const DEFAULT_CIRCULAR_BUFFER_SIZE: usize = 1024 * 1024 * 32;
const DEFAULT_MIN_RECORD_SIZE: usize = 4;
const DEFAULT_MAX_RECORD_SIZE: usize = 1952;
const DEFAULT_LEAF_PAGE_SIZE: usize = 4096;
const DEFAULT_MAX_KEY_LEN: usize = 16;

/// Bf-tree configuration for advanced usage.
/// Bf-tree is designed to work well on various workloads that you don't have to change the default configuration.
/// This configuration is more for advanced users who want to understand the different components of the system, rather than performance tuning.
#[derive(Debug)]
pub struct Config {
    pub(crate) read_promotion_rate: AtomicUsize,
    pub(crate) scan_promotion_rate: AtomicUsize,
    pub(crate) storage_backend: StorageBackend,
    pub(crate) cb_size_byte: usize,
    pub(crate) cb_min_record_size: usize,
    pub(crate) cb_max_record_size: usize,
    pub(crate) leaf_page_size: usize,
    pub(crate) cb_max_key_len: usize,
    pub(crate) max_fence_len: usize,
    pub(crate) cb_copy_on_access_ratio: f64,
    pub(crate) read_record_cache: bool,
    pub(crate) file_path: PathBuf,
    pub(crate) max_mini_page_size: usize,
    pub(crate) mini_page_binary_search: bool,
    pub(crate) write_ahead_log: Option<Arc<WalConfig>>,
    pub(crate) write_load_full_page: bool,
    pub(crate) cache_only: bool,
}

impl Clone for Config {
    fn clone(&self) -> Self {
        Self {
            read_promotion_rate: AtomicUsize::new(self.read_promotion_rate.load(Ordering::Relaxed)),
            scan_promotion_rate: AtomicUsize::new(self.scan_promotion_rate.load(Ordering::Relaxed)),
            storage_backend: self.storage_backend.clone(),
            cb_size_byte: self.cb_size_byte,
            cb_min_record_size: self.cb_min_record_size,
            cb_max_record_size: self.cb_max_record_size,
            leaf_page_size: self.leaf_page_size,
            cb_max_key_len: self.cb_max_key_len,
            max_fence_len: self.max_fence_len,
            cb_copy_on_access_ratio: self.cb_copy_on_access_ratio,
            read_record_cache: self.read_record_cache,
            file_path: self.file_path.clone(),
            max_mini_page_size: self.max_mini_page_size,
            mini_page_binary_search: self.mini_page_binary_search,
            write_ahead_log: self.write_ahead_log.clone(),
            write_load_full_page: self.write_load_full_page,
            cache_only: self.cache_only,
        }
    }
}

/// Where/how to store the leaf pages?
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub enum StorageBackend {
    Memory,
    #[default]
    Std,
    #[cfg(target_os = "linux")]
    StdDirect,
    #[cfg(target_os = "linux")]
    IoUringPolling,
    #[cfg(target_os = "linux")]
    IoUringBlocking,
    #[cfg(all(target_os = "linux", feature = "spdk"))]
    Spdk,
}

#[derive(Debug, Deserialize)]
pub struct ConfigFile {
    pub(crate) cb_size_byte: usize,
    pub(crate) cb_min_record_size: usize,
    pub(crate) cb_max_record_size: usize,
    pub(crate) cb_max_key_len: usize,
    pub(crate) leaf_page_size: usize,
    pub(crate) index_file_path: String,
    pub(crate) backend_storage: String,
    pub(crate) read_promotion_rate: usize,
    pub(crate) write_load_full_page: bool,
    pub(crate) cache_only: bool,
}

/// Default BfTree configuration
///
impl Default for Config {
    fn default() -> Self {
        let read_promotion_rate = if cfg!(debug_assertions) {
            DEFAULT_PROMOTION_RATE_DEBUG
        } else {
            DEFAULT_PROMOTION_RATE_RLEASE
        };
        let scan_promotion_rate = if cfg!(debug_assertions) {
            DEFAULT_PROMOTION_RATE_DEBUG
        } else {
            DEFAULT_PROMOTION_RATE_RLEASE
        };

        Self {
            read_promotion_rate: AtomicUsize::new(read_promotion_rate),
            scan_promotion_rate: AtomicUsize::new(scan_promotion_rate),
            cb_size_byte: DEFAULT_CIRCULAR_BUFFER_SIZE,
            cb_min_record_size: DEFAULT_MIN_RECORD_SIZE,
            cb_max_record_size: DEFAULT_MAX_RECORD_SIZE,
            leaf_page_size: DEFAULT_LEAF_PAGE_SIZE,
            cb_max_key_len: DEFAULT_MAX_KEY_LEN,
            max_fence_len: DEFAULT_MAX_KEY_LEN * 2,
            cb_copy_on_access_ratio: DEFAULT_COPY_ON_ACCESS_RATIO,
            file_path: PathBuf::new(),
            read_record_cache: true,
            max_mini_page_size: DEFAULT_MAX_MINI_PAGE_SIZE,
            mini_page_binary_search: true,
            storage_backend: StorageBackend::Memory,
            write_ahead_log: None,
            write_load_full_page: true,
            cache_only: false,
        }
    }
}
impl Config {
    pub fn new(file_path: impl AsRef<Path>, circular_buffer_size: usize) -> Self {
        let mut config = Self::default();
        let mut cache_only = false;
        let storage_backend = if file_path.as_ref().to_str().unwrap().starts_with(":memory:") {
            StorageBackend::Memory
        } else if file_path.as_ref().to_str().unwrap().starts_with(":cache:") {
            cache_only = true;
            StorageBackend::Memory
        } else {
            StorageBackend::default()
        };

        config
            .storage_backend(storage_backend)
            .cache_only(cache_only)
            .cb_size_byte(circular_buffer_size)
            .file_path(file_path);

        config
    }

    /// Constructor of Config based on a config TOML file
    /// The config file must have all fields defined ConfigFile
    pub fn new_with_config_file<P: AsRef<Path>>(config_file_path: P) -> Self {
        let config_file_str =
            fs::read_to_string(config_file_path).expect("couldn't read config file");
        let config_file: ConfigFile =
            toml::from_str(&config_file_str).expect("Fail to parse config file");
        let scan_promotion_rate = if cfg!(debug_assertions) {
            DEFAULT_PROMOTION_RATE_DEBUG
        } else {
            DEFAULT_PROMOTION_RATE_RLEASE
        };
        let mut storage = StorageBackend::Memory;
        if config_file.backend_storage == "disk" {
            storage = StorageBackend::default();
        }

        // Return the config
        Self {
            read_promotion_rate: AtomicUsize::new(config_file.read_promotion_rate),
            scan_promotion_rate: AtomicUsize::new(scan_promotion_rate),
            cb_size_byte: config_file.cb_size_byte,
            cb_min_record_size: config_file.cb_min_record_size,
            cb_max_record_size: config_file.cb_max_record_size,
            leaf_page_size: config_file.leaf_page_size,
            cb_max_key_len: config_file.cb_max_key_len,
            max_fence_len: config_file.cb_max_key_len * 2,
            cb_copy_on_access_ratio: DEFAULT_COPY_ON_ACCESS_RATIO,
            file_path: PathBuf::from(config_file.index_file_path),
            read_record_cache: true,
            max_mini_page_size: DEFAULT_MAX_MINI_PAGE_SIZE,
            mini_page_binary_search: true,
            storage_backend: storage,
            write_ahead_log: None,
            write_load_full_page: config_file.write_load_full_page,
            cache_only: config_file.cache_only,
        }
    }

    /// Default: Std
    ///
    /// Use std::fs::file to store/access disk data.
    /// For better performance, consider platform specific backends like: IoUringBlocking.
    pub fn storage_backend(&mut self, backend: StorageBackend) -> &mut Self {
        self.storage_backend = backend;
        self
    }

    /// Default: 30
    ///
    /// prob% of chance that a **page** will be promoted to buffer during scan operations.
    pub fn scan_promotion_rate(&mut self, prob: usize) -> &mut Self {
        self.scan_promotion_rate.store(prob, Ordering::Relaxed);
        self
    }

    /// Default: true
    ///
    /// By default bf-tree will cache the hot records in mini page.
    /// Setting this to false will try to cache the entire base page, which is less efficient.
    pub fn read_record_cache(&mut self, read_full_page_cache: bool) -> &mut Self {
        self.read_record_cache = read_full_page_cache;
        self
    }

    /// Default: 2048
    ///
    /// The maximum mini page size before it grows to a full page.
    pub fn max_mini_page_size(&mut self, size: usize) -> &mut Self {
        self.max_mini_page_size = size;
        self
    }

    /// Default: true
    ///
    /// If set to false, the mini page will use linear search instead of binary search.
    pub fn mini_page_binary_search(&mut self, binary_search: bool) -> &mut Self {
        self.mini_page_binary_search = binary_search;
        self
    }

    /// Default: 30
    ///
    /// prob% of chance that a record will be promoted from leaf page to mini page.
    pub fn read_promotion_rate(&mut self, prob: usize) -> &mut Self {
        self.read_promotion_rate.store(prob, Ordering::Relaxed);
        self
    }

    /// Default: 0.1
    ///
    /// The ratio of copy-on-access region for circular buffer.
    /// - 0.0 means the circular buffer is a FIFO.
    /// - 1.0 means the circular buffer is a LRU.
    ///
    /// You don't want to change this unless you know what you are doing.
    pub fn cb_copy_on_access_ratio(&mut self, ratio: f64) -> &mut Self {
        self.cb_copy_on_access_ratio = ratio;
        self
    }

    /// Default: false
    ///
    /// Whether to enable write ahead log, for persistency and recovery.
    pub fn enable_write_ahead_log(&mut self, wal_config: Arc<WalConfig>) -> &mut Self {
        // let write_ahead_log_path = self.file_path.parent().unwrap().join("wal.log");
        self.write_ahead_log = Some(wal_config);
        self
    }

    /// Default: false
    ///
    /// Similar to `enable_write_ahead_log`, but with default WAL configuration.
    /// The path to write the write ahead log.
    /// Advanced users may want to change the WAL to point to a different location
    /// to leverage different storage patterns
    /// (WAL is always sequence write and requires durability).
    pub fn enable_write_ahead_log_default(&mut self) -> &mut Self {
        let wal_config = WalConfig::new(self.file_path.parent().unwrap().join("wal.log"));
        self.write_ahead_log = Some(Arc::new(wal_config));
        self
    }

    /// Default: false
    pub fn cache_only(&mut self, cache_only: bool) -> &mut Self {
        self.cache_only = cache_only;
        self
    }

    /// Default: 32 * 1024 * 1024
    pub fn cb_size_byte(&mut self, cb_size_byte: usize) -> &mut Self {
        self.cb_size_byte = cb_size_byte;
        self
    }

    pub fn file_path<P: AsRef<Path>>(&mut self, file_path: P) -> &mut Self {
        self.file_path = file_path.as_ref().to_path_buf();
        self
    }

    pub fn cb_max_key_len(&mut self, max_key_len: usize) -> &mut Self {
        self.cb_max_key_len = max_key_len;
        self.max_fence_len = max_key_len * 2;
        self
    }

    pub fn cb_min_record_size(&mut self, min_record_size: usize) -> &mut Self {
        self.cb_min_record_size = min_record_size;
        self
    }

    pub fn cb_max_record_size(&mut self, max_record_size: usize) -> &mut Self {
        self.cb_max_record_size = max_record_size;
        self
    }

    /// Returns the current max record size
    pub fn get_cb_max_record_size(&self) -> usize {
        self.cb_max_record_size
    }

    pub fn leaf_page_size(&mut self, leaf_page_size: usize) -> &mut Self {
        self.leaf_page_size = leaf_page_size;
        self
    }

    /// Returns the current leaf page size
    pub fn get_leaf_page_size(&self) -> usize {
        self.leaf_page_size
    }

    /// Validate the configuration and report any invalid parameter, if found.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Sanity check of the input parameters
        if self.cb_min_record_size <= 1 {
            return Err(ConfigError::MinimumRecordSize(
                "cb_min_record_size (key + value in bytes) needs to be at least 2 bytes"
                    .to_string(),
            ));
        }

        if self.cb_min_record_size > self.cb_max_record_size {
            return Err(ConfigError::MaximumRecordSize("cb_min_record_size (key + value in bytes) cannot be greater than cb_max_record_size".to_string()));
        }

        if self.max_fence_len == 0 {
            return Err(ConfigError::MaxKeyLen(
                "cb_max_key_len cannot be zero".to_string(),
            ));
        }

        if self.max_fence_len / 2 > self.cb_max_record_size {
            return Err(ConfigError::MaxKeyLen(
                "cb_max_key_len cannot be greater than cb_max_record_size".to_string(),
            ));
        }

        if self.leaf_page_size > MAX_LEAF_PAGE_SIZE {
            return Err(ConfigError::LeafPageSize(format!(
                "leaf_page_size cannot be larger than {}",
                MAX_LEAF_PAGE_SIZE
            )));
        }

        if self.max_fence_len / 2 > MAX_KEY_LEN {
            return Err(ConfigError::MaxKeyLen(format!(
                "cb_max_key_len cannot be larger than {}",
                MAX_KEY_LEN
            )));
        }

        if !self.cb_size_byte.is_power_of_two() {
            return Err(ConfigError::CircularBufferSize(
                "cb_size_byte should be a power of two".to_string(),
            ));
        }

        if self.leaf_page_size / self.cb_min_record_size > 4096 {
            return Err(ConfigError::MinimumRecordSize(
                "leaf_page_size/min_record_size cannot exceed 2^12.".to_string(),
            ));
        }

        // Page alignment checks
        if !self.cache_only && !self.leaf_page_size.is_multiple_of(DISK_PAGE_SIZE) {
            return Err(ConfigError::LeafPageSize(format!(
                "In non cache-only mode leaf_page_size should be multiple of {}",
                DISK_PAGE_SIZE
            )));
        } else if self.cache_only && !self.leaf_page_size.is_multiple_of(CACHE_LINE_SIZE) {
            return Err(ConfigError::LeafPageSize(format!(
                "In cache-only mode leaf_page_size should be multiple of {}",
                CACHE_LINE_SIZE
            )));
        }

        // Mini-page merge/split operation safety guarantee checks
        let max_record_size_with_meta = self.cb_max_record_size + std::mem::size_of::<LeafKVMeta>();
        let mut max_mini_page_size: usize;

        if self.cache_only {
            if self.leaf_page_size < 2 * max_record_size_with_meta + std::mem::size_of::<LeafNode>()
            {
                return Err(ConfigError::MaximumRecordSize(format!(
                    "In cache-only mode, given the leaf_page_size the corresponding cb_max_record_size should be <= {}",
                    (self.leaf_page_size - std::mem::size_of::<LeafNode>()) / 2
                        - std::mem::size_of::<LeafKVMeta>()
                )));
            }
        } else {
            if max_record_size_with_meta
                > self.leaf_page_size - self.max_fence_len - 2 * std::mem::size_of::<LeafKVMeta>()
            {
                return Err(ConfigError::MaximumRecordSize(format!(
                    "In non cache-only mode, given the leaf_page_size the corresponding cb_max_record_size should be <= {}",
                    self.leaf_page_size
                        - self.max_fence_len
                        - 2 * std::mem::size_of::<LeafKVMeta>()
                )));
            }
            max_mini_page_size = self.leaf_page_size
                - max_record_size_with_meta
                - self.max_fence_len
                - 2 * std::mem::size_of::<LeafKVMeta>();
            max_mini_page_size = (max_mini_page_size / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

            if max_mini_page_size < max_record_size_with_meta + std::mem::size_of::<LeafNode>() {
                return Err(ConfigError::MaximumRecordSize(format!(
                    "In non cache-only mode, given the leaf_page_size the corresponding cb_max_record_size should be <= {}",
                    max_mini_page_size
                        - std::mem::size_of::<LeafNode>()
                        - std::mem::size_of::<LeafKVMeta>()
                )));
            }
        }

        // Circular buffer size checks
        // In cache-only mode, the circular buffer size >= 4 * leaf_page_size
        // This is because during page split, two full sized mini-page need to be in memory
        // Note that 2 * leaf_page_size only guarantees one full sized mini-page due to AllocMeta
        //
        // In non cache-only mode, the circular buffer size >= 2 * leaf_page_size
        // As at most one full sized leaf page is required
        if self.cache_only {
            if self.cb_size_byte < 4 * self.leaf_page_size {
                return Err(ConfigError::CircularBufferSize(
                    "In cache-only mode, cb_size_byte should be at least 4 times of leaf_page_size"
                        .to_string(),
                ));
            }
        } else if self.cb_size_byte < 2 * self.leaf_page_size {
            return Err(ConfigError::CircularBufferSize(
                "In non cache-only mode, cb_size_byte should be at least 2 times of leaf_page_size"
                    .to_string(),
            ));
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]

pub struct WalConfig {
    pub(crate) file_path: PathBuf,
    pub(crate) flush_interval: Duration,
    pub(crate) segment_size: usize,
    pub(crate) storage_backend: StorageBackend,
}

impl WalConfig {
    /// Default: same directory as the file_path
    ///
    /// The path to write the write ahead log.
    /// Advanced users may want to change the WAL to point to a different location
    /// to leverage different storage patterns
    /// (WAL is always sequence write and requires durability).
    pub fn new(file_path: impl AsRef<Path>) -> Self {
        Self {
            file_path: file_path.as_ref().to_path_buf(),
            flush_interval: Duration::from_millis(1),
            segment_size: 1024 * 1024 * 1024,
            storage_backend: StorageBackend::Std,
        }
    }

    /// Default: 1ms
    pub fn flush_interval(&mut self, interval: Duration) -> &mut Self {
        self.flush_interval = interval;
        self
    }

    /// Default: 1MB
    pub fn segment_size(&mut self, size: usize) -> &mut Self {
        self.segment_size = size;
        self
    }

    /// Default: Std
    ///
    /// Change the storage backend for potentially better performance, e.g., IoUring on Linux.
    pub fn storage_backend(&mut self, backend: StorageBackend) -> &mut Self {
        self.storage_backend = backend;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CONFIG_FILE: &str = "src/sample_config.toml";
    #[test]
    fn test_new_with_config_file() {
        let config = Config::new_with_config_file(SAMPLE_CONFIG_FILE);

        assert_eq!(config.cb_size_byte, 8192);
        assert_eq!(config.read_promotion_rate.load(Ordering::Relaxed), 100);
        assert_eq!(config.write_load_full_page, true);
        assert_eq!(config.file_path, PathBuf::from("c/d/E"));
        assert_eq!(config.cache_only, false);
    }

    #[test]
    fn test_leaf_page_size_getter_setter() {
        let mut config = Config::default();

        // Check default value
        assert_eq!(config.get_leaf_page_size(), DEFAULT_LEAF_PAGE_SIZE);

        // Set a new value and verify it
        let new_leaf_page_size = 8192;
        config.leaf_page_size(new_leaf_page_size);
        assert_eq!(config.get_leaf_page_size(), new_leaf_page_size);

        // Set another value and verify
        let another_leaf_page_size = 16384;
        config.leaf_page_size(another_leaf_page_size);
        assert_eq!(config.get_leaf_page_size(), another_leaf_page_size);
    }

    #[test]
    fn test_cb_max_record_size_getter_setter() {
        let mut config = Config::default();

        // Check default value
        assert_eq!(config.get_cb_max_record_size(), DEFAULT_MAX_RECORD_SIZE);

        // Set a new value and verify it
        let new_max_record_size = 4096;
        config.cb_max_record_size(new_max_record_size);
        assert_eq!(config.get_cb_max_record_size(), new_max_record_size);

        // Set another value and verify
        let another_max_record_size = 8192;
        config.cb_max_record_size(another_max_record_size);
        assert_eq!(config.get_cb_max_record_size(), another_max_record_size);
    }
}
