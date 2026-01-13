// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

pub mod bench_bftree;
pub mod bench_e2e;
mod common;
mod wrappers;

/// Uniformed interface for a key-value store that can be benchmarked by us.
pub trait KvStore: Sync + Send {
    fn new(file_path: impl AsRef<std::path::Path>, memory_size: usize) -> Self
    where
        Self: Sized;

    /// This is for splinterdb only, very unfortunate.
    fn register_thread(&self) {}

    fn deregister_thread(&self) {}

    fn scan(&self, key: &[u8], cnt: usize, out_buffer: &mut [u8]);

    fn insert(&self, key: &[u8], value: &[u8]);

    /// Read operation is always positive read, the benchmark system will ensure reading a existing key.
    /// Returning 0 means the key is not found and will fail the benchmark.
    fn read(&self, key: &[u8], value: &mut [u8]) -> usize;

    fn update(&self, key: &[u8], value: &[u8]);
}
