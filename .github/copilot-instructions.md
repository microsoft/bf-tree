# Copilot Instructions for bf-tree

## Build, Test, and Lint

```bash
# Build (release mode recommended — debug builds are large)
cargo build --release

# Run all unit tests
cargo test

# Run a single test by name
cargo test test_tree_insert_read_1

# Run shuttle concurrency tests (deterministic thread interleaving exploration, ~5 min)
cargo test --features "shuttle" --release shuttle_bf_tree_concurrent_operations

# Run benchmarks
cd benchmark
env SHUMAI_FILTER="inmemory" MIMALLOC_LARGE_OS_PAGES=1 cargo run --bin bftree --release

# Lint (pre-commit hooks enforce formatting)
cargo fmt --check
cargo clippy
```

Rust toolchain is pinned to **1.95.0** via `rust-toolchain.toml`.

## Architecture

Bf-Tree is a concurrent, larger-than-memory B+-tree range index. The core data flow:

1. **`BfTree`** (`src/tree.rs`) — The top-level API struct. Owns root page, config, storage, WAL, and snapshot manager.
2. **Circular Buffer** (`src/circular_buffer/`) — A custom lock-free memory allocator for in-memory "mini pages." Pages transition through lifecycle states: NotReady → Ready → BeginTombstone → Tombstone → Evicted. The freelist reclaims tombstoned memory.
3. **Nodes** (`src/nodes/`) — Inner nodes (4KB fixed) and leaf nodes (variable, up to 32KB). Leaf nodes hold KV data in "mini pages" that can be promoted to full pages.
4. **Storage** (`src/storage.rs`) — `PageTable` maps `PageID` → `PageLocation` (Mini, Full, Base/disk, Null). `LeafStorage` coordinates between the circular buffer and the backing VFS.
5. **VFS layer** (`src/fs/`) — Pluggable filesystem backends: `MemoryVfs`, `StdVfs`, `IoUringVfs` (Linux), `SpdkVfs` (Linux, optional).
6. **WAL** (`src/wal/`) — Optional write-ahead log for durability.
7. **Snapshot/Recovery** (`src/snapshot.rs`) — CPR-style consistent snapshots taken concurrently with reads/writes.

### Concurrency model

- The `sync` module (`src/sync.rs`) conditionally re-exports either `std::sync` or `shuttle::sync` based on the `shuttle` feature flag. This allows deterministic concurrency testing without code changes.
- Custom synchronization primitives live in `src/utils/` (`rw_lock.rs`, `inner_lock.rs`, `atomic_wait.rs`).
- Optimistic locking with version checks is the primary concurrency pattern — see the `check_parent!` macro.

## Key Conventions

- **Shuttle-aware concurrency**: All synchronization imports (`Arc`, `Mutex`, `thread`, atomics) come from `crate::sync`, not directly from `std`. This enables shuttle-based deterministic testing. New concurrent code must follow this pattern.
- **Feature-gated metrics**: Metrics are compiled out by default. Use the `counter!`, `histogram!`, and `timer!` macros which expand to no-ops without the appropriate feature flag.
- **Config uses builder-style setters**: `Config` fields are `pub(crate)` with public setter methods (e.g., `config.cb_min_record_size(4)`).
- **`unsafe impl Send + Sync`** on `BfTree` — the tree manages raw pointers internally; concurrency safety is maintained by the custom locking protocol.
- **Copyright header**: Every source file starts with `// Copyright (c) Microsoft Corporation.\n// Licensed under the MIT license.`
- **Test organization**: Tests live in `src/tests/` as a `#[cfg(test)]` module. Proptest is used for property-based testing; `rstest` for parameterized tests.
- **Platform-specific code**: Linux-specific I/O (io_uring, SPDK) is behind `#[cfg(target_os = "linux")]`. Windows uses `windows-sys` for threading primitives.
