# Bf-Tree

Bf-Tree is a modern read-write-optimized concurrent larger-than-memory range index in Rust from MSR.

## Design Details

You can find the Bf-Tree research paper [here](https://badrish.net/papers/bftree-vldb2024.pdf). You can find more design docs [here](/doc).
## User Guide

### Rust

Bf-Tree is written in Rust, and is available as a Rust crate. You can add Bf-Tree to your `Cargo.toml` like this:
```bash
$ cargo add bf_tree
```
Which will add bf_tree as a dependency to your Cargo.toml
```toml
[dependencies]
bf-tree = "0.5.5"
```

An example use of Bf-Tree:

```rust
use bf_tree::BfTree;
use bf_tree::LeafReadResult;

let mut config = bf_tree::Config::default();
config.cb_min_record_size(4);
let tree = BfTree::with_config(config, None).unwrap();
tree.insert(b"key", b"value");

let mut buffer = [0u8; 1024];
let read_size = tree.read(b"key", &mut buffer);

assert_eq!(read_size, LeafReadResult::Found(5));
assert_eq!(&buffer[..5], b"value");
```

### Snapshots and Recovery

Bf-Tree supports CPR-style consistent snapshots. A snapshot captures a
consistent prefix of all in-flight transactions to a file, and a tree can later
be reconstructed from such a snapshot file.

To enable snapshots, set `use_snapshot(true)` on the `Config` before creating
the tree. Once enabled, call [`BfTree::cpr_snapshot`] with the destination path
to take a snapshot at any point. Snapshots can be taken concurrently with
ongoing reads and writes; only one snapshot may be in progress at a time.

To recover a tree from a snapshot file, use
[`BfTree::new_from_cpr_snapshot`]. The snapshot file embeds most configuration
fields (record sizes, leaf page size, cache-only flag, etc.), so the caller
only needs to specify the recovery-time options:

- `recovery_snapshot_file_path`: path of the snapshot file to recover from.
- `use_snapshot`: whether the recovered tree should itself support taking new
  snapshots.
- `buffer_ptr`: optional pointer to a pre-allocated cache buffer; pass `None`
  to let Bf-Tree allocate one.
- `buffer_size`: optional override of the cache size stored in the snapshot.
  If smaller than the original, recovery may fail because cached pages from
  the snapshot must fit in memory.
- `wal`: optional write-ahead log configuration for the recovered tree.

```rust
use bf_tree::{BfTree, Config};
use std::path::PathBuf;

let mut config = Config::default();
config.use_snapshot(true);
let tree = BfTree::with_config(config, None).unwrap();

tree.insert(b"key", b"value");

// Take a CPR snapshot of the current tree state.
tree.cpr_snapshot("snapshot.bftree");

// Recover a bf-tree from a CPR snapshot
let tree = BfTree::new_from_cpr_snapshot(
    "snapshot.bftree",
    /* use_snapshot */ true,
    /* buffer_ptr */ None,
    /* buffer_size */ None,
    /* wal */ None,
).unwrap();
std::fs::remove_file("snapshot.bftree");
```

You can check whether all active threads have transitioned to the next
snapshot version during a snapshot (i.e., all threads are operating in v + 1) with:

```rust,ignore
let ready = tree.are_all_threads_in_next_snapshot_version();
```

This is useful for coordinating external systems that need to wait until a
snapshot's data is fully consistent but not the whole snapshot to finish
before proceeding. Note that, this function returns false if no ongoing snapshot.

Notes:
- The snapshot file path passed to `cpr_snapshot` must be different from the
  path used by any concurrent recovery.
- For more on the snapshot/recovery design, see
  [doc/snapshot-recovery.md](doc/snapshot-recovery.md).

PRs are accepted and preferred over feature requests. Feel free to reach out if you have any design questions.

## Developer Guide

### Building

#### Prerequisite

Bf-Tree supports Linux, Windows, and macOS, although only a recently version of Linux is rigorously tested. Bf-Tree is written in Rust, which you can install [here](https://rustup.rs).

Please install pre-commit hooks to ensure that your code is formatted and linted in the same way as the rest of the project; the coding style will be enforced in CI, these hooks act as a pre-filter.

```bash
# If on Ubuntu
sudo apt update && sudo apt install pre-commit
pre-commit install
```

#### Build

```bash
cargo build --release
```

### Testing

#### Unit Tests

```bash
cargo test
```

#### Shuttle Tests

Concurrent systems are nondeterministic, and subject to exponential amount of different thread interleaving. We use [shuttle](https://github.com/awslabs/shuttle)
to deterministically and systematically explore different thread interleaving to uncover the bugs caused by subtle multithread interactions.

```bash
# Core Bf-tree concurrent operations (~5 minutes)
cargo test --features "shuttle" --release shuttle_bf_tree_concurrent_operations

# CPR snapshot with disk-backed storage (fast, < 1s)
cargo test --features "shuttle" --release shuttle_cpr_snapshot_disk

# CPR snapshot with cache-only (in-memory) storage
cargo test --features "shuttle" --release shuttle_cpr_snapshot_cache_only
```

To replay a specific failing schedule (generated automatically on failure into `target/schedule000.txt`):

```bash
cargo test --features "shuttle" --release shuttle_replay -- --nocapture
```

#### Fuzz Tests

Fuzz testing is a bug finding technique that generates random inputs to the system and test for crash. Bf-Tree employs fuzzing to generate random operation sequences
(e.g., insert, read, scan) to the system and check that none of the operation sequence will crash the system or lead to inconsistent state. Check the 
[fuzz](fuzz/README.md) folder for more details.


### Benchmarking

Check the [benchmark](benchmark/README.md) folder for more details.

```bash
cd benchmark
env SHUMAI_FILTER="inmemory" MIMALLOC_LARGE_OS_PAGES=1 cargo run --bin bftree --release
```

More advanced benchmarking, with metrics collecting, numa-node binding, huge page, etc:
```bash
env MIMALLOC_SHOW_STATS=1 MIMALLOC_LARGE_OS_PAGES=1 MIMALLOC_RESERVE_HUGE_OS_PAGES_AT=0 numactl --membind=0 --cpunodebind=0 cargo bench --features "metrics-rt" micro
```

### Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

### Security

See [SECURITY.md](SECURITY.md) for security reporting details.


### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks
or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in
modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party 
trademarks or logos are subject to those third-party’s policies.

### Contact

- bftree@microsoft.com
