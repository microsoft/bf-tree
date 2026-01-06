# Bf-Tree

Bf-Tree is a modern read-write-optimized concurrent larger-than-memory range index in Rust from MSR.

## Design Details

You can find the Bf-Tree research paper [here](https://badrish.net/papers/bftree-vldb2024.pdf). You can find more design docs [here](/doc).
## User Guide

### Rust

Bf-Tree is written in Rust, and is available as a Rust crate. You can add Bf-Tree to your `Cargo.toml` like this:
```toml
[dependencies]
bf-tree = "0.1.0"
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

PRs are accepted and preferred over feature requests. Feel free to reach out if you have any design questions.


## Developer Guide

### Build & Test

#### Prerequisite

Bf-Tree supports Linux, Windows, and macOS, although only a recently version of Linux is rigorously tested.

Bf-Tree is written in Rust, which you can install [here](https://rustup.rs), note that Bf-Tree requires a nightly version of Rust.
```bash
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly
```

Please kindly install pre-commit hooks to ensure that your code is formatted and linted in the same way as the rest of the project;
the coding style will be enforced in CI, these hooks act as a pre-filtering to save us some CI cycles.

```bash
# If on Ubuntu
sudo apt update && sudo apt install pre-commit
pre-commit install
```

#### Build

```bash
cargo build --release
```

#### Test

```bash
cargo test
```

#### Concurrency test using [shuttle](https://github.com/awslabs/shuttle)
```bash
cargo test --features "shuttle" --release shuttle_bf_tree_concurrent_operations
```
(Takes about 5 minute to run)

#### In-memory benchmark

Check the [benchmark](benchmark/README.md) folder for more details.

```bash
cd benchmark
env SHUMAI_FILTER="inmemory" MIMALLOC_LARGE_OS_PAGES=1 cargo run --bin bftree --release
```

More advanced benchmarking, with metrics collecting, numa-node binding, huge page, etc:
```bash
env MIMALLOC_SHOW_STATS=1 MIMALLOC_LARGE_OS_PAGES=1 MIMALLOC_RESERVE_HUGE_OS_PAGES_AT=0 numactl --membind=0 --cpunodebind=0 cargo bench --features "metrics-rt" micro
```

#### Fuzz testing

Check the [fuzz](fuzz/README.md) folder for more details.

**Fuzzing** is a bug finding technique that generates random inputs to the system and test for crash.
Bf-Tree employs fuzzing to generate random operation sequences (e.g., insert, read, scan) to the system and check that none of the operation sequence will crash the system or lead to inconsistent state.

**Property-based testing** checks that a system behaves according to its specification.
In particular, Bf-Tree should semantically behave like any range index, e.g., a previously inserted record should be found in the future, a deleted record should not be found in the future, etc.
We use property-based testing to check that Bf-Tree always behaves like the B-tree in Rust's standard library, which is a battle-tested in-memory B-tree implementation.
Like fuzzing, property-based testing generates random inputs to Bf-Tree and std B-tree and checks that they always return consistent results.

**Shuttle testing**
Concurrent systems are nondeterministic, and subject to exponential amount of different thread interleaving.
We use shuttle to deterministically and systematically explore different thread interleaving to uncover the bugs caused by subtle multithread interactions.

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

- Badrish Chandramouli (badrishc@microsoft.com)
- bftree@microsoft.com
