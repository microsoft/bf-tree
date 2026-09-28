# Performance and safety validation

The development toolchain is Rust 1.98.1. It fixes the vtable miscompilation
reported in [Rust 1.98.0](https://blog.rust-lang.org/2026/09/03/Rust-1.98.1/).
The library keeps its existing edition, public tree API and page layout.

## Changes

- Leaf lookup retains the equality result from binary search for reads and
  updates. New metadata initializes its reference bit directly, eliminating an
  atomic read-modify-write before publication.
- Leaf consolidation stages records in two owned contiguous buffers instead of
  allocating separate keys and values per record. Source data stays alive until
  copying completes, including when fence prefixes change.
- Inner insertion reuses its first search. Inner consolidation packs payloads
  through a 4 KiB stack buffer without allocating keys or searching/reinserting
  records. Sorted metadata and child identities are preserved. The inner-node
  Rust type now covers all 4096 allocated bytes; compile-time assertions preserve
  the existing header and payload offsets.
- Mapping-table storage uses fixed slots, separate interior-mutability cells,
  and an initialized prefix published with Release/Acquire. Reads do not lock or
  create mutable references to shared allocations. Invalid IDs are checked;
  capacity failures and invalid recovery sequences cannot expose uninitialized
  records. Allocation and record destruction have explicit owners.
- Fresh persisted pages initialize unused bytes. Header initialization precedes
  reference creation; an empty leaf prefix does not follow a sentinel offset.
  Inner capacity checks occur before narrowing sizes to `u16`, and updating an
  existing child works even when a node is full.
- The circular buffer's approximate tail is a separate atomic value, avoiding
  an unlocked read of ordinary allocator state. Inner write-guard references
  are tied to the guard borrow; recovery copies one child ID before updating
  its slot, so it needs no overlapping immutable/mutable borrows.

These changes prioritize fewer allocations and repeated operations. The latest
[Rust 1.98 APIs](https://blog.rust-lang.org/2026/08/20/Rust-1.98.0/), including
algebraic floating-point operations and buffered integer formatting, do not
target this byte-key index's critical paths. There is no reason to change its
numeric semantics to use them. Stable
[`Box::new_uninit_slice`](https://doc.rust-lang.org/std/boxed/struct.Box.html#method.new_uninit_slice)
supports deferred initialization of fixed mapping-table storage.

## Reproduce measurements

The root package has a portable benchmark with no additional dependency:

```sh
cargo bench --bench in_memory
cargo bench --profile perf --bench in_memory
```

`perf` opts into ThinLTO and one codegen unit. The normal release profile stays
unchanged, and the optional profile preserves panic unwinding for RAII cleanup.
Consumers need to configure their own root manifest because
[Cargo ignores profiles in dependencies](https://doc.rust-lang.org/cargo/reference/profiles.html).

The benchmark uses 16,384 keys, a seeded shuffled order, 8-byte keys and
32-/128-byte keys with long common prefixes, and a 64 MiB cache-only tree.
It measures insertion, hit/miss reads, growing-value updates, and scans.
Inputs and tree setup are prepared outside the measured region; insertion and
update samples use fresh trees. Read and scan samples reuse a warmed tree.
`std::hint::black_box` prevents unused results from being optimized away.
Output is CSV with median, minimum and maximum nanoseconds per operation
(per returned record for scans). One warmup is discarded before nine samples.
Results are checked outside timed regions; this harness supplements tests.

Set `BFTREE_BENCH_SAMPLES` to change sample count and `BFTREE_BENCH_FILTER` to
select a workload substring, such as `update_grow` or `k128`.

Use the same compiler, lockfile, profile, machine and power settings for both
revisions. Build first, then alternate baseline and candidate executables with
other CPU-intensive work stopped. Small differences inside the timing spread
are inconclusive. This synthetic single-thread benchmark does not establish
disk throughput, concurrent scaling or production tail latency; use the Linux
`benchmark/` workload suite and production traces for those decisions.

### Local results, 2026-09-08

Baseline: `e6843109e2ad40a6494caa1f7eb7d6f1fabaf82c`, built with the same
Rust 1.98.1 MSVC toolchain and Cargo.lock as the candidate. Machine: Intel
Core i7-11700K, Windows x86-64. Both use the unchanged release/bench profile,
without target-native flags. Three process runs per revision alternated order;
each run discarded a warmup and measured 11 samples. All other task tests and
compilation had finished before measurement. Reported values are the median of
the three process medians, in ns/op; positive percentages mean lower latency.

| Workload | Baseline | Candidate | Latency reduction |
| --- | ---: | ---: | ---: |
| Insert, 8-byte keys | 606.41 | 444.34 | 26.73% |
| Insert, 32-byte keys | 649.29 | 495.61 | 23.67% |
| Insert, 128-byte keys | 891.57 | 814.99 | 8.59% |
| Growing update, 8-byte keys | 967.39 | 615.00 | 36.43% |
| Growing update, 32-byte keys | 1037.90 | 709.69 | 31.62% |
| Growing update, 128-byte keys | 1177.64 | 875.54 | 25.65% |
| Read hit, 8-byte keys | 210.56 | 206.66 | 1.86% |
| Read hit, 32-byte keys | 265.01 | 241.75 | 8.78% |
| Read hit, 128-byte keys | 374.24 | 363.70 | 2.81% |
| Read miss, 8-byte keys | 212.75 | 213.29 | -0.25% |
| Read miss, 32-byte keys | 284.97 | 267.62 | 6.09% |
| Read miss, 128-byte keys | 367.42 | 377.44 | -2.73% |
| Scan, 8-byte keys | 16.22 | 15.91 | 1.89% |
| Scan, 32-byte keys | 29.29 | 22.80 | 22.18% |
| Scan, 128-byte keys | 41.82 | 38.95 | 6.86% |

[CSV measurements](performance-results-2026-09-08.csv) include the individual
process medians. The `perf_profile_ns` column is one exploratory 11-sample run
with the optional ThinLTO profile, not part of the three-run comparison. Small
read/scan differences need more controlled hardware measurements; the miss
results do not support claiming universal speedups. These timings do not remove
the existing safety limitations described below.

## Further deployment optimization

[Profile-guided optimization](https://doc.rust-lang.org/rustc/profile-guided-optimization.html)
can improve inlining and code layout using representative traffic. Instrument
the consuming application with `-Cprofile-generate`, run the real workload,
merge the data with the matching `llvm-profdata`, and rebuild with
`-Cprofile-use` and otherwise identical settings. Validate on held-out traffic.
Training only on this microbenchmark risks overfitting and is not enabled by
default. CPU-specific SIMD/prefetching and target-native builds also require
hardware-specific measurements before adopting them in a portable library.

## Correctness and memory checks

```sh
cargo test
cargo test --release
cargo test --features shuttle --release shuttle_bf_tree_concurrent_operations
cargo test --features shuttle --release shuttle_cpr_snapshot
cargo +nightly miri test --lib utils::mapping_table::tests
```

Tests cover binary/prefix key ordering against ordered-map models, consolidation
and splits, full-node updates, oversized keys, short restore buffers, initialized
page bytes, and mapping-table lifetime/publication/panic paths. The integration
test mixes growing/shrinking values, deletion, and ordered scans across pages in
both cache-only and memory-backed storage modes.

The CI AddressSanitizer job exercises the test suite, and targeted Miri checks
inspect the allocation and aliasing paths. Miri and sanitizers provide evidence
for the executed paths; they are not a proof of soundness for the entire existing
concurrent storage engine. See [Miri's scope](https://github.com/rust-lang/miri).

Local validation completed on 2026-09-08:

| Check | Result |
| --- | --- |
| Final `cargo test` on Windows | 107 unit tests, 1 integration test, 10 doctests passed; 2 existing doctests ignored |
| Release validation | Full suite passed during implementation; final inner-node tests (8) and mixed-workload integration test passed |
| Shuttle core operations and both CPR snapshot modes | All 3 passed |
| Shuttle mapping-table tests | 11 passed, including 100 randomized publication schedules |
| Linux AddressSanitizer, final code | 8 inner-node tests, 6 leaf-node tests and mixed-workload integration test passed |
| Default Miri, nightly 2026-09-07 | 11 mapping-table tests and 2 deterministic inner-node tests passed |
| Clippy with warnings denied | Library in all 3 runtime metrics configurations; portable benchmark and new integration test passed |
| Formatting and whitespace | `cargo fmt --all -- --check` and `git diff --check` passed |

The final test suite adds 25 unit tests and one integration test over the
82-unit-test baseline. The new model-based node tests each run 256 generated
cases; the existing 1000-case node property tests remain in place.

### Leaf allocation bounds: follow-up on 2026-09-27

The original zero-tail representation was reproduced at `f8206af` with default
Stacked Borrows checks on nightly 2026-09-07:

```sh
cargo +nightly miri test --lib leaf_empty_prefix_and_empty_consolidation_are_valid
```

`LeafNode::write_initial_kv_meta` wrote past the borrow derived from
`data: [u8; 0]`, during initial fence setup. The follow-up replaces it with an
allocation-bounded DST backed by `[UnsafeCell<MaybeUninit<u8>>]`. This preserves
the 32-byte header and thin page-table pointers while allowing atomic reference
bits to change through shared leaf references. Fresh allocations use their
owner's length; published pages retain a validated page size. The historical
reproducer and deterministic leaf tests now pass with default alias checks.
See [implementation, validation and performance evidence](leaf-safety-2026-09-27.md).

### Remaining safety scope

These targeted checks are **not a clean memory-safety certification** of the
whole tree.

The existing optimistic inner read protocol also allows ordinary metadata and
payload reads to overlap non-atomic writes: `ReadGuard::try_read` samples a
version without excluding writers, `as_ref` exposes ordinary references, and
`check_version` validates only afterwards. Such validation cannot remove a data
race that already occurred. A complete fix needs a consistent atomic snapshot
representation or shared reader exclusion, followed by concurrency and
performance revalidation. That concurrency protocol rewrite is not claimed by
the leaf allocation-bounds fix.
