# Checked Bf-Tree C interface

This is an **unreleased package candidate**.
It wraps the existing Rust read, scan, insert, delete, and CPR operations.
The native operation implementations remain unchanged.
Recovery validation requires two read-only `Config` getters in the adjacent BF-tree source.
The native crate version currently reads `0.5.7`.
That version number alone does not identify this candidate's additional configuration getters.
Do not publish this package or replace an existing integration until compatible native and FFI versions receive approval.

## Build

Use Rust 1.93.0 and the platform's native linker.
The package builds a static library and an `rlib`.
`Cargo.lock` fixes dependency resolution.
Keep it with the package.

```powershell
cargo +1.93.0 build --manifest-path bf-tree-ffi\Cargo.toml --locked --release
cargo +1.93.0 test --manifest-path bf-tree-ffi\Cargo.toml --locked
```

`include\bf_tree.h` is the public header.
ABI-v1 function signatures, structures, and status values remain unchanged.
Capability queries and bounded scans use additional versioned entry points.
The package contains no application indexing, tenant, authorization, or WAL-admission policy.

The CMake project builds C and C++ consumers against the same static library:

```powershell
cmake -S bf-tree-ffi -B bf-tree-ffi\build -A x64
cmake --build bf-tree-ffi\build --config Release
ctest --test-dir bf-tree-ffi\build -C Release --output-on-failure
```

Pass `-DCARGO_EXECUTABLE=<path>` if Cargo is not on `PATH`.
For Linux, omit `-A x64` and set `-DCMAKE_BUILD_TYPE=Release`.
For Windows ARM64, use `-A ARM64` and install the matching Rust target first.
The standard native backend is the baseline.
The package does not require `io_uring`.
Other supported-target claims require separate build and runtime evidence.
The CI job covers Windows x64 and Linux x64.
Windows ARM64 runtime qualification remains required.

The local candidate passed Windows x64 checks with Rust 1.93.0 and MSVC 19.51.
Those checks include a C++ consumer compiled against the unchanged ABI-v1 header.
They do not establish Linux, Windows ARM64, performance, or physical power-loss qualification.

## Layout and ownership

All layouts use ordinary C alignment.
Pointers and `size_t` follow the target platform.
These structures are not disk or network formats.

The caller owns all input and output memory.
Input buffers remain borrowed for one call.
Read and scan results are copied into caller buffers.
Rust references, iterators, and allocator-owned buffers do not cross the interface.

The wrapper sizes scratch buffers from the effective tree configuration, including CPR-restored limits.
It supplies that scratch space to native reads and scans instead of passing caller output buffers.
Caller-capacity errors therefore do not depend on a native panic.

`BfTreeScanRecord` holds offsets and lengths within the caller's byte buffer.
`BUFFER_TOO_SMALL` reports required capacities without partially changing byte or descriptor output.
A sizing retry is a fresh operation and can observe concurrent mutations.

`BfTreeError` provides an optional caller-owned UTF-8 buffer.
`message_required` includes the null terminator for a nonempty diagnostic.
Truncated text remains valid UTF-8 and null-terminated.

Rust allocates opaque tree handles.
Close waits for active calls and leaves an allocated, closed handle.
Repeated close returns `CLOSED`.
Destroy requires caller-side exclusion, releases the allocation, and clears the supplied pointer.
Destroying an already null pointer returns `OK`.
Other copies of a destroyed pointer are invalid.

An operation panic returns `PANIC` and fences the handle.
Later data operations return `CLOSED`.
Other calls already in progress cannot report successful results after observing the fence.
Close and destroy remain available for cleanup.
The build requires `panic=unwind`.
Process aborts, invalid caller pointers, and native undefined behavior are not recoverable Rust panics.

## Capabilities and scans

Initialize `abi_version` and `struct_size` before querying `BfTreeCapabilitiesV1`.
The result describes package capabilities, not the current handle's permissions.
Read-only recovery still rejects insert, delete, and snapshot capture.

The package does not advertise existing-tree open, atomic multi-record batches,
native I/O counters, or power-loss durability.
The reserved `bf_tree_open` function returns `UNSUPPORTED` without creating or changing files.

`bf_tree_scan` retains its lower-bound, count-limited behavior.
Its hard limits are 65,536 records and 64 MiB of staged bytes.

`bf_tree_scan_bounded_v1` adds:

- An inclusive lower bound, or an exclusive lower bound for continuation.
- An optional exclusive upper bound.
- A record limit and a staged-byte limit.

Equal bounds produce an empty range.
Reversed bounds, invalid flags, and invalid limits fail explicitly.
When the byte budget stops a scan, the result contains complete records only.
An oversized first record returns `BUFFER_TOO_SMALL` with that record's required capacities.
Continue from the last returned key with `lower_exclusive = 1`.
An empty successful result ends the selected range.

Budgets limit staged output, not native disk I/O or traversal time.
Scratch space also includes one configured native record buffer.
Stable pagination requires a stable input view supplied by the owner.
The C ABI does not create an application snapshot across separate calls.

## Persistence and recovery

The package disables native WAL when creating or recovering a handle.
It does not supply a replacement application WAL.
Neither close nor snapshot completion establishes durability of acknowledged application mutations.

Snapshot capture refuses an existing output path and serializes requests per handle.
The snapshot mutex prevents concurrent requests from reaching the native capture operation.
The completed generation advances only after capture succeeds.
Recovery requires a nonzero completed generation.
Writable recovery enables subsequent capture.

Recovery uses the snapshot's stored format.
Nonzero requested record, key, and leaf limits must match that format.
A zero limit accepts the corresponding stored limit.
The circular-buffer size remains an optional recovery override.

The caller must supply an exclusively owned, validated native CPR artifact.
Native recovery can use that file as working storage, even for a read-only application view.
Use a private copy to preserve the published snapshot.
This package checks the request and reports native failures.
It does not replace native-format integrity validation by the artifact owner.
Arbitrary corrupt native snapshots and physical power-loss behavior remain separate qualification requirements.

## Distribution and migration

The repository's MIT license applies to this package.
Keep the Microsoft attribution in source distributions.
No package upload or upstream acceptance is implied by this candidate.

Before distribution:

1. Approve compatible native and FFI releases or another immutable source mechanism.
2. Update the exact dependency versions and retain the lockfile.
3. Verify package contents and exported C symbols.
4. Run Rust, C, and C++ tests on supported Windows ARM64 and Linux targets.
5. Measure overhead and preserve application snapshot compatibility.
6. Rehearse rollback using data produced after migration.

Do not delete the previous integration before those requirements pass.
