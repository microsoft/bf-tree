# Checked API contract

This contract defines the Rust foundation for a generic C ABI package.
The existing Rust API and snapshot format remain compatible.
The [C ABI candidate](../bf-tree-ffi/README.md) implements the foreign-language boundary.
It remains unpublished and requires separate release qualification.

## Initial scope

The first package will not open ordinary existing-tree files.
Its reserved `open` operation will return `UNSUPPORTED`.
Recovery will use an explicit CPR snapshot.
Opening a cleanly closed tree and recovering crash-left working files remain outside this scope.

The package must not introduce application-specific records, transactions, or WAL policy.
A tree snapshot does not prove a consistent checkpoint across application components.
The snapshot generation does not identify a logical application transaction.

## Checked Rust operations

| Operation | Result | Compatibility |
|---|---|---|
| `BfTree::read_checked` | Returns the existing `LeafReadResult`, or `BufferTooSmall` with the required value capacity. | `read` retains its existing return type and short-buffer panic. |
| `ScanIter::next_checked` | Returns one record, end of scan, or `BufferTooSmall`. A short-buffer result does not consume the record. | `next` retains its existing return type and short-buffer panic. |
| `BfTree::delete_checked` | Returns `LeafDeleteResult::Success` or `InvalidKey`. | `delete` retains its existing signature and ignores invalid keys. |
| `BfTree::cpr_snapshot_checked` | Returns completion, `SnapshotError::Disabled`, or `InProgress`. | `cpr_snapshot` retains its existing disabled-snapshot panic and concurrent-request behavior. |
| `BfTree::record_limits` | Returns effective record, key, and leaf limits, including snapshot-restored values. | Adds inspection without changing stored formats. |

The native record format requires a nonempty value for an insert.
Upstream `BfTree::insert` rejects an empty value through `InvalidKV`, before the leaf assertion.
Valid inserts and the public result type remain unchanged.

Checked operations do not convert internal failures into successful outcomes.
Native I/O failures and internal defects can still panic.
A foreign-language boundary must contain unwinding and fence a handle when its safety remains uncertain.
These operations do not establish physical power-loss durability.

### Reads

`BufferTooSmall.required` is the value length observed during that call.
Another writer can change that length before a retry.
The caller must use output bytes only after `Found`.
A failed read can leave output bytes from an earlier internal attempt.
The C ABI must stage results before changing caller output.

An invalid key returns `LeafReadResult::InvalidKey`.
Missing and deleted records retain their existing distinct outcomes.
The checked API does not change key ordering, cache policy, or record limits.

### Scans

Existing scan constructors retain inclusive start and end keys.
Count-limited scans retain their existing record limit.
`next_checked` reports the capacity for the requested fields.
That capacity includes both key and value bytes for `KeyAndValue`.

An insufficient buffer changes neither output bytes nor the pending live record.
Deleted entries can be skipped before that result.
A retry returns the pending record before advancing to another record.
End of scan remains terminal.
Several calls do not establish an application snapshot.

### Deletes

`Success` means that the delete operation completed.
It does not mean that a live record existed before the operation.
The native engine can install a delete marker for an absent key.
A caller that needs an atomic existence-and-delete result must coordinate concurrent writers.
The C ABI must preserve its existing serialized existence check for `NOT_FOUND`.

### Snapshots

`Disabled` means that the configuration does not enable CPR snapshots.
`InProgress` means that another capture already owns the snapshot operation.
Neither result creates the requested output file.
Completion means that the native capture completed.
It does not establish an application-wide checkpoint or a power-loss guarantee.

An internal panic can leave snapshot state unsafe.
Do not retry a capture on that handle without a separate safety guarantee.
Recovery must preserve the published snapshot when the engine uses its input as writable backing.

## C ABI requirements

Preserve existing ABI-v1 layouts, function signatures, and status values.
Use additive versioned entry points for new capabilities and bounded scans.
Do not expose Rust references, iterators, allocator-owned buffers, or unwinding.

The package must report capabilities and supported limits explicitly.
It must distinguish unsupported operations from missing records.
It must not report unavailable native metrics as measured zeroes.

The additional bounded scan will use an inclusive lower bound and an exclusive upper bound.
It will support an exclusive lower bound for continuation after a returned key.
Equal bounds will produce an empty range.
Reversed bounds will fail validation.
Record and byte budgets will limit staged output.
These budgets will not claim a bound on native disk I/O or traversal time.

Each successful scan will return complete records only.
An individually oversized first record will return a sizing or resource error.
A sizing failure must not partially change byte or descriptor output.
The caller must supply a stable recovered view when continuation requires snapshot consistency.

Close must wait for active calls.
Later operations must reject a closed or fenced handle.
Destroy requires caller-side exclusion and clears the caller's handle variable.
The package must document idempotency separately for each operation.

## Verification

The checked Rust tests cover:

- Empty, insufficient, exact, and larger output buffers.
- Missing records, deleted records, and invalid keys.
- Reads across updates, splits, and snapshot recovery.
- Key-only, value-only, and combined scans.
- Retrying a scan without skipping its pending record.
- Deleted records, range endpoints, and terminal scan state.
- Explicit delete outcomes and legacy API compatibility.
- Disabled, concurrent, and completed snapshots.

The package includes Rust/C/C++ conformance tests, panic fencing, and source packaging.
Supported-target, failure-path, and performance qualification remain release requirements.
Application migration also requires snapshot compatibility and rollback from newly produced artifacts.

The legacy-only `scan_cached_records` regression also reproduced a failure on the older `0.5.6` baseline.
Current upstream already loads the base page after a clean mini-page skips merging.
It also falls back to the base page when optional promotion exhausts the cache.
The regression retains coverage for both enabled and disabled scan promotion.
The candidate preserves these upstream corrections rather than replacing their implementation.
