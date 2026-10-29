# Bf-Tree WAL, snapshot, and recovery

## WAL format

## Snapshot file format
Bf-Tree now supports persisting data to a snapshot file or recovering from a snapshot file.
The doc describes the file format of Bf-Tree snapshot files.

Code is at https://github.com/gim-home/Bf-Tree/blob/main/src/snapshot.rs#L189-L199

The metadata is stored on the file's first page, with sufficient padding to align to 4096 bytes.
The metadata is directly written in the file (i.e., with C ABI struct layout) without serialization.

```rust
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
```

### Magic words
```rust
const BF_TREE_MAGIC_BEGIN: &[u8; 16] = b"BF-TREE-V0-BEGIN";
const BF_TREE_MAGIC_END: &[u8; 14] = b"BF-TREE-V0-END";
```

### Inner mapping
Inner nodes are pinned to memory (I don't think we will change this in the future).
We write the inner nodes to disk for ease of recovery when snapshotting.

To snapshot, we first traverse all inner nodes, write the nodes to disk, and track their `(ptr, offset)` pairs, where the ptr is the virtual memory address of that inner node, and the offset is the corresponding disk offset in the snapshot file.

To recover, we read the inner nodes mapping using the `(inner_offset, inner_size)` pair in the metadata, which gives us the `(ptr, offset)` pairs.
Then, we start from the root node and read the node from the disk.
At this time, the node from the disk contains child pointers that are not valid (old virtual memory address); we need to correct all the child pointers.
This is done by looking up the inner mapping we just loaded; it tells us where to load the child node from disk (rather than from the virtual address); after we load the child node to disk, we update the child ptr to the corrected memory address.
If the child ptr points to an inner node, we need to correct the child node's child pointers recursively; if it points to a leaf node, i.e., it is a page ID rather than a virtual memory address, we don't need to correct it,
as page ID translations are handled by the page table (described below).

#### Locking private inner nodes

Root growth and recovery upgrade freshly allocated, unpublished inner nodes to
write guards. Their version cannot have changed through another tree operation,
so upgrading must not report `Locked` solely because an atomic operation failed
spuriously. `ReadGuard::upgrade` uses strong `compare_exchange`: an actual
version mismatch (including a held write lock or an intervening writer) still
fails, with the same memory orderings as before.

The deterministic integration tests in `tests/inner_lock_spurious.rs` compile
the production guard implementation against a test-only node/atomic shim that
injects a permitted unchanged-value weak-CAS failure. This avoids depending on
the host ISA or probabilistic stress. The native
`inner_root_split_preserves_complete_state_through_recovery` regression forces
an inner-root split (at least three levels), checks every key/value and the full
ordered scan, then repeats snapshot/recovery with further inner-node splits.
Recovery uses a private working copy and checks that the sealed snapshot is unchanged.
Read/scan promotion is disabled in that test to isolate root growth and recovery
from full-cache promotion behavior.

Run the deterministic regression with `cargo test --test inner_lock_spurious`
and the native regression with `cargo test --lib inner_root_split`.
Shuttle's integer weak CAS currently delegates to strong CAS rather than
injecting spurious failures. The deterministic test does not depend on Shuttle;
Shuttle tests should use `cargo test --lib --features shuttle` because the
library's Shuttle platform shims require `cfg(test)`.

### Leaf mapping
Leaf mapping is a page table that maps page ID to disk offset.

To snapshot, we serialize the page table to disk; nothing needs to be changed for the last level inner nodes or leaf page.

To recover, we also need to reconstruct the page table.
Specifically, we use the `(leaf_offset, leaf_size)` pair in the metadata to read the (PageID, offset) pairs from the disk; we then use it to reconstruct the page table.
