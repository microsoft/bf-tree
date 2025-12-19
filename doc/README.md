# Bf-Tree

<hr>

**Bf-Tree is a modern read-write-optimized concurrent larger-than-memory range index.**

- **Modern**: designed for modern SSDs, implemented with modern programming languages (Rust).

- **Read-write-optimized**: 2.5× faster than RocksDB for scan operations, 6× faster than a B-Tree for write operations, and 2× faster than both B-Trees and LSM-Trees for point lookup -- for small records (e.g. ~100 bytes).

- **Concurrent**: scale to high thread count.

- **Larger-than-memory**: scale to large data sets.

- **Range index**: records are sorted.

The core of Bf-Tree are the mini-page abstraction and the buffer pool to support it.

## Mini-pages

#### 1. As a record-level cache
More fine-grained than page-level cache, which is more efficient at identifying individual hot records.
![Mini-pages serves as a record level cache](figures/bf-tree-cache-records.gif)

#### 2. As a write buffer
Mini pages absorb writes records and batch flush them to disk.
![Mini-pages serves as a write buffer](figures/bf-tree-buffer-writes.gif)

#### 3. Grow/shrink in size
Mini-pages grow and shrink in size to be more precise in memory usage. 

![Mini-pages grow and shrink in size](figures/bf-tree-grow-larger.gif)

#### 4. Flush to disk
Mini-pages are flushed to disk when they are too large or too cold.

![Mini-pages are flushed to disk](figures/bf-tree-batch-write.gif)

## Buffer pool for mini-pages

The buffer pool is a circular buffer, allocated space is defined by head and tail address.

#### 1. Allocate mini-pages
![Allocate mini-pages](figures/buffer-pool-alloc.gif)

#### 2. Evict mini-pages when full
![Evict mini-pages when full](figures/buffer-pool-evict.gif)

#### 3. Track hot mini-pages
Naive circular buffer is a fifo queue, we make it a LRU-approximation using the second chance region.

Mini-pages in the second-chance region are:
- Copy-on-accessed to the tail address
- Evicted to disk if not being accessed while in the region

![Track hot mini-pages](figures/buffer-pool-lru.gif)

#### 4. Grow mini-pages
Mini-pages are copied to a larger mini-page when they need to grow.
The old space is added to a free list for future allocations.

![Grow mini-pages](figures/buffer-pool-grow.gif)

## Design Topics

- [Snapshot Recovery](snapshot-recovery.md)
- [Debugging Tips](debugging-tips.md)
- [SPDK](spdk-support.md)


## What are the drawbacks of Bf-Tree?

- Bf-Tree only works for modern SSDs where parallel random 4KB writes have similar throughput to sequential writes. While not all SSDs have this property, it is not uncommon for modern SSDs.

- Bf-Tree is heavily optimized for small records (e.g., 100 bytes, a common size when used as secondary indexes). Large records will have a similar performance to B-Trees or LSM-Trees.

- Bf-Tree's buffer pool is more complex than B-Tree and LSM-Trees, as it needs to handle variable length mini-pages.

## Future directions

- Bf-Tree in-place writes to disk pages, which can burden SSD garbage collection. If it is indeed a problem, we should consider using log-structured write to disk pages.

- Bf-Tree's mini-page eviction/promotion policy is dead simple. More advanced policies can be used to ensure fairness, improve hit rate, and reduce copying. Our current paper focus on mini-page/buffer pool mechanisms, and exploring advanced policies is left as future work.

- Better async. Bf-Tree relies on OS threads to interleave I/O operations, many people believe this is not ideal. Implement Bf-Tree with user-space async I/O (e.g., tokio) might be useful.

- Go lock/latch free. Bf-Tree is lock-based, and is carefully designed so that no dead lock is possible. 

- Future hardware. Apply Bf-Tree to modern hardware such as CXL, RDMA, PM, GPU, and SmartNic.

## Cite Bf-Tree

```bibtex
@article{bf-tree,
  title={Bf-Tree: A Modern Read-Write-Optimized Concurrent Larger-Than-Memory Range Index},
  author={Hao, Xiangpeng and Chandramouli, Badrish},
  journal={Proceedings of the VLDB Endowment},
  volume={17},
  number={11},
  pages={3442--3455},
  year={2024},
  publisher={VLDB Endowment}
}
```
