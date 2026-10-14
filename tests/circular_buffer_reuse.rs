// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use bf_tree::circular_buffer::{CircularBuffer, CircularBufferError};

#[test]
fn repeated_reuse_keeps_one_allocation_and_remains_evictable() {
    let buffer = CircularBuffer::new(65536, 0.1, 64, 1952, 4096, 32, None, false);
    let allocation = buffer.alloc(4096).unwrap();
    let original = allocation.as_ptr();
    drop(allocation);
    buffer.dealloc(unsafe { buffer.acquire_exclusive_dealloc_handle(original).unwrap() });

    for _ in 0..100_000 {
        let allocation = buffer.alloc(4096).unwrap();
        let reused = allocation.as_ptr();
        drop(allocation);
        buffer.dealloc(unsafe { buffer.acquire_exclusive_dealloc_handle(reused).unwrap() });
        assert_eq!(reused, original);
    }

    let metrics = buffer.get_metrics();
    assert_eq!(metrics.allocated_cnt, 1);
    assert_eq!(metrics.free_listed_cnt, 1);
    buffer
        .evict_n(usize::MAX, |_| {
            panic!("free-list eviction must not call the callback")
        })
        .unwrap();
    assert_eq!(buffer.get_metrics().head_tail_distance, 0);
}

#[test]
fn reuse_under_capacity_pressure_preserves_values_across_wraparound() {
    let buffer = CircularBuffer::new(16384, 0.1, 64, 1952, 4096, 32, None, false);

    for round in 0..128u8 {
        let mut pointers = Vec::new();
        for index in 0..3u8 {
            let allocation = buffer.alloc(4096).unwrap();
            let ptr = allocation.as_ptr();
            unsafe { *ptr = round.wrapping_add(index) };
            pointers.push(ptr);
        }
        assert!(matches!(buffer.alloc(4096), Err(CircularBufferError::Full)));

        buffer.dealloc(unsafe {
            buffer
                .acquire_exclusive_dealloc_handle(pointers[1])
                .unwrap()
        });
        let allocation = buffer.alloc(4096).unwrap();
        let reused = allocation.as_ptr();
        assert_eq!(reused, pointers[1]);
        unsafe { *reused = round.wrapping_add(1) };
        drop(allocation);

        let mut seen = [false; 3];
        buffer
            .evict_n(usize::MAX, |handle| {
                let index = pointers
                    .iter()
                    .position(|&ptr| ptr == handle.as_ptr())
                    .unwrap();
                assert!(!seen[index]);
                assert_eq!(unsafe { *handle.as_ptr() }, round.wrapping_add(index as u8));
                seen[index] = true;
                Ok(handle)
            })
            .unwrap();
        assert_eq!(seen, [true; 3]);
        let metrics = buffer.get_metrics();
        assert_eq!(metrics.head_tail_distance, 0);
        assert_eq!(metrics.allocated_cnt, 0);
        assert_eq!(metrics.free_listed_cnt, 0);
    }
}
