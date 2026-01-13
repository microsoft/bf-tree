// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#![no_main]

use std::collections::BTreeSet;

use arbitrary::Arbitrary;
use bf_tree::circular_buffer::TombstoneHandle;
use libfuzzer_sys::fuzz_target;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Arbitrary, Debug)]
enum Methods {
    Alloc(u8),
    Dealloc,
    Evict,
}

fn write_initial_ptr(ptr: *mut u8, size: u8) {
    for i in 0..size {
        unsafe {
            *ptr.add(i as usize) = size;
        }
    }
}

fn check_ptr(ptr: *mut u8) {
    let size = unsafe { *ptr };
    for i in 0..size {
        unsafe {
            assert_eq!(*ptr.add(i as usize), size);
        }
    }
}

fuzz_target!(|data: Vec<Methods>| {
    let buf =
        bf_tree::circular_buffer::CircularBuffer::new(8192, 0.1, 64, 2048, 4096, 64, None, true);

    let mut allocated = BTreeSet::new();

    for method in data {
        match method {
            Methods::Alloc(size) => {
                if size == 0 {
                    // we don't support zero sized allocations
                    continue;
                }
                let v = buf.alloc(size as usize);
                if let Ok(v) = v {
                    let ptr = v.as_ptr();
                    write_initial_ptr(ptr, size);
                    let not_present = allocated.insert(v.as_ptr());
                    assert!(not_present);
                }
            }
            Methods::Dealloc => {
                let ptr = allocated.iter().next().copied();
                if let Some(v) = ptr {
                    check_ptr(v);
                    let present = allocated.remove(&v);
                    assert!(present);
                    let h = unsafe { buf.acquire_exclusive_dealloc_handle(v).unwrap() };
                    buf.dealloc(h);
                }
            }
            Methods::Evict => {
                let mut callback = |h: TombstoneHandle| {
                    check_ptr(h.as_ptr());
                    let present = allocated.remove(&h.as_ptr());
                    assert!(present);
                    Ok(h)
                };
                let _ = buf.evict_one(&mut callback);
            }
        }
    }

    buf.evict_n(usize::MAX, |h| {
        check_ptr(h.as_ptr());
        let present = allocated.remove(&h.as_ptr());
        assert!(present);
        Ok(h)
    })
    .unwrap();
});
