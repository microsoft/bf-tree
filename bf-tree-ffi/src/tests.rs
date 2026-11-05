// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use super::*;
use std::sync::{mpsc, Arc, Barrier};
use std::thread;

fn memory_config() -> BfTreeAbiConfig {
    BfTreeAbiConfig {
        abi_version: ABI_VERSION,
        struct_size: size_of::<BfTreeAbiConfig>() as u32,
        storage: 0,
        reserved: 0,
        path: ptr::null(),
        path_length: 0,
        circular_buffer_bytes: 2 * 1024 * 1024,
        minimum_record_bytes: 2,
        maximum_record_bytes: 0,
        maximum_key_bytes: 0,
        leaf_page_bytes: 0,
    }
}

struct Handle(*mut BfTreeHandle);

impl Handle {
    fn new() -> Self {
        let mut handle = ptr::null_mut();
        assert_eq!(
            unsafe { bf_tree_create(&memory_config(), &mut handle, ptr::null_mut()) },
            BfTreeStatus::Ok
        );
        Self(handle)
    }

    fn put(&self, key: &[u8], value: &[u8]) {
        assert_eq!(
            unsafe {
                bf_tree_insert(
                    self.0,
                    key.as_ptr(),
                    key.len(),
                    value.as_ptr(),
                    value.len(),
                    ptr::null_mut(),
                )
            },
            BfTreeStatus::Ok
        );
    }

    fn read(&self, key: &[u8], value: &mut [u8]) -> (BfTreeStatus, usize) {
        let mut length = usize::MAX;
        let status = unsafe {
            bf_tree_read(
                self.0,
                key.as_ptr(),
                key.len(),
                value.as_mut_ptr(),
                value.len(),
                &mut length,
                ptr::null_mut(),
            )
        };
        (status, length)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        assert_eq!(
            unsafe { bf_tree_destroy(&mut self.0, ptr::null_mut()) },
            BfTreeStatus::Ok
        );
        assert!(self.0.is_null());
    }
}

fn options(lower: &[u8], upper: &[u8], records: usize, bytes: usize) -> BfTreeScanOptionsV1 {
    BfTreeScanOptionsV1 {
        abi_version: ABI_VERSION,
        struct_size: size_of::<BfTreeScanOptionsV1>() as u32,
        lower: lower.as_ptr(),
        lower_length: lower.len(),
        upper: upper.as_ptr(),
        upper_length: upper.len(),
        maximum_records: records,
        maximum_bytes: bytes,
        lower_exclusive: 0,
        reserved: 0,
    }
}

fn bounded(
    handle: &Handle,
    request: &BfTreeScanOptionsV1,
    bytes: &mut [u8],
    records: &mut [BfTreeScanRecord],
) -> (BfTreeStatus, usize, usize) {
    let mut byte_count = usize::MAX;
    let mut record_count = usize::MAX;
    let status = unsafe {
        bf_tree_scan_bounded_v1(
            handle.0,
            request,
            bytes.as_mut_ptr(),
            bytes.len(),
            &mut byte_count,
            records.as_mut_ptr(),
            records.len(),
            &mut record_count,
            ptr::null_mut(),
        )
    };
    (status, byte_count, record_count)
}

fn record() -> BfTreeScanRecord {
    BfTreeScanRecord {
        key_offset: 99,
        key_length: 99,
        value_offset: 99,
        value_length: 99,
    }
}

#[test]
fn abi_layout_and_capabilities_preserve_v1() {
    assert_eq!(bf_tree_abi_version(), 1);
    assert_eq!(size_of::<BfTreeStatus>(), 4);
    assert_eq!(std::mem::offset_of!(BfTreeAbiConfig, path), 16);
    assert_eq!(size_of::<BfTreeScanRecord>(), 4 * size_of::<usize>());
    let mut capabilities = BfTreeCapabilitiesV1 {
        abi_version: 1,
        struct_size: size_of::<BfTreeCapabilitiesV1>() as u32,
        flags: 0,
        maximum_scan_records: 0,
        maximum_scan_bytes: 0,
    };
    assert_eq!(
        unsafe { bf_tree_get_capabilities_v1(&mut capabilities, ptr::null_mut()) },
        BfTreeStatus::Ok
    );
    assert_eq!(capabilities.flags, CAPABILITIES);
    assert_eq!(capabilities.flags & (0b1111 << 10), 0);
    assert_eq!(capabilities.maximum_scan_records, MAX_SCAN_RECORDS);
    assert_eq!(capabilities.maximum_scan_bytes, MAX_SCAN_BYTES);
    capabilities.abi_version = 2;
    capabilities.flags = 123;
    assert_eq!(
        unsafe { bf_tree_get_capabilities_v1(&mut capabilities, ptr::null_mut()) },
        BfTreeStatus::InvalidConfig
    );
    assert_eq!(capabilities.flags, 123);
}

#[test]
fn configuration_and_open_errors_preserve_outputs_and_files() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing.db");
    fs::write(&path, b"existing-data").unwrap();
    let path_text = path.to_str().unwrap();
    let mut config = memory_config();
    config.storage = 1;
    config.path = path_text.as_ptr();
    config.path_length = path_text.len();
    let mut handle = ptr::dangling_mut::<BfTreeHandle>();
    let mut message = [0 as c_char; 5];
    let mut detail = BfTreeError {
        message: message.as_mut_ptr(),
        message_capacity: message.len(),
        message_required: 0,
    };
    assert_eq!(
        unsafe { bf_tree_open(&config, &mut handle, &mut detail) },
        BfTreeStatus::Unsupported
    );
    assert!(handle.is_null());
    assert!(detail.message_required > message.len());
    assert_eq!(message[4], 0);
    assert_eq!(
        unsafe { bf_tree_create(&config, &mut handle, &mut detail) },
        BfTreeStatus::AlreadyExists
    );
    assert_eq!(fs::read(&path).unwrap(), b"existing-data");
    config.abi_version = 2;
    assert_eq!(
        unsafe { bf_tree_create(&config, &mut handle, &mut detail) },
        BfTreeStatus::InvalidConfig
    );
    config.abi_version = 1;
    config.reserved = 1;
    assert_eq!(
        unsafe { bf_tree_create(&config, &mut handle, &mut detail) },
        BfTreeStatus::InvalidConfig
    );
    config.reserved = 0;
    config.maximum_key_bytes = usize::MAX;
    assert_eq!(
        unsafe { bf_tree_create(&config, &mut handle, &mut detail) },
        BfTreeStatus::InvalidConfig
    );
    assert!(handle.is_null());
}

#[test]
fn caller_buffers_and_delete_statuses_remain_checked() {
    let handle = Handle::new();
    handle.put(b"key", b"value");
    let mut short = [0xA5; 4];
    assert_eq!(
        handle.read(b"key", &mut short),
        (BfTreeStatus::BufferTooSmall, 5)
    );
    assert_eq!(short, [0xA5; 4]);
    let mut value = [0; 5];
    assert_eq!(handle.read(b"key", &mut value), (BfTreeStatus::Ok, 5));
    assert_eq!(&value, b"value");
    assert_eq!(
        unsafe {
            bf_tree_insert(
                handle.0,
                b"key".as_ptr(),
                3,
                ptr::null(),
                0,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::InvalidArgument
    );
    assert_eq!(handle.read(b"key", &mut value), (BfTreeStatus::Ok, 5));
    assert_eq!(
        unsafe { bf_tree_delete(handle.0, b"key".as_ptr(), 3, ptr::null_mut()) },
        BfTreeStatus::Ok
    );
    assert_eq!(
        unsafe { bf_tree_delete(handle.0, b"key".as_ptr(), 3, ptr::null_mut()) },
        BfTreeStatus::NotFound
    );
    assert_eq!(
        handle.read(b"missing", &mut []),
        (BfTreeStatus::NotFound, 0)
    );
}

#[test]
fn bounded_scans_preserve_ranges_budgets_and_continuation() {
    let handle = Handle::new();
    for key in [b"aa", b"bb", b"cc", b"dd"] {
        handle.put(key, b"123");
    }
    let mut bytes = [0xA5; 32];
    let mut records = [record(); 4];
    let mut request = options(b"aa", b"dd", 4, 10);
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::Ok, 10, 2)
    );
    assert_eq!(&bytes[..10], b"aa123bb123");
    request.lower = b"bb".as_ptr();
    request.lower_exclusive = 1;
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::Ok, 5, 1)
    );
    assert_eq!(&bytes[..5], b"cc123");
    request.lower = b"cc".as_ptr();
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::Ok, 0, 0)
    );
    request = options(b"aa", b"aa", 1, 1);
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::Ok, 0, 0)
    );
    request = options(b"dd", b"aa", 4, 32);
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidArgument, 0, 0)
    );
    request = options(b"aa", b"", 4, 4);
    bytes.fill(0xA5);
    records.fill(record());
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::BufferTooSmall, 5, 1)
    );
    assert_eq!(bytes, [0xA5; 32]);
    assert_eq!(records, [record(); 4]);
    request.maximum_bytes = 32;
    assert_eq!(
        bounded(&handle, &request, &mut bytes[..4], &mut records),
        (BfTreeStatus::BufferTooSmall, 20, 4)
    );
    assert_eq!(bytes, [0xA5; 32]);
    assert_eq!(records, [record(); 4]);
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records[..1]),
        (BfTreeStatus::BufferTooSmall, 20, 4)
    );
    assert_eq!(bytes, [0xA5; 32]);
    request.maximum_records = 0;
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidArgument, 0, 0)
    );
    request.maximum_records = MAX_SCAN_RECORDS + 1;
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidArgument, 0, 0)
    );
    request.maximum_records = 1;
    request.lower_exclusive = 2;
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidArgument, 0, 0)
    );
}

#[test]
fn legacy_scan_preserves_whole_output_on_capacity_failure() {
    let handle = Handle::new();
    handle.put(b"key", b"value");
    let mut bytes = [0xA5; 8];
    let mut records = [record(); 1];
    let mut byte_count = 0;
    let mut record_count = 0;
    assert_eq!(
        unsafe {
            bf_tree_scan(
                handle.0,
                b"key".as_ptr(),
                3,
                1,
                bytes.as_mut_ptr(),
                2,
                &mut byte_count,
                records.as_mut_ptr(),
                records.len(),
                &mut record_count,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::BufferTooSmall
    );
    assert_eq!((byte_count, record_count), (8, 1));
    assert_eq!(bytes, [0xA5; 8]);
    assert_eq!(records, [record(); 1]);
    assert_eq!(
        unsafe {
            bf_tree_scan(
                handle.0,
                b"key".as_ptr(),
                3,
                1,
                bytes.as_mut_ptr(),
                bytes.len(),
                &mut byte_count,
                records.as_mut_ptr(),
                records.len(),
                &mut record_count,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Ok
    );
    assert_eq!(&bytes, b"keyvalue");
    assert_eq!(
        records[0],
        BfTreeScanRecord {
            key_offset: 0,
            key_length: 3,
            value_offset: 3,
            value_length: 5
        }
    );
}

#[test]
fn bounded_scan_sizing_is_fresh_and_invalid_options_fail_explicitly() {
    let handle = Handle::new();
    handle.put(b"key", b"one");
    let mut request = options(b"key", b"", 1, 32);
    assert_eq!(
        bounded(&handle, &request, &mut [], &mut []),
        (BfTreeStatus::BufferTooSmall, 6, 1)
    );
    handle.put(b"key", b"longer");
    let mut bytes = [0xA5; 6];
    let mut records = [record()];
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::BufferTooSmall, 9, 1)
    );
    assert_eq!(bytes, [0xA5; 6]);
    assert_eq!(records, [record()]);
    request.maximum_bytes = 0;
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidArgument, 0, 0)
    );
    request.maximum_bytes = MAX_SCAN_BYTES + 1;
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidArgument, 0, 0)
    );
    request.maximum_bytes = 32;
    request.reserved = 1;
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidArgument, 0, 0)
    );
    request.reserved = 0;
    request.abi_version = 2;
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidConfig, 0, 0)
    );
    request.abi_version = 1;
    let oversized = [b'z'; 1024];
    request.upper = oversized.as_ptr();
    request.upper_length = oversized.len();
    assert_eq!(
        bounded(&handle, &request, &mut bytes, &mut records),
        (BfTreeStatus::InvalidArgument, 0, 0)
    );
}

#[test]
fn snapshot_recovery_preserves_modes_generation_and_private_backing() {
    let directory = tempfile::tempdir().unwrap();
    let snapshot = directory.path().join("snapshot.cpr");
    let copy = directory.path().join("private.cpr");
    let next = directory.path().join("next.cpr");
    let original = Handle::new();
    original.put(b"key", b"old");
    let snapshot_text = snapshot.to_str().unwrap();
    let mut generation = 0;
    assert_eq!(
        unsafe {
            bf_tree_cpr_snapshot(
                original.0,
                snapshot_text.as_ptr(),
                snapshot_text.len(),
                &mut generation,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Ok
    );
    assert_eq!(generation, 1);
    assert_eq!(
        unsafe {
            bf_tree_cpr_snapshot(
                original.0,
                snapshot_text.as_ptr(),
                snapshot_text.len(),
                &mut generation,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::AlreadyExists
    );
    assert_eq!(generation, 0);
    let original_bytes = fs::read(&snapshot).unwrap();
    fs::copy(&snapshot, &copy).unwrap();
    let copy_text = copy.to_str().unwrap();
    let mut raw = ptr::null_mut();
    assert_eq!(
        unsafe {
            bf_tree_recover(
                &memory_config(),
                copy_text.as_ptr(),
                copy_text.len(),
                1,
                0,
                &mut raw,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Ok
    );
    let restored = Handle(raw);
    original.put(b"key", b"new");
    let mut value = [0; 3];
    assert_eq!(restored.read(b"key", &mut value), (BfTreeStatus::Ok, 3));
    assert_eq!(&value, b"old");
    assert_eq!(
        unsafe {
            bf_tree_insert(
                restored.0,
                b"key".as_ptr(),
                3,
                b"bad".as_ptr(),
                3,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Unsupported
    );
    assert_eq!(
        unsafe { bf_tree_delete(restored.0, b"key".as_ptr(), 3, ptr::null_mut()) },
        BfTreeStatus::Unsupported
    );
    let next_text = next.to_str().unwrap();
    assert_eq!(
        unsafe {
            bf_tree_cpr_snapshot(
                restored.0,
                next_text.as_ptr(),
                next_text.len(),
                &mut generation,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Unsupported
    );
    assert!(!next.exists());
    drop(restored);
    assert_eq!(fs::read(&snapshot).unwrap(), original_bytes);
    assert_eq!(
        unsafe {
            bf_tree_recover(
                &memory_config(),
                copy_text.as_ptr(),
                copy_text.len(),
                1,
                1,
                &mut raw,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Ok
    );
    let writable = Handle(raw);
    writable.put(b"key", b"new");
    assert_eq!(
        unsafe {
            bf_tree_cpr_snapshot(
                writable.0,
                next_text.as_ptr(),
                next_text.len(),
                &mut generation,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Ok
    );
    assert_eq!(generation, 2);
}

#[test]
fn recovery_checks_limits_and_uses_snapshot_scratch_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let snapshot = directory.path().join("snapshot.cpr");
    let working = directory.path().join("working.cpr");
    let mut config = memory_config();
    config.minimum_record_bytes = 8;
    config.maximum_record_bytes = 3000;
    config.maximum_key_bytes = 24;
    config.leaf_page_bytes = 8192;
    let mut raw = ptr::null_mut();
    assert_eq!(
        unsafe { bf_tree_create(&config, &mut raw, ptr::null_mut()) },
        BfTreeStatus::Ok
    );
    let original = Handle(raw);
    let key = [b'k'; 20];
    let value = vec![0x5A; 2500];
    original.put(&key, &value);
    let snapshot_text = snapshot.to_str().unwrap();
    let mut generation = 0;
    assert_eq!(
        unsafe {
            bf_tree_cpr_snapshot(
                original.0,
                snapshot_text.as_ptr(),
                snapshot_text.len(),
                &mut generation,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Ok
    );

    fs::copy(&snapshot, &working).unwrap();
    let working_text = working.to_str().unwrap();
    let mut request = memory_config();
    request.minimum_record_bytes = 0;
    assert_eq!(
        unsafe {
            bf_tree_recover(
                &request,
                working_text.as_ptr(),
                working_text.len(),
                generation,
                0,
                &mut raw,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Ok
    );
    let restored = Handle(raw);
    let mut output = vec![0; value.len()];
    assert_eq!(
        restored.read(&key, &mut output),
        (BfTreeStatus::Ok, value.len())
    );
    assert_eq!(output, value);
    let mut bytes = vec![0; key.len() + value.len()];
    assert_eq!(
        bounded(
            &restored,
            &options(&key, b"", 1, 3000),
            &mut bytes,
            &mut [record()],
        ),
        (BfTreeStatus::Ok, bytes.len(), 1)
    );
    assert_eq!(&bytes[..key.len()], &key);
    assert_eq!(&bytes[key.len()..], &value);

    for (minimum, maximum, key, leaf) in [
        (4, 0, 0, 0),
        (0, 1952, 0, 0),
        (0, 0, 16, 0),
        (0, 0, 0, 4096),
    ] {
        let mut request = memory_config();
        request.minimum_record_bytes = minimum;
        request.maximum_record_bytes = maximum;
        request.maximum_key_bytes = key;
        request.leaf_page_bytes = leaf;
        let copy = directory
            .path()
            .join(format!("mismatch-{minimum}-{maximum}-{key}-{leaf}.cpr"));
        fs::copy(&snapshot, &copy).unwrap();
        let text = copy.to_str().unwrap();
        let mut rejected = ptr::dangling_mut::<BfTreeHandle>();
        assert_eq!(
            unsafe {
                bf_tree_recover(
                    &request,
                    text.as_ptr(),
                    text.len(),
                    generation,
                    0,
                    &mut rejected,
                    ptr::null_mut(),
                )
            },
            BfTreeStatus::InvalidConfig
        );
        assert!(rejected.is_null());
    }
}

#[test]
fn panic_fences_all_data_operations_and_overlapping_success() {
    let handle = Handle::new();
    handle.put(b"key", b"value");
    let raw = handle.0 as usize;
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let other = thread::spawn(move || unsafe {
        with_tree(raw as *mut BfTreeHandle, false, |_| {
            entered_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
            Ok(())
        })
        .unwrap_err()
        .status
    });
    entered_rx.recv().unwrap();
    let status = unsafe {
        ffi_call(ptr::null_mut(), || {
            with_tree(handle.0, false, |state| {
                state.tree.read(b"key", &mut []);
                Ok(BfTreeStatus::Ok)
            })
        })
    };
    assert_eq!(status, BfTreeStatus::Panic);
    resume_tx.send(()).unwrap();
    assert_eq!(other.join().unwrap(), BfTreeStatus::Closed);
    assert_eq!(handle.read(b"key", &mut [0; 5]), (BfTreeStatus::Closed, 0));
    assert_eq!(
        unsafe {
            bf_tree_insert(
                handle.0,
                b"key".as_ptr(),
                3,
                b"value".as_ptr(),
                5,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Closed
    );
    assert_eq!(
        unsafe { bf_tree_delete(handle.0, b"key".as_ptr(), 3, ptr::null_mut()) },
        BfTreeStatus::Closed
    );
    let request = options(b"key", b"", 1, 16);
    assert_eq!(
        bounded(&handle, &request, &mut [0; 16], &mut [record()]),
        (BfTreeStatus::Closed, 0, 0)
    );
    assert_eq!(
        unsafe { bf_tree_close(handle.0, ptr::null_mut()) },
        BfTreeStatus::Ok
    );
    assert_eq!(
        unsafe { bf_tree_close(handle.0, ptr::null_mut()) },
        BfTreeStatus::Closed
    );
}

#[test]
fn concurrent_deletes_have_one_success_and_one_missing_result() {
    let handle = Handle::new();
    handle.put(b"key", b"value");
    let barrier = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let barrier = Arc::clone(&barrier);
        let raw = handle.0 as usize;
        workers.push(thread::spawn(move || {
            barrier.wait();
            unsafe {
                bf_tree_delete(
                    raw as *mut BfTreeHandle,
                    b"key".as_ptr(),
                    3,
                    ptr::null_mut(),
                )
            }
        }));
    }
    barrier.wait();
    let results: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    assert_eq!(
        results
            .iter()
            .filter(|status| **status == BfTreeStatus::Ok)
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|status| **status == BfTreeStatus::NotFound)
            .count(),
        1
    );
}

#[test]
fn close_waits_for_active_operations_and_destroy_is_null_idempotent() {
    let handle = Handle::new();
    let raw = handle.0 as usize;
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let reader = thread::spawn(move || unsafe {
        with_tree(raw as *mut BfTreeHandle, false, |_| {
            entered_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();
    });
    entered_rx.recv().unwrap();
    let raw = handle.0 as usize;
    let (closed_tx, closed_rx) = mpsc::channel();
    let closer = thread::spawn(move || {
        closed_tx
            .send(unsafe { bf_tree_close(raw as *mut BfTreeHandle, ptr::null_mut()) })
            .unwrap();
    });
    assert!(closed_rx.try_recv().is_err());
    resume_tx.send(()).unwrap();
    reader.join().unwrap();
    assert_eq!(closed_rx.recv().unwrap(), BfTreeStatus::Ok);
    closer.join().unwrap();
    assert_eq!(handle.read(b"key", &mut []), (BfTreeStatus::Closed, 0));
    drop(handle);
    let mut null = ptr::null_mut();
    assert_eq!(
        unsafe { bf_tree_destroy(&mut null, ptr::null_mut()) },
        BfTreeStatus::Ok
    );
}

#[test]
fn invalid_paths_and_snapshot_limits_report_errors_without_publication() {
    let mut config = memory_config();
    config.storage = 1;
    let mut raw = ptr::null_mut();
    assert_eq!(
        unsafe { bf_tree_create(&config, &mut raw, ptr::null_mut()) },
        BfTreeStatus::InvalidConfig
    );
    assert!(raw.is_null());
    let invalid_utf8 = [0xFF];
    config.path = invalid_utf8.as_ptr();
    config.path_length = 1;
    assert_eq!(
        unsafe { bf_tree_create(&config, &mut raw, ptr::null_mut()) },
        BfTreeStatus::InvalidConfig
    );
    assert!(raw.is_null());

    let directory = tempfile::tempdir().unwrap();
    let handle = Handle::new();
    handle.put(b"key", b"value");
    let path = directory.path().join("missing").join("snapshot.cpr");
    let text = path.to_str().unwrap();
    let mut generation = 99;
    assert_eq!(
        unsafe {
            bf_tree_cpr_snapshot(
                handle.0,
                text.as_ptr(),
                text.len(),
                &mut generation,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::InvalidConfig
    );
    assert_eq!(generation, 0);
    assert!(!path.exists());
    assert_eq!(handle.read(b"key", &mut [0; 5]), (BfTreeStatus::Ok, 5));
    unsafe {
        *(*handle.0).snapshot_generation.lock().unwrap() = u64::MAX;
    }
    assert_eq!(
        unsafe {
            bf_tree_cpr_snapshot(
                handle.0,
                text.as_ptr(),
                text.len(),
                &mut generation,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::ResourceExhausted
    );
    assert!(!path.exists());
    assert_eq!(
        unsafe { bf_tree_close(handle.0, ptr::null_mut()) },
        BfTreeStatus::Ok
    );
    assert_eq!(
        unsafe {
            bf_tree_cpr_snapshot(
                handle.0,
                text.as_ptr(),
                text.len(),
                &mut generation,
                ptr::null_mut(),
            )
        },
        BfTreeStatus::Closed
    );
}

#[test]
fn error_truncation_preserves_utf8_and_null_termination() {
    let mut bytes = [0 as c_char; 4];
    let mut target = BfTreeError {
        message: bytes.as_mut_ptr(),
        message_capacity: bytes.len(),
        message_required: 0,
    };
    unsafe {
        diagnostics(&mut target, "\u{e9}\u{e9}");
    }
    assert_eq!(target.message_required, 5);
    let output: Vec<u8> = bytes.iter().map(|byte| *byte as u8).collect();
    assert_eq!(&output[..3], &[0xC3, 0xA9, 0]);
    assert!(str::from_utf8(&output[..2]).is_ok());
}
