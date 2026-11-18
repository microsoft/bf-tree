// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#![doc = include_str!("../README.md")]

#[cfg(not(panic = "unwind"))]
compile_error!("bf-tree-ffi requires panic=unwind for its checked C boundary");

use bf_tree::{BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField, StorageBackend};
use std::any::Any;
use std::ffi::c_char;
use std::fs::{self, OpenOptions};
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::str;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};

pub const ABI_VERSION: u32 = 1;
pub const MAX_SCAN_RECORDS: usize = 65_536;
pub const MAX_SCAN_BYTES: usize = 64 * 1024 * 1024;
pub const CAPABILITIES: u64 = (1 << 10) - 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BfTreeStatus {
    Ok = 0,
    NotFound = 1,
    Deleted = 2,
    BufferTooSmall = 3,
    InvalidArgument = 4,
    InvalidConfig = 5,
    Closed = 6,
    Unsupported = 7,
    Panic = 8,
    InternalError = 9,
    AlreadyExists = 10,
    ResourceExhausted = 11,
}

#[repr(C)]
pub struct BfTreeError {
    pub message: *mut c_char,
    pub message_capacity: usize,
    pub message_required: usize,
}

#[repr(C)]
pub struct BfTreeAbiConfig {
    pub abi_version: u32,
    pub struct_size: u32,
    pub storage: u32,
    pub reserved: u32,
    pub path: *const u8,
    pub path_length: usize,
    pub circular_buffer_bytes: usize,
    pub minimum_record_bytes: usize,
    pub maximum_record_bytes: usize,
    pub maximum_key_bytes: usize,
    pub leaf_page_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BfTreeScanRecord {
    pub key_offset: usize,
    pub key_length: usize,
    pub value_offset: usize,
    pub value_length: usize,
}

#[repr(C)]
pub struct BfTreeCapabilitiesV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub flags: u64,
    pub maximum_scan_records: usize,
    pub maximum_scan_bytes: usize,
}

#[repr(C)]
pub struct BfTreeScanOptionsV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub lower: *const u8,
    pub lower_length: usize,
    pub upper: *const u8,
    pub upper_length: usize,
    pub maximum_records: usize,
    pub maximum_bytes: usize,
    pub lower_exclusive: u32,
    pub reserved: u32,
}

struct TreeState {
    tree: BfTree,
    maximum_record_bytes: usize,
    maximum_key_bytes: usize,
    writable: bool,
}

pub struct BfTreeHandle {
    state: RwLock<Option<TreeState>>,
    snapshot_generation: Mutex<u64>,
    fenced: AtomicBool,
}

#[derive(Debug)]
struct ApiError {
    status: BfTreeStatus,
    message: String,
}

type ApiResult<T> = Result<T, ApiError>;

fn error(status: BfTreeStatus, message: impl Into<String>) -> ApiError {
    ApiError {
        status,
        message: message.into(),
    }
}

fn invalid(message: impl Into<String>) -> ApiError {
    error(BfTreeStatus::InvalidArgument, message)
}

fn closed() -> ApiError {
    error(BfTreeStatus::Closed, "handle is closed or fenced")
}

fn default_if_zero(value: usize, default: usize) -> usize {
    if value == 0 {
        default
    } else {
        value
    }
}

fn header(version: u32, size: u32, expected: usize) -> ApiResult<()> {
    if version != ABI_VERSION || size as usize != expected {
        return Err(error(
            BfTreeStatus::InvalidConfig,
            "ABI version or structure size mismatch",
        ));
    }
    Ok(())
}

unsafe fn reference<'a, T>(value: *const T) -> ApiResult<&'a T> {
    if value.is_null() || !value.is_aligned() {
        return Err(invalid("null or unaligned structure pointer"));
    }
    Ok(&*value)
}

fn output<T>(value: *mut T, capacity: usize) -> ApiResult<()> {
    if capacity > isize::MAX as usize / size_of::<T>() {
        return Err(invalid("output capacity exceeds the addressable range"));
    }
    if capacity != 0 && (value.is_null() || !value.is_aligned()) {
        return Err(invalid("null or unaligned output pointer"));
    }
    Ok(())
}

unsafe fn input<'a>(value: *const u8, length: usize) -> ApiResult<&'a [u8]> {
    if length > isize::MAX as usize || (length != 0 && value.is_null()) {
        return Err(invalid("invalid input pointer or length"));
    }
    if length == 0 {
        Ok(&[])
    } else {
        Ok(slice::from_raw_parts(value, length))
    }
}

unsafe fn path<'a>(value: *const u8, length: usize, status: BfTreeStatus) -> ApiResult<&'a Path> {
    let bytes = input(value, length)?;
    if bytes.is_empty() || bytes.contains(&0) {
        return Err(error(
            status,
            "path must be nonempty UTF-8 without a null byte",
        ));
    }
    Ok(Path::new(
        str::from_utf8(bytes).map_err(|_| error(status, "path is not UTF-8"))?,
    ))
}

fn buffer(length: usize) -> ApiResult<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|cause| {
        error(
            BfTreeStatus::ResourceExhausted,
            format!("buffer allocation failed: {cause}"),
        )
    })?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn panic_detail(value: Box<dyn Any + Send>) -> String {
    if let Some(text) = value.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = value.downcast_ref::<String>() {
        text.clone()
    } else {
        "panic without a string payload".to_string()
    }
}

unsafe fn diagnostics(target: *mut BfTreeError, text: &str) {
    if target.is_null() {
        return;
    }
    let target = &mut *target;
    target.message_required = if text.is_empty() { 0 } else { text.len() + 1 };
    if target.message.is_null() || target.message_capacity == 0 {
        return;
    }
    let mut count = text.len().min(target.message_capacity - 1);
    while !text.is_char_boundary(count) {
        count -= 1;
    }
    ptr::copy_nonoverlapping(text.as_ptr(), target.message.cast(), count);
    *target.message.add(count) = 0;
}

unsafe fn ffi_call(
    target: *mut BfTreeError,
    action: impl FnOnce() -> ApiResult<BfTreeStatus>,
) -> BfTreeStatus {
    diagnostics(target, "");
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(status)) => status,
        Ok(Err(cause)) => {
            diagnostics(target, &cause.message);
            cause.status
        }
        Err(cause) => {
            diagnostics(target, &format!("Rust panic: {}", panic_detail(cause)));
            BfTreeStatus::Panic
        }
    }
}

fn checked_action<T>(
    handle: &BfTreeHandle,
    state: &Option<TreeState>,
    action: impl FnOnce(&TreeState) -> ApiResult<T>,
) -> ApiResult<T> {
    if handle.fenced.load(Ordering::Acquire) {
        return Err(closed());
    }
    let state = state.as_ref().ok_or_else(closed)?;
    match catch_unwind(AssertUnwindSafe(|| action(state))) {
        Ok(Err(cause)) => Err(cause),
        Ok(Ok(value)) => {
            if handle.fenced.load(Ordering::Acquire) {
                Err(closed())
            } else {
                Ok(value)
            }
        }
        Err(cause) => {
            handle.fenced.store(true, Ordering::Release);
            Err(error(
                BfTreeStatus::Panic,
                format!("handle fenced after panic: {}", panic_detail(cause)),
            ))
        }
    }
}

unsafe fn with_tree<T>(
    handle: *mut BfTreeHandle,
    exclusive: bool,
    action: impl FnOnce(&TreeState) -> ApiResult<T>,
) -> ApiResult<T> {
    let handle = reference(handle)?;
    if handle.fenced.load(Ordering::Acquire) {
        return Err(closed());
    }
    if exclusive {
        let guard = handle.state.write().map_err(|_| {
            handle.fenced.store(true, Ordering::Release);
            error(BfTreeStatus::InternalError, "handle lock is poisoned")
        })?;
        checked_action(handle, &guard, action)
    } else {
        let guard = handle.state.read().map_err(|_| {
            handle.fenced.store(true, Ordering::Release);
            error(BfTreeStatus::InternalError, "handle lock is poisoned")
        })?;
        checked_action(handle, &guard, action)
    }
}

fn writable(state: &TreeState) -> ApiResult<()> {
    if !state.writable {
        return Err(error(
            BfTreeStatus::Unsupported,
            "read-only recovery rejects this operation",
        ));
    }
    Ok(())
}

struct ReservedPath {
    path: PathBuf,
    active: bool,
}

impl ReservedPath {
    fn reserve(path: &Path, create_parents: bool) -> ApiResult<Self> {
        if create_parents {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                fs::create_dir_all(parent).map_err(|cause| {
                    error(
                        BfTreeStatus::InvalidConfig,
                        format!("create parent failed: {cause}"),
                    )
                })?;
            }
        }
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|cause| {
                let status = if cause.kind() == std::io::ErrorKind::AlreadyExists {
                    BfTreeStatus::AlreadyExists
                } else {
                    BfTreeStatus::InvalidConfig
                };
                error(status, format!("reserve output failed: {cause}"))
            })?;
        Ok(Self {
            path: path.to_owned(),
            active: true,
        })
    }

    fn cleanup(&mut self) -> ApiResult<()> {
        self.active = false;
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(cause) => Err(error(
                BfTreeStatus::InternalError,
                format!("output cleanup failed: {cause}"),
            )),
        }
    }

    fn finish<T>(&mut self, result: ApiResult<T>) -> ApiResult<T> {
        match result {
            Ok(value) => {
                self.active = false;
                Ok(value)
            }
            Err(cause) => {
                if let Err(cleanup) = self.cleanup() {
                    return Err(error(
                        cleanup.status,
                        format!("{}. {}", cause.message, cleanup.message),
                    ));
                }
                Err(cause)
            }
        }
    }
}

impl Drop for ReservedPath {
    fn drop(&mut self) {
        if self.active {
            if let Err(cause) = self.cleanup() {
                eprintln!("bf-tree-ffi: {}", cause.message);
            }
        }
    }
}

unsafe fn config<'a>(value: *const BfTreeAbiConfig) -> ApiResult<&'a BfTreeAbiConfig> {
    let value = reference(value)?;
    header(
        value.abi_version,
        value.struct_size,
        size_of::<BfTreeAbiConfig>(),
    )?;
    if value.reserved != 0 || value.storage > 1 {
        return Err(error(
            BfTreeStatus::InvalidConfig,
            "invalid storage or reserved configuration field",
        ));
    }
    Ok(value)
}

fn owned_handle(tree: BfTree, writable: bool, generation: u64) -> *mut BfTreeHandle {
    let maximum_record_bytes = tree.config().get_cb_max_record_size();
    let maximum_key_bytes = tree.config().get_max_key_len();
    Box::into_raw(Box::new(BfTreeHandle {
        state: RwLock::new(Some(TreeState {
            tree,
            maximum_record_bytes,
            maximum_key_bytes,
            writable,
        })),
        snapshot_generation: Mutex::new(generation),
        fenced: AtomicBool::new(false),
    }))
}

#[no_mangle]
pub extern "C" fn bf_tree_abi_version() -> u32 {
    ABI_VERSION
}

/// # Safety
/// Pointers must satisfy the ownership, alignment, and buffer rules in the C header.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_create(
    value: *const BfTreeAbiConfig,
    out: *mut *mut BfTreeHandle,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        output(out, 1)?;
        *out = ptr::null_mut();
        let value = config(value)?;
        let maximum = default_if_zero(value.maximum_record_bytes, 1952);
        let maximum_key = default_if_zero(value.maximum_key_bytes, 16);
        if maximum_key.checked_mul(2).is_none() {
            return Err(error(
                BfTreeStatus::InvalidConfig,
                "maximum key size overflows the native fence size",
            ));
        }
        let storage_path = if value.storage == 0 {
            Path::new(":memory:")
        } else {
            path(value.path, value.path_length, BfTreeStatus::InvalidConfig)?
        };
        let mut reservation = if value.storage == 1 {
            Some(ReservedPath::reserve(storage_path, true)?)
        } else {
            None
        };
        let mut native = Config::new(
            storage_path,
            default_if_zero(value.circular_buffer_bytes, 32 * 1024 * 1024),
        );
        native
            .storage_backend(if value.storage == 0 {
                StorageBackend::Memory
            } else {
                StorageBackend::Std
            })
            .cb_min_record_size(default_if_zero(value.minimum_record_bytes, 4))
            .cb_max_record_size(maximum)
            .cb_max_key_len(maximum_key)
            .leaf_page_size(default_if_zero(value.leaf_page_bytes, 4096))
            .use_snapshot(true)
            .snapshot_version(0);
        let created = BfTree::with_config(native, None).map_err(|cause| {
            error(
                BfTreeStatus::InvalidConfig,
                format!("create rejected configuration: {cause:?}"),
            )
        });
        let tree = if let Some(reservation) = reservation.as_mut() {
            reservation.finish(created)?
        } else {
            created?
        };
        *out = owned_handle(tree, true, 0);
        Ok(BfTreeStatus::Ok)
    })
}

/// # Safety
/// Pointers must satisfy the ownership, alignment, and buffer rules in the C header.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_open(
    value: *const BfTreeAbiConfig,
    out: *mut *mut BfTreeHandle,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        output(out, 1)?;
        *out = ptr::null_mut();
        config(value)?;
        Err(error(
            BfTreeStatus::Unsupported,
            "existing-tree open is unsupported; use CPR recovery",
        ))
    })
}

/// # Safety
/// Pointers must satisfy the C header. The caller must own a validated native CPR
/// artifact exclusively. Use a private copy if the original must remain immutable.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_recover(
    value: *const BfTreeAbiConfig,
    snapshot_path: *const u8,
    snapshot_path_length: usize,
    completed_generation: u64,
    write_mode: u8,
    out: *mut *mut BfTreeHandle,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        output(out, 1)?;
        *out = ptr::null_mut();
        let value = config(value)?;
        if completed_generation == 0 || write_mode > 1 {
            return Err(invalid(
                "recovery needs a nonzero generation and a boolean writable flag",
            ));
        }
        let path = path(
            snapshot_path,
            snapshot_path_length,
            BfTreeStatus::InvalidArgument,
        )?;
        let metadata = fs::metadata(path)
            .map_err(|cause| invalid(format!("recovery file unavailable: {cause}")))?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(invalid("recovery requires a nonempty CPR file"));
        }
        let cache = if value.circular_buffer_bytes == 0 {
            None
        } else {
            Some(value.circular_buffer_bytes)
        };
        let tree = BfTree::new_from_cpr_snapshot(path, write_mode != 0, None, cache, None)
            .map_err(|cause| {
                error(
                    BfTreeStatus::InvalidConfig,
                    format!("CPR recovery failed: {cause:?}"),
                )
            })?;
        let stored = tree.config();
        for (requested, stored) in [
            (value.minimum_record_bytes, stored.get_cb_min_record_size()),
            (value.maximum_record_bytes, stored.get_cb_max_record_size()),
            (value.maximum_key_bytes, stored.get_max_key_len()),
            (value.leaf_page_bytes, stored.get_leaf_page_size()),
        ] {
            if requested != 0 && requested != stored {
                return Err(error(
                    BfTreeStatus::InvalidConfig,
                    "recovery limits disagree with the snapshot",
                ));
            }
        }
        *out = owned_handle(tree, write_mode != 0, completed_generation);
        Ok(BfTreeStatus::Ok)
    })
}

/// # Safety
/// Pointers must satisfy the C header. The handle must remain allocated for this call.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_insert(
    handle: *mut BfTreeHandle,
    key: *const u8,
    key_length: usize,
    value: *const u8,
    value_length: usize,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        let key = input(key, key_length)?;
        let value = input(value, value_length)?;
        with_tree(handle, false, |state| {
            writable(state)?;
            match state.tree.insert(key, value) {
                LeafInsertResult::Success => Ok(BfTreeStatus::Ok),
                LeafInsertResult::InvalidKV(message) => Err(invalid(message)),
            }
        })
    })
}

/// # Safety
/// Pointers must satisfy the C header. The handle must remain allocated for this call.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_read(
    handle: *mut BfTreeHandle,
    key: *const u8,
    key_length: usize,
    value: *mut u8,
    value_capacity: usize,
    value_length: *mut usize,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        output(value_length, 1)?;
        *value_length = 0;
        output(value, value_capacity)?;
        let key = input(key, key_length)?;
        let (status, bytes) = with_tree(handle, false, |state| {
            let mut bytes = buffer(state.maximum_record_bytes)?;
            match state.tree.read(key, &mut bytes) {
                LeafReadResult::Found(length) => {
                    bytes.truncate(length as usize);
                    Ok((BfTreeStatus::Ok, bytes))
                }
                LeafReadResult::NotFound => Ok((BfTreeStatus::NotFound, Vec::new())),
                LeafReadResult::Deleted => Ok((BfTreeStatus::Deleted, Vec::new())),
                LeafReadResult::InvalidKey => Err(invalid("invalid read key")),
            }
        })?;
        *value_length = bytes.len();
        if value_capacity < bytes.len() {
            return Ok(BfTreeStatus::BufferTooSmall);
        }
        if !bytes.is_empty() {
            ptr::copy_nonoverlapping(bytes.as_ptr(), value, bytes.len());
        }
        Ok(status)
    })
}

/// # Safety
/// Pointers must satisfy the C header. The handle must remain allocated for this call.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_delete(
    handle: *mut BfTreeHandle,
    key: *const u8,
    key_length: usize,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        let key = input(key, key_length)?;
        with_tree(handle, true, |state| {
            writable(state)?;
            let mut scratch = buffer(state.maximum_record_bytes)?;
            match state.tree.read(key, &mut scratch) {
                LeafReadResult::NotFound | LeafReadResult::Deleted => {
                    return Ok(BfTreeStatus::NotFound)
                }
                LeafReadResult::InvalidKey => return Err(invalid("invalid delete key")),
                LeafReadResult::Found(_) => {}
            }
            state.tree.delete(key);
            Ok(BfTreeStatus::Ok)
        })
    })
}

struct ScanRequest<'a> {
    lower: &'a [u8],
    upper: Option<&'a [u8]>,
    exclusive: bool,
    records: usize,
    bytes: usize,
    bounded: bool,
}

struct ScanOutput {
    bytes: Vec<u8>,
    records: Vec<BfTreeScanRecord>,
    required_bytes: usize,
}

fn scan(state: &TreeState, request: ScanRequest<'_>) -> ApiResult<ScanOutput> {
    if request.records == 0
        || request.records > MAX_SCAN_RECORDS
        || request.bytes == 0
        || request.bytes > MAX_SCAN_BYTES
        || request.lower.is_empty()
    {
        return Err(invalid("invalid scan bounds or limits"));
    }
    if request.lower.len() > state.maximum_key_bytes
        || request
            .upper
            .is_some_and(|upper| upper.len() > state.maximum_key_bytes)
    {
        return Err(invalid("scan bound exceeds the native key limit"));
    }
    if request.upper.is_some_and(|upper| request.lower > upper) {
        return Err(invalid("scan lower bound exceeds upper bound"));
    }
    let mut result = ScanOutput {
        bytes: Vec::new(),
        records: Vec::new(),
        required_bytes: 0,
    };
    if request.upper == Some(request.lower) {
        return Ok(result);
    }
    let mut iterator = state
        .tree
        .scan_with_count(
            request.lower,
            request.records + usize::from(request.exclusive),
            ScanReturnField::KeyAndValue,
        )
        .map_err(|cause| invalid(format!("native scan rejected request: {cause:?}")))?;
    let mut scratch = buffer(state.maximum_record_bytes)?;
    while result.records.len() < request.records {
        let Some((key_length, value_length)) = iterator.next(&mut scratch) else {
            break;
        };
        let key = &scratch[..key_length];
        if request.exclusive && key == request.lower {
            continue;
        }
        if request.upper.is_some_and(|upper| key >= upper) {
            break;
        }
        let length = key_length
            .checked_add(value_length)
            .ok_or_else(|| invalid("scan record length overflow"))?;
        let next = result
            .bytes
            .len()
            .checked_add(length)
            .ok_or_else(|| invalid("scan output length overflow"))?;
        if next > request.bytes {
            if !request.bounded {
                return Err(error(
                    BfTreeStatus::ResourceExhausted,
                    "scan exceeds the ABI byte limit",
                ));
            }
            if result.records.is_empty() {
                result.required_bytes = length;
            }
            break;
        }
        result.bytes.try_reserve(length).map_err(|cause| {
            error(
                BfTreeStatus::ResourceExhausted,
                format!("scan byte allocation failed: {cause}"),
            )
        })?;
        result.records.try_reserve(1).map_err(|cause| {
            error(
                BfTreeStatus::ResourceExhausted,
                format!("scan descriptor allocation failed: {cause}"),
            )
        })?;
        let key_offset = result.bytes.len();
        result.bytes.extend_from_slice(&scratch[..length]);
        result.records.push(BfTreeScanRecord {
            key_offset,
            key_length,
            value_offset: key_offset + key_length,
            value_length,
        });
    }
    Ok(result)
}

unsafe fn write_scan(
    result: ScanOutput,
    bytes: *mut u8,
    bytes_capacity: usize,
    bytes_length: *mut usize,
    records: *mut BfTreeScanRecord,
    records_capacity: usize,
    records_length: *mut usize,
) -> BfTreeStatus {
    if result.required_bytes != 0 {
        *bytes_length = result.required_bytes;
        *records_length = 1;
        return BfTreeStatus::BufferTooSmall;
    }
    *bytes_length = result.bytes.len();
    *records_length = result.records.len();
    if bytes_capacity < result.bytes.len() || records_capacity < result.records.len() {
        return BfTreeStatus::BufferTooSmall;
    }
    if !result.bytes.is_empty() {
        ptr::copy_nonoverlapping(result.bytes.as_ptr(), bytes, result.bytes.len());
    }
    if !result.records.is_empty() {
        ptr::copy_nonoverlapping(result.records.as_ptr(), records, result.records.len());
    }
    BfTreeStatus::Ok
}

unsafe fn scan_buffers(
    bytes: *mut u8,
    bytes_capacity: usize,
    bytes_length: *mut usize,
    records: *mut BfTreeScanRecord,
    records_capacity: usize,
    records_length: *mut usize,
) -> ApiResult<()> {
    output(bytes_length, 1)?;
    *bytes_length = 0;
    output(records_length, 1)?;
    *records_length = 0;
    output(bytes, bytes_capacity)?;
    output(records, records_capacity)
}

/// # Safety
/// Pointers must satisfy the C header. Output buffers must not overlap each other or inputs.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_scan(
    handle: *mut BfTreeHandle,
    start: *const u8,
    start_length: usize,
    maximum_records: usize,
    bytes: *mut u8,
    bytes_capacity: usize,
    bytes_length: *mut usize,
    records: *mut BfTreeScanRecord,
    records_capacity: usize,
    records_length: *mut usize,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        scan_buffers(
            bytes,
            bytes_capacity,
            bytes_length,
            records,
            records_capacity,
            records_length,
        )?;
        let request = ScanRequest {
            lower: input(start, start_length)?,
            upper: None,
            exclusive: false,
            records: maximum_records,
            bytes: MAX_SCAN_BYTES,
            bounded: false,
        };
        let result = with_tree(handle, false, |state| scan(state, request))?;
        Ok(write_scan(
            result,
            bytes,
            bytes_capacity,
            bytes_length,
            records,
            records_capacity,
            records_length,
        ))
    })
}

/// # Safety
/// Pointers must satisfy the C header. Output buffers must not overlap each other or inputs.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_scan_bounded_v1(
    handle: *mut BfTreeHandle,
    options: *const BfTreeScanOptionsV1,
    bytes: *mut u8,
    bytes_capacity: usize,
    bytes_length: *mut usize,
    records: *mut BfTreeScanRecord,
    records_capacity: usize,
    records_length: *mut usize,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        scan_buffers(
            bytes,
            bytes_capacity,
            bytes_length,
            records,
            records_capacity,
            records_length,
        )?;
        let options = reference(options)?;
        header(
            options.abi_version,
            options.struct_size,
            size_of::<BfTreeScanOptionsV1>(),
        )?;
        if options.lower_exclusive > 1 || options.reserved != 0 {
            return Err(invalid("invalid scan flags"));
        }
        let upper = input(options.upper, options.upper_length)?;
        let request = ScanRequest {
            lower: input(options.lower, options.lower_length)?,
            upper: if upper.is_empty() { None } else { Some(upper) },
            exclusive: options.lower_exclusive != 0,
            records: options.maximum_records,
            bytes: options.maximum_bytes,
            bounded: true,
        };
        let result = with_tree(handle, false, |state| scan(state, request))?;
        Ok(write_scan(
            result,
            bytes,
            bytes_capacity,
            bytes_length,
            records,
            records_capacity,
            records_length,
        ))
    })
}

/// # Safety
/// The capability and diagnostic structures must be valid and writable.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_get_capabilities_v1(
    capabilities: *mut BfTreeCapabilitiesV1,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        let current = reference(capabilities)?;
        header(
            current.abi_version,
            current.struct_size,
            size_of::<BfTreeCapabilitiesV1>(),
        )?;
        *capabilities = BfTreeCapabilitiesV1 {
            abi_version: ABI_VERSION,
            struct_size: size_of::<BfTreeCapabilitiesV1>() as u32,
            flags: CAPABILITIES,
            maximum_scan_records: MAX_SCAN_RECORDS,
            maximum_scan_bytes: MAX_SCAN_BYTES,
        };
        Ok(BfTreeStatus::Ok)
    })
}

/// # Safety
/// Pointers must satisfy the C header. The caller must exclusively own the output path.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_cpr_snapshot(
    handle: *mut BfTreeHandle,
    destination: *const u8,
    destination_length: usize,
    completed_generation: *mut u64,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        output(completed_generation, 1)?;
        *completed_generation = 0;
        let path = path(
            destination,
            destination_length,
            BfTreeStatus::InvalidArgument,
        )?;
        let owner = reference(handle)?;
        with_tree(handle, false, writable)?;
        let mut generation = owner.snapshot_generation.lock().map_err(|_| {
            owner.fenced.store(true, Ordering::Release);
            error(BfTreeStatus::InternalError, "snapshot lock is poisoned")
        })?;
        let next = generation.checked_add(1).ok_or_else(|| {
            error(
                BfTreeStatus::ResourceExhausted,
                "snapshot generation exhausted",
            )
        })?;
        let mut reservation = ReservedPath::reserve(path, false)?;
        let captured = with_tree(handle, false, |state| {
            writable(state)?;
            state.tree.cpr_snapshot(path);
            let metadata = fs::metadata(path).map_err(|cause| {
                owner.fenced.store(true, Ordering::Release);
                error(
                    BfTreeStatus::InternalError,
                    format!("snapshot output unavailable: {cause}"),
                )
            })?;
            if metadata.len() == 0 {
                owner.fenced.store(true, Ordering::Release);
                return Err(error(
                    BfTreeStatus::InternalError,
                    "snapshot output is empty",
                ));
            }
            Ok(())
        });
        reservation.finish(captured)?;
        *generation = next;
        *completed_generation = next;
        Ok(BfTreeStatus::Ok)
    })
}

/// # Safety
/// The handle must remain allocated. Destruction must not run concurrently.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_close(
    handle: *mut BfTreeHandle,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        let owner = reference(handle)?;
        let mut guard = owner
            .state
            .write()
            .map_err(|_| error(BfTreeStatus::InternalError, "handle lock is poisoned"))?;
        let state = guard.take().ok_or_else(closed)?;
        owner.fenced.store(true, Ordering::Release);
        drop(guard);
        drop(state);
        Ok(BfTreeStatus::Ok)
    })
}

/// # Safety
/// The caller must exclude all concurrent use and must not reuse copied handle pointers.
#[no_mangle]
pub unsafe extern "C" fn bf_tree_destroy(
    handle: *mut *mut BfTreeHandle,
    target: *mut BfTreeError,
) -> BfTreeStatus {
    ffi_call(target, || {
        output(handle, 1)?;
        if (*handle).is_null() {
            return Ok(BfTreeStatus::Ok);
        }
        let mut owner = Box::from_raw(*handle);
        *handle = ptr::null_mut();
        let state = owner
            .state
            .get_mut()
            .map_err(|_| error(BfTreeStatus::InternalError, "handle lock is poisoned"))?
            .take();
        drop(owner);
        drop(state);
        Ok(BfTreeStatus::Ok)
    })
}

#[cfg(test)]
mod tests;
