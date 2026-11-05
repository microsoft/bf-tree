// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BF_TREE_ABI_VERSION 1u

typedef struct BfTreeHandle BfTreeHandle;

typedef enum BfTreeStatus {
    BF_TREE_OK = 0,
    BF_TREE_NOT_FOUND = 1,
    BF_TREE_DELETED = 2,
    BF_TREE_BUFFER_TOO_SMALL = 3,
    BF_TREE_INVALID_ARGUMENT = 4,
    BF_TREE_INVALID_CONFIG = 5,
    BF_TREE_CLOSED = 6,
    BF_TREE_UNSUPPORTED = 7,
    BF_TREE_PANIC = 8,
    BF_TREE_INTERNAL_ERROR = 9,
    BF_TREE_ALREADY_EXISTS = 10,
    BF_TREE_RESOURCE_EXHAUSTED = 11
} BfTreeStatus;

typedef enum BfTreeStorage {
    BF_TREE_STORAGE_MEMORY = 0,
    BF_TREE_STORAGE_STANDARD = 1
} BfTreeStorage;

typedef struct BfTreeError {
    char* message;
    size_t message_capacity;
    size_t message_required;
} BfTreeError;

typedef struct BfTreeABIConfig {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t storage;
    uint32_t reserved;
    const uint8_t* path;
    size_t path_length;
    size_t circular_buffer_bytes;
    size_t minimum_record_bytes;
    size_t maximum_record_bytes;
    size_t maximum_key_bytes;
    size_t leaf_page_bytes;
} BfTreeABIConfig;

typedef struct BfTreeScanRecord {
    size_t key_offset;
    size_t key_length;
    size_t value_offset;
    size_t value_length;
} BfTreeScanRecord;

uint32_t bf_tree_abi_version(void);
BfTreeStatus bf_tree_create(
    const BfTreeABIConfig* config, BfTreeHandle** out_handle, BfTreeError* error);
BfTreeStatus bf_tree_open(
    const BfTreeABIConfig* config, BfTreeHandle** out_handle, BfTreeError* error);
BfTreeStatus bf_tree_recover(
    const BfTreeABIConfig* config,
    const uint8_t* snapshot_path, size_t snapshot_path_length,
    uint64_t completed_generation, uint8_t writable,
    BfTreeHandle** out_handle, BfTreeError* error);
BfTreeStatus bf_tree_insert(
    BfTreeHandle* handle, const uint8_t* key, size_t key_length,
    const uint8_t* value, size_t value_length, BfTreeError* error);
BfTreeStatus bf_tree_read(
    BfTreeHandle* handle, const uint8_t* key, size_t key_length,
    uint8_t* value, size_t value_capacity, size_t* value_length, BfTreeError* error);
BfTreeStatus bf_tree_scan(
    BfTreeHandle* handle, const uint8_t* start_key, size_t start_key_length,
    size_t maximum_records,
    uint8_t* bytes, size_t bytes_capacity, size_t* bytes_length,
    BfTreeScanRecord* records, size_t records_capacity, size_t* records_length,
    BfTreeError* error);
BfTreeStatus bf_tree_delete(
    BfTreeHandle* handle, const uint8_t* key, size_t key_length, BfTreeError* error);
BfTreeStatus bf_tree_cpr_snapshot(
    BfTreeHandle* handle, const uint8_t* path, size_t path_length,
    uint64_t* completed_generation, BfTreeError* error);
BfTreeStatus bf_tree_close(BfTreeHandle* handle, BfTreeError* error);
BfTreeStatus bf_tree_destroy(BfTreeHandle** handle, BfTreeError* error);

#define BF_TREE_CAP_CREATE_MEMORY       (UINT64_C(1) << 0)
#define BF_TREE_CAP_CREATE_STANDARD     (UINT64_C(1) << 1)
#define BF_TREE_CAP_READ                (UINT64_C(1) << 2)
#define BF_TREE_CAP_SCAN                (UINT64_C(1) << 3)
#define BF_TREE_CAP_INSERT              (UINT64_C(1) << 4)
#define BF_TREE_CAP_DELETE              (UINT64_C(1) << 5)
#define BF_TREE_CAP_CPR_SNAPSHOT        (UINT64_C(1) << 6)
#define BF_TREE_CAP_CPR_RECOVERY        (UINT64_C(1) << 7)
#define BF_TREE_CAP_BOUNDED_SCAN_V1     (UINT64_C(1) << 8)
#define BF_TREE_CAP_PANIC_FENCING       (UINT64_C(1) << 9)
#define BF_TREE_CAP_OPEN_EXISTING       (UINT64_C(1) << 10)
#define BF_TREE_CAP_ATOMIC_BATCH        (UINT64_C(1) << 11)
#define BF_TREE_CAP_NATIVE_IO_METRICS   (UINT64_C(1) << 12)
#define BF_TREE_CAP_POWER_LOSS_DURABLE  (UINT64_C(1) << 13)

// Package capabilities, not permission to operate on a closed or read-only handle.
// Initialize abi_version and struct_size. Errors leave this structure unchanged.
typedef struct BfTreeCapabilitiesV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t flags;
    size_t maximum_scan_records;
    size_t maximum_scan_bytes;
} BfTreeCapabilitiesV1;

BfTreeStatus bf_tree_get_capabilities_v1(
    BfTreeCapabilitiesV1* capabilities, BfTreeError* error);

// The range is [lower, upper), or (lower, upper) when lower_exclusive is one.
// A zero upper_length means no upper bound. An empty lower key is invalid.
// Limits bound staged output, not native I/O or traversal time.
typedef struct BfTreeScanOptionsV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint8_t* lower;
    size_t lower_length;
    const uint8_t* upper;
    size_t upper_length;
    size_t maximum_records;
    size_t maximum_bytes;
    uint32_t lower_exclusive;
    uint32_t reserved;
} BfTreeScanOptionsV1;

// Returns a complete-record prefix. Continue with an exclusive last-returned key.
// An oversized first record returns BUFFER_TOO_SMALL with its required sizes.
// Caller-capacity errors write no byte/descriptor output. Sizing retries are fresh
// scans and can observe mutations. Stable pagination requires a stable input view.
BfTreeStatus bf_tree_scan_bounded_v1(
    BfTreeHandle* handle, const BfTreeScanOptionsV1* options,
    uint8_t* bytes, size_t bytes_capacity, size_t* bytes_length,
    BfTreeScanRecord* records, size_t records_capacity, size_t* records_length,
    BfTreeError* error);

// All pointers must remain valid, aligned, and appropriately non-overlapping.
// Input memory is borrowed for one call. Output memory remains caller-owned.
// A handle panic returns PANIC and fences further data operations with CLOSED.
// close waits for active calls; destroy requires exclusive caller ownership.
// Recovery can write its input file. Use an exclusively owned private copy to
// preserve an immutable checkpoint. Neither close nor capture proves durability.

#ifdef __cplusplus
}
#endif
