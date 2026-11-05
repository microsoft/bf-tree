// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "bf_tree.h"

#include <stdio.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

_Static_assert(sizeof(BfTreeStatus) == 4, "ABI status width changed");
_Static_assert(offsetof(BfTreeABIConfig, path) == 16, "ABI config prefix changed");
_Static_assert(sizeof(BfTreeScanRecord) == 4 * sizeof(size_t), "ABI scan layout changed");

int main(void) {
    BfTreeCapabilitiesV1 capabilities = {0};
    capabilities.abi_version = BF_TREE_ABI_VERSION;
    capabilities.struct_size = sizeof(capabilities);
    CHECK(bf_tree_abi_version() == 1);
    CHECK(bf_tree_get_capabilities_v1(&capabilities, NULL) == BF_TREE_OK);
    CHECK((capabilities.flags & BF_TREE_CAP_BOUNDED_SCAN_V1) != 0);
    CHECK((capabilities.flags & BF_TREE_CAP_PANIC_FENCING) != 0);
    CHECK((capabilities.flags & (BF_TREE_CAP_OPEN_EXISTING |
        BF_TREE_CAP_ATOMIC_BATCH | BF_TREE_CAP_NATIVE_IO_METRICS |
        BF_TREE_CAP_POWER_LOSS_DURABLE)) == 0);

    BfTreeABIConfig config = {0};
    config.abi_version = BF_TREE_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.storage = BF_TREE_STORAGE_MEMORY;
    config.circular_buffer_bytes = 2 * 1024 * 1024;
    config.minimum_record_bytes = 2;
    BfTreeHandle* handle = NULL;
    char message[128] = {0};
    BfTreeError error = {message, sizeof(message), 0};
    CHECK(bf_tree_open(&config, &handle, &error) == BF_TREE_UNSUPPORTED);
    CHECK(handle == NULL);
    CHECK(error.message_required > 1);
    CHECK(bf_tree_create(&config, &handle, &error) == BF_TREE_OK);
    CHECK(handle != NULL);

    CHECK(bf_tree_insert(handle, (const uint8_t*)"aa", 2,
        (const uint8_t*)"one", 3, &error) == BF_TREE_OK);
    CHECK(bf_tree_insert(handle, (const uint8_t*)"bb", 2,
        (const uint8_t*)"two", 3, &error) == BF_TREE_OK);
    CHECK(bf_tree_insert(handle, (const uint8_t*)"cc", 2,
        (const uint8_t*)"three", 5, &error) == BF_TREE_OK);

    size_t length = 0;
    CHECK(bf_tree_read(handle, (const uint8_t*)"aa", 2,
        NULL, 0, &length, &error) == BF_TREE_BUFFER_TOO_SMALL);
    CHECK(length == 3);
    uint8_t value[8] = {0};
    CHECK(bf_tree_read(handle, (const uint8_t*)"aa", 2,
        value, sizeof(value), &length, &error) == BF_TREE_OK);
    CHECK(length == 3 && memcmp(value, "one", 3) == 0);

    BfTreeScanOptionsV1 options = {0};
    options.abi_version = BF_TREE_ABI_VERSION;
    options.struct_size = sizeof(options);
    options.lower = (const uint8_t*)"aa";
    options.lower_length = 2;
    options.upper = (const uint8_t*)"cc";
    options.upper_length = 2;
    options.maximum_records = 3;
    options.maximum_bytes = 10;
    uint8_t bytes[32];
    BfTreeScanRecord records[3];
    size_t byte_count = 0;
    size_t record_count = 0;
    CHECK(bf_tree_scan_bounded_v1(handle, &options, bytes, sizeof(bytes),
        &byte_count, records, 3, &record_count, &error) == BF_TREE_OK);
    CHECK(byte_count == 10 && record_count == 2);
    CHECK(memcmp(bytes, "aaonebbtwo", 10) == 0);
    CHECK(records[1].key_offset == 5 && records[1].value_offset == 7);

    options.lower = (const uint8_t*)"bb";
    options.lower_exclusive = 1;
    CHECK(bf_tree_scan_bounded_v1(handle, &options, bytes, sizeof(bytes),
        &byte_count, records, 3, &record_count, &error) == BF_TREE_OK);
    CHECK(byte_count == 0 && record_count == 0);
    options.upper = NULL;
    options.upper_length = 0;
    CHECK(bf_tree_scan_bounded_v1(handle, &options, bytes, sizeof(bytes),
        &byte_count, records, 3, &record_count, &error) == BF_TREE_OK);
    CHECK(byte_count == 7 && record_count == 1);
    CHECK(memcmp(bytes, "ccthree", 7) == 0);

    options.lower = (const uint8_t*)"aa";
    options.lower_exclusive = 0;
    options.maximum_bytes = 4;
    memset(bytes, 0xA5, sizeof(bytes));
    memset(records, 0xA5, sizeof(records));
    uint8_t old_bytes[sizeof(bytes)];
    BfTreeScanRecord old_records[3];
    memcpy(old_bytes, bytes, sizeof(bytes));
    memcpy(old_records, records, sizeof(records));
    CHECK(bf_tree_scan_bounded_v1(handle, &options, bytes, sizeof(bytes),
        &byte_count, records, 3, &record_count, &error) == BF_TREE_BUFFER_TOO_SMALL);
    CHECK(byte_count == 5 && record_count == 1);
    CHECK(memcmp(bytes, old_bytes, sizeof(bytes)) == 0);
    CHECK(memcmp(records, old_records, sizeof(records)) == 0);

    CHECK(bf_tree_delete(handle, (const uint8_t*)"aa", 2, &error) == BF_TREE_OK);
    CHECK(bf_tree_delete(handle, (const uint8_t*)"aa", 2, &error) == BF_TREE_NOT_FOUND);
    CHECK(bf_tree_close(handle, &error) == BF_TREE_OK);
    CHECK(bf_tree_read(handle, (const uint8_t*)"bb", 2,
        value, sizeof(value), &length, &error) == BF_TREE_CLOSED);
    CHECK(bf_tree_destroy(&handle, &error) == BF_TREE_OK);
    CHECK(handle == NULL);
    CHECK(bf_tree_destroy(&handle, &error) == BF_TREE_OK);
    return 0;
}
