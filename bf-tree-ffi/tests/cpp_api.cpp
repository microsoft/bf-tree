// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef BF_TREE_HEADER
#define BF_TREE_HEADER "bf_tree.h"
#endif
#include BF_TREE_HEADER

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << #expression << '\n'; \
        return 1; \
    } \
} while (0)

static_assert(sizeof(BfTreeStatus) == 4);
static_assert(offsetof(BfTreeABIConfig, path) == 16);
static_assert(sizeof(BfTreeScanRecord) == 4 * sizeof(size_t));

class Handle {
public:
    ~Handle() {
        if (bf_tree_destroy(&value, nullptr) != BF_TREE_OK) {
            std::abort();
        }
    }
    Handle() = default;
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    BfTreeHandle* value = nullptr;
};

int main() {
    BfTreeABIConfig config{};
    config.abi_version = BF_TREE_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.circular_buffer_bytes = 2 * 1024 * 1024;
    config.minimum_record_bytes = 2;
    Handle handle;
    CHECK(bf_tree_create(&config, &handle.value, nullptr) == BF_TREE_OK);
    const auto* key = reinterpret_cast<const uint8_t*>("key");
    const auto* value = reinterpret_cast<const uint8_t*>("value");
    CHECK(bf_tree_insert(handle.value, key, 3, value, 5, nullptr) == BF_TREE_OK);

    std::array<uint8_t, 8> bytes{};
    std::array<BfTreeScanRecord, 1> records{};
    size_t byte_count = 0;
    size_t record_count = 0;
    CHECK(bf_tree_scan(handle.value, key, 3, 1, bytes.data(), bytes.size(),
        &byte_count, records.data(), records.size(), &record_count, nullptr) == BF_TREE_OK);
    CHECK(byte_count == 8 && record_count == 1);
    CHECK(std::memcmp(bytes.data(), "keyvalue", 8) == 0);
    CHECK(records[0].key_length == 3 && records[0].value_offset == 3);

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
        ("bf-tree-ffi-cpp-" + std::to_string(unique));
    CHECK(std::filesystem::create_directory(directory));
    const auto snapshot = directory / "snapshot.cpr";
    const auto recovery = directory / "private.cpr";
    const auto snapshot_text = snapshot.u8string();
    const auto recovery_text = recovery.u8string();
    uint64_t generation = 0;
    CHECK(bf_tree_cpr_snapshot(handle.value,
        reinterpret_cast<const uint8_t*>(snapshot_text.data()), snapshot_text.size(),
        &generation, nullptr) == BF_TREE_OK);
    CHECK(generation == 1);
    CHECK(std::filesystem::copy_file(snapshot, recovery));
    {
        Handle restored;
        CHECK(bf_tree_recover(&config,
            reinterpret_cast<const uint8_t*>(recovery_text.data()), recovery_text.size(),
            generation, 0, &restored.value, nullptr) == BF_TREE_OK);
        std::array<uint8_t, 5> output{};
        size_t length = 0;
        CHECK(bf_tree_read(restored.value, key, 3, output.data(), output.size(),
            &length, nullptr) == BF_TREE_OK);
        CHECK(length == 5 && std::memcmp(output.data(), "value", 5) == 0);
        CHECK(bf_tree_insert(restored.value, key, 3, value, 5, nullptr) == BF_TREE_UNSUPPORTED);
    }
    CHECK(std::filesystem::remove(recovery));
    CHECK(std::filesystem::remove(snapshot));
    CHECK(std::filesystem::remove(directory));

    std::atomic<unsigned> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> readers;
    for (unsigned index = 0; index < 4; ++index) {
        readers.emplace_back([&] {
            ready.fetch_add(1);
            while (!start.load()) {
                std::this_thread::yield();
            }
            for (unsigned iteration = 0; iteration < 1000; ++iteration) {
                std::array<uint8_t, 5> output{};
                size_t length = 0;
                const auto status = bf_tree_read(handle.value, key, 3,
                    output.data(), output.size(), &length, nullptr);
                if (status == BF_TREE_CLOSED) {
                    return;
                }
                if (status != BF_TREE_OK || length != 5 ||
                    std::memcmp(output.data(), "value", 5) != 0) {
                    failed.store(true);
                    return;
                }
            }
        });
    }
    while (ready.load() != 4) {
        std::this_thread::yield();
    }
    start.store(true);
    const auto closed = bf_tree_close(handle.value, nullptr);
    for (auto& reader : readers) {
        reader.join();
    }
    CHECK(closed == BF_TREE_OK && !failed.load());
    CHECK(bf_tree_close(handle.value, nullptr) == BF_TREE_CLOSED);
    CHECK(bf_tree_insert(handle.value, key, 3, value, 5, nullptr) == BF_TREE_CLOSED);
    return 0;
}
