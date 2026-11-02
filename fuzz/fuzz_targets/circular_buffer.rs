// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| bf_tree_fuzz::circular_buffer(data));
