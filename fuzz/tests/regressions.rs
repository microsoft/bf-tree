// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[test]
fn tree_split_overwrite_delete_recovery_seeds() {
    for policy in [0, 1] {
        let mut data = vec![policy];
        data.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0]);
        for id in 0_u16..192 {
            data.extend_from_slice(&[0, id as u8, (id >> 8) as u8, 8, 1, 2, 3, 4]);
        }
        for size in [0, 8, 1] {
            data.extend_from_slice(&[0, 1, 0, size, 7, 9, 11, 13]);
        }
        data.extend_from_slice(&[2, 2, 0, 0, 0, 0, 0, 0]);
        data.extend_from_slice(&[3, 0, 0, 255, 0, 0, 0, 0]);
        data.extend_from_slice(&[4, 0, 0, 0, 191, 0, 0, 0]);
        bf_tree_fuzz::tree(&data);
    }
}

#[test]
fn circular_buffer_wrap_reuse_boundary_seeds() {
    let mut data = Vec::new();
    for tag in 0..64 {
        data.extend_from_slice(&[0, 7, tag, 2, 0, 0]);
    }
    data.extend_from_slice(&[0, 5, 42, 1, 0, 0, 0, 5, 99]);
    for size in 0..8 {
        data.extend_from_slice(&[0, size, size]);
    }
    bf_tree_fuzz::circular_buffer(&data);
}
