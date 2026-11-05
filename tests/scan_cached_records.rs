// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use bf_tree::{BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField, StorageBackend};

#[test]
fn scans_load_base_pages_after_merging_clean_cached_records() {
    for scan_promotion in [0, 100] {
        let mut config = Config::new(":memory:", 1024 * 1024);
        config
            .storage_backend(StorageBackend::Memory)
            .cb_min_record_size(4)
            .read_promotion_rate(100)
            .scan_promotion_rate(scan_promotion);
        let tree = BfTree::with_config(config, None).unwrap();
        let value = [0x5A; 512];

        for index in 0_u64..4096 {
            assert_eq!(
                tree.insert(&index.to_be_bytes(), &value),
                LeafInsertResult::Success
            );
        }
        let mut output = [0; 520];
        for index in 0_u64..4096 {
            assert_eq!(
                tree.read(&index.to_be_bytes(), &mut output),
                LeafReadResult::Found(512)
            );
            assert_eq!(&output[..512], &value);
        }

        let mut scan = tree
            .scan_with_count(&0_u64.to_be_bytes(), 4096, ScanReturnField::KeyAndValue)
            .unwrap();
        for index in 0_u64..4096 {
            assert_eq!(scan.next(&mut output), Some((8, 512)));
            assert_eq!(&output[..8], &index.to_be_bytes());
            assert_eq!(&output[8..], &value);
        }
        assert_eq!(scan.next(&mut output), None);
    }
}
