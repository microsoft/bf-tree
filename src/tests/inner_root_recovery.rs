// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::{
    utils::{BfsVisitor, NodeInfo},
    BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField,
};
use std::collections::BTreeMap;

fn check_state(tree: &BfTree, model: &BTreeMap<Vec<u8>, Vec<u8>>) {
    let mut buffer = vec![0u8; 1024];
    for (key, value) in model {
        assert_eq!(
            tree.read(key, &mut buffer),
            LeafReadResult::Found(value.len() as u32)
        );
        assert_eq!(&buffer[..value.len()], value);
    }
    let mut scan = tree
        .scan_with_count(
            &0u64.to_be_bytes(),
            usize::MAX,
            ScanReturnField::KeyAndValue,
        )
        .unwrap();
    for (key, value) in model {
        let (key_len, value_len) = scan.next(&mut buffer).expect("missing scan record");
        assert_eq!(&buffer[..key_len], key);
        assert_eq!(&buffer[key_len..key_len + value_len], value);
    }
    assert!(scan.next(&mut buffer).is_none(), "extra scan record");
}

#[test]
fn inner_root_split_preserves_complete_state_through_recovery() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = Config::new(directory.path().join("live"), 64 * 1024 * 1024);
    config
        .leaf_page_size(4096)
        .cb_max_key_len(8)
        .cb_max_record_size(1024)
        .read_promotion_rate(0)
        .scan_promotion_rate(0)
        .use_snapshot(true);
    let mut tree = BfTree::with_config(config, None).unwrap();
    let mut model = BTreeMap::new();
    for round in 0u64..3 {
        let inner_count_before = BfsVisitor::new_inner_only(&tree).count();
        for id in round * 6000..(round + 1) * 6000 {
            let key = id.to_be_bytes().to_vec();
            let value = key.repeat(64);
            assert_eq!(tree.insert(&key, &value), LeafInsertResult::Success);
            model.insert(key, value);
        }
        assert!(
            BfsVisitor::new_all_nodes(&tree)
                .any(|node| matches!(node, NodeInfo::Leaf { level, .. } if level >= 2)),
            "must split an inner root, not only the original leaf root"
        );
        assert!(
            BfsVisitor::new_inner_only(&tree).count() > inner_count_before,
            "each round must split inner nodes, including after recovery"
        );
        check_state(&tree, &model);
        let snapshot = directory.path().join(format!("snapshot-{round}"));
        tree.cpr_snapshot(&snapshot);
        let image = std::fs::read(&snapshot).unwrap();

        // Recovery writes to its working file; the sealed snapshot is immutable.
        let working = directory.path().join(format!("recovered-{round}"));
        std::fs::copy(&snapshot, &working).unwrap();
        drop(tree);
        tree = BfTree::new_from_cpr_snapshot(&working, true, None, None, None).unwrap();
        check_state(&tree, &model);
        assert_eq!(std::fs::read(&snapshot).unwrap(), image);
    }
}
