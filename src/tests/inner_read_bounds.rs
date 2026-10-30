// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::mem::{size_of, size_of_val};

use crate::{
    error::TreeError,
    nodes::{InnerNode, InnerNodeBuilder, PageID},
    snapshot::INVALID_SNAPSHOT_VERSION,
    sync::atomic::Ordering,
    utils::inner_lock::ReadGuard,
};

struct TestNode(*mut InnerNode);

impl TestNode {
    fn new() -> Self {
        let mut builder = InnerNodeBuilder::new();
        builder
            .set_children_is_leaf(true)
            .set_left_most_page_id(PageID::from_id(0))
            .add_record(b"same".to_vec(), PageID::from_id(1))
            .add_record(b"same-longer".to_vec(), PageID::from_id(2))
            .add_record(vec![b'z'; 256], PageID::from_id(3));
        let ptr = builder.build(INVALID_SNAPSHOT_VERSION);
        // Boundary cases also read unused, but in-bounds, payload bytes.
        unsafe {
            let node = &*ptr;
            let start =
                node.meta.meta_count_with_fence() as usize * size_of_val(node.get_kv_meta(0));
            let end = node.current_lowest_offset() as usize;
            ptr.cast::<u8>()
                .add(size_of::<InnerNode>() + start)
                .write_bytes(0, end - start);
        }
        Self(ptr)
    }

    fn read(&self) -> ReadGuard<'_> {
        ReadGuard::try_read(self.0).unwrap()
    }

    fn set_span(&mut self, index: u16, offset: u16, key_len: u16) {
        let mut meta = unsafe { *(*self.0).get_kv_meta(index) };
        let slot = size_of::<InnerNode>() + index as usize * size_of_val(&meta);
        meta.offset = offset;
        meta.key_len = key_len;
        unsafe {
            std::ptr::write(self.0.cast::<u8>().add(slot).cast(), meta);
        }
    }
}

impl Drop for TestNode {
    fn drop(&mut self) {
        InnerNode::free_node(self.0);
    }
}

#[test]
fn inner_read_bounds_preserve_routing_and_fences() {
    let node = TestNode::new();
    let reader = node.read();
    for key in [
        b"".as_slice(),
        b"s",
        b"same",
        b"same-",
        b"same-longer",
        b"same-longest",
        b"zzzz",
        &[b'z'; 257],
    ] {
        let pos = reader.lower_bound(key).unwrap();
        assert_eq!(pos as u64, reader.as_ref().lower_bound(key));
        for index in pos..std::cmp::min(pos + 2, reader.checked_record_count().unwrap()) {
            let record = reader.checked_record(index).unwrap();
            let meta = reader.as_ref().get_kv_meta(index);
            assert_eq!(record.full_key(), reader.as_ref().get_full_key(meta));
            assert_eq!(record.value(), reader.as_ref().get_value(meta));
        }
    }
    let (is_leaf, children) = reader.checked_children().unwrap();
    assert!(is_leaf);
    assert_eq!(children, (0..4).map(PageID::from_id).collect::<Vec<_>>());
    let stats = reader.get_stats().unwrap();
    assert_eq!(stats.child_id, children);
    assert_eq!(stats.child_keys[2], b"same-longer");
}

#[rstest::rstest]
#[case(8, 4, true)]
#[case(9, 5, true)]
#[case(8, 5, false)]
#[case(7, 4, false)]
#[case(0, 0, false)]
fn inner_read_bounds_payload_end(
    #[case] bytes_from_end: u16,
    #[case] key_len: u16,
    #[case] valid: bool,
) {
    let mut node = TestNode::new();
    node.set_span(
        1,
        InnerNode::max_data_size() as u16 - bytes_from_end,
        key_len,
    );
    let reader = node.read();
    let record = reader.checked_record(1);
    assert_eq!(record.is_ok(), valid);
    if let Ok(record) = record {
        assert_eq!(record.full_key().len(), key_len as usize);
        let _ = record.value();
    }
}

#[test]
fn inner_read_bounds_reject_mixed_valid_metadata() {
    let mut node = TestNode::new();
    let (offset, key_len) = {
        let reader = node.read();
        (
            reader.as_ref().get_kv_meta(1).offset,
            reader.as_ref().get_kv_meta(3).key_len,
        )
    };
    assert!(offset as usize + key_len as usize - 4 + 8 > InnerNode::max_data_size());
    node.set_span(1, offset, key_len);
    let reader = node.read();
    assert!(matches!(reader.checked_record(1), Err(TreeError::Locked)));
    assert!(matches!(
        reader.lower_bound(b"same"),
        Err(TreeError::Locked)
    ));
    assert!(matches!(reader.checked_children(), Err(TreeError::Locked)));
    assert!(matches!(reader.get_stats(), Err(TreeError::Locked)));
}

#[rstest::rstest]
#[case(0)]
#[case(4095)]
fn inner_read_bounds_reject_invalid_count(#[case] count: u16) {
    let node = TestNode::new();
    unsafe { (*node.0).meta.set_value_count(count) };
    let reader = node.read();
    assert!(matches!(
        reader.lower_bound(b"same"),
        Err(TreeError::Locked)
    ));
    assert!(matches!(reader.checked_children(), Err(TreeError::Locked)));
    assert!(matches!(reader.get_stats(), Err(TreeError::Locked)));
}

#[test]
fn inner_read_bounds_reject_metadata_outside_allocation() {
    let node = TestNode::new();
    let reader = node.read();
    let first_outside =
        (InnerNode::max_data_size() / size_of_val(reader.as_ref().get_kv_meta(0))) as u16;
    for index in [first_outside, first_outside + 1, u16::MAX] {
        assert!(matches!(
            reader.checked_record(index),
            Err(TreeError::Locked)
        ));
    }
}

#[test]
fn inner_read_bounds_capture_metadata_once() {
    let node = TestNode::new();
    let reader = node.read();
    let record = reader.checked_record(2).unwrap();
    let writer = node.read().upgrade().unwrap();
    let mut meta = *writer.as_ref().get_kv_meta(2);
    let slot = size_of::<InnerNode>() + 2 * size_of_val(&meta);
    meta.offset = u16::MAX;
    meta.key_len = u16::MAX;
    unsafe { std::ptr::write(node.0.cast::<u8>().add(slot).cast(), meta) };
    assert_eq!(record.full_key(), b"same-longer");
    assert_eq!(record.value(), PageID::from_id(2));
    drop(writer);
    assert!(matches!(reader.check_version(), Err(TreeError::Locked)));
}

#[test]
fn inner_read_bounds_collections_reject_changed_version() {
    let node = TestNode::new();
    let reader = node.read();
    reader.as_ref().version_lock.fetch_add(4, Ordering::Release);
    assert!(matches!(reader.checked_children(), Err(TreeError::Locked)));
    assert!(matches!(reader.get_stats(), Err(TreeError::Locked)));
}
