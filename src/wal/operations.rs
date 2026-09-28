// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::nodes::leaf_node::OpType;

use super::LogEntryImpl;

pub(crate) struct WriteOp<'a> {
    pub(crate) key: &'a [u8],
    pub(crate) value: &'a [u8],
    pub(crate) op_type: OpType,
}

impl<'a> WriteOp<'a> {
    pub(crate) fn make_insert(key: &'a [u8], value: &'a [u8]) -> WriteOp<'a> {
        WriteOp {
            key,
            value,
            op_type: OpType::Insert,
        }
    }

    pub(crate) fn make_delete(key: &'a [u8]) -> WriteOp<'a> {
        WriteOp {
            key,
            value: &[],
            op_type: OpType::Delete,
        }
    }

    pub(crate) fn try_read_from_buffer(buffer: &'a [u8]) -> Option<WriteOp<'a>> {
        const HEADER_SIZE: usize = 5;
        if buffer.len() < HEADER_SIZE {
            return None;
        }

        let key_size = u16::from_le_bytes(buffer[0..2].try_into().ok()?) as usize;
        let value_size = u16::from_le_bytes(buffer[2..4].try_into().ok()?) as usize;
        let expected_size = HEADER_SIZE.checked_add(key_size)?.checked_add(value_size)?;
        if buffer.len() != expected_size {
            return None;
        }

        let op_type = match buffer[4] {
            0 => OpType::Insert,
            1 => OpType::Delete,
            _ => return None,
        };
        let value_offset = HEADER_SIZE + key_size;
        Some(WriteOp {
            key: &buffer[HEADER_SIZE..value_offset],
            value: &buffer[value_offset..],
            op_type,
        })
    }
}

impl<'a> LogEntryImpl<'a> for WriteOp<'a> {
    fn log_size(&self) -> usize {
        // layout:
        // key_size | value_size | op_type | key | value
        let key_size = self.key.len();
        let value_size = self.value.len();
        let op_type_size = std::mem::size_of::<OpType>();

        std::mem::size_of::<u16>()
            + std::mem::size_of::<u16>()
            + op_type_size
            + key_size
            + value_size
    }

    fn write_to_buffer(&self, buffer: &mut [u8]) {
        debug_assert_eq!(buffer.len(), self.log_size());
        let key_len = self.key.len() as u16;
        let val_len = self.value.len() as u16;
        let op_u8 = self.op_type as u8;
        buffer[0..2].copy_from_slice(key_len.to_le_bytes().as_ref());
        buffer[2..4].copy_from_slice(val_len.to_le_bytes().as_ref());
        buffer[4] = op_u8;

        let key_offset = 5 + self.key.len();
        buffer[5..key_offset].copy_from_slice(self.key);
        buffer[key_offset..].copy_from_slice(self.value);
    }

    #[cfg(test)]
    fn read_from_buffer(buffer: &'a [u8]) -> WriteOp<'a> {
        Self::try_read_from_buffer(buffer).expect("invalid WAL write operation")
    }
}
