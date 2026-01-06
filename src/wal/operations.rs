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

    fn read_from_buffer(buffer: &'a [u8]) -> WriteOp<'a> {
        let key_size = u16::from_le_bytes(buffer[0..2].try_into().unwrap()) as usize;
        let value_size = u16::from_le_bytes(buffer[2..4].try_into().unwrap()) as usize;
        let op_type = unsafe { std::mem::transmute::<u8, OpType>(buffer[4]) };
        let key = &buffer[5..5 + key_size];
        let value = &buffer[5 + key_size..];
        assert_eq!(value.len(), value_size);
        WriteOp {
            key,
            value,
            op_type,
        }
    }
}

pub(crate) enum LogEntry<'a> {
    Write(WriteOp<'a>),
    Split(SplitOp),
}

#[repr(u8)]
enum LogEntryTagVal {
    Write = 0,
    Split = 1,
}

impl LogEntryTagVal {
    fn size() -> usize {
        std::mem::size_of::<u8>()
    }
}

impl<'a> From<&LogEntry<'a>> for LogEntryTagVal {
    fn from(entry: &LogEntry<'a>) -> Self {
        match entry {
            LogEntry::Write(_) => LogEntryTagVal::Write,
            LogEntry::Split(_) => LogEntryTagVal::Split,
        }
    }
}

impl From<u8> for LogEntryTagVal {
    fn from(val: u8) -> Self {
        match val {
            0 => LogEntryTagVal::Write,
            1 => LogEntryTagVal::Split,
            _ => unreachable!(),
        }
    }
}

pub(crate) struct SplitOp {}

impl<'a> LogEntryImpl<'a> for LogEntry<'a> {
    fn log_size(&self) -> usize {
        let tag_size = LogEntryTagVal::size();
        let data_size = match self {
            LogEntry::Write(op) => op.log_size(),
            LogEntry::Split(_) => todo!(),
        };
        tag_size + data_size
    }

    fn write_to_buffer(&self, buffer: &mut [u8]) {
        buffer[0] = LogEntryTagVal::from(self) as u8;
        match self {
            LogEntry::Write(op) => op.write_to_buffer(&mut buffer[1..]),
            LogEntry::Split(_) => {
                todo!()
            }
        }
    }

    fn read_from_buffer(buffer: &'a [u8]) -> Self {
        let tag = LogEntryTagVal::from(buffer[0]);
        match tag {
            LogEntryTagVal::Write => LogEntry::Write(WriteOp::read_from_buffer(&buffer[1..])),
            LogEntryTagVal::Split => LogEntry::Split(SplitOp {}),
        }
    }
}
