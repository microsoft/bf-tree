// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[derive(Debug)]
pub(crate) enum TreeError {
    Locked,
    CircularBufferFull,
    NeedRestart, // need to restart the operation, potentially will do SMO operations
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ConfigError {
    MinimumRecordSize(String),
    MaximumRecordSize(String),
    LeafPageSize(String),
    MaxKeyLen(String),
    CircularBufferSize(String),
    SnapshotFileInvalid(String),
    SnapshotDisabled,
}

/// The output buffer cannot hold the selected record fields.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BufferTooSmall {
    pub required: usize,
}

impl std::fmt::Display for BufferTooSmall {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "output buffer requires {} bytes", self.required)
    }
}

impl std::error::Error for BufferTooSmall {}

/// A snapshot request did not start.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SnapshotError {
    Disabled,
    InProgress,
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Disabled => "CPR snapshots are disabled",
            Self::InProgress => "another CPR snapshot is in progress",
        })
    }
}

impl std::error::Error for SnapshotError {}
