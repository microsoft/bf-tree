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
}
