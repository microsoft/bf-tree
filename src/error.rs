#[derive(Debug)]
pub(crate) enum TreeError {
    Locked,
    CircularBufferFull,
    NeedRestart, // need to restart the operation, potentially will do SMO operations
}
