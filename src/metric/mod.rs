// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use serde::Serialize;
use std::cell::UnsafeCell;
use std::ops::AddAssign;

pub(crate) mod counter;
pub(crate) mod histogram;
pub mod recorders;
pub mod timer;

#[allow(unused_imports)]
pub use crate::metric::counter::Counter;
#[allow(unused_imports)]
pub use crate::metric::histogram::Histogram;
#[allow(unused_imports)]
pub use crate::metric::timer::Timer;

pub use recorders::TlsRecorder;
pub use timer::DebugTimerGuard;

thread_local! {
    static LOCAL_RECORDER: UnsafeCell<TlsRecorder> = UnsafeCell::new(TlsRecorder::default());
}

pub trait RecorderImpl: Serialize + AddAssign + Sized {
    fn reset(&mut self);
}

pub fn get_tls_recorder() -> &'static mut TlsRecorder {
    LOCAL_RECORDER.with(|id| unsafe { &mut *id.get() })
}
