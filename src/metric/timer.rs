// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::{metric::recorders::TlsRecorder, metric::RecorderImpl};
use chrono::{Datelike, Timelike};
use core::array;
use hdrhistogram::serialization::Serializer as _;
use serde::{ser::SerializeStruct, Serialize, Serializer};
use std::fmt;
use std::io::Write as _;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use std::time::Instant;
use std::{cell::UnsafeCell, sync::Arc};
use thread_local::ThreadLocal;
use variant_count::VariantCount;

#[repr(u8)]
#[derive(Clone, Debug, VariantCount)]
pub enum Timer {
    Read = 0,
}

impl fmt::Display for Timer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Timer::Read => write!(f, "Read"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TimerRecorder {
    timers: [hdrhistogram::Histogram<u64>; Timer::VARIANT_COUNT],
}

impl Default for TimerRecorder {
    fn default() -> Self {
        let mut timers = array::from_fn(|_| {
            hdrhistogram::Histogram::<u64>::new_with_bounds(1, 100_000_000, 3).unwrap()
        });

        for t in timers.iter_mut() {
            t.auto(true);
        }

        TimerRecorder { timers }
    }
}

impl TimerRecorder {
    pub fn start(&self, event: Timer) -> TimerGuard {
        TimerGuard::new(event)
    }

    pub(crate) fn add_time(&mut self, event: Timer, time: Duration) {
        unsafe {
            self.timers
                .get_unchecked_mut(event as usize)
                .record(time.as_nanos() as u64)
                .unwrap();
        }
    }
}

impl RecorderImpl for TimerRecorder {
    fn reset(&mut self) {
        for timer in self.timers.iter_mut() {
            timer.reset();
        }
    }
}

impl_op_ex!(+= |a: &mut TimerRecorder, b: &TimerRecorder| {
    for(i, t) in a.timers.iter_mut().enumerate(){
        t.add(b.timers[i].clone()).unwrap();
    }
});

impl_op_ex!(+ |a: &TimerRecorder, b: &TimerRecorder| -> TimerRecorder{
    let mut c_a = a.clone();
    c_a += b;
    c_a
});

const QUANTILE: [(&str, f64); 11] = [
    ("15th", 0.15),
    ("30th", 0.30),
    ("45th", 0.45),
    ("60th", 0.6),
    ("75th", 0.75),
    ("90th", 0.9),
    ("95th", 0.95),
    ("99th", 0.99),
    ("99.9th", 0.999),
    ("99.99th", 0.9999),
    ("99.999th", 0.99999),
];

impl Serialize for TimerRecorder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("timers", self.timers.len())?;

        let mut buf = Vec::new();
        let mut v2_serializer = hdrhistogram::serialization::V2DeflateSerializer::new();

        let local_time = chrono::Local::now();
        let file = format!(
            "target/benchmark/{}-{:02}-{:02}/{:02}-{:02}.hdr",
            local_time.year(),
            local_time.month(),
            local_time.day(),
            local_time.hour(),
            local_time.minute()
        );
        let file_name = PathBuf::from_str(&file).unwrap();
        std::fs::create_dir_all(file_name.parent().unwrap()).unwrap();
        state.serialize_field("file", &file_name)?;

        for (i, timer) in self.timers.iter().enumerate() {
            let variant: Timer = unsafe { std::mem::transmute(i as u8) };
            match variant {
                Timer::Read => {
                    state.serialize_field("type", &variant.to_string())?;
                    for (qs, q) in QUANTILE.iter() {
                        let v = timer.value_at_quantile(*q);
                        state.serialize_field(qs, &v)?;
                    }

                    v2_serializer.serialize(timer, &mut buf).unwrap();
                }
            }
        }

        // write the buffer to file
        let mut file = std::fs::File::create(&file_name).unwrap();
        file.write_all(&buf).unwrap();

        state.end()
    }
}

pub struct TimerGuard {
    event: Timer,
    start: Instant,
}

impl TimerGuard {
    fn new(event: Timer) -> Self {
        TimerGuard {
            event,
            start: Instant::now(),
        }
    }
}

impl Drop for TimerGuard {
    fn drop(&mut self) {
        let elapsed = Instant::now() - self.start;

        crate::metric::get_tls_recorder().add_time(self.event.clone(), elapsed);
    }
}

/// Debugging timer guard that is used to record timers for each individual Bf-Tree.
pub struct DebugTimerGuard {
    event: Timer,
    start: Instant,
    // Reference to the metric recorder of a specific Bf-Tree
    tls_recorder: Option<Arc<ThreadLocal<UnsafeCell<TlsRecorder>>>>,
}

impl DebugTimerGuard {
    pub fn new(
        event: Timer,
        tls_recorder: Option<Arc<ThreadLocal<UnsafeCell<TlsRecorder>>>>,
    ) -> Self {
        DebugTimerGuard {
            event,
            start: Instant::now(),
            tls_recorder, // move out
        }
    }

    pub fn end(&mut self) {
        let elapsed = Instant::now() - self.start;

        if let Some(ref r) = self.tls_recorder {
            let rc = r.get_or(|| UnsafeCell::new(TlsRecorder::default()));
            let rc_mut = unsafe { &mut *rc.get() };
            rc_mut.add_time(self.event.clone(), elapsed);
        }
    }
}

use auto_ops::impl_op_ex;
