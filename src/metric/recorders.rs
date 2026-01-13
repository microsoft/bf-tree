// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::metric::{
    counter::{Counter, CounterRecorder},
    histogram::{Histogram, HistogramRecorder},
    timer::{Timer, TimerGuard, TimerRecorder},
    RecorderImpl,
};
use serde::Serialize;
use std::{default::Default, time::Duration};

#[derive(Debug, Default, Serialize, Clone)]
pub struct TlsRecorder {
    pub(crate) counters: CounterRecorder,
    pub(crate) histograms: HistogramRecorder,
    pub(crate) timers: TimerRecorder,
}

impl TlsRecorder {
    pub fn increment_counter(&mut self, event: Counter, amount: u64) {
        self.counters.increment(event, amount);
    }

    pub fn hit_histogram(&mut self, event: Histogram, key: u64) {
        self.histograms.hit(event, key);
    }

    #[must_use = "dropping the timer guard will immediately stop the timer"]
    pub fn timer_guard(&mut self, event: Timer) -> TimerGuard {
        self.timers.start(event)
    }

    pub fn add_time(&mut self, event: Timer, time: Duration) {
        self.timers.add_time(event, time);
    }

    pub fn reset(&mut self) {
        self.counters.reset();
        self.histograms.reset();
        self.timers.reset();
    }
}

use auto_ops::impl_op_ex;

impl_op_ex!(+= |a: &mut TlsRecorder, b: &TlsRecorder|{
    a.counters += &b.counters;
    a.histograms += &b.histograms;
    a.timers += &b.timers;
});

impl_op_ex!(+ |a: &TlsRecorder, b: &TlsRecorder| -> TlsRecorder{
    let mut c_a = a.clone();
    c_a += b;
    c_a
});
