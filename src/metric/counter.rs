// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::metric::RecorderImpl;
use serde::{ser::SerializeMap, Serialize, Serializer};
use variant_count::VariantCount;

#[repr(u8)]
#[derive(Debug, Hash, Eq, PartialEq, VariantCount)]
pub enum Counter {
    Insert = 0,
    Read = 1,
    UpgradeToFullPage = 2,
    MergeTriggerSplit = 3,
    MiniPageRead = 4,
    MiniPageReadMiss = 5,
    BasePageRead = 6,
    InsertToMiniPageSuccess = 7,
    InsertMiniPageUpgraded = 8,
    InsertCreatedMiniPage = 9,
    AllocDiskID = 10,
    IOReadRequest = 11,
    IOWriteRequest = 12,
    ReadPromotionOk = 13,
    ReadPromotionFailed = 14,
    InsertMergeFullPage = 15,
    EvictFromCircularBuffer = 16,
    FullPageRead = 17,
    MergeFailedDueToParentFull = 18,
    MoveMiniPageToTail = 19,
    MoveFullPageToTail = 20,
    LeafInsertDuplicate = 21,
    LeafInsertNew = 22,
    LeafNotFoundDueToRange = 23,
    LeafNotFoundDueToKey = 24,
    ScanGoNextLeaf = 25,
    ScanFullPage = 26,
    ScanBasePage = 27,
    ScanPromoteBaseToFull = 28,
    ScanMergeMiniPage = 29,
    InsertLocked = 30,
    InsertCircularBufferFull = 31,
    InsertNeedRestart = 32,
}

#[derive(Debug, Clone)]
pub(crate) struct CounterRecorder {
    pub(crate) counters: [u64; Counter::VARIANT_COUNT],
}

impl Default for CounterRecorder {
    fn default() -> Self {
        Self {
            counters: [0; Counter::VARIANT_COUNT],
        }
    }
}

impl CounterRecorder {
    const LENGTH: usize = Counter::VARIANT_COUNT;

    pub(crate) fn increment(&mut self, event: Counter, amount: u64) {
        let counter = unsafe { self.counters.get_unchecked_mut(event as usize) };
        *counter += amount;
    }
}

impl Serialize for CounterRecorder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(Self::LENGTH))?;
        for i in 0..self.counters.len() {
            let val = &self.counters[i];
            let key: Counter = unsafe { std::mem::transmute(i as u8) };
            state.serialize_key(&format!("{key:?}"))?;
            state.serialize_value(val)?;
        }
        state.end()
    }
}

impl RecorderImpl for CounterRecorder {
    fn reset(&mut self) {
        for i in self.counters.iter_mut() {
            *i = 0;
        }
    }
}

use auto_ops::impl_op_ex;

impl_op_ex!(+= |a: &mut CounterRecorder, b: &CounterRecorder| {
    for i in 0..Self::LENGTH {
        a.counters[i] += b.counters[i];
    }
});

impl_op_ex!(+ |a: &CounterRecorder, b: &CounterRecorder| -> CounterRecorder {
    let mut c_a = a.clone();
    c_a += b;
    c_a
});
