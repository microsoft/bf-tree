// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use crate::metric::RecorderImpl;
use serde::{ser::SerializeStruct, Serialize, Serializer};
use std::collections::HashMap;
use variant_count::VariantCount;

#[repr(u8)]
#[derive(Debug, VariantCount)]
pub enum Histogram {
    EvictNodeSize = 0,
    HitMiniPage = 1,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct HistogramRecorder {
    histograms: [HashMap<u16, usize>; Histogram::VARIANT_COUNT],
}

impl HistogramRecorder {
    pub(crate) fn hit(&mut self, event: Histogram, key: u64) {
        let hist = unsafe { self.histograms.get_unchecked_mut(event as usize) };
        if let Some(v) = hist.get_mut(&(key as u16)) {
            *v += 1;
        } else {
            hist.insert(key as u16, 1);
        }
    }
}

impl Serialize for HistogramRecorder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("histograms", self.histograms.len())?;
        for (i, h) in self.histograms.iter().enumerate() {
            let variant: Histogram = unsafe { std::mem::transmute(i as u8) };
            match variant {
                Histogram::EvictNodeSize => state.serialize_field("evict_node_size", h)?,
                Histogram::HitMiniPage => state.serialize_field("hit_mini_page", h)?,
            }
        }
        state.end()
    }
}

use auto_ops::impl_op_ex;

impl_op_ex!(+= |a: &mut HistogramRecorder, b: &HistogramRecorder| {
    for (i, h) in b.histograms.iter().enumerate(){
        for(k, v) in h.iter(){
            if let Some(value) = a.histograms[i].get_mut(k){
                *value += v;
            }else {
                a.histograms[i].insert(*k, *v);
            }
        }
    }
});

impl_op_ex!(+ |a: &HistogramRecorder, b: &HistogramRecorder| -> HistogramRecorder{
    let mut c_a = a.clone();
    c_a += b;
    c_a
});

impl RecorderImpl for HistogramRecorder {
    fn reset(&mut self) {
        for h in self.histograms.iter_mut() {
            h.clear();
        }
    }
}
