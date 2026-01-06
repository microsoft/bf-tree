// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::ops::Range;

use rand::distr::{Distribution as RandDistribution, Uniform};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Distribution {
    Uniform,
    HotSpot(f64),
    Zipf(f64),
}

pub enum Workload {
    Read,
    NegativeRead,
    Update,
    Scan,
    Insert,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkloadMix {
    read: u64,
    negative_read: u64,
    update: u64,
    scan: u64,
    insert: u64,
}

impl WorkloadMix {
    pub fn gen(&self, rng: &mut impl rand::Rng) -> Workload {
        debug_assert!(self.is_valid());
        let val = rng.random_range(0..100);

        let mut bar = self.read;
        if val < bar {
            return Workload::Read;
        }

        bar += self.negative_read;
        if val < bar {
            return Workload::NegativeRead;
        }

        bar += self.update;
        if val < bar {
            return Workload::Update;
        }

        bar += self.scan;
        if val < bar {
            return Workload::Scan;
        }

        bar += self.insert;
        if val < bar {
            return Workload::Insert;
        }

        panic!("Invalid workload mix")
    }

    pub fn is_valid(&self) -> bool {
        self.read + self.update + self.scan == 100
    }
}

pub enum Sampler {
    Uniform(Uniform<usize>),
    HotSpot(Uniform<usize>),
    Zipf((rand_distr::Zipf<f64>, usize)),
}

impl Sampler {
    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        match self {
            Self::Uniform(u) => u.sample(rng),
            Self::HotSpot(u) => u.sample(rng),
            Self::Zipf(z) => z.0.sample(rng) as usize - 1 + z.1, // Zipf distribution starts with 1
        }
    }

    pub fn from(dist: &Distribution, item_range: Range<usize>) -> Self {
        match dist {
            Distribution::Uniform => Sampler::Uniform(Uniform::try_from(item_range).unwrap()),
            Distribution::HotSpot(skew) => {
                assert!(*skew <= 1.0);

                let end = (item_range.start as f64
                    + (item_range.end - item_range.start) as f64 * skew)
                    as usize;
                let hotspot_range = item_range.start..end;
                Sampler::HotSpot(Uniform::try_from(hotspot_range).unwrap())
            }
            Distribution::Zipf(s) => {
                let num_element = item_range.end - item_range.start;
                Sampler::Zipf((
                    rand_distr::Zipf::new(num_element as f64, *s).unwrap(),
                    item_range.start,
                ))
            }
        }
    }
}

pub fn murmur64(mut h: usize) -> usize {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

use bf_tree::metric::TlsRecorder;

#[derive(Serialize, Default, Clone)]
pub struct MicroBenchResult {
    throughput: usize,
    metrics: Option<TlsRecorder>,
}

impl MicroBenchResult {
    pub fn new(throughput: usize, metrics: Option<TlsRecorder>) -> Self {
        Self {
            throughput,
            metrics,
        }
    }
}

impl std::fmt::Display for MicroBenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Throughput: {} ops/sec", self.throughput)
    }
}

impl std::ops::Add for MicroBenchResult {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let new_metric = match (&self.metrics, other.metrics) {
            (Some(m1), Some(m2)) => Some(m1 + m2),
            (None, Some(m2)) => Some(m2),
            _ => None,
        };
        Self {
            throughput: self.throughput + other.throughput,
            metrics: new_metric,
        }
    }
}

impl std::ops::AddAssign for MicroBenchResult {
    fn add_assign(&mut self, other: Self) {
        self.throughput += other.throughput;
        match (&self.metrics, other.metrics) {
            (Some(m1), Some(m2)) => {
                self.metrics = Some(m1 + m2);
            }
            (None, Some(m2)) => {
                self.metrics = Some(m2);
            }
            _ => {}
        }
    }
}

impl shumai::BenchResult for MicroBenchResult {
    fn short_value(&self) -> usize {
        self.throughput
    }

    fn normalize_time(self, dur: &std::time::Duration) -> Self {
        let dur = dur.as_secs_f64();
        let throughput = (self.throughput as f64) / dur;
        Self {
            throughput: throughput as usize,
            metrics: self.metrics,
        }
    }
}

#[derive(Copy, Debug, Clone, Deserialize, Serialize)]
pub enum StorageBackend {
    Memory,
    Std,
    StdDirect,
    IoUringPolling,
    IoUringBlocking,
    Spdk,
}

impl From<StorageBackend> for bf_tree::StorageBackend {
    fn from(value: StorageBackend) -> Self {
        match value {
            StorageBackend::Memory => bf_tree::StorageBackend::Memory,
            StorageBackend::Std => bf_tree::StorageBackend::Std,
            StorageBackend::StdDirect => {
                #[cfg(target_os = "linux")]
                {
                    bf_tree::StorageBackend::StdDirect
                }
                #[cfg(not(target_os = "linux"))]
                {
                    panic!("Direct IO is only supported on Linux")
                }
            }
            StorageBackend::IoUringPolling => {
                #[cfg(target_os = "linux")]
                {
                    bf_tree::StorageBackend::IoUringPolling
                }
                #[cfg(not(target_os = "linux"))]
                {
                    panic!("io_uring is only supported on Linux")
                }
            }
            StorageBackend::IoUringBlocking => {
                #[cfg(target_os = "linux")]
                {
                    bf_tree::StorageBackend::IoUringBlocking
                }
                #[cfg(not(target_os = "linux"))]
                {
                    panic!("io_uring is only supported on Linux")
                }
            }
            StorageBackend::Spdk => {
                #[cfg(all(target_os = "linux", feature = "spdk"))]
                {
                    bf_tree::StorageBackend::Spdk
                }
                #[cfg(not(all(target_os = "linux", feature = "spdk")))]
                {
                    panic!("SPDK is not enabled in this build")
                }
            }
        }
    }
}
