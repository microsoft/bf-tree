## Benchmark tips

```bash
env MIMALLOC_SHOW_STATS=1 MIMALLOC_LARGE_OS_PAGES=1 MIMALLOC_RESERVE_HUGE_OS_PAGES_AT=0 numactl --membind=0 --cpunodebind=0 cargo bench --features "metrics-rt" micro
```

Currently benchmark can only run on Linux.

## Prerequisites

All prerequisites are for Linux and are optional to run the benchmark.
Those steps make sure your results are consistent with the numbers in the paper.

### Setup huge pages
Use the following command to reserve huge pages (20GB).
```bash
sudo sysctl -w vm.nr_hugepages=10240
```

To confirm huge pages are populated
```bash
cat /proc/meminfo | grep HugePages
```
```
HugePages_Total:   10240
HugePages_Free:    10240
HugePages_Rsvd:        0
HugePages_Surp:        0
```


### Monitor disk usage
### Some useful commands:
Check the disk IO:
```bash
iostat 2 -x -h nvme0n1
```

## Benchmark guide

```bash
cd benchmark
```

### In-memory benchmark
```bash
env SHUMAI_FILTER="inmemory" MIMALLOC_LARGE_OS_PAGES=1 cargo run --bin bftree --release
```

### Benchmark different storage backends
```bash
env SHUMAI_FILTER="storage" MIMALLOC_LARGE_OS_PAGES=1 cargo run --bin bftree --release
```


### Guide to analyze the HdrHistogram

When benchmarked with `metrics-rt` feature, the benchmark will generate a histogram file in the `target\benchmark` directory. 
The file is in the [HdrHistogram](http://hdrhistogram.org) format. 

To read and analyze it using python:

```python
with open('18-52.hdr', 'rb') as f:
	de = base64.b64encode(f.read())
	histogram = HdrHistogram.decode(de)

percentiles = [0.1* x for x in range(1, 1000)]
values = [histogram.get_value_at_percentile(p) for p in percentiles]

tail_percentiles = [50, 90, 99, 99.9, 99.99, 99.999, 99.9999]
tail_values = [histogram.get_value_at_percentile(p) for p in tail_percentiles]

sns.barplot(x=tail_percentiles, y=tail_values)
sns.lineplot(x=values, y=percentiles)
```