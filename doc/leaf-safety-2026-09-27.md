# 叶页分配边界修复

2026-09-27 开始修复，2026-09-28 完成最终验证与运行库重建。

对照基线为 `f8206afbd0049196f32c3fa6620fb40fadd126ad`。本报告与此前以
`2bdb766` 为基线的性能优化报告分开记录。

## 表示与访问

叶页采用 `LeafNodeStorage<[UnsafeCell<MaybeUninit<u8>>]>` DST。页头仍为
32 字节、8 字节对齐，KV 元数据仍为 8 字节；`PageLocation` 保持两个机器字，
编译期断言防止页表项因宽指针而增大。持久化字段、偏移和页尺寸类别不变。

从原分配指针和页长构造引用，不从页头引用或零长度字段扩大访问范围。
TmpBuffer 使用自身容量，新分配 mini-page 检查分配器容量，已发布页使用初始化后
不变的页长。恢复 mini-page 时先检查长度范围、对齐及页头长度与分配长度一致。
`UnsafeCell` 保留共享读取设置原子引用位的合法性；`MaybeUninit` 允许未用空间
尚未初始化，不将其当成有效记录读取。

环形缓冲区驱逐回调从缓冲区原始基址计算页指针。释放操作保留原分配指针，
避免从叶页借用向前访问分配器元数据。缓存页可变借用接口明确标为 unsafe，
要求页写锁、有效页长、无重叠借用以及不超过锁与分配的生命周期。

快照复制叶页时持写锁，排除读取线程同时修改原子引用位。sweep 复用两个对齐
缓冲区，在同一写锁内采样 Mini+Base，释放页锁后才同步写出独立映像，避免磁盘
写入持续阻塞同页读取。Full 页的 next_level 在放锁前恢复，CPR 版本和映射优先级
保持不变。生成快照才清零
元数据与记录区之间的空闲间隙，普通读写没有新增整页清零。基础页临时缓冲区的
快照处理保留 dirty 状态，避免额外磁盘写回。修复了加载基础页、改变页表位置时
重复创建同页可变借用的顺序问题。

真实树 Miri 测试还触发了新页表项获取锁的单次 weak CAS 伪失败。
`try_write` 改用 strong CAS，保持原内存序，不通过关闭 Miri 伪失败注入来绕过。

为抵消表示变化的性能成本，命中读取从一次原子 load 同时取得引用位和值长度，
引用位未设置时仍执行原来的条件 `fetch_or`。扫描复用值长度，KeyAndValue
将页内连续的键后缀和值合并拷贝，减少一次小块复制。选择性内联保留在读取和
扫描入口；强制内联扫描、禁止查找内联的
实验未带来稳定收益，均未保留。

二分查找每轮的 0～2 字节预览原先调用 `memcmp`，现改为大端 `u16` 加真实长度
比较，按长度遮罩无效字节。预览相等且任一完整后缀不超过 2 字节时，长度已经
足够确定完整顺序，省去重复负载比较；其他情况保持完整键比较。BTreeMap 参照
测试覆盖 85 种短后缀、共同前缀、空键、零字节及四字节查询，并刻意污染未用
预览字节，验证不依赖零填充。

## 验证与范围

使用官方 `nightly-2026-09-07`，Windows 主机解释 Linux 目标。
基线默认 Miri 在 fence 初始化的 `[0x20..0x28]` 写入处失败，报告零长
`[0x20..0x20]` retag。修复后的历史用例、五种页长（64/128/448/832/4096）、
精确填满、共享引用位更新、合并、分裂及未初始化分配升级均通过默认别名检查。
直接快照序列化测试逐字节读取未初始化分配生成的结果，并验证空闲间隙为零。

- 最终代码的确定性叶页、原历史用例和内联叶页测试：默认 Miri 26 项通过。
- 最终 sweep 独立映像测试：默认 Miri 1 项通过。写出映像时直接验证页锁已释放、
  对齐保持、原树可读且映像不受原树读取修改影响。合计 27 项默认 Miri 通过。
- 真实环形缓冲区全部尺寸类别逐级增长、合并、快照与释放：默认 Miri 通过。
- 16 条记录的 cache-only / 内存后端真实树读写、扫描与销毁：默认 Miri 2 项通过。
- 两模式的碎片页快照往返在此前 DST 实现上通过过 Miri，仅设置
  `-Zmiri-disable-isolation` 允许临时文件 I/O。最终 sweep 放锁改动后的同项复测，
  该命令被自动审批拒绝（只返回 `blocked by policy`，未启动）；默认隔离重试在
  创建临时目录处被 Miri 拒绝。因此不把历史结果计作最终代码通过。
  最终普通测试、Windows 互操作与 Linux 原生测试中的快照恢复通过。
  MemoryVfs 既有整数指针转换警告保留。
- 96 条记录跨内节点分裂：普通测试通过；默认 Miri 在既有
  `inner_lock.rs` 的 `&InnerNode` 到 `&UnsafeCell<InnerNode>` 转换报 UB。
  该结果保留为失败，没有关闭别名检查，也没有把小型用例的通过替代它。
- 最终代码完整普通测试 130 项、混合工作负载集成测试 1 项、doctest 10 项通过；
  原有微基准及 2 个 doctest 忽略。
- Clippy 库、`in_memory` 基准和三个 runtime metrics 配置均在 `-D warnings`
  下通过。全测试目标的 Clippy 仍有既有测试风格告警，不声明全目标零告警。

复现（先加载本机工具链环境）：

```powershell
. .\target\audit-env.ps1
cargo test --locked
cargo clippy --locked --lib -- -D warnings
cargo clippy --locked --bench in_memory -- -D warnings
$env:RUSTUP_TOOLCHAIN = 'nightly-2026-09-07-x86_64-pc-windows-msvc'
cargo miri test --target x86_64-unknown-linux-gnu --lib tests::leaf_node -- --skip test_leaf_insert_read --skip leaf_search_update_and_consolidation_match_model --skip leaf_allocation_cache_only_tree_lifecycle --skip leaf_allocation_memory_backed_tree_lifecycle
cargo miri test --target x86_64-unknown-linux-gnu --lib nodes::leaf_node::tests
cargo miri test --target x86_64-unknown-linux-gnu --lib sweep_writes_independent_images_after_releasing_leaf_locks
$env:MIRIFLAGS = '-Zmiri-disable-isolation'
cargo miri test --target x86_64-unknown-linux-gnu --lib snapshot_roundtrip_preserves_fragmented_small_leaf_pages
```

最后一个命令跨解释 Linux 时还需要将 `TMPDIR` 指向解释器可访问的宿主路径；
本机验证用 `/H:/git/bf-tree/target/leaf-miri-temp`。跳过的两个属性测试分别固定
1000/256 个案例，已在普通完整测试中执行；两个跨内节点用例的 Miri 失败单独记录。

Miri 与普通测试验证已覆盖的路径，不能证明整个并发引擎无 UB。内部节点乐观读的
普通字段并发访问，以及驱逐回调取得页锁前读取路由键的既有并发风险不在本次
变长叶页边界修复中解决。快照输入检查也不是任意损坏负载的全面验证。

## 性能复核方法

同一台 i7-13700KF、Windows、Rust 1.98.1 MSVC，相同 Cargo.lock 和普通 release
配置。修复前基准程序单独保存在 `target/leaf-safety-perf-baseline/in_memory.exe`。
分别固定进程到 CPU 0、CPU 4，每对交替基线/候选和候选/基线，暂停其他构建与
Miri 后测量。最终完整源码的两个核心各做 8 对，每次进程运行取 9 次测量的中位数。
覆盖 8/32/128B 键的插入、命中/未命中读取、增长更新和扫描；读取使用原有固定
随机序列。配对中位耗时的候选/基线差值为正表示变慢，负表示变快。

基准直接调用 Rust 公共 API；互操作 DLL 使用单独的 fat-LTO 发布配置，
这里的耗时不等同于 .NET P/Invoke 延迟。

### 最终完整基准结果

| 工作负载 | CPU 0 均值 | CPU 0 中位数 | CPU 4 均值 | CPU 4 中位数 |
|---|---:|---:|---:|---:|
| insert/k8 | -0.82% | -1.56% | -2.24% | -2.26% |
| read_hit/k8 | +2.55% | +0.98% | -8.41% | -2.36% |
| read_miss/k8 | -0.81% | -2.09% | -3.03% | -3.55% |
| update_grow/k8 | +0.72% | -0.04% | +0.16% | -1.01% |
| scan/k8 | -9.44% | -8.92% | -8.72% | -9.06% |
| insert/k32 | -1.25% | -0.71% | -0.99% | -1.09% |
| read_hit/k32 | +1.53% | +1.40% | +0.44% | -0.85% |
| read_miss/k32 | +0.69% | +1.52% | -0.97% | -1.18% |
| update_grow/k32 | -1.23% | -0.20% | -2.65% | -0.90% |
| scan/k32 | -6.23% | -5.52% | -4.92% | -5.28% |
| insert/k128 | +1.23% | -0.20% | -1.23% | +0.42% |
| read_hit/k128 | +1.90% | +1.35% | +0.86% | +1.34% |
| read_miss/k128 | +1.76% | +1.14% | +1.80% | +2.43% |
| update_grow/k128 | +3.54% | +2.16% | -0.80% | +0.19% |
| scan/k128 | -8.58% | -8.83% | -7.21% | -7.45% |

以上均值和中位数均基于 8 个配对百分比差值；所有样本完整保存在
`leaf-safety-perf-2026-09-28.csv`。CPU 0 前半轮读取耗时明显偏高，CPU 4
命中读取还出现基线异常慢值，因此同时列出配对中位数，没有删掉异常样本。
扫描的改善在各键长和两个核心上均可重复。128B 读取等项目仍有约 1%～2%
的慢值，不能把本次修复描述为所有工作负载都零回退。

基线可执行文件 SHA256：
`3eddc199aaf40fe42339590383fa3343d4f4a151465624f257cf8c5bf1a559e5`。
最终候选可执行文件 SHA256：
`4a8d8761c1bc2335776ad6a6759575de1089327633eef30c3816e74e6db39f1d`。

该基准未覆盖并发快照期间的吞吐和尾延迟，不能据此推断所有并发场景均无退化。

### 8B 专项确认

针对完整基准中读取噪声较大的现象，使用同一最终二进制，先分别完整运行基线
和候选预热，再在每个核心执行 8 对 8B 工作负载，每个进程取 21 次测量的中位数。
原完整基准样本保留；本表是追加实验，不替换上表。

| 工作负载 | CPU 0 均值 | CPU 0 中位数 | CPU 4 均值 | CPU 4 中位数 |
|---|---:|---:|---:|---:|
| insert/k8 | -2.26% | -1.71% | -2.63% | -2.48% |
| read_hit/k8 | -3.24% | -3.09% | -2.04% | -2.04% |
| read_miss/k8 | -3.90% | -4.18% | -2.86% | -3.22% |
| update_grow/k8 | -1.24% | -1.75% | -1.68% | -1.44% |
| scan/k8 | -8.85% | -8.56% | -10.60% | -10.72% |

两核的 8B 命中、未命中、插入、增长更新和扫描平均耗时均低于基线。
命中读取均值降低 2.04%～3.24%，未命中降低 2.86%～3.90%，扫描降低
8.85%～10.60%。此结论限于本机基准和已测工作负载。

## 原生运行库

Windows x64、Linux x64、macOS x64/arm64 均从本次工作区源码通过本地路径依赖
重新编译，保留 13 个 C ABI 导出。没有使用注册表中的旧版 bf-tree。
最终 4 个运行库和 Windows PDB 已替换到
`H:\git\SuperTank\SuperTank\libs\BfTreeInterop\runtimes`，原文件备份在
`target/interop-leaf-safety/previous-runtimes`。替换后再次运行 Windows 50 项测试，
并核对测试输出 DLL 与部署 DLL 的 SHA256 一致。
构建源码补丁及 SHA256、原生文件 SHA256 和验证记录保存在
`target/interop-leaf-safety`，用于区分基线提交与尚未提交的修复。

NuGet 包为 `target/interop-leaf-safety/nuget/SuperTank.BfTreeInterop.1.0.0-preview.1.nupkg`，
4 个原生资产与部署文件逐项 SHA256 相同，PDB 没有打入包中；没有发布到包源。
完整清单位于 `target/interop-leaf-safety/build-manifest.json`。

Windows 新 DLL 在 .NET 11 下通过 50/50 互操作测试。Linux 新库通过现有 WSL
及隔离 glibc 2.36 的原生运行测试，两后端各 512 条记录，覆盖 13 个导出、CRUD、
命中/未命中、扫描、快照恢复和释放。Linux 最高 GLIBC 符号要求仍为 2.28，但没有
执行 glibc 2.28 最低版本实机验证。macOS 最低版本实际为 13.0，架构、导出、系统
依赖与 arm64 签名全部验证；没有 Mac 实机测试。
