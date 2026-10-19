# Scalar benchmark: run, read, explain | 两路线实验框架

This stage measures full-matrix and rowwise scalar FP32 attention on the same inputs.
The FP64 oracle, interface and numerical tolerance are unchanged. The full-matrix
route now uses three complete phases: materialize scores, normalize all rows in
place to form P, then compute P*V. The rowwise route uses the same row arithmetic,
but finishes each row before moving to the next one. No new buffer is allocated
inside either kernel. Blocked streaming, SIMD, OpenMP, PyTorch and roofline are later work.

这一阶段只接好两条标量路线的“同一份输入 → 验答案 → 计时 → 存记录”。
沿用现有接口、FP64参考答案与数值容差；旧原型的结果不作为本版本结果。

本轮修正了整表基线的执行阶段：原代码虽分配L²空间，却按行完成全部运算；
现在明确先存完整scores，再原地变成完整P，最后算全部输出。逐行版使用相同的
逐行算术，但算完一行立即输出。两边都先逐元素除以denom，再累加P*V。
旧二进制的舍入和时间可能不同，旧结果不能作为这版的测量。

## Run | 怎么运行

From the repository root, with a C++17 compiler and Python 3 available:

```text
build.bat
build\attn_tests.exe
build\attn_blocks.exe
python tests/test_benchmark.py build/attn_bench.exe
python tests/test_contract_gate.py
python tools/run_scalar_bench.py --out results/run1
```

On Linux/macOS, build with `bash build.sh`, use binaries without `.exe`, and pass
`build/attn_bench` to the Python test. That platform still needs its own validation.
On Windows `CXX` may select an existing compiler; otherwise the build uses `g++`.
No additional Python packages are needed.
The optional independent NumPy checker needs NumPy: run `python tests/test_crosscheck.py`
after building. It checks a valid dump and deliberately corrupts C++ FP32 output to
verify that finite errors, NaN and infinity cannot be reported as passed.
The fault-gate test uses the same C++ compiler, selected by `CXX` or found as `g++`.
It builds a separate test binary; the injected route is never in the normal build.

默认只测 L=128、512，d=dv=64，单线程，每条路线预热10次、正式20次。
两条路线在同一进程里交替测，轮流先跑。每次使用新的输出前缀，例如第二次用
`results/run2`，程序不会覆盖旧数据。输出前缀建议使用简单英文相对路径。

To try a small odd shape without changing code:

```text
python tools/run_scalar_bench.py --out results/odd --L 31 --d 31 --dv 33 --query-block 3 --warmup 2 --reps 7 --seed 42
```

`build/attn_bench --help` lists the options and bounds. `--L` selects one length;
without it the two default lengths run. Direct execution also works, but the Python
runner additionally saves the environment. `--csv` prints the summary CSV to stdout.
Very small inputs can finish within a timer tick on some platforms. A zero-time sample
is rejected as `Invalid timer sample`; it is not replaced with a made-up positive time.
The benchmark tests use L=31, d=31, dv=33 to avoid that timing limitation. The original
numerical tests still cover tiny shapes without relying on timer resolution.

## Files and code blocks | 文件与代码块

| File / block | Input → work → output | 中文解释 |
| --- | --- | --- |
| `tools/bench_scalar.cpp`: `Options`, `parse` | CLI → check numeric bounds → settings | 只解析本阶段要用的尺寸、次数和输出位置；不是通用框架 |
| `measure`: input/oracle | shape + seed → one Q/K/V set and FP64 oracle | Q/K/V只生成一次，两路线读取同一数组；参考答案不计时 |
| `Route`, `call` | existing buffers → one existing kernel call → output | 一个普通if选择两函数，不用插件、继承或复杂分发 |
| `validate` | FP32 output + FP64 reference → fixed contract check | 每元素须有限，且满足绝对+相对容差；计时前两条都要过 |
| warmup loop | validated input/buffers → repeated calls | 先让代码、缓存等进入重复运行状态；预热不进入raw |
| timed loop | same prepared buffers → complete kernel → milliseconds | 只把函数调用夹在两个时钟之间；每轮交换先后顺序 |
| post-check | last output → same contract → pass/fail | 正式调用之后再验；失败非零退出，不写性能CSV |
| `quantile`, `write_results` | individual samples → sorted statistics + CSV | raw保留全部读数，summary能从raw重新算出来 |
| `tools/run_scalar_bench.py` | built binary + source + machine → environment JSON | 记实际源码版本/是否有改动、源码与二进制哈希、编译记录、CPU和系统 |
| `tests/test_benchmark.py` | small actual run → independent Python statistics | 用标准库独立复算四分位数、吞吐和工作区，并检查无效参数与覆盖保护 |
| `tests/test_contract_gate.py` | test-only wrong outputs → rejection and no CSV | 分别制造有限超差、NaN和末次调用错误，检查计时前后两道数值门 |

The helper's `source_dirty` describes the complete working tree, not just the kernel.
Its file hashes identify the relevant source bytes even during development. Build records
come from the build scripts; after editing a source file, rebuild before measuring.
If a detailed Windows CPU query is unavailable, the environment records that fact and
keeps the portable CPU description and logical processor count. It does not change permissions.
The CSV test checks each route's finite error fields; it does not require different
valid implementations to have identical error summaries as proof of identical inputs.

自测：把“输入生成”“内存分配”“oracle”“状态复位”“CSV输出”放到计时内/外两栏。
答案：前3项和CSV在外；内核自身必须做的复位在调用内。当前两个内核在自己的循环里
重置局部累加器并重写工作区和输出，harness不会偷用上一次答案。

## Timing boundary | 秒表到底测了什么

```text
same Q/K/V + one FP64 reference
  → allocate workspace/output separately for each route
  → call and validate BOTH routes
  → warm up BOTH routes
  → round 0: full-matrix then rowwise
  → round 1: rowwise then full-matrix
  → repeat, recording each complete call
  → validate final outputs
  → write raw and summary
```

Allocation, input generation, oracle evaluation, validation, sorting and file output
are outside the timer. Necessary work inside the kernel stays inside the timer.
These are native scalar timings, not a PyTorch API comparison or cold-start timings.
The first pre-check call is also outside timing; `warmup` counts the additional calls.

预热和交替顺序只能减少某些干扰，不能保证无噪声。这里不设置CPU亲和性、不切电源模式，
环境文件明确写“未控制”；请保留IQR，不把一次很小的差别说成稳定加速。

自测：如果一条路线计入每次分配内存，另一条提前分配，能直接比较算法吗？
不能。先统一边界；将来API比较应把双方所需allocation/dispatch对等计入，并和native表分开。

## CSV | 怎样读结果

The runner writes `<prefix>_raw.csv`, `<prefix>_summary.csv`, and
`<prefix>_environment.json`. The environment's final `status` and `exit_code`
must be checked; a failed or interrupted run is not a valid result.
Use a unique prefix for each run, including concurrent runs. The overwrite check is
not a cross-process lock, and the two CSV writes are not an atomic transaction.
The post-check validates the final output of each route, not every timed sample.

| Column | Meaning / 中文 |
| --- | --- |
| `route,L,d,dv,seed` | 实现与输入身份；同shape两路线用相同seed和同一份数组 |
| `query_block` | rowwise实际Bq=min(请求Bq,L)；full-matrix为0（不适用） |
| `sample,order,ms` | 第几次正式样本（从0开始）、本轮先/后（0/1）、单次毫秒 |
| `warmup,reps` | 额外预热次数与正式样本数 |
| `median_ms,min_ms` | 中间水平与最快一次，均由raw得到 |
| `q1_ms,q3_ms,iqr_ms` | 第25/75百分位及其差，表示波动 |
| `gflops` | 主矩阵乘有效运算量2L²(d+dv)除以median时间 |
| `workspace_bytes` | 本路线分配的分数临时区，不含输入、输出、FP64或另一条路线 |
| `pre_*`, `post_*` | 计时前/后的最大绝对与相对误差 |
| `contract_pass` | 1表示两个检查都通过；失败run不生成性能CSV |

Quartiles use inclusive linear interpolation: sort n values and interpolate at
position `p*(n-1)` (zero-based). For `[1,2,3,4,5]` ms, median=3, Q1=2, Q3=4,
IQR=2. The Python test recomputes these statistics using `statistics`.

有效GFLOP/s分子只数两次矩阵乘；softmax时间仍在分母里，不代表数了exp的FLOPs。
`max_rel`在参考值接近0时可能很大；通过条件仍是逐元素
`abs(diff) <= 1e-4 + 1e-3*abs(ref)`，不是单独看最大相对误差。

L=512、Bq=64时，full-matrix工作区为1,048,576字节，rowwise为131,072字节，
是这项存储8倍差。不能据此说DRAM流量或速度也差8倍。

自测：哪一行raw属于哪一行summary？用route、L、d、dv、seed、query_block匹配，
每组必须恰好有reps个样本。再自己算median和Q1/Q3。

## Review questions | 理解检查

1. Why do both routes use one Q/K/V set rather than two independent random sets?
2. Which calls occur before the timer, and why?
3. Where does the last output get checked? What happens if it fails?
4. Why do we alternate which route is measured first?
5. What do median and IQR each tell us?
6. Why is less workspace not proof of less DRAM traffic?

New routes can enter the same call/validation/timing chain once their implementations
and tests are available. The frozen numerical contract remains the same.
