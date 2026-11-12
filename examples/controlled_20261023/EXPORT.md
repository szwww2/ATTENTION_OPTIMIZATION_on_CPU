# 导出与校验

来源为同一仓库的 `results/formal_20261023/study_steady/`。只导出这次完成的实验；未收录中止的实验目录。

- 197 个源文件进入本包：所有原始/汇总 CSV、环境与 manifest JSON、协议、冻结记录、控制器状态、分析结果及 5 张 PNG 图。
- 相对首次导出来源，194 个源文件逐字节保留；`roofline_analysis/README.md` 只修正文档链接；另两份 Roofline 派生文件按下述标签修正重新生成。
- 15 个控制台 `.log` 未收录；其中 12 个 API 日志含机器上的个人绝对路径。完整日志与原件保留在来源目录。结构化状态、起止时间、命令、退出码仍在 JSON 中。
- `protocol.json`、`frozen_settings.json` 及全部测量环境/manifest JSON 保留原字节，协议 SHA256 与冻结记录仍一致。manifest 的 `prefix`、控制器历史命令中的 `results/...` 是采集时的相对路径记录；分析器按旁边的 job ID 寻找数据，不依赖原目录。协议内 `build/`、`src/` 等相对路径以仓库根目录为基准。
- PyTorch 的 `torch_build` 保留第三方软件构建信息，包括发行包自身的 CI 编译器路径；它不是本机账户路径，也不是重新分析依赖。
- `export_hashes.csv` 的 `source_sha256` 保留首次导出时全部 212 个源文件的 SHA256，`exported_sha256` 是当前导出字节 SHA256；未导出的日志仅有源哈希。`SHA256SUMS.txt` 覆盖本包除该清单自身以外的全部文件。运行 `verify_package.py` 检查当前包。
- 根目录 `.gitattributes` 对本目录设置 `-text`，防止 Git 换行转换改变 CSV、JSON 等证据字节。

2026-10-23 标签修正：Rowwise 的两个流量场景后缀由 `score_spill` 改为 `cached_row`，与既有的 row scratch cached 假设一致；Full matrix 名称保留。只重新生成 `roofline_analysis/traffic_scenarios.csv`（10 行场景名）和 `analysis.json`（分析器 SHA256），所有数值、5 张图、原始计时、协议及冻结设置均未变。修正前 Roofline 文件已在本地另存；当前 `study_steady/roofline_analysis` 与本包使用修正版。计时仍来自 `f7b4b09`，后续分析器标签修正不代表重新采集实验。

包内没有可执行的实验二进制。源码版本和构建二进制哈希用于追溯原测量；在另一台机器重新编译不会保证得到相同二进制或时间。
