# EXP-046 PROCESS

写法按 ASD-STE100 (simplified technical writing)。时间是。机器 gpu-server-2（Llama 家族）和 gpu-server-1（OLMo 家族）。远端目录两台都是 `/data/exp046`。

## 揭盲条件（分析 agent 先读这一节）

- 分析只能在下面两件事都完成后开始（SPEC 第 7 节）：
  1. EXP-047 的预测量 SPEC 的 sha256 已经冻结，并记在 `../EXP-047/PROCESS.md`。
  2. 最终预测量已经写入 `../EXP-047/results/FINAL_PREDICTOR.json`，它的 sha256 也记在 `../EXP-047/PROCESS.md`。
- 分析 agent 先核对这两项（重算两个文件的 sha256，和 EXP-047 PROCESS 里的值比较），再把两个 sha256 和核对时间写到本文件。核对不过就不分析，不打开 `blind/`。
- 揭盲前，任何人不打开两台机器上 `/data/exp046/blind/` 下的文件，不运行 `extract46.py`。看 DONE / FAILED 标记和 ts 队列不算揭盲。

## 261010 布置（实验 agent）

- 09:5x：读 EXP-039、EXP-041 的 SPEC、PROCESS、SUMMARY 和代码，读 EXP-047 SPEC 的结果量定义（O1–O4）。两台机器的 ts 队列都满：gpu-server-2 有 EXP-045 的 14 个 1 卡任务，gpu-server-1 有 EXP-042 和 EXP-045 的 17 个 1 卡任务。
- 10:0x：选 judge。Llama-3.2-1B、Llama-3.1-8B（加已有的 3B，成 3 个尺寸）；OLMo-2 1B（0425）和 7B（1124）-Instruct（Apache-2.0，不 gated，≤ 8B 有两个尺寸）。meta-llama 仓库是 gated，没有凭据：用 a model mirror 官方副本，备用 unsloth 副本。HF API 显示 unsloth 的权重 LFS sha256 和 meta-llama 相同，小文件不同（unsloth 改了 tokenizer），所以小文件只从 a model mirror 取，按 meta-llama 的 blob id 核对。
- 10:0x：在 gpu-server-2 的 CPU 上检查 3 个新 tokenizer（OLMo 1B、OLMo 7B、Llama-3.1-8B）："Yes" / "No" 都是单个 token（9642 / 2822），OLMo 有 pad token，模板正常。`judge39.py` 不用改。
- 10:1x：gpu-server-1 的 venv 和 gpu-server-2 不同（torch 2.14 cu130，Python 3.10）。决定在 gpu-server-1 新建 venv，按 gpu-server-2 的 75 个包锁定版本（Python 3.12.3 用 uv 安装）。gpu-server-1 上的 sft-s2 sha256（1596e1d3…）和 gpu-server-2 的（1ef340d1…）不同，所以从 gpu-server-2 复制。J15 的权重两台相同（dd924a11…），用来做环境核对。
- 10:2x：代码：EXP-041 代码的副本，改动见 SPEC 第 2 节。`lenient_static46.py` 在 gpu-server-2 上自测：J15 / J3 / J7 / JL 的 16 个格和 EXP-041 的 `static_lenient.json` 完全相同。
- **10:3x：SPEC.md 写好，在任何 EXP-046 的 RL 运行之前。sha256 = e67f105ca25aad7f5540e751b1d192858a4887cad34fc6bab9a5be4187c3282b。**
- 10:3x：部署代码和第 0 步数据（`/data/exp041/data` 的副本，不是硬链接）到两台机器。写 `markers/WAKE_WHEN`（每台 8 个标签）。提交链的第一步：

| 机器 | 标签 | ts ID | 卡 | 做什么 |
|---|---|---|---|---|
| gpu-server-2 | exp046-dl-JL1 | 42 | 0 | 下载、核对 JL1 → 提交 exp046-gate-JL1 |
| gpu-server-2 | exp046-dl-JL8 | 43 | 0 | 下载、核对 JL8 → 提交 exp046-gate-JL8 |
| gpu-server-1 | exp046-setup2-gpu-server-1 | 51 | 0 | 建 venv、复制并核对策略 → 下载 JO1 → 提交 exp046-gate-JO1 → 下载 JO7 → 提交 exp046-gate-JO7 |

- 10:2x：第一次提交的 gpu-server-1 准备任务（exp046-setup-gpu-server-1，ts 50）卡在从 gpu-server-2 用 scp 复制 uv（46 / 48 MB 处停住）。我用 `ts -k 50` 停掉它（没有写 FAILED 标记），改为：uv 从阿里云 PyPI 镜像安装，策略用带重试和超时的 rsync 复制（`setup46_gpu-server-1.sh` v2）。重新提交为 exp046-setup2-gpu-server-1（ts 51）。45 秒后：uv、Python 3.12.3 已装好，包在下载。
- 准备任务还会比较 gpu-server-1 自己的 sft-s2 和 gpu-server-2 的副本（sha256 不同、大小相同），逐个张量比较，写到 `logs/sft_compare_gpu-server-1.json`。只作诊断：EXP-039 / 041 / 046 都用 gpu-server-2 的副本。
- 每个 `exp046-gate-<J>`（1 卡）：（gpu-server-1 第一次先做环境核对）静态闸门 → 冒烟 → 用 expq 提交 `exp046-<J>-P05-seed301..304`。
- 闸门不过：退出码 3，FAILED 标记会唤醒协调 agent；这个 judge 的标签从 WAKE_WHEN 删掉。冒烟不过：退出码 4。环境核对不过：退出码 5。下载失败：退出码 1。
- 261010 10:46 修复 JL1 下载（修复 agent）：ts 42 失败的原因是 a model mirror 对小文件的单连接下载很慢（约 2 KB/s），`dl46.py` 对 tokenizer.json（9 MB）和 tokenizer_config.json 的 300 秒超时到了，文件不完整，核对失败，退出码 1。权重已核对通过。修复：`dl46.py` 对小于 1 MB 的文件改成重试 20 次（每次重新下载，因为 a model mirror 不支持续传，curl 返回 33），对 1 MB 到 100 MB 的文件改成 1 MB 分块并行下载，分块的最低速度门限从 30 KB/s 降到 1 KB/s。旧版保存为 `code/server/dl46.py.bak261010`。第一次重新提交的 ts 44 因为续传错误被我停掉（`ts -k`，没有写标记）。第二次重新提交用相同标签 `exp046-dl-JL1`，ts 46 在 8 秒内完成，所有文件核对通过（READY），并提交了 `exp046-gate-JL1`（ts 47）。旧的 FAILED 标记改名为 `markers/exp046-dl-JL1.FAILED.old261010`（内容 `rc=1 261010-103045`）。JL8 的 ts 43 没有动，它已经完成（DONE）。

## 最终 RL 标签（WAKE_WHEN）

| 机器 | 标签 | 预计 A100 小时 / run |
|---|---|---|
| gpu-server-2 | exp046-JL1-P05-seed301 … seed304 | 约 0.9 |
| gpu-server-2 | exp046-JL8-P05-seed301 … seed304 | 约 1.25 |
| gpu-server-1 | exp046-JO1-P05-seed301 … seed304 | 约 0.95 |
| gpu-server-1 | exp046-JO7-P05-seed301 … seed304 | 约 1.2 |

- 合计约 18 A100 小时（上限 25）。
- 预计墙钟（不确定，取决于前面的队列）：gpu-server-2 上 EXP-045 约 13:00 结束，Llama 的 8 个 run 约 18:00 完成。gpu-server-1 上 EXP-042 和 EXP-045 下午到晚上结束，OLMo 的 8 个 run 约 261011 00:00 完成。
- 标记：`/data/exp046/markers/<标签>.DONE` 或 `.FAILED`。每个 run 的输出：`/data/exp046/blind/runs/exp046-<host>-P05<J>-s<seed>/`（盲区）。
- 第 0 步的结果（不盲）：`/data/exp046/results/static_<J>.json`、`gate_<J>.json`、`smoke_<J>.json`、`static_lenient.json`、`envcheck_gpu-server-1.json`；日志在 `/data/exp046/logs/`。

## 分析 agent 要做的事（揭盲条件满足后，按 SPEC 第 6–8 节，不改定义）

1. 在每台机器上：`source code/server/env.sh; python code/extract46.py blind/runs/exp046-*-P05J*-s30?` → `blind/compact/`。复制到本地 `results/compact/`；小文件（run.json、config.json、steps.jsonl、evals.jsonl、stdout.log）复制到本地 `runs/`。`static_lenient.json` 两台各一份（gpu-server-2 有 JL1 / JL8，gpu-server-1 有 JO1 / JO7），都复制。
2. E1、E2、E5、E6：按 SPEC 第 6 节，用本格的 FPR0_len。
3. E3、E4：用 `../EXP-041/code/analyze41.py` 的 onset()（thr 0.5、0.3）和 AFT（同 σ 先验），合并 EXP-039 的 20 个、EXP-041 的 6 个和本实验的 run。
4. 描述：SPEC 6 节末尾的列表。
5. 停止规则核对（SPEC 第 8 节）和算力（墙钟 × 卡数，从 run.json 的 total_seconds，加闸门和冒烟）。
6. 然后按 EXP-047 SPEC 的检验规则，用 FINAL_PREDICTOR.json 做检验。
7. 在最终回复里给出 SUMMARY 全文，由协调 agent核对后写入 `results/SUMMARY.md`。

- 261010 11:03（协调 agent）：exp046-gate-JL1 退出码 3，闸门不过。AUC(G) 在 d8 上：P05 0.560，P15 0.518，门槛 0.60（gpu-server-2 `/data/exp046/results/gate_JL1.json`）。按 SPEC 第 3 节，JL1 不跑 RL，不换 judge。gpu-server-2 的 WAKE_WHEN 已经只剩 JL8 的 4 个标签。

- 261010 11:40（协调 agent）：EXP-047 的 FINAL_PREDICTOR.json 已锁定，sha256 `8c1eb85a…138af1`（全文见 EXP-047/PROCESS.md）。EXP-046 的 run 全部完成后即可揭盲。

- 261010 11:23 修复 gpu-server-1 准备任务（修复 agent）：ts 51（exp046-setup2-gpu-server-1）在 10:46 失败，退出码 26。原因：`sft.sha256` 列了 `training_args.bin`，但 gpu-server-2 的 `/data/exp029/models/sft` 里没有这个文件，所以 `sha256sum -c` 对它报 FAILED。其余 6 个文件（含 model.safetensors，1ef340d1…）全部核对通过。venv 已经建好（`venv/.OK`），没有重做。修复：(1) `setup46_gpu-server-1.sh` 核对时跳过 `training_args.bin`（用 `models/sft.sha256.use`；旧版存为 `setup46_gpu-server-1.sh.bak261010b`）。(2) 把 gpu-server-2 上打过补丁的 `dl46.py` 复制到 gpu-server-1（旧版存为 `dl46.py.bak261010`）。用相同标签 `exp046-setup2-gpu-server-1` 重新提交，新 ts id = 52。10:5x 起，策略核对通过，sft_compare 已写，JO1 权重已从 Hugging Face 下载并核对通过，在下载其余小文件。到 11:23，ts 52 仍在运行（还没提交 gate-JO1，JO7 还没开始）。旧 FAILED 标记：expq 重新提交时已把它删掉，我补写了 `markers/exp046-setup2-gpu-server-1.FAILED.old261010b`（内容 rc=26，时间约 261010-104605）。没有碰 `blind/`。

- 261010 14:15 修复 gpu-server-1 分词器核对失败（修复 agent）：ts 52（exp046-setup2-gpu-server-1）14:03 退出码 1。原因：Hugging Face 忽略 Range 请求头，对每个请求都返回整个文件（状态码 200）。`dl46.py` 对 1 MB 到 100 MB 的文件（JO1 / JO7 的 tokenizer.json 和 vocab.json）用 1 MB 分块下载，每块大小都不对，文件没有生成，核对报 BAD。镜像内容本身没有问题：直接下载的 vocab.json 的 git blob sha1 与 HF API 一致。修复：在 gpu-server-1 上用 curl 整文件下载这 4 个文件（JO1、JO7 各 2 个），核对 git blob sha1 全部通过；权重没有重新下载。`dl46.py` 没有改（之后遇到同样的镜像再改：1 MB 到 100 MB 的文件应整文件下载）。用新标签 `exp046-setup3-gpu-server-1`（0 卡，ts 53）重新运行 `dl_then_gate46.sh` JO1、JO7，JO1 的 .READY 已写，exp046-gate-JO1 = ts 54（已在运行）；exp046-gate-JO7 在 ts 53 完成 JO7 的核对后由它提交，id 待定（用 `ts` 查）。WAKE_WHEN 没有改。旧 FAILED 标记改名为 `markers/exp046-setup2-gpu-server-1.FAILED.old261010c`（rc=1 261010-140313）。没有碰 `blind/`。

- 261010 14:30 修复 gpu-server-1 冒烟失败（修复 agent）：exp046-gate-JO1（ts 54）和 exp046-gate-JO7（ts 55）都是退出码 4（冒烟失败），闸门本身通过。原因和 OLMo 无关：gpu-server-1 的 `/lib/x86_64-linux-gnu` 只有 `libcuda.so.1`，没有 `libcuda.so`。TRL 的 triton 内核（`selective_log_softmax_and_entropy`）第一次训练步要编译 `cuda_utils.c`，链接 `-lcuda` 失败（ld 退出码 1），所以每个 run 在第 1 步后崩溃（steps 1 行、rollouts 64 行）。gpu-server-2 的 venv 在系统里能找到 `libcuda.so`，所以 EXP-039 / 041 没遇到。修复：在 gpu-server-1 的 venv 里建符号链接 `venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/libcuda.so -> /lib/x86_64-linux-gnu/libcuda.so.1`。没有改任何代码、judge 提示词或评分。验证：在 `/data/exp046/diag`（不在 blind 下）用相同命令跑 JO1 的 4 步冒烟，退出码 0，4 步、256 条 rollout；诊断目录已删除。我曾误把一次诊断 run 写到 `blind/runs/diag-JO1`，没有打开，立刻按路径删除（该目录和 ckpts/diag-JO1 已删）。新增 `code/server/smoke_submit46.sh`（只含 gate46.sh 的冒烟加提交部分，闸门结果不重算）。旧 FAILED 标记改名为 `markers/exp046-gate-JO1.FAILED.old261010d` 和 `exp046-gate-JO7.FAILED.old261010d`。用原标签重新提交：exp046-gate-JO1 = ts 56，exp046-gate-JO7 = ts 57（冒烟通过后各自提交 4 个 RL run，标签同 WAKE_WHEN）。WAKE_WHEN 没有改。没有碰 `blind/` 的内容。

- 261010 14:3x（协调 agent）：修复 agent 报告，它曾把一次诊断 run 的输出误写进 `blind/runs/diag-JO1`，没有打开或列出，已按路径删除。这是诊断 run，不是正式 run，正式结果没有被看过。揭盲时如果看到 diag 相关的痕迹，按此说明处理。

- 261010 18:06（分析 agent）：揭盲核对通过。重算 `../EXP-047/SPEC.md` sha256 = 45513193ec9778591359a61b95b78ae482a968e93621a412621b16fafc0c5b20，`../EXP-047/results/FINAL_PREDICTOR.json` sha256 = 8c1eb85a2e601c8ed563ad70824b4291dec3de9929ec0d50c225ba750d138af1，都和 EXP-047 PROCESS 的记录相同。本 SPEC sha256 = e67f105c…3282b，和上文相同。两台机器的 12 个 RL 标签（JL8、JO1、JO7 各 4 个）都是 DONE。`blind/runs/` 下没有 diag-* 目录。从现在开始揭盲，运行 extract46.py。
- 261010 18:3x（分析 agent）：揭盲分析完成，没有改 SPEC 第 6 节的定义。
  - `extract46.py` 在两台机器上各跑一次（gpu-server-2 4 个 run，gpu-server-1 8 个 run，每个 200 步）。compact 复制到 `results/compact/`，小文件复制到 `runs/`，两份 `static_lenient.json` 和闸门、冒烟文件复制到 `results/gpu-server-2/`、`results/gpu-server-1/`。
  - 预测量：EXP-047 的 `predictors47.py`（sha256 1b1c761c…，未改）在本地 CPU 上对 8 个 judge 运行，输出 `results/predictors46.json`。旧 judge 的值和 EXP-047 `predictors47.json` 最大差 2.2e-16。新格的 B1、B2、τ 和本实验 `static_lenient.json` 完全相同。
  - 新代码：`code/analysis46.py`（E1–E6、AFT、检验，输出 `results/analysis46.json`），`code/desc46.py`（格式漂移、评测正确率、被骗回答抽样，两台机器各跑一次，输出 `results/desc46_<host>.json`、`results/fooled_<cell>.jsonl`）。`results/t_half46.json` 是 desc46 的输入。
  - EXP-047 SPEC 没有单独写 EXP-046 的通过规则。本 agent 用 EXP-047 SPEC 第 7 节的合格条件作为通过规则（新格上 O1、O2 的符号调整 ρ 都 > 0，并且锁定拟合的 O1 平均绝对误差小于三个基线中最小的）。这是事后的操作化，写在 `analysis46.py` 的说明里。
  - 结论草稿在最终回复里，由协调 agent核对后写 `results/SUMMARY.md`。
