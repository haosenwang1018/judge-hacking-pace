# EXP-046 SPEC：留出 judge 的盲跑 GRPO（Llama 1B / 8B，OLMo-2 1B / 7B），给 EXP-047 的第 0 步预测量做检验（Experiment）

写于 261010 10:3x，在任何 EXP-046 的 RL 运行之前。研究线 11。上游：EXP-039、EXP-041（SPEC、SUMMARY、代码），EXP-047（预测量的筛选，另一个 agent）。
写法按 ASD-STE100 (simplified technical writing)。本文件的 sha256 记在 `PROCESS.md`。之后的改动只写成带时间的补充节。

## 0. 问题和本实验的角色

- 线 11 的新问题：judge 在第 0 步的测量，能否预测它在 GRPO 里被骗的程度和速度？
- EXP-047 只用 EXP-039 / EXP-041 的数据选出一个预测量，并冻结它（`../EXP-047/results/FINAL_PREDICTOR.json`）。
- **本实验只产生结果量**：在新的、留出的 judge 上跑同一配方的 RL，盲着保存结果。检验在揭盲后做，规则在 EXP-047 SPEC 里，不在本文件里。
- 留出的含义：这些 judge 的 RL 结果在 EXP-047 冻结之前没有人看过（第 7 节）。

## 1. 术语（新的只有两个）

- 其余术语（P05、G、G_cal、宽松金标准、宽容金标准、τ_len、FPR0_len、h、t½、删失、AFT、采样 KL、H̄）同 EXP-039 SPEC 第 1 节和 EXP-041 SPEC 第 1 节。
- **JL1、JL8、JO1、JO7**：冻结的 judge，不看参考答案，prompt 同 EXP-041（`code/judge39.py`，Llama 的日期固定为模板默认值 26 Jul 2024）。
  - JL1 = Llama-3.2-1B-Instruct（1.24B），JL8 = Llama-3.1-8B-Instruct（8.03B）。和已有的 JL（Llama-3.2-3B）组成 3 个尺寸的 Llama 家族。
  - JO1 = OLMo-2-0425-1B-Instruct（1.48B），JO7 = OLMo-2-1124-7B-Instruct（7.30B）。整个家族是新的。
- **上限 C**：一个 run 里 10 步块 h 的最大值（第 6 节 E6）。

## 2. 格点（唯一的改动是 judge）

| 格 | 种子 | 步数 | 卡 | 机器 |
|---|---|---|---|---|
| P05×JL1 | 301–304 | 200 | 1 | gpu-server-2 |
| P05×JL8 | 301–304 | 200 | 1 | gpu-server-2 |
| P05×JO1 | 301–304 | 200 | 1 | gpu-server-1 |
| P05×JO7 | 301–304 | 200 | 1 | gpu-server-1 |

- 策略、SFT 起点（sft-s2，sha256 1ef340d1…）、任务（GSM8K，revision 740312ad）、prompt、超参（lr 3e-6，8 题 × 8 条，T = 1.0，最长 256 token，β = 0，dapo，组内 std，micro-batch 2，fused AdamW）、评测（200 题，贪心；采样 KL 用前 100 题）、G_cal 的拟合方法、种子和步数都同 EXP-041 的 P05×JL 格。同一种子的题目顺序相同。
- 只用 0.5B 策略：EXP-041 的 P05 一行是 4 个种子 × 200 步；P15 一行只有 2 个种子，不作对照。
- 代码是 EXP-041 代码的副本。改动只有：
  1. `grpo46.py`：run 的输出目录改为 `/data/exp046/blind/runs/<id>`（第 7 节）。
  2. `lenient_static46.py`：judge 名单加 JL1、JL8、JO1、JO7。
  3. `extract46.py`：输出改到 `blind/compact/`。
  4. 新增 `envcheck46.py`（gpu-server-1 的环境核对）、`gpu-server-1/` 下的下载、闸门和提交脚本。
- 记录的量同 EXP-041：每步每条 rollout 的 G_cal、严格和宽松对错、格式、长度、尾巴标记；每次评测的正确率、格式率、judge 分数、FPR、贪心 KL、采样 KL、参数距离。

### 2.1 机器和环境

- Llama 家族在 gpu-server-2 上跑：同 EXP-039 / EXP-041 的 venv（Python 3.12.3，torch 2.9.1 cu128，transformers 5.18.0，trl 1.14.1）。
- OLMo 家族在 gpu-server-1 上跑：新建 `/data/exp046/venv`，按 gpu-server-2 venv 的 75 个包逐个锁定版本（`code/server/freeze_gpu-server-2.txt`），装完逐个核对。策略权重从 gpu-server-2 复制，按 sha256 核对。
- **gpu-server-1 环境核对**（在 gpu-server-1 的第一个闸门里，RL 之前）：用 J15（两台机器 sha256 相同）给 P05 / P15 的 d8 打分，和 gpu-server-2 上 EXP-039 的 `d8.G_J15.npy` 比较。AUC 相差 < 0.005 才通过。不通过 → gpu-server-1 不跑 RL，OLMo 格交给协调 agent决定（例如移到 gpu-server-2）。

## 3. 闸门（运行前，每个 judge 一次）

- 同 EXP-041 的静态闸门：`static46.py`（= static41），用 judge 给两个策略的第 0 步样本 d0、d8 打分；G_cal 在两个策略的 d0 上合并拟合；τ 和 FPR0 按 d8。
- **通过条件**：d8(P05) 上 AUC(G) ≥ 0.60（严格金标准，同 EXP-041）。只跑 P05 一行，所以只用 P05 的 AUC 判；P15 的 AUC 只报告。
- 不通过 → 这个 judge 不跑 RL，报告它的闸门数字，不换 judge。
- 闸门的输出（AUC、G_cal 系数、τ_len、FPR0_len）都是第 0 步的量，不是 RL 结果，不在盲区里。它们也是 EXP-047 的预测量在新 judge 上的输入。

## 4. 资产和许可

| judge | HF 仓库 | revision | 许可 | 来源和核对 |
|---|---|---|---|---|
| JL1 | meta-llama/Llama-3.2-1B-Instruct | 92131767… | Llama 3.2 Community License | HF 仓库是 gated。小文件从 a model mirror LLM-Research 官方副本下载，按 meta-llama 的 git blob id 核对；权重从 a model mirror 下载，备用不需申请的 unsloth 副本，按 meta-llama 的 LFS sha256（1ff795ff…）核对 |
| JL8 | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f2… | Llama 3.1 Community License | 同上（4 个分片的 sha256 2b1879f3… / 09d433f6… / fc1cdddd… / 92ecfe1a…） |
| JO1 | allenai/OLMo-2-0425-1B-Instruct | 48d788ec… | Apache-2.0，不 gated | Hugging Face，按 HF 的 sha256 / blob id 核对 |
| JO7 | allenai/OLMo-2-1124-7B-Instruct | 470b1fba… | Apache-2.0，不 gated | 同上 |

- 下载和核对：`code/server/dl46.py` + `code/server/manifest_<J>.json`（哈希在 261010 从 huggingface.co API 取得）。核对通过才写 `.READY`，并在 `/data/models/REGISTRY.md` 加一行。
- Llama 许可允许研究用途；gated 只是下载方式，许可条款照样适用。本实验是研究用途。
- 没有用任何凭据。

## 5. 运行

- 每个 judge 一条链，全部用 `expq`，不挂着等：
  1. 0 卡任务：下载并核对 judge（gpu-server-1 上先建 venv、复制策略）。完成后提交第 2 步。
  2. 1 卡任务 `exp046-gate-<J>`：（gpu-server-1 第一次先做环境核对）静态闸门 → 冒烟 → 用 `expq` 提交 4 个正式 run。
  3. 正式 run：`code/server/run46.sh <J> <seed> 200`，标签 `exp046-<J>-P05-seed<seed>`。
- 显存上限和占位（不影响计算）：JL1、JO1 用 EXP-039 P05×J15 的值（上限 20.5 GiB）；JL8、JO7 用 P05×J7 的值加 1 GiB（上限 34.5 GiB）。
- **冒烟**（种子 1，4 步，id `smoke-<host>-<J>`，不进分析）：只检查 rc = 0、steps.jsonl 4 行、rollouts.jsonl 256 行、evals.jsonl ≥ 2 行、显存峰值。不看任何结果量。
- 崩溃：同 EXP-039 / EXP-041。崩溃重跑一次；外部 OOM 最多重跑 3 次；第 0 步之前的本实验代码错误不算崩溃。

## 6. 端点（运行前写定，定义同 EXP-041）

所有端点用宽松金标准，τ_len 和 FPR0_len 取本格的值（`results/static_lenient.json` 的 `P05x<J>`），被骗 = G_cal + 1e-6 ≥ τ_len（EXP-039 补充 7）。h = (f/n − FPR0_len)/(1 − FPR0_len)。

| 编号 | 名称 | 定义 | 格的值 |
|---|---|---|---|
| E1 | 平台高度 H̄(101–150) | 第 101–150 步全部宽松错答合并算 h（EXP-041 第 6.1 节） | 种子算术平均 |
| E2 | 后段平台 H̄(151–200) | 同 E1，第 151–200 步 | 种子算术平均 |
| E3 | 起点 t½(0.5) | EXP-039 第 6.1 节主定义：5 步窗口，h ≥ 0.5 连续两个不重叠窗口，窗口错答 ≥ 20 条；没有 → 删失于 200 | AFT 格均值 μ（EXP-039 第 7.1 节，σ 先验 ln σ ~ N(ln 0.25, 0.5²)）；也报告几何均值 |
| E4 | 起点 t½(0.3) | 同 E3，阈值 0.3 | 同 E3 |
| E5 | 后段 h(126–150) | 第 126–150 步合并算 h（= EXP-047 的 O4） | 种子算术平均 |
| E6 | 上限 C | 20 个不重叠的 10 步块（1–10, …, 191–200），每块合并算 h，错答 < 20 条的块不计；C = 最大值 | 种子算术平均 |

- 主端点是 E1 和 E3 / E4，和 EXP-047 的 O1、O2、O3 对应。哪个用于检验，由 EXP-047 的冻结文件决定。
- 计算代码：`extract46.py`（= extract41）→ `blind/compact/`；E1–E4 用 `../EXP-041/code/analyze41.py` 的函数（onset() 和 AFT 抄自 analyze39），AFT 合并 EXP-039 的 20 个 run、EXP-041 的 6 个 run 和本实验的 run。
- 描述（不进检验）：
  - h 随步数的曲线（5 步窗口）。
  - 宽容金标准下的 E1（EXP-039 补充 2）；严格金标准下的 E1。
  - 格式：每 50 步块里带 "####" 的比例、写 "The answer is" 的比例（EXP-041 第 6.2 节的漂移）。
  - 第 0 步和最后的评测正确率（严格和宽松），t½ 处和最后的采样 KL。
  - 每格读 20 条被骗回答（`../EXP-039/code/fooled39.py` 的方法）。
  - 同家族里尺寸的顺序：Llama 1B / 3B / 8B，OLMo 1B / 7B。

## 7. 盲法（严格）

- 所有 RL 输出（stdout、rollouts、steps、evals、eval 输出、checkpoint）只写在两台机器的 `/data/exp046/blind/` 下。run 脚本不向 ts 日志打印任何内容。
- 本实验的 agent 不打印、不总结、不画任何结果量（h 曲线、起点、平台、正确率、KL），冒烟也一样。冒烟只报告"能跑、有日志"和显存。
- 闸门、环境核对的数字是第 0 步的量，可以报告。
- **揭盲条件**：分析只能在下面两件事都完成后开始：
  1. EXP-047 的预测量 SPEC 的 sha256 已经冻结并记在 `../EXP-047/PROCESS.md`；
  2. 最终预测量已经写入 `../EXP-047/results/FINAL_PREDICTOR.json`，它的 sha256 也记在 `../EXP-047/PROCESS.md`。
- 分析 agent 在开始前核对这两项，并把两个 sha256 和核对时间写进本实验的 `PROCESS.md`。核对不过就不分析。
- 揭盲前，任何人不打开 `blind/` 下的文件，也不运行 `extract46.py`。看完成标记（DONE / FAILED）和 ts 队列状态不算揭盲。
- 揭盲后不改第 6 节的端点定义。附加分析只作描述，写明是事后的。

## 8. 停止规则

- 每个通过闸门的 judge，4 个 run 由链一次提交。中途不看结果，不加种子，不加步数，不删格。
- 有 run 失败：按第 5 节重跑。重跑后，一个格完成的种子 < 3 → 这个格不进检验，只报告。
- 一个格第 101–150 步的宽松错答总数 < 200 条 → E1 在这个格不可识别，只报告起点。
- 闸门不过 → 不跑（第 3 节）。gpu-server-1 环境核对不过 → gpu-server-1 不跑（第 2.1 节）。
- **算力上限**：主任务 25 A100 小时（墙钟 × 卡数，含闸门、冒烟和失败的启动）。全部 EXP-046（含第 10 节的可选部分）≤ 60。重跑前检查"已用 + 预计 ≤ 上限"。

## 9. 算力

| 项 | 估计（A100 小时） | 依据 |
|---|---|---|
| P05×JL1，4 个 run | 3.6 | EXP-041 P05×JL 约 1.0 小时 / run；EXP-039 P05×J15 150 步约 0.75 |
| P05×JL8，4 个 run | 5.0 | EXP-039 P05×J7 250 步 1.46 小时 / run → 200 步约 1.2，8B 略大 |
| P05×JO1，4 个 run | 3.8 | 同 JL1 |
| P05×JO7，4 个 run | 4.8 | 同 JL8 |
| 闸门、环境核对、4 个冒烟 | 0.8 | EXP-041 闸门约 90 秒；EXP-047 实测 J7 打分约 184 秒 |
| **合计** | **约 18** | 上限 25 |

- 下载和建 venv 不占卡。

## 10. 可选的第二个任务（MATH）：本 SPEC 不包含

- 审稿人会要求 ≥ 2 个任务。计划是 MATH 子集上 J3 和一个新 judge。
- 它需要新的第 0 步样本、新的 G_cal 拟合、新的闸门，还要先确认 0.5B 策略（GSM8K 格式的 SFT）在 MATH 子集上的正确率不在地板。这些是新的设计决定。
- 所以它不在本 SPEC 里。主任务提交后，如果算力允许，另写带时间的补充节（或新的 EXP），先做第 0 步闸门，再预登记，再跑。

## 11. 已知局限（运行前）

- 每个格只有 4 个种子；格是检验的单位，留出的格只有 4 个（或更少，如果有 judge 没过闸门）。
- 两个家族分在两台机器上。gpu-server-1 的 venv 版本相同，但 GPU 机器不同。环境核对只检查 judge 的打分，不检查训练轨迹。
- OLMo-2 1B 是 0425 版，7B 是 1124 版，训练配方不完全相同。
- JL8 的 chat 模板有系统段（知识日期），JL1 / JL 也有。OLMo 的模板没有系统段。prompt 正文相同。
- 对照（EXP-039、EXP-041）不是同时跑的。
- 宽松和宽容金标准都不是人工判分。
