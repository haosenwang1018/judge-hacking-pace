# EXP-041 PROCESS

写法按 ASD-STE100 (simplified technical writing)。时间是。机器 gpu-server-2。远端目录 `/data/exp041`。

## 261010 布置（实验 agent）

- 05:0x：读 EXP-039 的 SPEC、PROCESS、SUMMARY 和代码。gpu-server-2 两卡 0 MiB，ts 队列（`/root/.ts.sock`）里没有任务。
- 05:0x：Llama-3.2-3B-Instruct 在 gpu-server-2 的 `/data/exp029/models/Llama-3.2-3B-Instruct`（EXP-029 下载）。直接读，不重新下载。
- 05:0x：建 `/data/exp041`。第 0 步数据从 `/data/exp039/data/step0` 硬链接；J15 / J3 / J7 的 readouts 复制过来。代码是 EXP-039 代码的副本（改动见 SPEC 第 2 节）。
- 05:07：JL 静态闸门（`code/static41.py JL`，GPU0，90 秒）**通过**：AUC(G) 0.763 / 0.698。宽松：τ_len 0.255 / 0.305，FPR0_len 0.450 / 0.539（`results/static_JL.json`、`results/static_lenient.json`）。J15 / J3 / J7 的宽松数字和 EXP-039 完全相同。
- 05:07–05:11：冒烟 P05×JL（种子 1，6 步，GPU0）：rc 0。每步 11–14 秒，评测约 41 秒。显存最大分配 23.2 GiB（含加载时的占位张量），上限 25.5。第 0 步正确率 0.33（同 EXP-039），第 0 步采样 KL −0.012（应为 0），第 5 步 0.70。
- 05:12–05:15：冒烟 P15×JL（种子 1，3 步，两卡）：rc 0。每步约 14 秒，评测约 67 秒。策略卡最大分配 29.0 GiB（上限 32.5）。第 0 步正确率 0.61（同 EXP-039）。
- 05:16：SPEC.md 写好，在任何正式 run 之前（SPEC 页头写的"05:2x"是估计，实际是 05:16）。sha256 = 57483b3b11bd4bbc1c9590b1c43e583571d4f7c477228adcc8cf18dc7eca5b15。代码：`grpo41.py` aad34467…，`code/server/run41.sh` 5318f179…，`extract41.py` 0e166434…。
- 05:17：用 `expq` 提交 6 个 run（ts ID 4–9）。提交后 20 秒：P05 s301、s302 在跑（各一卡）。

## 提交的任务

| 标签 | ts ID | run_id（`/data/exp041/runs/`） | 卡 | 步数 | 预计 A100 小时 |
|---|---|---|---|---|---|
| exp041-JL-P05-seed301 | 4 | 20261010-gpu-server-2-e41-P05JL-s301 | 1 | 200 | 1.2 |
| exp041-JL-P05-seed302 | 5 | 20261010-gpu-server-2-e41-P05JL-s302 | 1 | 200 | 1.2 |
| exp041-JL-P05-seed303 | 6 | 20261010-gpu-server-2-e41-P05JL-s303 | 1 | 200 | 1.2 |
| exp041-JL-P05-seed304 | 7 | 20261010-gpu-server-2-e41-P05JL-s304 | 1 | 200 | 1.2 |
| exp041-JL-P15-seed301 | 8 | 20261010-gpu-server-2-e41-P15JL-s301 | 2 | 150 | 2.2 |
| exp041-JL-P15-seed302 | 9 | 20261010-gpu-server-2-e41-P15JL-s302 | 2 | 150 | 2.2 |

- 合计约 9.2 A100 小时，加静态闸门和冒烟约 0.25。上限 12（SPEC 第 8 节）。
- 预计墙钟：P05 两批约 06:30、07:45 结束；P15 两个依次约 08:55、10:05 结束。**全部约 261010 10:05 完成**（别的任务占卡会推迟）。
- 标记文件：`/data/exp041/markers/<标签>.DONE` 或 `.FAILED`。ts 输出：`/data/ts-logs`。每个 run 的 stdout：`runs/<run_id>/stdout.log`。
- 查看队列要用 `TS_SOCKET=/root/.ts.sock ts`（expq 用这个 socket；不设时 `ts` 显示的是另一个空队列）。

## 结果在哪里

- 每个 run：`/data/exp041/runs/<run_id>/`：`rollouts.jsonl`（每条训练 rollout，含 G_cal、correct_lenient）、`steps.jsonl`、`evals.jsonl`、`eval_outputs_stepNNN.jsonl`、`config.json`、`run.json`。权重：`/data/exp041/ckpts/<run_id>/`。
- 第 0 步量：`/data/exp041/results/static_lenient.json`（cells `P05xJL`、`P15xJL`，也含 J15 / J3 / J7），本地副本在 `results/`。
- EXP-039 对照：本地 `../EXP-039/results/compact/*.json`（20 个 run 的逐步计数），`../EXP-039/results/analysis39.json`。

## 分析 agent 要做的事（按 SPEC 第 6–8 节，不改定义）

1. 在 gpu-server-2 上：`source code/server/env.sh; python code/extract41.py runs/20261010-gpu-server-2-e41-P*JL-s3??` → `results/compact/`。复制到本地 `results/compact/`，同时把小文件（run.json、config.json、steps.jsonl、evals.jsonl、stdout.log）复制到本地 `runs/`。
2. **主端点 H̄**（SPEC 6.1）：每个 run，第 101–150 步合计 f_wl / n_wl，h = (f/n − FPR0_len)/(1 − FPR0_len)，FPR0_len 用 `static_lenient.json` 的 `P05xJL` / `P15xJL`。格平均。
   - D_P05 = H̄(P05×JL) − H̄(P05×J3)，Welch t 95% 区间（J3 的 4 个值在 SPEC 6.1 表里：0.824、0.849、0.783、0.808）。用同一代码从 `../EXP-039/results/compact/` 重算 EXP-039 的 6 格，核对 SPEC 表里的数字。
   - P15 同样算（2 对 4），只作描述。
   - 判决按 SPEC 7.3（V1–V4）。
3. **次要 t½(0.5) 和 t½(0.3)**（SPEC 6.2）：用 `../EXP-039/code/analyze39.py` 的 `onset()`（thr = 0.5、0.3）。把 JL 两格加进 AFT 模型（CELLS 加 "P05JL"、"P15JL"；SIZE 加 "JL": 3.21；compact 目录合并 EXP-039 的 20 个和 EXP-041 的 6 个），同 σ 先验。报告 μ(P×JL) − μ(P×J3) 和 μ(P×JL) − μ(P×J15)，95% 剖面区间；对照 SPEC 7.2 的预测（P05×JL 几何均值 17–39 步，|Δ vs J3| < 0.41）。
4. 描述（SPEC 6.3）：h 曲线图（JL 叠在 J15、J3、J7 上，按 `~/.claude/` 记忆里的论文图风格）；每格读 20 条被骗回答（`../EXP-039/code/fooled39.py` 的方法）；第 0 步和最后的正确率、格式率；t½ 处采样 KL。
5. 停止规则核对（SPEC 第 8 节）：失败的 run、重跑、P05 第 101–150 步错答总数 ≥ 200、算力（墙钟 × 卡数，从 run.json 的 total_seconds 或 stdout 的 start/end）。
6. 在最终回复里给出 SUMMARY 全文（协调 agent核对后写入 `results/SUMMARY.md`）。分析 agent 不写 SUMMARY 文件。

## 风险（布置时已知）

- JL 的第 0 步能力（AUC 0.76 / 0.70，FPR0 0.45 / 0.54）接近 Qwen 1.5B judge，不接近 Qwen 3B。所以"按大小"和"按能力"两种 judge 主导的预测不同。SPEC 7.1 已把两种都写成判决。
- P15 只有 2 个种子，只作描述。

## 261010 分析（分析 agent）

- 6 个 run 都完成（DONE 标记，没有 FAILED，没有重跑）。`extract41.py` → `results/compact/`；小文件复制到 `runs/`。
- `code/analyze41.py`（主端点 H̄、Welch 区间、t½ 的 AFT，onset 和 AFT 抄自 analyze39，同 σ 先验）→ `results/analysis41.json`、`results/plateau41.csv`。EXP-039 的 6 格 H̄ 重算，和 SPEC 6.1 表一致。
- 判决 **V1**：D_P05 = +0.018，95% [−0.106, +0.143]。
- 事后描述（不进判决）：`results/blocks41_posthoc.csv`（各 50 步块的 h）、`results/format_drift41.json`（格式漂移、宽松评测正确率）、`results/fooled/`、`results/fooled_stats.json`。图：`figures/fig1_h_curves_P05`、`fig2_plateau_vs_auc`（`code/figures41.py`）。
- 算力：6 个 run 8.30 A100 小时，加静态闸门和冒烟约 8.55。SUMMARY 全文在分析 agent 的最终回复里，由协调 agent写入。
