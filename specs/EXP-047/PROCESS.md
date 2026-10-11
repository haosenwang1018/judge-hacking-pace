# EXP-047 PROCESS（追加式）

研究线 11，阶段 Question → Finding。写法按 ASD-STE100 (simplified technical writing)。

## 261010 布置（实验 agent）

### 冻结的哈希（提交 GPU 任务之前）

- **SPEC.md sha256 = `45513193ec9778591359a61b95b78ae482a968e93621a412621b16fafc0c5b20`**
- 代码（本地 `code/`，同一份在 gpu-server-2 `/data/exp047/code/`）：
  - `perturb47.py` ef1e5f88bac697318feb40c8c6c445690f248f7616a8f84e97c0e8454b4d7dc0
  - `probe47.py` 35301b1100d30b8dc407a7dbb765fb92065d5371c3bac61723a0d95734db1119
  - `predictors47.py` 1b1c761cb6cad4183f7027ed3d897f6630a9aeba7cb30996fc5d71a3ccfd5fe2
  - `common.py` 69ca8ff0…、`judge39.py` 89477e8a…、`stats.py` b05832f9…（EXP-039 原样复制）
  - `gpu-server-1/env.sh` 7987dc38…、`gpu-server-1/probe.sh` 06bf4a35…
- 改写文本：gpu-server-2 `/data/exp047/data/perturb/manifest.json` sha256 d9e62209d320cf66f8f2948920e93a4e1e7a66f985e32303f8d49d29070c1f54（里面有 16 个 jsonl 各自的 sha256）。
- **EXP-046 的揭盲条件**：本 SPEC 哈希不变，加上分析 agent 写出的 `results/FINAL_PREDICTOR.json` 的 sha256（SPEC 第 7 节第 5 步）。两者都记在本文件之后，EXP-046 才能揭盲。
- 本 agent 没有算过任何候选预测量的真实值。`predictors47.py` 只在一个随机的假 judge 上测过（输出已删除）。

### 关卡 1：撞车检查（Sonnet 子 agent，一手来源，12 个月和不限年份各查一次）

- **判决：没有撞车。** 没有已有工作用第 0 步、只看 judge 的测量，去预测生成式 LLM judge 在 RL 下被骗的起点、平台或上限，也没有跨 judge × 策略格做这件事。
- 最接近的工作（全文读过的标"全文"）：
  - 2607.05904（More Convincing, Not More Correct，全文）：第 0 步 FPR_base × (1 − EM) 的"风险分"能给被骗的**程度**排序。设置是自博弈 DPO，judge 和策略是同一个模型，结果量是差距大小，不是起点。→ 已加为基线 B3。
  - 2503.15477（Razin 等，What Makes a Reward Model a Good Teacher?，全文）：第 0 步策略样本上的奖励方差预测早期优化速度，比准确率好。标量 RM，结果量不是 hacking。→ 已作为候选 P4。
  - 2512.17375（AdvJudge-Zero，全文）：静态 FPR 低的加固 judge，RL 塌缩的种子更少。只有一个组合，不跨 judge 预测。
  - 2410.14872（PPE，摘要）：12 个静态指标对一次端到端 RLHF。不是 hacking 起点或平台。
  - 2210.10760（Gao 等，摘要）：过优化的缩放律，按 RM 大小事后拟合，没有第 0 步探针。
  - 2605.02909（Delay, Plateau, or Collapse，摘要）：结果类型（延迟、平台、塌缩）和我们一样，但预测量是注入的 verifier 错误模式，不是 LLM judge 的实测性质。
  - 2603.06621（PRM 三层攻击，全文）：没有检验扰动敏感度和 GRPO 结果的关系。
  - 2507.08794（One Token to Fool，摘要）：master key FPR，没有 RL 结果。
- **我们新的部分**：冻结、不给参考答案的生成式 judge；GRPO；judge × 策略的格点；起点 t½、平台 H̄ 作为结果量；最小错误改写、写法改写、表面改写这几种第 0 步探针；并且必须胜过已有的基线（AUC、FPR0、风险分、奖励方差）。
- X（推特）：在内置浏览器里只读搜了 3 次，1 次结果无关，2 次页面空白。**没有查成**，记为未完成。
- 来源已追加到 `~/.claude/skills/source-ledger/sources.md`。

### 关卡 1 其他项（简短）

- 重要性：如果有一个第 0 步测量能预测被骗程度，用 LLM judge 做 RL 的人可以在训练前挑 judge。没有它，"judge 越大越难骗"只是已有结论（2210.10760 等）。
- 可行性：只需要 judge 前向，约 0.6 A100 小时。但 n 只有 4 个 judge、8 格（SPEC 第 10 节）。本实验只能筛选，检验靠 EXP-046。
- 停止规则：SPEC 第 8 节 S1。

### 冒烟

- `perturb47.py` 在 gpu-server-2 CPU 上运行完成（改写已生成，计数见 manifest）。P05：d8 2048 条，含 "####" 1960 条，宽松判对 400 条。F1–F4 各保留 1950 条，S1、S3 1960 条，S2 939 条，E 397 条。P15：含 "####" 2033 条，宽松判对 964 条；F 各 2024 条，S2 834 条，E 963 条。E 改写里 m 之前找到数字的比例 0.98 / 0.97。没有超过 1500 字符的改写。
- 读了 2 条 E 改写和 F2、S2 各 1 条：只改了要改的地方。E 改写里，更早一行重复说的旧数字不会改（例："averaged 5 points … averaged 6 points … #### 6"）。这是"最后一步不一致"的流畅错答，符合 SPEC 3.3 的意思。
- `probe47.py` 在 CPU 上用 J15 跑 8 条（`--limit 8`）：所有改写文件都能打分；和已存 G 的差最大 0.18 logit（CPU float32 对 GPU bf16，预期内）。冒烟输出带 `smoke` 后缀，分析时忽略。

### 提交的 GPU 任务（gpu-server-2，expq，各 1 张卡）

| 标签 | ts ID | judge | 预计 |
|---|---|---|---|
| exp047-probe-J15 | 38 | Qwen2.5-1.5B-Instruct | 约 5 分钟 |
| exp047-probe-J3 | 39 | Qwen2.5-3B-Instruct | 约 6 分钟 |
| exp047-probe-JL | 40 | Llama-3.2-3B-Instruct | 约 6 分钟 |
| exp047-probe-J7 | 41 | Qwen2.5-7B-Instruct（bs 8） | 约 12 分钟 |

- 合计约 0.5 A100 小时，上限 2。队列前面有 EXP-045 的 10 个任务，开始时间取决于它们。
- 标记：`/data/exp047/markers/<标签>.DONE` 或 `.FAILED`。`markers/WAKE_WHEN` 列出上面 4 个标签。
- 日志：`/data/exp047/logs/probe_<J>.log`。输出：`/data/exp047/data/probe/<P>/<改写>.G_<J>.npy`、`d8.G_<J>.npy`，`/data/exp047/results/probe_<J>.json`。

## 分析 agent 要做的事（按 SPEC 第 4–8 节，不改定义）

0. 只读 SPEC、本文件和结果文件。不打开 `/data/exp046`。先核对 SPEC.md 的 sha256 等于上面的值。
1. 核对 4 个任务都是 DONE。看 `results/probe_<J>.json` 的 consistency（重打的 d8 和已存 G 的差，GPU 上应很小）。有 FAILED：按 SPEC 第 8 节 S2 重跑一次（`expq /data/exp047 1 exp047-probe-<J>-r1 -- bash /data/exp047/code/server/probe.sh <J>`）。
2. 在 gpu-server-2 上：`source /data/exp047/code/server/env.sh; python code/predictors47.py $STEP0 $READOUTS J15 J3 J7 JL` → `results/predictors47.json`（8 格 × B1–B3、P1–P8，带 bootstrap 区间）。复制到本地 `results/`。
   - 核对：B1、B2 应等于 `EXP-041/results/static_lenient.json` 的 auc_len、FPR0_len；`_tau` 等于 tau_len。
3. 结果量（SPEC 第 5 节），8 格：
   - O1 H̄：`EXP-041/results/analysis41.json` → `plateau_cells`。
   - O2 ln t½(0.3)：用 `EXP-041/code/analyze41.py` 的 onset(thr = 0.3) 和 AFT（同 σ 先验）算格均值 μ。
   - O3 ln t½(0.5)：`analysis41.json` → `onset`。
   - O4 第 126–150 步的 h，从 `EXP-039/results/compact/` 和 `EXP-041/results/compact/` 算，种子平均。
   - 种子层的值也留着，用于 SPEC 6.1 的"并列"判断（Welch 95%）。
4. 排序（SPEC 6.1）：符号调整后的 8 格 Spearman；每行（n = 4）的 Spearman 和是否全对；行内两两一致率（去掉并列对）；关键对 JL–J15、JL–J3。只给 8 格 Spearman 的精确置换分布作参考，不当证据。
5. 留出家族（SPEC 6.2）：6 个 Qwen 格拟合 OLS（结果 ~ a + b × 候选 + c × [P15]），预测 2 个 JL 格。O1、O2 各一次。和 B1、B2、B3 比误差。
6. 改写的机制核对（SPEC 6.3）：计数、丢弃、截断；每个 judge 读 10 条 E、10 条 F1。
7. 按 SPEC 第 7 节选出最终预测量，写 `results/FINAL_PREDICTOR.json`（候选名、代码路径和 sha256、8 格 OLS 系数和残差标准差，O1、O2 各一个）。算它的 sha256，写进最终回复（协调 agent记到本文件）。没有合格的候选 → 文件里写 "none" 和三个基线的拟合。
8. 停止规则 S1–S3 和算力（`probe_<J>.json` 的 seconds）逐条对照。
9. 最终回复里给出 SUMMARY 全文，必须写明：n = 4 个 judge / 8 格；候选是在知道结果后定的（尤其 P6 来自 EXP-041 的写法漂移）；这里只是筛选，检验靠 EXP-046。分析 agent 不写 SUMMARY 文件，协调 agent核对数字后写入 `results/SUMMARY.md`。

- 261010 11:40（协调 agent）：分析完成。最终预测量 P1（校准斜率）。锁定：`results/FINAL_PREDICTOR.json` sha256 `8c1eb85a2e601c8ed563ad70824b4291dec3de9929ec0d50c225ba750d138af1`；`code/analysis47.py` `b68aa2ddb941503a320a19a0d0714dfc9df70d005bdc564225221b20ae73e414`；`code/mech47.py` `e1a693613469c5b9951a41635596c1ce9742551c4fe1e5db7caf5a705fb79406`。协调 agent已核对这三个哈希。从这一刻起，EXP-046 满足揭盲条件中的"预测量已锁定"一项。
