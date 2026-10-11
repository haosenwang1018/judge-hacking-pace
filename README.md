# Who Sets the Pace of Judge Hacking? Judge Capability versus the Policy–Judge Gap

Code, per-step logs, analysis scripts and read answer samples for the paper of the same title.

**Summary.** We train small Qwen2.5 policies (0.5B and 1.5B, SFT on GSM8K) with GRPO against a frozen,
reference-free LLM judge, and measure how fast and how deeply the judge is fooled (the fraction *h* of wrong
answers that the judge accepts). We cross policy size with judge size and family (Qwen2.5 1.5B/3B/7B,
Llama-3.2-3B, Llama-3.1-8B, OLMo-2 1B/7B). Onset and plateau cluster by judge, not by the policy–judge
capability gap: equal-gap pairs differ strongly, and within a family the larger judge is fooled less deeply.
Judge-RL did not detectably raise held-out accuracy. A step-0 predictor selected and hash-locked on screening
judges was not better than step-0 AUC in a pre-registered blind test on three held-out judges; three cells cannot separate the two.
The study ran in five pre-registered stages (38 GRPO runs, about 57 A100-hours).

## Repository layout

```
code/<stage>/        scripts actually used (training, judge scoring, extraction, analysis, plotting)
code/<stage>/server/ launch and environment scripts as run on the GPU servers (paths are server paths)
code/paper/          paper-level analyses (analysis/) and figure scripts (figures/)
specs/<stage>/       pre-registration documents: SPEC.md and PROCESS.md (run log)
results/<stage>/     summaries, analysis JSON, per-run compact logs (compact/), per-step logs (runs/<run_id>/)
results/paper/       outputs of code/paper/analysis (all figures and paper tables are built from these)
data/EXP-046/        step-0 samples (256 problems x 8 per policy) and judge scores used for step-0 predictors
data/EXP-034/        model file manifests (HF API listings used to verify weight hashes)
data/answer_samples/ the 520 answers read for the paper (see below)
tools/make_workdir.sh  rebuilds the original directory layout with symlinks so the scripts run unchanged
```

`paper/` will be added after the arXiv release.

**Language note.** `SPEC.md`, `PROCESS.md` and `results/*/SUMMARY.md` are the original pre-registration
documents and run logs. They are written partly in Chinese and are kept as they were (only redacted, see
"Anonymization").

## Stages

Internal stage IDs (`EXP-0NN`) are used as directory names because the documents and scripts refer to them.

| Paper stage | Directory | What it is | Runs | SPEC sha256 recorded before the runs (addenda) |
|---|---|---|---|---|
| Stage 1 | `EXP-034` | P0.5/P1.5 x Qwen2.5-1.5B/7B judges, GRPO | 8 | `5daa7235b121d5e45cc02a0aeca098f224696e061c8a1595d9391c010c6e041d` (addendum 1 `a908d589387dc93f584ae79499d0bbbed5f9ef2a249db84c09c91bdabb310179`, addendum 2 `d1cbaef6…`) |
| Stage 2 | `EXP-039` | adds Qwen2.5-3B judge and seeds; lenient label, AFT onset model, rules R1–R4 | 12 | `fa81fb17df68e4081f724eb51ae7538ab6fb885dc7cbef16ebf80a971c82f689` (addenda 1–7; last: `f60e4c688a10b2fb9b76548125383713e5a3b23148edb751dc549dcfbb833847`) |
| Stage 3 | `EXP-041` | Llama-3.2-3B judge, plateau endpoint | 6 | `57483b3b11bd4bbc1c9590b1c43e583571d4f7c477228adcc8cf18dc7eca5b15` |
| Stage 4 | `EXP-047` | screening of step-0 predictors (judge forward passes only, no RL) | 0 | `45513193ec9778591359a61b95b78ae482a968e93621a412621b16fafc0c5b20`; locked predictor `results/EXP-047/FINAL_PREDICTOR.json` sha256 `8c1eb85a2e601c8ed563ad70824b4291dec3de9929ec0d50c225ba750d138af1` |
| Stage 5 | `EXP-046` | blind test on held-out judges (Llama-3.1-8B, OLMo-2 1B/7B; Llama-3.2-1B failed the gate) | 12 | `e67f105ca25aad7f5540e751b1d192858a4887cad34fc6bab9a5be4187c3282b` |
| Post-hoc | `paper` | gold accuracy, capability gap, onset vs rollout peak, parser audit (no new runs) | 0 | — |

Not reported in the paper, included for completeness:

| Directory | What it is | SPEC sha256 recorded |
|---|---|---|

**Hash note.** The hashes above are of the unredacted originals. The copies in `specs/` were redacted for
anonymous review (host names, local paths), so their sha256 differs. The unredacted files will be released
after review.

## Cell names

`P05`, `P15` = Qwen2.5-0.5B / 1.5B base after SFT (paper: P0.5, P1.5).
Judges: `J15`, `J3`, `J7` = Qwen2.5-1.5B/3B/7B-Instruct (Q1.5, Q3, Q7); `JL` = Llama-3.2-3B-Instruct (L3);
`JL8` = Llama-3.1-8B-Instruct (L8); `JL1` = Llama-3.2-1B-Instruct (failed the gate); `JO1`, `JO7` =
OLMo-2-0425-1B-Instruct / OLMo-2-1124-7B-Instruct (O1, O7).
A cell is policy x judge, e.g. `P05JL8`. Run IDs: `<date>-<host>-e<stage>-<cell>-s<seed>`.

## Map from the paper to files

| Paper item | Script | Inputs / outputs |
|---|---|---|
| Fig. 1 (onset vs step-0 capability gap), Fig. 5 (fooled fraction curves) | `code/paper/figures/make_fig1_gap.py` | `results/paper/gap261011.json`, `results/EXP-046/analysis46.json`, `results/EXP-041/static_lenient.json`, `results/EXP-039/compact/`, `results/EXP-041/compact/` |
| Table 1 (pre-specified pairs), Sec. 4, Table of contrasts | `code/paper/analysis/gap261011.py`, `extra261011.py` | `results/paper/gap261011.json`, `extra261011.json` |
| Sec. 3, Fig. 3 (held-out gold accuracy), gold table | `code/paper/analysis/extract_remote.py` (run on the servers over saved greedy eval outputs) → `gold261011.py` → `code/paper/figures/make_fig3_gold.py` | `results/paper/evals_lenient.json` → `gold261011.json` |
| Fig. 4 (training-rollout accuracy) | `code/paper/figures/make_fig4_rollout.py` | `gold261011.json`, `analysis46.json` |
| Onset vs rollout-peak count | `code/paper/analysis/onset_peak261011.py` | `results/paper/onset_peak261011.json` |
| Sec. 5, Fig. 2 (blind test of step-0 predictors) | `code/EXP-047/predictors47.py`, `analysis47.py`; `code/EXP-046/analysis46.py`; `code/paper/figures/make_fig2_blind.py` | `results/EXP-047/{predictors47,analysis47,FINAL_PREDICTOR}.json`, `results/EXP-046/{analysis46,predictors46}.json` |
| Sec. 5.1 (judge families), all-cells table, 38-run AFT fit | `code/EXP-046/analysis46.py` (AFT and stats code in `stats.py`); `desc46.py` (`t_half46.json`) | `results/EXP-046/analysis46.json`, `t_half46.json`, `results/EXP-04*/static_lenient.json` |
| Stage verdicts (App. A) | `code/EXP-034/analyze34.py`, `code/EXP-039/analyze39.py`, `code/EXP-041/analyze41.py` | `results/EXP-034/analysis_stage1.json`, `results/EXP-039/analysis39.json`, `results/EXP-041/analysis41.json` |
| Offline transfer to Llama | `code/EXP-039/llama39.py` | `results/EXP-039/llama_rescore.json` |
| Format drift | no saved script (computed inline from the compact logs during stage 3; see `specs/EXP-041/PROCESS.md`) | `results/EXP-041/format_drift41.json` |
| Fooled answers read (20 per cell) | `code/EXP-039/fooled39.py`, `read_fooled39.py`; `code/EXP-041/fooled41.py`; `code/EXP-046/extract46.py` | `results/EXP-039/fooled/`, `results/EXP-041/fooled/`, `results/EXP-046/fooled_*.jsonl`, `fooled_read_counts.json` |
| Parser audit (300 answers), App. examples | `code/paper/analysis/extract_remote.py`, `show.py` | `results/paper/audit_sample.json`, `parser_audit261011.json`, `data/answer_samples/answer_samples.json` |
| Compute | `code/EXP-039/compute39.py` | `results/EXP-039/runs/driver.log`, `pids.txt` |

## Answer samples

`data/answer_samples/answer_samples.json` holds the 520 answers read for the paper:

- `set = "parser_audit"` (300): 50 per cell for P05J7, P15J7, P05JL, P15JL, P05JL8, P05JO7. Lenient-wrong
  training rollouts with `G_cal >= tau` in steps 101–150, pooled over seeds, sampled with `random.Random(0)`.
  Fields `audit_parser_error` and `audit_label` give the hand label (a parser error = the stated final answer
  equals the reference but the lenient parser read another number). `results/paper/audit_sample.json` is the
  compact version that was read (question and answer tails only); the full texts were re-read from the saved
  rollouts and matched to it.
- `set = "fooled_read"` (220): 20 fooled wrong answers per cell for 11 cells.

Each record has the cell (internal and paper name), policy, judge, run, seed, training step, question,
reference answer (`gold`), the verbatim answer, and the calibrated judge score `judge_score_G_cal`.

## Environment

Training and judge scoring (GPU servers):

- Python 3.12.3, torch 2.9.1 (CUDA 12.8), transformers 5.18.0, trl 1.14.1 (GRPOTrainer, SFTTrainer),
  accelerate 1.15.0, datasets 5.1.0, tokenizers 0.23.2, safetensors 0.8.0, numpy 2.5.3, pandas 3.0.6.
  Full list (75 packages): `code/EXP-046/server/freeze_gpu-server-2.txt`.
- NVIDIA A100-PCIE-40GB GPUs.

Analysis and figures (CPU): Python 3.13.12, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2.

## How to reproduce

The scripts were written for our servers. Model paths, `/data/exp0NN` directories and the job queue
(`expq`, a wrapper around task-spooler) appear in `code/*/server/*.sh`; adapt them to your machine.

1. **Models and data.** Download the models listed under "Licenses" at the revisions given in the SPECs.
   GSM8K is `openai/gsm8k` (revision `740312ad`), shuffled with seed 0: 2,000 training problems for SFT, the
   other 5,473 for RL, the first 200 shuffled test problems for held-out evaluation (`code/*/common.py`).
2. **SFT policies.** `code/EXP-034/sft34.py <run_id> <seed> <out_dir>` with `EXP013_BASE_MODEL` set to
   Qwen2.5-0.5B or Qwen2.5-1.5B (seed 2 gives the policies used here; `code/EXP-034/server/step0.sh`).
3. **Step-0 samples and judge calibration.** `gen_step0.py` (samples), `static*.py` (judge scores, AUC,
   calibration `G_cal`, readouts JSON), `lenient_static*.py` (tau and FPR0 under the lenient label).
4. **GRPO.** `code/EXP-046/grpo46.py` (latest version; earlier stages use `grpo34.py`, `grpo39.py`,
   `grpo41.py`). Example (see `code/EXP-046/server/run46.sh`):
   ```
   python code/grpo46.py --steps 200 --eval-every 10 --lr 3e-6 --scale group --readout G_cal --micro-bs 2 \
     --sft-path <P05> --judge-path <judge> --readouts-json data/readouts_<J>_P05.json --seed 301 --run-id <id>
   ```
   Each run writes `steps.jsonl`, `evals.jsonl`, `rollouts.jsonl` and greedy `eval_outputs_step*.jsonl`.
5. **Extraction.** `extract39.py` / `extract41.py` / `extract46.py` turn a run directory into
   `results/<stage>/compact/<run_id>.json`. `code/paper/analysis/extract_remote.py` builds
   `evals_lenient.json` and the parser-audit pool from the saved eval outputs and rollouts.
6. **Analysis and figures.** From the repository root:
   ```
   bash tools/make_workdir.sh
   cd workdir/PAPER-T11/rev261011 && python gold261011.py && python gap261011.py && python extra261011.py && python onset_peak261011.py
   cd ../figures && for f in make_fig*.py; do python $f; done
   ```
   Stage analyses run the same way, e.g. `cd workdir && python EXP-046/code/analysis46.py`.
   We checked that `gold261011.py`, `gap261011.py`, `extra261011.py`, `onset_peak261011.py`,
   `analyze39.py`, `analyze41.py` and `analysis46.py` reproduce the included JSON files byte for byte, and
   that all figure scripts run. Note that these scripts overwrite the JSON files in `results/` through the
   symlinks.

Not included (too large, kept on the servers): model weights and checkpoints, `rollouts.jsonl` (every
training rollout), greedy `eval_outputs_step*.jsonl`, and stdout logs.

## Compute

A100-hours = wall-clock hours x GPUs held, including failed launches. Stage 1: 13.45; stage 2: 22.9;
stage 3: 8.55; stage 4: 0.44; stage 5: about 11.5. Total: about 57 A100-hours. The post-hoc analyses used
no GPU.

## Licenses

- **Code:** MIT (`LICENSE`).
- **Generated model outputs and result files** (answers, logs, JSON): CC BY 4.0.
- **GSM8K** questions and reference answers that appear in the data files: MIT. Cobbe et al. (2021),
  "Training Verifiers to Solve Math Word Problems", arXiv:2110.14168.
- **Models** (license checked from the LICENSE file or model card shipped with the downloaded weights, unless
  marked otherwise):

| Model | Role | License |
|---|---|---|
| Qwen2.5-0.5B, Qwen2.5-1.5B (base) | policies | Apache-2.0 |
| Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct | judges | Apache-2.0 |
| Qwen2.5-3B-Instruct | judge | **Qwen Research License** (2024-09-19), research and non-commercial use only; not Apache-2.0 |
| Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct | judges | Llama 3.2 Community License (1B: per the stage-5 SPEC; license file not re-read) |
| Llama-3.1-8B-Instruct | judge | Llama 3.1 Community License |
| OLMo-2-0425-1B-Instruct, OLMo-2-1124-7B-Instruct | judges | Apache-2.0 (Hugging Face model card) |

## Anonymization

For anonymous review, host names were replaced by `gpu-server-1/2`, local home paths by `<HOME>`, and a
reference to a private style guide by its public name. Run IDs, file names and paths inside logs were changed
the same way, so all files stay consistent. Some SPEC/PROCESS documents mention internal notes (for example
（内部笔记）, `NOTES/…`, `PAPER-T11/reviews/…`) that are not part of this release.

## AI use

The code and the documents in this repository were written with heavy use of Claude (Anthropic) agents;
see the AI use disclosure in the paper.
