# Spindle vs Tinker: GRPO/MACRO format-failure gap

Repro bundle for one anomaly in our Agent-STAR (TravelPlanner) replication on Qwen3.5-4B. In the GRPO/MACRO arm, the run started from the **Spindle** SFT checkpoint had a much higher format-failure rate than the run started from the **Tinker** SFT checkpoint: **40.2% vs 6.7%** of rollouts in batches 53–62. The other arms (GRPO SUM, DAPO SUM/MACRO) did not show this gap.

A "format failure" is a rollout that ends with the environment error `Format error occurs in the previous turn!` (`failure_categories.model_format_failure` in `data/format_failures_by_batch.csv`).

## Measured gap

Training rollouts (32 groups × 8 paths = 256 per batch), from `data/format_failures_by_batch.csv`:

| Batches | Spindle-SFT → GRPO/MACRO | Tinker-SFT → GRPO/MACRO |
|---|---|---|
| 1–10 | 14.3% | 8.6% |
| 11–20 | 27.7% | 11.8% |
| 21–31 (epoch 1 end) | 34.9% | 7.0% |
| 32–42 | 38.3% | 6.6% |
| 43–52 | 41.6% | 6.1% |
| **53–62** | **40.2%** (1029/2560) | **6.7%** (172/2560) |

Validation (180 cases, temp 0.6), from `data/*_validation_by_epoch.json`:

| Checkpoint | Spindle: successes / format terminations | Tinker: successes / format terminations |
|---|---|---|
| SFT init (epoch 0) | 24 / 13 | 33 / 18 |
| After epoch 1 (batch 31) | 45 / 62 | 36 / 17 |
| After epoch 2 (batch 62) | 55 / 56 | 69 / 4 |

Both SFT checkpoints start at a similar format-failure rate (13 vs 18 of 180 cases). The gap develops during RL, so the RL updates on Spindle drive it.

## SFT checkpoints

Both are Qwen3.5-4B, LoRA rank 32, trained on the same 1140 traces (`sft-checkpoints/spindle/training.jsonl`, SHA-256 `9a0f189c…83bc08`).

| | Spindle SFT (the anomalous arm's init) | Tinker SFT (control arm's init) |
|---|---|---|
| Training run | 144/144 steps, 4×H200, 65k context | Tinker managed LoRA |
| Native state | `tinker://3ceade808c2e312f65ef52da42069ccf/weights/sft-final` (private Spindle volume) | `tinker://d40167e7-406d-5be8-801c-80bc5c2a9029:train:0/weights/sft-final` (private Tinker account) |
| PEFT export | release asset `spindle-sft1140-adapter_model.safetensors` (96 MB, BF16) | release asset `tinker-sft1140-adapter_model.safetensors` (278 MB, F32) |
| Config | `sft-checkpoints/spindle/adapter_config.json` | `sft-checkpoints/tinker/adapter_config.json` |
| SFT validation | 36/180 | 38/180 (McNemar p = 0.87) |

The two adapters have different structures. These differences carry into RL, so they are the main suspects:

- **Target modules.** The Spindle/Miles adapter has **no LoRA on Qwen3.5's linear-attention (Gated DeltaNet) layers**. It has 129 exported projection pairs (162 native tensors across 81 pairs), covering MLP, full-attention q/k/v/o and `lm_head`. The Tinker adapter is `all-linear`, with 249 pairs, including 120 linear-attention pairs.
- **Shared input factors.** Miles MultiLoRA fuses QKV and gate/up, so those projections share one LoRA A matrix. Tinker trains independent factors for each projection.
- **Precision.** The Spindle export is BF16. The Tinker export is F32.
- **Init seed.** Miles multi-LoRA rejects a per-model seed, so `sft_backend.py` passes `seed=None` on Spindle. The data-shuffle seed (42) is unchanged.

## Code

`code/` is the exact tree of the Spindle arm's commit (`6d51ec6`, W&B run dirs removed). The repo has two redactions. The live Modal frontend URL is replaced with `https://YOUR-SPINDLE-FRONTEND.modal.run`, and local artifact paths are replaced with `~/star-artifacts`. `upstream/` is the vendored [Agent-STAR](https://arxiv.org/abs/2603.21972) source, which provides the environment, tools, scorer and prompts.

Files that matter for this bug:

| File | Role |
|---|---|
| `code/rl_config.json` | Spindle GRPO/MACRO config: lr 2e-5 constant, 32×8, clip 0.2/0.28, 30,720 context, temp 1.0 |
| `code/rl_run.py` | RL controller (Tinker SDK API, pointed at Spindle with `spindle_base_url` / `SPINDLE_BASE_URL` + `SPINDLE_API_KEY`) |
| `code/rl_rollout.py` | Multi-turn rollouts, formatter calls, crash-recovery replay of journals |
| `code/rl_objective.py` | GRPO advantage / loss weighting (source-dynamic microbatch, virtual DP 8) |
| `code/rl_cost.py` | Reward formatter client (deepseek-chat via OpenRouter), retry policy |
| `code/sft_backend.py` | SFT/LoRA client creation (`seed=None` on Spindle) |
| `code/spindle_sft_65k.py` | Spindle deployment config (Qwen3.5-4B LoRA preset, 65k ctx, H200, ≤6 rollout replicas) |
| `code/RL_PROTOCOL.md` | Full frozen RL protocol |
| `spindle-to-tinker-arm.diff` | Everything that differs in the Tinker control arm (commit `d3b0c8b`) |

### Differences between the two RL arms besides the backend

These come from `spindle-to-tinker-arm.diff`. They should be ruled out before blaming numerics:

1. **Init checkpoint**: the Spindle SFT vs the Tinker SFT (see above).
2. **Formatter retry policy**: the Tinker arm retries **HTTP 503** from the formatter with backoff. The Spindle arm retries only 429. Formatter outages on Spindle therefore surface as errors instead of being retried. Check whether any formatter 503s land in reward or format failures.
3. **Crash/resume (Spindle only)**: the Spindle run was interrupted in batch 19. 25 truncated gzip group journals were set aside, and the run restored optimizer update 18 and regenerated the affected paths (runs `56b5a7a4` → `521da198` → `8a23dd89` → `f1e4ea11`). The Tinker run had resumes too (`cf554528` → `e4cc8a6e` → `9cd75ae3` → `58267069`). The divergence is already visible in batches 1–10 (14.3% vs 8.6%), so the resume probably does not cause the gap on its own.
4. **Deployment**: on Spindle, 4 trainer and up to 6 rollout H200s were shared by four concurrent LoRA clients (the other arms ran alongside it). Tinker used native shared LoRA workers.

## Reproducing

```bash
cd code
uv sync --locked
# RL reads the SFT dataset + checkpoint pointer from here:
mkdir -p ~/.local/share/openresearch/files/modal-blog/agent-star/sft-1140-spindle-65k-4h200-2026-11-06/pipeline
cp ../sft-checkpoints/spindle/{training.jsonl,checkpoint.json} \
   ~/.local/share/openresearch/files/modal-blog/agent-star/sft-1140-spindle-65k-4h200-2026-11-06/pipeline/
# edit rl_config.json: spindle_base_url, budget_path; rewrite checkpoint.json to your own restored SFT state
export SPINDLE_API_KEY=... OPENROUTER_API_KEY=... WANDB_API_KEY=... ORX_RUN_ID=$(uuidgen)  # TINKER_API_KEY for the control arm
uv run --locked python rl_run.py
```

`checkpoint.json` must point to a training state that your backend can restore. The `tinker://` paths above are private. Load the released PEFT adapter into your own Spindle/Miles deployment (or Tinker), save a native state, and put that path into `checkpoint.json` and `rl_config.json` (`sft_state` / `sft_sampler`). For the control arm, apply `spindle-to-tinker-arm.diff` and start from the Tinker adapter.

Cheap checks before a full 62-batch run:

- Restore each adapter on both engines and compare token log-probs on fixed inputs. A mismatch in sampler vs learner log-probs on Spindle would bias the GRPO ratio.
- Run batches 1–10 only. The gap is already 14.3% vs 8.6% there.
- Swap only the init: Tinker SFT → Spindle RL, and Spindle SFT → Tinker RL. This separates "checkpoint structure" from "RL engine".

## Data

- `data/format_failures_by_batch.csv`: per-batch counts for both arms (batches 1–62)
- `data/{spindle,tinker}_validation_by_epoch.json`: full validation metrics at init, epoch 1 and epoch 2
- `sft-checkpoints/spindle/comparison.json`: Spindle SFT validation summary
