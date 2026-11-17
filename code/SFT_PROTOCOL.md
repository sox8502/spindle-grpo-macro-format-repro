# SFT1140 on Spindle: 65k context, 4 H200 trainer

Same frozen 1140 dataset and recipe as Tinker run 4623cb09, retrained from base on a separate Spindle v0.1.4 frontend (`spindle_sft_65k.py`, deployed with `spindle deploy`). Deliberate differences: context override to 65536 instead of the stock 32k, and H200s on the stock 4-GPU trainer (DP4); no per-model seed (Miles rejects it), config seed still fixes shuffle order; Miles LoRA parameterization is not equivalent to Tinker's. Evaluate validation180 only. Output: sft-1140-spindle-65k-4h200-2026-11-06/pipeline.

# SFT1140: exclude the exact earlier RL1K

Retrain from the original Qwen/Qwen3.5-4B base, never from SFT1200 weights. The previous1200 checkpoint has seen60 queries from the prior RL1K and is ineligible for this disjoint sweep.

## Frozen datasets

Original1200 accepted source: teacher-1200-2026-10-21/collection/accepted.jsonl, SHA256 ef4c114e95614b4d0d483c95cfc4925c04a31c175023bed37faec1c876ae1a6e.

Exact earlier RL1K is committed as rl_training.jsonl, byte-identical to saved earlier RL runs and published Mixed1K; SHA256 721616fca5835a20c28feb650ba26c012b7a26cd069e460e10aae6fca37d12f2. The user clarified this exact dataset and authorized retraining SFT to remove its60 overlaps. Do not select a new1K or substitute the separate2000-query reserve.

Exclude exactly60 matching IDs (30easy,18medium,12hard), preserve every retained original JSONL line byte-for-byte. Exactly1140 examples; SHA256 9a0f189c2d92d717e70af8c8942058c6a1d1e64705f85ef617a173030783bc08. No changes to reasoning, actions, observations, rejected-turn masks, conversion, or scoring. Verify exact-query and ID disjointness with RL1K and the prior2000 reserve; verify train/RL queries disjoint from validation180/test1000. Manifest records all excluded IDs.

## SFT and evaluation

Use unchanged previous encoder, pinned tokenizer851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, rank32 LoRA on attention/MLP/unembedding, LR5e-5, seed42, batch32, four epochs, cosine LR with10%warmup, original optimizer and complete65536-token context. This gives144updates and4560 trajectory visits. No truncation. Optimizer-state checkpoints every6updates and final state/sampler live on Tinker. Fresh durable output sft-1140-rl-disjoint-2026-10-23/pipeline cannot reuse1200-trained weights. Disk safety reserve remains1GiB. No spending cap.

After full SFT, evaluate validation180 and officialtest1000 using unchanged matched settings: temperature.6,top_p.95,max8192tokens perturn,60turns,visible32768/native65536context,concurrency12,seed42+caseindex*1000+turnindex. Preserve all model failures; infrastructure/formatter/scorer failures block final comparison. Reuse only completed integrity-checked cases after interruption. Same official online scoring authorization and ASCII-space insertion after Flight Number: as prior SFT runs; preserve original submission and exact edits. Compare with historical185-example results; also compare completed1200 results in final handoff. Private service label revision cannot independently be pinned.

## Authorized RL continuation, after SFT evaluation

Use this new1140-trained checkpoint for all eight arms, independently loading its weights with fresh RL optimizers. Freeze exact rl_training.jsonl across all eight arms. Seed/settings fixed, zero injected tool failures, track attributable costs and throughput without a hard spending cap.

Pair1: GRPO SUM; GRPO Curriculum. Pair2: DAPO SUM; DAPO Curriculum. Pair3: GRPO MACRO; GRPO SUCCESS. Pair4: DAPO MACRO; DAPO SUCCESS.

Initially launch only pair1, concurrently. Report each pair and pause for explicit approval before the next pair. Train epochs1–3 only while preserving the original five-epoch learning-rate horizon and reward schedule. Curriculum epochs1–2 SUM,epochs3–4 MACRO,epoch5 SUCCESS; standalone rewards fixed. Do not use the old progress-thirds schedule unchanged. Evaluate initialization and each epoch; save resumable optimizer state, data cursor, and schedule position. Report validation success by trip length, model vs infrastructure failures, usage costs and throughput. No epoch4/5 continuation without approval. This SFT launcher does not auto-launch RL; the agent must verify implementation and launch the first pair after reviewing SFT evaluation.
