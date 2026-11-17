# Clean SFT1140 RL sweep: first pair

## Scope and approval boundary

Eight configurations are planned: GRPO/DAPO × SUM/MACRO/SUCCESS/Curriculum. Only first pair (GRPO SUM + GRPO Curriculum) is authorized now, concurrent. Later pairs, in order: DAPO SUM+Curriculum;GRPO MACRO+SUCCESS;DAPO MACRO+SUCCESS. After each pair, report and PAUSE for explicit approval. Do not launch any later pair automatically or extend beyond three epochs.

## Frozen initialization and data

Each arm independently loads weights (fresh RL optimizer) from clean1140 SFT final state tinker://d40167e7-406d-5be8-801c-80bc5c2a9029:train:0/weights/sft-final. This is NOT SFT1200, SFT185 or priorRL66. SFT1140 exact dataset SHA256 9a0f189c2d92d717e70af8c8942058c6a1d1e64705f85ef617a173030783bc08. Clean SFT validation38/180; officialtest154/1000. The RL validation harness/prompt/context differs, so its required initialization evaluation is separately measured.

Exact prior RL1K is committed rl_training.jsonl, SHA256 721616fca5835a20c28feb650ba26c012b7a26cd069e460e10aae6fca37d12f2. The user clarified this was the intended set, then authorized removing60 overlapping SFT examples and retraining. Preserve all1000 ordered IDs and full rows. It is the earlier published Mixed1K, not a new sample or the separate2000-query tail reserve. Runtime verifies hash, uniqueness, and SFT/RL ID/query disjointness plus RL/validation/test query disjointness before paid calls. Dataset manifest and source/config hashes frozen per condition.

## Training recipe and explicit epoch schedule

Retain prior Tinker implementation of original GRPO: 32 query groups/batch,8 independent paths/group,seed42,temperature1,top_p1,zero intentional tool failures, original travel tools/database/scorers/prompts,2048 initial prompt,28672 generated/context additions,30720 total context,60tool turns,1200s trajectory timeout,whole overlong paths masked. Original sample-standard-deviation advantages. One on-policy weighted-CE update is the source PPO gradient at ratio1;globaltokenmean is an existing disclosed Tinker adaptation. No stale repeated updates. Rank32 LoRA is inherited from SFT, not fullparameter training.

The original source uses drop_last=True: floor(1000/32)=31batches/epoch. Seed42 in-place shuffle each epoch;8 query rows dropped each epoch, differing with shuffle. Five-epoch horizon155batches stays intact. Authorized stop at93batches, three epochs,2976 query-group visits and23808 paths/arm. Standalone SUM stays SUM. Curriculum epochs1–2 SUM (batches1–62),epochs3–4 MACRO (63–124),epoch5 SUCCESS (125–155). Current run stops afterbatch93 without compressing transitions. Explicit pure-function boundary tests cover62→63,124→125 and allfixed modes.

Original optimizer default is constant2e-6 LR with zero warmup (upstream/RL/verl/verl/workers/fsdp_workers.py429–453 and config/actor/dp_actor.yaml). There is no original LR decay to compress; preserve constant LR and record full155-step horizon. Adam(.9,.999,eps1e-8),weightdecay.01,gradclip1. Initial clipped PPO gradient matches previous implementation; no new objective proposed.

## Evaluation, resources and recovery

Evaluate initialization and after each epoch31/62/93. All180 validation queries,originalRL harness,macro reward only for judging,1rollout,temperature.6/top_p.95. This follows original source validation temperature, correcting the old Tinker controller0.0 override; identical across both arms. Seed/config/scorer unchanged between arms. Report3/5/7-day performance,task/model/parser/token/turn/wallclock failures,formatter-content failures and scorer errors separately. Transport/accounting exceptions are infrastructure failures and block progress; preserve successful concurrent results before stopping.

Runtime concurrency8 groups×8 paths perarm (64 independent paths,128 across pair);evaluationconcurrency12perarm. Concurrency is an operational choice to bound host memory, not a scientific change. Disk2GiB reserve,FDpreflight retained. Controller is local; all model computation/checkpoints are on Tinker.

Save resumable optimizer state and separate sampler after every acknowledged batch/update, plus epoch/reward/LR/data cursor,evaluation results and source/config hashes. Each condition has an atomic latest-checkpoint pointer. Restore optimizer only within that samecondition; replay deterministic shuffled data and saved realAPI responses with exact controls. Validation resume uses original saved sampler identity, and searches all ancestor validation journals. Never resample completed cases or select successes. Unknown in-flight optimizer state is never snapshotted; restart from last acknowledged checkpoint. Checkpoints use30dayTTL; transient rollout samplers1day.

## Costs

No hard spending cap. A NEW dedicated usage ledger tracks reservations/settlements, retaining unknown charges without modifying historical capped ledgers. Verify current published token prices0.33prefill/0.066cached/1.005sample/0.737train USD per million at https://tinker-docs.thinkingmachines.ai/tinker/models.json; snapshot stored with artifacts. Formatter costs use provider-reported usage when returned. Distinguish estimates,unknowns and invoices; storage charges are not measured by token event accounting. Deduplicate recovered payments by reservation ID across run ancestry. Per-condition run lineage records costs and infrastructure attempts; report combined pair total after completion. Log per-batch elapsed time, query-group rate and actual sampled-token counts.
