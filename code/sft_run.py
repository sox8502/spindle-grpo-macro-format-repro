import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd
import tinker
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from sft_backend import MODEL, TOKENIZER_REVISION, encode_training, train
from sft_data import OUTPUT, build_dataset

ROOT = Path(__file__).resolve().parent
BENCHMARK_REVISION = '8736504ecfc31b7f8b7e40122873c337e83fff7c'
DATABASE_REVISION = '985c8c7ba3e6beb986b3b0a55a2b064d18f2e68e'


def load_credentials():
    path = Path.home() / '.openresearch/env'
    if path.exists():
        for line in path.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                name, value = line.split('=', 1)
                name = name.strip().removeprefix('export ')
                if name in {'TINKER_API_KEY', 'OPENROUTER_API_KEY', 'HF_TOKEN'}:
                    os.environ[name] = value.strip().strip('\"\'')
    os.environ['DASHSCOPE_API_KEY'] = os.environ['OPENROUTER_API_KEY']
    os.environ.pop('STAR_API_EVIDENCE_DIR', None)


def validate_local_labels(cases):
    required = {'org', 'dest', 'days', 'date', 'query', 'level', 'budget',
                'people_number', 'local_constraint', 'visiting_city_number'}
    for index, row in enumerate(cases):
        missing = required - row.keys()
        if missing:
            raise ValueError(f'Local evaluation case {index} lacks labels: {sorted(missing)}; use the official evaluator for unlabeled test inputs')


def prepare_evaluation(split="validation"):
    if split not in {"validation", "test"}:
        raise ValueError("Unknown benchmark split")
    hashes = {}
    for name in ['attractions.csv', 'clean_Flights_2022.csv', 'clean_accommodations_2022.csv',
                 'clean_restaurant_2022.csv', 'distance.csv']:
        cached = Path(hf_hub_download('xxwu/Agent-STAR-TravelDatabase', name,
                      repo_type='dataset', revision=DATABASE_REVISION))
        target = ROOT / 'upstream/database' / name
        if not target.exists():
            target.symlink_to(cached)
        hashes[name] = hashlib.sha256(cached.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[name]
    path = hf_hub_download('osunlp/TravelPlanner', f'{split}.csv', repo_type='dataset', revision=BENCHMARK_REVISION)
    cases = pd.read_csv(path, keep_default_na=False).to_dict(orient='records')
    assert len(cases) == {'validation': 180, 'test': 1000}[split]
    if split == 'validation':
        validate_local_labels(cases)
    for index, row in enumerate(cases):
        row.update(case_id=f'{split}-{index}', query_index=index)
        row.pop('reference_information', None)
        for key in ['date', 'local_constraint']:
            if isinstance(row.get(key), str):
                row[key] = ast.literal_eval(row[key])
    return cases, {'database_hashes': hashes, 'benchmark_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   'benchmark_revision': BENCHMARK_REVISION, 'database_revision': DATABASE_REVISION, 'split': split}


async def main():
    config = json.loads((ROOT / 'sft_config.json').read_text())
    assert config['phase'] in {'smoke', 'full', 'recover_base', 'official_test'}
    load_credentials()
    run_id = os.environ['ORX_RUN_ID']
    output = OUTPUT / 'runs' / run_id
    output.mkdir(parents=True, exist_ok=False)
    print('SFT_CONFIG ' + json.dumps(config), flush=True)
    print('ARTIFACT_DIRECTORY ' + str(output), flush=True)
    if config['phase'] == 'official_test':
        from sft_test import generate_test
        await generate_test(config, output)
        return
    if config['phase'] == 'recover_base':
        from sft_recover import recover
        await recover(config, output)
        return
    examples, manifest = build_dataset(output)
    assert manifest['dataset_sha256'] == config['training_dataset_sha256']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)
    lengths = [len(encode_training(tokenizer, row['messages'], config['cutoff'])[0]) for row in examples]
    cases, provenance = prepare_evaluation(config['eval_split'])
    validate_local_labels(cases)
    assert not ({r['messages'][1]['content'].strip() for r in examples} & {c['query'].strip() for c in cases})
    if config['phase'] == 'smoke':
        indices = [lengths.index(min(lengths)), lengths.index(max(lengths))]
        examples = [examples[i] for i in indices]
        cases = cases[:config['smoke_eval_cases']]
    (output / 'evaluation-queries.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in cases))
    provenance.update(model=MODEL, tokenizer_revision=TOKENIZER_REVISION, config=config,
                      adaptations={'loss_mask': 'whole rejected assistant turns', 'retained_prior_reasoning': True,
                                   'protocol': 'inherited JSON correction hint', 'formatter': 'direct format_planning',
                                   'evaluation_seed': 'seed + case_index*1000 + turn_index',
                                   'source_context_threshold': 'visible contents only; native total separately bounded',
                                   'training': 'rank32 LoRA, LR5e-5, batch32 vs released full5e-6',
                                   'length': 'complete conversations up to65536 vs paper20480'},
                      all_training_examples=manifest['examples'], all_max_sequence_length=max(lengths),
                      actual_training_examples=len(examples), training_manifest_sha256=manifest['dataset_sha256'])
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    os.chdir(ROOT / 'upstream/Inference')
    sys.path.insert(0, str(ROOT / 'upstream/Inference'))
    from sft_evaluate import evaluate
    service = tinker.ServiceClient(user_metadata={'orx_run_id': run_id,
        'project_id': '93d7d223-6800-419c-ad90-89836e18cfe8', 'purpose': '185 masked SFT'})
    capabilities = await service.get_server_capabilities_async()
    (output / 'capabilities.json').write_text(capabilities.model_dump_json(indent=2))
    available = [m for m in capabilities.supported_models if m.model_name == MODEL]
    assert available and available[0].trainable and available[0].sampleable
    assert available[0].max_context_length >= config['cutoff']
    if config['phase'] == 'smoke' and config.get('smoke_checkpoint'):
        checkpoint = config['smoke_checkpoint']
        print('REUSED_SMOKE_CHECKPOINT ' + json.dumps(checkpoint), flush=True)
    else:
        checkpoint = await train(service, examples, tokenizer, output, config)
    student = await service.create_sampling_client_async(model_path=checkpoint['sampler_path'])
    base = await service.create_sampling_client_async(base_model=MODEL)
    base_summary = await asyncio.to_thread(evaluate, base, tokenizer, cases, output / 'base-evaluation', config)
    sft_summary = await asyncio.to_thread(evaluate, student, tokenizer, cases, output / 'sft-evaluation', config)
    summary = {'run_id': run_id, 'phase': config['phase'], 'training_examples': len(examples),
               'checkpoint': checkpoint, 'base': base_summary, 'sft': sft_summary,
               'success_rate_delta': sft_summary['task_success_rate'] - base_summary['task_success_rate']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('FINAL_SUMMARY ' + json.dumps(summary), flush=True)
    if config['phase'] == 'smoke' and any(s['generation_errors'] or s['formatter_errors'] or s['scoring_errors'] for s in [base_summary, sft_summary]):
        raise RuntimeError('End-to-end smoke has operational failures; inspect saved evidence before full training')


if __name__ == '__main__':
    asyncio.run(main())
