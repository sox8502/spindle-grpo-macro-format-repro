"""Recover infrastructure-failed baseline cases without selecting on task scores."""
import asyncio
import json
import os
import sys

import tinker
from transformers import AutoTokenizer

from sft_backend import MODEL, TOKENIZER_REVISION
from sft_data import OUTPUT, file_hash

CONNECTION_ERROR = 'APIConnectionError: No progress made in 7200s. Requests appear to be stuck.'


def recovery_cases(original, cases):
    by_id = {row['case_id']: row for row in original}
    if len(original) != len(cases) or len(by_id) != len(cases):
        raise ValueError('Original baseline is not a complete unique benchmark')
    selected = []
    for case in cases:
        row = by_id[case['case_id']]
        if row['item'] != case or row['query'] != case['query']:
            raise ValueError('Recovery benchmark differs from original baseline')
        if row.get('generation_error') == CONNECTION_ERROR:
            selected.append(case)
    return selected


def merge_recovery(original, recovered, selected):
    expected = {row['case_id'] for row in selected}
    replacements = {row['case_id']: row for row in recovered}
    if len(recovered) != len(replacements) or set(replacements) != expected:
        raise ValueError('Recovery is incomplete or contains unselected cases')
    for row in recovered:
        error = row.get('generation_error', '')
        if error and not error.startswith(('ValueError: Generation ended before closing its reasoning block',
                                           'ValueError: Complete evaluation context exceeds model context',
                                           'ValueError: Generation has no visible action after reasoning')):
            raise ValueError('Recovery still contains operational failures; comparison remains unresolved')
        if 'formatter_error' in row or 'scoring_error' in row or not isinstance(row.get('plan'), list):
            raise ValueError('Recovery formatting/scoring failed')
    return [replacements.get(row['case_id'], row) for row in original]


async def recover(config, output):
    from sft_run import ROOT, prepare_evaluation, validate_local_labels
    source = OUTPUT / 'runs' / config['recovery_source_run']
    original_config = json.loads((source / 'provenance.json').read_text())['config']
    keys = ['seed', 'eval_split', 'eval_concurrency', 'eval_context', 'eval_steps',
            'model_context', 'max_turn_tokens', 'temperature', 'top_p']
    if any(config[key] != original_config[key] for key in keys):
        raise ValueError('Recovery changed the baseline generation protocol')
    cases, provenance = prepare_evaluation(config['eval_split'])
    validate_local_labels(cases)
    path = source / 'base-evaluation/predictions.jsonl'
    original = [json.loads(line) for line in path.read_text().splitlines()]
    selected = recovery_cases(original, cases)
    if len(selected) != config['recovery_expected_cases']:
        raise ValueError('Frozen infrastructure-failure count changed')
    manifest = {'source_run': config['recovery_source_run'], 'original_predictions_sha256': file_hash(path),
                'selection': 'exact SDK connection-watchdog error only; independent of success labels',
                'case_ids': [row['case_id'] for row in selected], 'cases': len(selected),
                'fresh_sampler': True, 'training_steps': 0, 'config': config, **provenance}
    (output / 'recovery-manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'evaluation-queries.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in selected))
    print('RECOVERY_MANIFEST ' + json.dumps(manifest), flush=True)
    os.chdir(ROOT / 'upstream/Inference')
    sys.path.insert(0, str(ROOT / 'upstream/Inference'))
    from sft_evaluate import evaluate, summarize_results
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)
    service = tinker.ServiceClient(user_metadata={'orx_run_id': os.environ['ORX_RUN_ID'],
        'project_id': '93d7d223-6800-419c-ad90-89836e18cfe8', 'purpose': 'baseline connection recovery'})
    capabilities = await service.get_server_capabilities_async()
    (output / 'capabilities.json').write_text(capabilities.model_dump_json(indent=2))
    if not any(m.model_name == MODEL and m.sampleable for m in capabilities.supported_models):
        raise RuntimeError('Baseline model currently unavailable for sampling')
    sampler = await service.create_sampling_client_async(base_model=MODEL)
    await asyncio.to_thread(evaluate, sampler, tokenizer, selected, output / 'recovered-base', config)
    recovered = [json.loads(line) for line in (output / 'recovered-base/predictions.jsonl').read_text().splitlines()]
    merged = merge_recovery(original, recovered, selected)
    (output / 'corrected-base-predictions.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in merged))
    base = summarize_results(merged, cases)
    sft = json.loads((source / 'sft-evaluation/summary.json').read_text())
    if base['query_sha256'] != sft['query_sha256']:
        raise ValueError('Corrected baseline and SFT benchmark differ')
    result = {'run_id': os.environ['ORX_RUN_ID'], 'phase': config['phase'],
              'source_run': config['recovery_source_run'], 'recovered_cases': len(recovered),
              'training_steps': 0, 'base': base, 'sft': sft,
              'success_rate_delta': sft['task_success_rate'] - base['task_success_rate']}
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print('FINAL_SUMMARY ' + json.dumps(result), flush=True)
