"""Repeat the prior SFT recipe on the frozen1140 export and matched evaluations."""
import asyncio
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

import requests
import tinker
from transformers import AutoTokenizer

from official_score import score_file, HOST
from sft_backend import MODEL, TOKENIZER_REVISION, encode_training, train
from sft_data import OUTPUT, ARTIFACTS, append_jsonl, build_dataset, file_hash
from sft_run import ROOT, load_credentials, prepare_evaluation
from sft_test import MATCHED_SETTINGS, resume_predictions, submission_rows

PREVIOUS = ARTIFACTS / 'sft-185-masked-2026-10-18'
PREVIOUS_TRAIN = PREVIOUS / 'runs/4f63e8e6-e30d-4422-b37a-50c416e62324'


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2))


def fix_submission_spacing(rows):
    fixed, changes = copy.deepcopy(rows), []
    for row in fixed:
        for day in row['plan']:
            value = day.get('transportation') if isinstance(day, dict) else None
            if isinstance(value, str):
                replacement = re.sub(r'Flight Number:(?=\S)', 'Flight Number: ', value)
                if replacement != value:
                    changes.append({'idx': row['idx'], 'day': day.get('day'),
                                    'before': value, 'after': replacement})
                    day['transportation'] = replacement
    return fixed, changes


async def evaluate_stage(service, checkpoint, tokenizer, cases, output, config, variant):
    from sft_evaluate import evaluate, summarize_results
    folder = output / (variant + '-test')
    saved, missing = [], cases
    if (folder / 'predictions.jsonl').exists():
        saved, missing, evidence = resume_predictions(output, variant, cases)
        json_file(output / (variant + '-resume.json'), evidence)
    if missing:
        for case in missing:
            for suffix in ['-api.jsonl', '-api.jsonl.gz']:
                journal = folder / (case['case_id'] + suffix)
                if journal.exists():
                    archive = output / (variant + '-interrupted') / os.environ['ORX_RUN_ID']
                    archive.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(journal), archive / journal.name)
        sampler = await service.create_sampling_client_async(model_path=checkpoint['sampler_path'])
        await asyncio.to_thread(evaluate, sampler, tokenizer, missing, folder, config)
    rows, unresolved, evidence = resume_predictions(output, variant, cases)
    assert not unresolved, 'Evaluation cases incomplete'
    if any('scoring_error' in row for row in rows):
        raise RuntimeError('Local scoring failed; comparison remains unresolved')
    submission_rows(rows, cases)
    summary = summarize_results(rows, cases)
    json_file(folder / 'summary.json', summary)
    return rows, summary


async def main():
    load_credentials()
    config = json.loads((ROOT / 'sft_config.json').read_text())
    output = OUTPUT / 'pipeline'
    output.mkdir(parents=True, exist_ok=True)
    examples, manifest = build_dataset(output)
    assert manifest['dataset_sha256'] == config['training_dataset_sha256']
    assert len(examples) == config['training_examples'] == 1140
    old = json.loads((PREVIOUS_TRAIN / 'provenance.json').read_text())
    for key in MATCHED_SETTINGS + ['epochs', 'batch_size', 'learning_rate', 'warmup_ratio', 'lora_rank', 'cutoff']:
        assert config[key] == old['config'][key], ('Previous recipe changed', key)
    source_names = ['sft_pipeline.py', 'sft_backend.py', 'sft_data.py', 'sft_evaluate.py',
                    'sft_run.py', 'sft_test.py', 'official_score.py', 'uv.lock', 'rl_training.jsonl']
    frozen = {'config': config, 'dataset_sha256': manifest['dataset_sha256'],
              'source_hashes': {name: file_hash(ROOT / name) for name in source_names},
              'previous_provenance_sha256': file_hash(PREVIOUS_TRAIN / 'provenance.json')}
    frozen['source_hashes'].update({str(path.relative_to(ROOT)): file_hash(path)
                                   for path in (ROOT / 'upstream/Inference').rglob('*.py')})
    provenance = output / 'frozen-provenance.json'
    if provenance.exists():
        assert json.loads(provenance.read_text()) == frozen, 'Resume source/config changed'
    else:
        json_file(provenance, frozen)
    append_jsonl(output / 'controller-runs.jsonl', {'run_id': os.environ['ORX_RUN_ID'], **frozen})
    print('SFT_CONFIG ' + json.dumps(config), flush=True)
    print('ARTIFACT_DIRECTORY ' + str(output), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)
    lengths = [len(encode_training(tokenizer, r['messages'], config['cutoff'])[0]) for r in examples]
    validation, val_provenance = prepare_evaluation('validation')
    test, test_provenance = prepare_evaluation('test')
    holdout_file = ROOT / 'rl_holdout.jsonl'
    assert file_hash(holdout_file) == config['rl_holdout_sha256']
    holdout = [json.loads(line) for line in holdout_file.read_text().splitlines()]
    assert not {r['case_id'] for r in examples} & {r['id'] for r in holdout}
    queries = {r['messages'][1]['content'].strip() for r in examples}
    assert not queries & {r['query'].strip() for r in validation + test + holdout}
    rl_rows = [json.loads(line) for line in (ROOT / 'rl_training.jsonl').read_text().splitlines()]
    rl_queries = {row['query'].strip() for row in rl_rows}
    assert not rl_queries & (queries | {row['query'].strip() for row in validation + test})
    json_file(output / 'evaluation-provenance.json', {'validation': val_provenance, 'test': test_provenance})
    json_file(output / 'length-summary.json', {'examples': len(lengths), 'min': min(lengths), 'max': max(lengths)})
    spindle = config.get('spindle_base_url')
    connection = {'base_url': spindle, 'api_key': os.environ['SPINDLE_API_KEY']} if spindle else {}
    service = tinker.ServiceClient(**connection, user_metadata={'orx_run_id': os.environ['ORX_RUN_ID'],
        'project_id': '93d7d223-6800-419c-ad90-89836e18cfe8', 'purpose': '1140-example SFT and matched evaluation'})
    caps = await service.get_server_capabilities_async()
    (output / 'capabilities.json').write_text(caps.model_dump_json(indent=2))
    available = [m for m in caps.supported_models if m.model_name == MODEL]
    assert available and (spindle or (available[0].trainable and available[0].sampleable))
    assert available[0].max_context_length >= config['cutoff']
    checkpoint_file = output / 'checkpoint.json'
    checkpoint = json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else None
    expected_steps = math.ceil(len(examples) / config['batch_size']) * config['epochs']
    if not checkpoint or checkpoint['step'] != expected_steps or 'sampler_path' not in checkpoint:
        checkpoint = await train(service, examples, tokenizer, output, config)
    assert checkpoint['step'] == expected_steps == 144
    os.chdir(ROOT / 'upstream/Inference')
    sys.path.insert(0, str(ROOT / 'upstream/Inference'))
    val_rows, val_summary = await evaluate_stage(service, checkpoint, tokenizer, validation, output, config, 'validation')
    old_rows = [json.loads(line) for line in (PREVIOUS_TRAIN / 'sft-evaluation/predictions.jsonl').read_text().splitlines()]
    previous_by_id = {r['case_id']: r for r in old_rows}
    assert len(previous_by_id) == len(val_rows) == 180
    assert all(previous_by_id[r['case_id']]['query'] == r['query'] for r in val_rows)
    previous_success = sum(r['success'] for r in old_rows)
    comparison = {'previous_examples': 185, 'new_examples': 1140,
                  'validation': {'previous_success': previous_success, 'new_success': val_summary['task_success'],
                    'cases': 180, 'delta_pp': (val_summary['task_success'] - previous_success) / 180 * 100},
                  'new_validation_summary': val_summary, 'checkpoint': checkpoint}
    json_file(output / 'comparison.json', comparison)
    print('VALIDATION_COMPARISON ' + json.dumps(comparison), flush=True)
    if config.get('stop_after_validation'):
        print('FINAL_SUMMARY ' + json.dumps(comparison), flush=True)
        return
    test_config = {**config, 'official_test': True, 'eval_split': 'test'}
    test_rows, test_summary = await evaluate_stage(service, checkpoint, tokenizer, test, output, test_config, 'sft')
    raw = submission_rows(test_rows, test)
    raw_file = output / 'sft-official-test-original.jsonl'
    raw_file.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in raw))
    fixed, changes = fix_submission_spacing(raw)
    submission = output / 'sft-official-test.jsonl'
    submission.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in fixed))
    json_file(output / 'submission-manifest.json', {'cases': 1000, 'raw_sha256': file_hash(raw_file),
        'submitted_sha256': file_hash(submission), 'changes': changes,
        'adaptation': 'Same ASCII-space insertion after Flight Number: as prior185 SFT evaluation'})
    scoring = output / 'official-scoring'
    if not scoring.exists():
        with requests.Session() as session:
            for name in ['app.py', 'eval.py']:
                response = session.get('https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard/raw/main/' + name, timeout=60)
                response.raise_for_status()
                (output / ('official-service-' + name)).write_bytes(response.content)
            score_file(session, submission, scoring)
    if not (scoring / 'official-report.json').exists() or not (scoring / 'metrics.md').exists():
        raise RuntimeError('Official scoring pending/failed; inspect saved queue before any resubmission')
    old_official = json.loads((PREVIOUS / 'official-scoring-flight-spacing-fixed/comparison.json').read_text())
    previous_metrics = old_official['variants']['sft']['metrics']
    returned = (scoring / 'metrics.md').read_text()
    metrics = ast.literal_eval(re.search(r'\{[^{}]+\}', returned).group())
    assert set(metrics) == set(previous_metrics)
    assert all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in metrics.values())
    comparison.update(test_cases=1000, test_generation=test_summary,
                      previous_official_metrics=previous_metrics,
                      new_official_metrics=metrics,
                      official_delta_pp=(metrics['Final Pass Rate'] - previous_metrics['Final Pass Rate']) * 100,
                      new_official_metrics_markdown=returned,
                      official_report_sha256=file_hash(scoring / 'official-report.json'),
                      status='training and matched evaluation complete')
    json_file(output / 'comparison.json', comparison)
    json_file(output / 'summary.json', comparison)
    report = ['# SFT1140 versus SFT185', '',
              f"Validation: {previous_success}/180 → {val_summary['task_success']}/180.", '',
              '| Official test metric | SFT185 | SFT1140 |', '|---|---:|---:|']
    report.extend(f'| {key} | {previous_metrics[key]*100:.2f}% | {metrics[key]*100:.2f}% |' for key in metrics)
    report.extend(['', f"Final-pass difference: {comparison['official_delta_pp']:+.2f} percentage points.", '',
                   'Both models use the same documented generation and formatting settings. Prior results are historical; the official service private-label revision cannot be independently pinned.', '',
                   'Training used all1140 examples from original base weights, rank32 LoRA, four passes and144 optimizer updates.'])
    (output / 'comparison.md').write_text('\n'.join(report) + '\n')
    print('FINAL_SUMMARY ' + json.dumps(comparison), flush=True)
