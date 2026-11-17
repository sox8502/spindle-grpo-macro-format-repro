"""Prepare source-format official test submissions from the frozen SFT checkpoint."""
import asyncio
import gzip
import json
import os
import sys

import tinker
from transformers import AutoTokenizer

from sft_backend import MODEL, TOKENIZER_REVISION
from pathlib import Path

from sft_data import OUTPUT, append_jsonl, file_hash

MATCHED_SETTINGS = ['seed', 'eval_concurrency', 'eval_context', 'eval_steps', 'model_context',
                    'max_turn_tokens', 'temperature', 'top_p']


def resume_predictions(source, variant, cases):
    path = source / (variant + '-test') / 'predictions.jsonl'
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {case['case_id']: case for case in cases}
    prior = source / 'resume-manifest.json'
    inherited = json.loads(prior.read_text())['variants'][variant]['api_origins'] if prior.exists() else {}
    origins, seen = {}, set()
    for row in saved:
        case_id = row['case_id']
        if case_id in seen or row['item'] != expected[case_id] or row['query'] != expected[case_id]['query']:
            raise ValueError('Resume predictions duplicate or differ from benchmark')
        seen.add(case_id)
        error = row.get('generation_error', '')
        if error and error not in {
            'ValueError: Generation ended before closing its reasoning block',
            'ValueError: Complete evaluation context exceeds model context; refusing truncation',
            'ValueError: Generation has no visible action after reasoning',
        }:
            raise ValueError('Cannot reuse an infrastructure-failed prediction')
        if 'formatter_error' in row or not isinstance(row['plan'], list):
            raise ValueError('Cannot reuse a formatter-failed prediction')
        journal = path.parent / (case_id + '-api.jsonl.gz')
        if not journal.exists():
            journal = path.parent / (case_id + '-api.jsonl')
        if not journal.exists():
            origin = inherited[case_id]
            journal = Path(origin['path'])
            if file_hash(journal) != origin['sha256']:
                raise ValueError('Inherited API evidence changed')
        opener = gzip.open if journal.suffix == '.gz' else open
        pending, records = False, 0
        with opener(journal, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n'):
                    raise ValueError('Incomplete API evidence line')
                record = json.loads(line)
                records += 1
                if 'request' in record:
                    if pending:
                        raise ValueError('Unanswered request in API evidence')
                    pending = True
                if 'response' in record or 'error' in record:
                    pending = False
        if not records or pending:
            raise ValueError('Incomplete API evidence')
        origins[case_id] = {'path': str(journal), 'sha256': file_hash(journal)}
    return saved, [case for case in cases if case['case_id'] not in seen], {
        'predictions_sha256': file_hash(path), 'api_origins': origins,
        'reused_cases': len(saved), 'missing_cases': len(cases) - len(saved)}


def submission_rows(predictions, cases):
    by_id = {row['case_id']: row for row in predictions}
    if len(predictions) != len(cases) or len(by_id) != len(cases):
        raise ValueError('Test predictions are not complete and unique')
    result = []
    for case in cases:
        row = by_id[case['case_id']]
        if row['query'] != case['query'] or row['item'] != case:
            raise ValueError('Official test prediction/query alignment changed')
        if not isinstance(row['plan'], list):
            raise ValueError('Official test plan must be a list')
        result.append({'idx': case['query_index'], 'query': case['query'], 'plan': row['plan']})
    return result


async def generate_test(config, output):
    from sft_run import ROOT, prepare_evaluation
    if not config.get('official_test') or config['eval_split'] != 'test':
        raise ValueError('Official test preparation requires the test formatter and test split')
    source = OUTPUT / 'runs' / config['test_checkpoint_run']
    source_provenance = json.loads((source / 'provenance.json').read_text())
    original = source_provenance['config']
    if original['phase'] != 'full' or source_provenance['actual_training_examples'] != 185:
        raise ValueError('Official test must use the full185-trace SFT checkpoint')
    if source_provenance['model'] != MODEL or source_provenance['training_manifest_sha256'] != config['training_dataset_sha256']:
        raise ValueError('Source model or training dataset differs')
    for key in MATCHED_SETTINGS:
        if config[key] != original[key]:
            raise ValueError('Official test changed the matched generation settings')
    checkpoint = json.loads((source / 'checkpoint.json').read_text())
    if checkpoint['step'] != 24:
        raise ValueError('Full SFT checkpoint has not completed24steps')
    cases, provenance = prepare_evaluation('test')
    assert len(cases) == 1000 and [c['query_index'] for c in cases] == list(range(1000))
    dataset = OUTPUT / 'training.jsonl'
    assert file_hash(dataset) == config['training_dataset_sha256']
    training_queries = {json.loads(line)['messages'][1]['content'].strip() for line in dataset.read_text().splitlines()}
    assert not training_queries & {case['query'].strip() for case in cases}
    provenance.update(config=config, checkpoint=checkpoint, source_run=config['test_checkpoint_run'],
                      training_steps=0, checkpoint_sha256=file_hash(source / 'checkpoint.json'), scoring='pending official evaluator; no test labels inferred',
                      formatter='released format_planning_for_official_test')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    (output / 'evaluation-queries.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in cases))
    cached = {}
    if config.get('resume_test_run'):
        previous = OUTPUT / 'runs' / config['resume_test_run']
        old = json.loads((previous / 'provenance.json').read_text())
        if (any(old['config'][key] != config[key] for key in MATCHED_SETTINGS + ['official_test', 'eval_split', 'test_checkpoint_run'])
                or old['checkpoint_sha256'] != provenance['checkpoint_sha256']
                or (previous / 'evaluation-queries.jsonl').read_bytes() != (output / 'evaluation-queries.jsonl').read_bytes()):
            raise ValueError('Resume changed generation settings, checkpoint or benchmark')
        manifest = {'source_run': config['resume_test_run'], 'source_provenance_sha256': file_hash(previous / 'provenance.json'),
                    'selection': 'reuse every complete prediction, including model failures; no selection on quality', 'variants': {}}
        for variant in ['base', 'sft']:
            saved, missing, evidence = resume_predictions(previous, variant, cases)
            cached[variant] = saved, missing
            manifest['variants'][variant] = evidence
        (output / 'resume-manifest.json').write_text(json.dumps(manifest, indent=2))
        print('RESUME_COUNTS ' + json.dumps({v: {'reused': len(x[0]), 'remaining': len(x[1])} for v, x in cached.items()}), flush=True)
    os.chdir(ROOT / 'upstream/Inference')
    sys.path.insert(0, str(ROOT / 'upstream/Inference'))
    from sft_evaluate import evaluate, summarize_results
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)
    summaries, files = {}, {}
    for variant in ['base', 'sft']:
        folder = output / (variant + '-test')
        folder.mkdir(parents=True, exist_ok=True)
        saved, missing = cached.get(variant, ([], cases))
        for row in saved:
            append_jsonl(folder / 'predictions.jsonl', row)
        if not missing:
            predictions = saved
            summaries[variant] = summarize_results(predictions, cases)
            (folder / 'summary.json').write_text(json.dumps(summaries[variant], indent=2))
        else:
            summaries[variant] = await generate_variant(variant, checkpoint, tokenizer, missing, folder, config, output, evaluate)
            predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
            summaries[variant] = summarize_results(predictions, cases)
            (folder / 'summary.json').write_text(json.dumps(summaries[variant], indent=2))
        rows = submission_rows(predictions, cases)
        path = output / f'{variant}-official-test.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        files[variant] = {'path': str(path), 'sha256': file_hash(path), 'cases': len(rows)}
    summary = {'run_id': os.environ['ORX_RUN_ID'], 'phase': config['phase'], 'training_steps': 0,
               'variants': summaries, 'submission_files': files,
               'official_scores': None, 'status': 'test outputs ready; no external submission performed'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('FINAL_SUMMARY ' + json.dumps(summary), flush=True)


async def generate_variant(variant, checkpoint, tokenizer, cases, folder, config, output, evaluate):
    service = tinker.ServiceClient(user_metadata={'orx_run_id': os.environ['ORX_RUN_ID'],
        'project_id': '93d7d223-6800-419c-ad90-89836e18cfe8', 'purpose': 'official test preparation ' + variant})
    capabilities = await service.get_server_capabilities_async()
    (output / f'{variant}-capabilities.json').write_text(capabilities.model_dump_json(indent=2))
    if not any(m.model_name == MODEL and m.sampleable for m in capabilities.supported_models):
        raise RuntimeError('Model currently unavailable for sampling')
    kwargs = {'base_model': MODEL} if variant == 'base' else {'model_path': checkpoint['sampler_path']}
    sampler = await service.create_sampling_client_async(**kwargs)
    return await asyncio.to_thread(evaluate, sampler, tokenizer, cases, folder, config)
