import hashlib
import json
import struct
import os
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rl_budget import Budget, BudgetExceeded
import rl_run


class ScheduleTests(unittest.TestCase):
    def test_boundaries_preserve_five_epoch_recipe(self):
        config = json.loads(Path('rl_config.json').read_text())
        config.pop('diagnostic', None)
        config['learning_rate'] = 2e-6
        self.assertEqual([rl_run.reward_for_epoch('curriculum', e) for e in range(1, 6)],
                         ['sum', 'sum', 'macro', 'macro', 'success'])
        for mode in ['sum', 'macro', 'success']:
            self.assertEqual([rl_run.reward_for_epoch(mode, e) for e in range(1, 6)], [mode] * 5)
        for completed, reward in [(0, 'sum'), (31, 'sum'), (61, 'sum'), (62, 'macro'), (93, 'macro'), (124, 'success')]:
            cursor = rl_run.schedule_cursor({**config, 'reward_type': 'curriculum'}, completed)
            self.assertEqual(cursor['next_reward'], reward)
            self.assertEqual(cursor['planned_batches'], 155)
            self.assertEqual(cursor['authorized_batches'], 93)
            self.assertEqual(cursor['learning_rate'], 2e-6)
        all_batches = list(rl_run.dataset_batches(list(range(1000)), 5, 32, 42))
        self.assertEqual(len(all_batches), 155)
        self.assertEqual(all_batches[31][0], 2)
        self.assertEqual(all_batches[62][0], 3)
        self.assertEqual(len(all_batches[:93]), 93)

    def test_diagnostic_panel_selection_ignores_outcomes_and_order(self):
        cases = [{'case_id': f'{days}-{i}', 'days': days, 'success': i % 2}
                 for days in (3, 5, 7) for i in range(5)]
        config = {'seed': 42, 'diagnostic': {'validation_per_trip_length': 2}}
        first = rl_run.diagnostic_cases(cases, config)
        changed = [{**case, 'success': 1 - case['success']} for case in reversed(cases)]
        second = rl_run.diagnostic_cases(changed, config)
        self.assertEqual([r['case_id'] for r in first], [r['case_id'] for r in second])
        self.assertEqual([r['days'] for r in first], [3, 3, 5, 5, 7, 7])

    def test_diagnostic_reference_matches_frozen_query_order(self):
        config = json.loads(Path('rl_config.json').read_text())
        tasks = [json.loads(line) for line in Path('rl_training.jsonl').read_text().splitlines()]
        reference = json.loads(Path('rl_diagnostic_reference.json').read_text())
        batches = list(rl_run.dataset_batches(tasks, 5, config['batch_size'], config['seed']))[:10]
        self.assertEqual([[task['id'] for task in batch] for _, batch in batches],
                         [row['task_ids'] for row in reference['rows']])

    def test_no_cap_and_no_duplicate_recovered_payment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Budget(root / 'usage.sqlite3')
            identity = ledger.reserve('uncapped', 2000)
            ledger.settle(identity, 2500)
            ledger.settle(identity, 2500)
            self.assertIsNone(ledger.snapshot()['spending_cap_usd'])
            self.assertEqual(ledger.snapshot()['committed_usd'], 2500)
            for name in ['a', 'b']:
                p = root / name;p.mkdir()
                (p / 'cost-events.jsonl').write_text(json.dumps({'reservation_id': 'same',
                    'cost_usd': None, 'token_rate_estimate_usd': 2}) + '\n')
            result = rl_run.summarize_lineage([root / 'a', root / 'b'])
            self.assertEqual(result['combined_usage_estimate_usd'], 2)
            self.assertEqual(result['unique_payment_reservations'], 1)

    def test_failure_attribution(self):
        def row(error=None, termination='ENV_DONE', success=0):
            return {'reward_error': error, 'termination': termination, 'judgement': {'success_pass': success}}
        self.assertEqual(rl_run.failure_category(row('Error evaluating one sample: X')), 'scorer_failure')
        self.assertEqual(rl_run.failure_category(row('Error formatting planning: X')), 'formatter_content_failure')
        self.assertEqual(rl_run.failure_category(row(termination='TRUNCATION')), 'token_limit')
        self.assertEqual(rl_run.failure_category(row(success=1)), 'success')


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_refills_paths_and_preserves_group_order(self):
        next_group_started = threading.Event()
        calls = []
        def group(sampler, tokenizer, task, config, output, step, identity, budget, *, _path_id):
            calls.append((task['id'], _path_id, identity, sampler))
            if task['id'] == 'a' and _path_id == 0:
                if not next_group_started.wait(5):
                    raise RuntimeError('A finished path did not admit the next group')
            if task['id'] == 'b':
                next_group_started.set()
            return [(task['id'], _path_id)]
        sampler = object()
        with patch('rl_rollout.collect_group', side_effect=group), ThreadPoolExecutor(max_workers=2) as pool:
            groups = await rl_run.collect_batch(pool, sampler, None, [{'id': 'a'}, {'id': 'b'}],
                {'group_size': 2}, None, 7, None)
        self.assertEqual(groups, [[('a', 0), ('a', 1)], [('b', 0), ('b', 1)]])
        self.assertEqual({identity for _, _, identity, _ in calls}, {'batch-7-0-a', 'batch-7-1-b'})
        self.assertTrue(all(client is sampler for _, _, _, client in calls))

    async def test_batch_error_waits_for_other_paths(self):
        finished = threading.Event()
        def group(*args, _path_id):
            if _path_id == 0:
                raise RuntimeError('offline sampling failure')
            finished.set()
            return ['saved trajectory']
        with patch('rl_rollout.collect_group', side_effect=group), ThreadPoolExecutor(max_workers=2) as pool:
            groups = await rl_run.collect_batch(pool, None, None, [{'id': 'a'}],
                {'group_size': 2}, None, 7, None)
            self.assertTrue(finished.is_set())
        self.assertIsInstance(groups[0], RuntimeError)

    async def test_batch_preserves_budget_failure_priority(self):
        def group(*args, _path_id):
            raise RuntimeError('sampling') if _path_id == 0 else BudgetExceeded('accounting')
        with patch('rl_rollout.collect_group', side_effect=group), ThreadPoolExecutor(max_workers=2) as pool:
            groups = await rl_run.collect_batch(pool, None, None, [{'id': 'a'}],
                {'group_size': 2}, None, 7, None)
        self.assertIsInstance(groups[0], BudgetExceeded)

    async def test_validation_drains_after_guard_or_worker_error(self):
        for guard_failure in [False, True]:
            with self.subTest(guard_failure=guard_failure):
                release = threading.Event()
                called = []
                checks = []
                cases = [{'case_id': str(i), 'days': 3} for i in range(3)]
                def preflight(*args, **kwargs):
                    checks.append(1)
                    if guard_failure and len(checks) == 3:
                        release.set()
                        raise RuntimeError('offline guard failure')
                def group(sampler, tokenizer, task, *args):
                    called.append(task['id'])
                    if not guard_failure and task['id'] == '0':
                        raise RuntimeError('offline worker failure')
                    if task['id'] == ('0' if guard_failure else '1') and not release.wait(5):
                        raise RuntimeError('Active validation case was not drained')
                    return [{'task_id': task['id'], 'judgement': {'success_pass': 1},
                             'termination': 'ENV_DONE', 'turns': [], 'reward_error': None}]
                append = rl_run.append_jsonl
                def save(path, value):
                    append(path, value)
                    if path.name == 'infrastructure-errors.jsonl':
                        release.set()
                service = Mock(create_sampling_client_async=AsyncMock(return_value=object()))
                config = {'eval_concurrency': 2, 'eval_temperature': .6, 'min_free_disk_bytes': 0}
                with tempfile.TemporaryDirectory() as directory, \
                     patch('rl_rollout.collect_group', side_effect=group), \
                     patch.object(rl_run, 'resource_preflight', side_effect=preflight), \
                     patch.object(rl_run, 'append_jsonl', side_effect=save):
                    output = Path(directory)
                    with self.assertRaisesRegex(RuntimeError, 'offline (guard|worker) failure'):
                        await rl_run.evaluate_checkpoint(service, 'frozen-policy', None, cases,
                            output, config, None, 'validation')
                    rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
                self.assertEqual(set(called), {'0', '1'})
                self.assertEqual({row['task_id'] for row in rows}, {'0', '1'} if guard_failure else {'1'})

    async def test_validation_refills_slots_and_saves_completed_cases(self):
        third_started = threading.Event()
        cases = [{'case_id': str(i), 'days': 3} for i in range(3)]
        def group(sampler, tokenizer, task, *args):
            if task['id'] == '0' and not third_started.wait(5):
                raise RuntimeError('Validation is still waiting on a fixed wave')
            if task['id'] == '2':
                third_started.set()
            return [{'task_id': task['id'], 'judgement': {'success_pass': 1},
                     'termination': 'ENV_DONE', 'turns': [], 'reward_error': None}]
        service = Mock(create_sampling_client_async=AsyncMock(return_value=object()))
        config = {'eval_concurrency': 2, 'eval_temperature': .6, 'min_free_disk_bytes': 0}
        with tempfile.TemporaryDirectory() as directory, \
             patch('rl_rollout.collect_group', side_effect=group), patch.object(rl_run, 'resource_preflight'):
            output = Path(directory)
            metrics = await rl_run.evaluate_checkpoint(service, 'frozen-policy', None, cases,
                output, config, None, 'validation')
            rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
        self.assertEqual(metrics['successes'], 3)
        self.assertEqual({row['task_id'] for row in rows}, {'0', '1', '2'})

    async def test_epoch_boundary_resume_and_three_epoch_stop(self):
        await self.check_controller(False)

    async def test_diagnostic_resume_evaluations_and_ten_update_stop(self):
        await self.check_controller(True)

    async def test_diagnostic_early_stop_after_resumed_validation(self):
        await self.check_controller(True, early_stop=True)

    async def test_source_weighting_resume_and_ten_update_stop(self):
        await self.check_controller(True, source_weighting=True)

    async def test_sum_three_epoch_gates(self):
        await self.check_controller(False, source_weighting=True, epoch_gate=True, interrupt=False)

    async def test_curriculum_resumes_pending_validation_before_next_epoch(self):
        await self.check_controller(False, source_weighting=True, epoch_gate=True, curriculum=True)

    async def test_resume_after_sampler_failure_past_completed_epoch_gates(self):
        await self.check_controller(False, source_weighting=True, epoch_gate=True,
                                    interrupt=False, retry_after_gates=True)

    async def test_dapo_sum_three_epoch_gates(self):
        await self.check_controller(False, source_weighting=True, epoch_gate=True,
                                    interrupt=False, dapo=True)

    async def test_dapo_curriculum_skipped_batch_resume_and_stop(self):
        await self.check_controller(False, source_weighting=True, epoch_gate=True,
                                    interrupt=True, curriculum=True, dapo=True, skip_batch=31)

    async def test_spindle_fresh_base_controller(self):
        await self.check_controller(False, source_weighting=True, epoch_gate=True, interrupt=False, spindle=True)

    async def test_spindle_initial_validation_resume(self):
        await self.check_controller(False, source_weighting=True, epoch_gate=True, interrupt=False, spindle=True, initial_resume=True)

    async def check_controller(self, diagnostic, early_stop=False, source_weighting=False,
                               epoch_gate=False, interrupt=True, curriculum=False, retry_after_gates=False, dapo=False, skip_batch=None, spindle=False, initial_resume=False):
        actual = Path.cwd()
        config = json.loads((actual / 'rl_config.json').read_text())
        config.pop('diagnostic', None)
        if not spindle:
            for key in ['compute_backend', 'initialization', 'lora_rank', 'budget_path']:
                config.pop(key, None)
            trained = json.loads((rl_run.SFT_ROOT / 'checkpoint.json').read_text())
            config.update(sft_state=trained['state_path'], sft_sampler=trained['sampler_path'])
        config.pop('epoch_audit_gate', None)
        config.pop('rollout_priority', None)
        config.update(algorithm='GRPO', condition='offline-curriculum', reward_type='sum' if source_weighting else 'curriculum',
                      learning_rate=2e-5 if source_weighting else 2e-6)
        if dapo:
            config['algorithm'] = 'DAPO'
        def applied(completed):
            return completed - int(skip_batch is not None and completed >= skip_batch)
        if epoch_gate:
            config.update(epoch_audit_gate=True, rollout_priority='longer_trips_first',
                          reward_type='curriculum' if curriculum else 'sum')
        if not source_weighting:
            config.pop('loss_weighting', None)
        if diagnostic:
            config['diagnostic'] = dict(max_updates=10, evaluation_updates=[2, 5, 10],
                validation_per_trip_length=1, early_stop_validation_drop=10, reference_file='reference.json')
            if early_stop:
                config['diagnostic']['early_stop_validation_drop'] = 1
        boundary, stop = (2, 10) if diagnostic else (31, 93)
        if early_stop:
            stop = boundary
        tokenizer = Mock();tokenizer.get_vocab.return_value = {'test': 1}
        def trainer(name):
            t = Mock(model_id=name);t.get_tokenizer.return_value = tokenizer
            def future(value):
                return Mock(result_async=AsyncMock(return_value=value))
            t.save_state_async = AsyncMock(side_effect=lambda n, **k: future(SimpleNamespace(path=f'tinker://{name}/state/{n}')))
            t.save_weights_for_sampler_async = AsyncMock(side_effect=lambda n, **k: future(SimpleNamespace(path=f'tinker://{name}/sampler/{n}')))
            t.forward_backward_async = AsyncMock(side_effect=lambda *a, **k: future(SimpleNamespace(metrics={})))
            t.optim_step_async = AsyncMock(side_effect=lambda *a, **k: future(SimpleNamespace(metrics={})))
            return t
        first, second = trainer('first'), trainer('restored')
        caps = SimpleNamespace(supported_models=[SimpleNamespace(model_name=rl_run.MODEL,
            trainable=True, sampleable=True, max_context_length=65536)], model_dump_json=lambda **k: '{}')
        service = Mock(get_server_capabilities_async=AsyncMock(return_value=caps),
            create_training_client_from_state_async=AsyncMock(return_value=first),
            create_lora_training_client_async=AsyncMock(return_value=first),
            create_training_client_from_state_with_optimizer_async=AsyncMock(return_value=second),
            create_sampling_client_async=AsyncMock(return_value=Mock()))
        tasks = [{'id': str(i), 'level': 'easy', 'query': str(i), 'days': (3, 5, 7)[i % 3]} for i in range(1000)]
        cases = [{'case_id': f'validation-{days}', 'query': 'q', 'days': days} for days in (3, 5, 7)]
        if epoch_gate:
            cases = [{'case_id': f'validation-{i}', 'query': 'q', 'days': (3, 5, 7)[i % 3]} for i in range(180)]
        observed = []
        if epoch_gate:
            def forward(data, **kwargs):
                self.assertEqual(len(observed), observed[-1][0] * 256)
                self.assertEqual(len(data), 240 if dapo else 256)
                return Mock(result_async=AsyncMock(return_value=SimpleNamespace(metrics={})))
            first.forward_backward_async.side_effect = second.forward_backward_async.side_effect = forward
        def group(sampler, tokenizer, task, config, output, step, identity, budget, *, _path_id=None):
            observed.append((step, task['id'], config['reward_type']))
            return [{'turns': [{'prompt_tokens': [1], 'response_tokens': [2], 'logprobs': [-.1]}],
                     'reward': (0 if step == skip_batch or dapo and int(identity.split('-')[2]) == 0
                                else {'sum': 5, 'macro': 3}[config['reward_type']]
                                if dapo and int(identity.split('-')[2]) == 1 else i % 2), 'overlong': False, 'judgement': {'success_pass': i % 2},
                     'reward_error': None, 'termination': 'ENV_DONE', 'task_id': task['id'],
                     'response_token_len': 1} for i in (range(8) if _path_id is None else [_path_id])]
        validation_paths = []
        async def evaluate(*args):
            sampler_path = args[1];folder = args[4]
            validation_paths.append(sampler_path)
            if initial_resume:
                folder.mkdir(parents=True, exist_ok=True)
                if len(validation_paths) == 1:
                    (folder / 'predictions.jsonl').write_text(json.dumps({'task_id': cases[0]['case_id'], 'judgement': {'success_pass': 0}}) + '\n')
                    raise RuntimeError('offline initial interruption')
                self.assertEqual(sampler_path, validation_paths[0])
                rl_run.restore_validation(args[8], folder, cases, sampler_path)
                self.assertEqual(json.loads((folder / 'recovery.json').read_text())['reused_cases'], 1)
                raise RuntimeError('offline initial recovery verified')
            if interrupt and len(validation_paths) == 2:
                raise RuntimeError('offline interruption at epoch1 validation')
            metrics = {'cases': len(cases), 'successes': 0 if early_stop and len(validation_paths) > 2 else 1,
                       'sampler_path': sampler_path}
            folder.mkdir(parents=True, exist_ok=True)
            rl_run.write_json(folder / 'summary.json', metrics)
            return metrics
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);code=root/'code';code.mkdir()
            (code/'rl_config.json').write_text(json.dumps(config))
            if diagnostic:
                reference = {'rl1k_sha256': rl_run.RL_HASH, 'rows': [
                    {'batch_index': i, 'task_ids': [task['id'] for task in batch]}
                    for i, (_, batch) in enumerate(list(rl_run.dataset_batches(tasks, 5, 32, 42))[:10], 1)]}
                (code/'reference.json').write_text(json.dumps(reference))
            for name in ['rl_training.jsonl', 'rl_run.py', 'rl_rollout.py', 'rl_cost.py', 'rl_budget.py',
                         'rl_objective.py', 'sft_evaluate.py', 'sft_backend.py', 'sft_data.py', 'sft_run.py', 'run_original.py', 'pyproject.toml', 'uv.lock']:
                (code/name).symlink_to(actual/name)
            (code/'upstream/RL').mkdir(parents=True)
            (code/'upstream/RL/rllm').symlink_to(actual/'upstream/RL/rllm')
            (code/'upstream/RL/verl').symlink_to(actual/'upstream/RL/verl')
            factory=lambda **k: Budget(**{**k, 'path': root/'usage.sqlite3'})
            with patch.object(rl_run, 'ROOT', code), patch.object(rl_run, 'STUDY', root/'study'), \
                 patch.object(rl_run, 'load_credentials'), patch.object(rl_run, 'load_tasks', return_value=tasks), \
                 patch.object(rl_run, 'prepare_evaluation', return_value=(cases, {})), \
                 patch.object(rl_run, 'resource_preflight'), patch.object(rl_run, 'Budget', side_effect=factory), \
                 patch.object(rl_run, 'evaluate_checkpoint', side_effect=evaluate), \
                 patch('rl_rollout.initialize_source'), patch('rl_rollout.collect_group', side_effect=group), \
                 patch('rl_run.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                 patch('rl_run.tinker.ServiceClient', return_value=service) as service_factory, patch('builtins.print'), \
                 patch.dict(os.environ, SPINDLE_BASE_URL='https://spindle.invalid', SPINDLE_API_KEY='test-key', STAR_COMPUTE_BACKEND='tinker'):
                if initial_resume:
                    for name, error in [('interrupted-initial', 'offline initial interruption'), ('resumed-initial', 'offline initial recovery verified')]:
                        with patch.dict(os.environ, ORX_RUN_ID=name), self.assertRaisesRegex(RuntimeError, error):
                            await rl_run.main()
                    service.create_lora_training_client_async.assert_awaited_once()
                    self.assertEqual(service.create_training_client_from_state_with_optimizer_async.await_count, 1)
                    return
                with patch.dict(os.environ, ORX_RUN_ID='run-one'):
                    if interrupt:
                        with self.assertRaisesRegex(RuntimeError, 'offline interruption'):
                            await rl_run.main()
                    else:
                        await rl_run.main()
                pointer=root/'study/offline-curriculum/latest-checkpoint.json'
                prior=json.loads(pointer.read_text())
                self.assertEqual(prior['batch_index'], boundary)
                self.assertEqual(set(prior['evaluations']), {'0'} if interrupt else {'0', '1'})
                if source_weighting:
                    changed = {**config, 'loss_weighting': {'mode': 'global_token_mean'}}
                    (code/'rl_config.json').write_text(json.dumps(changed))
                    with patch.dict(os.environ, ORX_RUN_ID='rejected-mode-change'):
                        with self.assertRaisesRegex((ValueError, AssertionError), 'Resume changed source/config|Epoch resume'):
                            await rl_run.main()
                    (code/'rl_config.json').write_text(json.dumps(config))
                async def fail_after_gate(completed):
                    summary = pointer.parent / 'summary.json'
                    before = summary.read_bytes()
                    validation_count = len(validation_paths)
                    service.create_sampling_client_async.side_effect = RuntimeError('offline sampler failure')
                    try:
                        with patch.dict(os.environ, ORX_RUN_ID=f'failed-after-gate-{completed}'):
                            with self.assertRaisesRegex(RuntimeError, 'offline sampler failure'):
                                await rl_run.main()
                    finally:
                        service.create_sampling_client_async.side_effect = None
                    failed = json.loads(pointer.read_text())
                    self.assertEqual(failed['status'], 'initialized')
                    self.assertEqual(failed['batch_index'], completed)
                    self.assertEqual(first.optim_step_async.await_count + second.optim_step_async.await_count, completed)
                    self.assertEqual(len(validation_paths), validation_count)
                    self.assertEqual(summary.read_bytes(), before)
                if retry_after_gates:
                    await fail_after_gate(31)
                with patch.dict(os.environ, ORX_RUN_ID='run-two'):
                    await rl_run.main()
                if epoch_gate:
                    restored = json.loads(pointer.read_text())
                    self.assertEqual(restored['batch_index'], 31 if interrupt else 62)
                    self.assertEqual(restored['status'], 'epoch_complete_audit_required')
                    self.assertEqual(second.optim_step_async.await_count, 0 if interrupt else applied(62) - applied(31))
                    if interrupt:
                        with patch.dict(os.environ, ORX_RUN_ID='run-epoch-two'):
                            await rl_run.main()
                        restored = json.loads(pointer.read_text())
                        self.assertEqual(restored['batch_index'], 62)
                        self.assertEqual(restored['status'], 'epoch_complete_audit_required')
                    if retry_after_gates:
                        await fail_after_gate(62)
                    with patch.dict(os.environ, ORX_RUN_ID='run-epoch-three'):
                        await rl_run.main()
                    terminal = json.loads(pointer.read_text())
                    before = {p: p.read_bytes() for p in (root / 'study').rglob('*') if p.is_file()}
                    calls_before = service_factory.call_count
                    with patch.dict(os.environ, ORX_RUN_ID='terminal-no-op'):
                        await rl_run.main()
                    self.assertEqual(service_factory.call_count, calls_before)
                    self.assertEqual(before, {p: p.read_bytes() for p in (root / 'study').rglob('*') if p.is_file()})
                    self.assertFalse((pointer.parent / 'runs/terminal-no-op').exists())
                    for field, value in [('schedule', {}), ('config', {}), ('source_hashes', {}), ('evaluations', {})]:
                        rl_run.write_json(pointer, {**terminal, field: value})
                        with patch.dict(os.environ, ORX_RUN_ID=f'reject-{field}'):
                            with self.assertRaisesRegex(ValueError, 'Epoch resume'):
                                await rl_run.main()
                        self.assertEqual(service_factory.call_count, calls_before)
                    rl_run.write_json(pointer, terminal)
                    summary = pointer.parent / 'summary.json'
                    saved_summary = summary.read_bytes()
                    for value in [None, {'checkpoint': {}}]:
                        if value is None:
                            summary.unlink()
                        else:
                            rl_run.write_json(summary, value)
                        with patch.dict(os.environ, ORX_RUN_ID='reject-summary'):
                            with self.assertRaisesRegex(ValueError, 'terminal summary missing or stale'):
                                await rl_run.main()
                        self.assertEqual(service_factory.call_count, calls_before)
                    summary.write_bytes(saved_summary)
            final=json.loads(pointer.read_text())
            self.assertEqual(final['batch_index'], stop)
            self.assertEqual(final['optimizer_step'], applied(stop))
            self.assertEqual(final['status'], 'diagnostic_early_stop' if early_stop else
                             'diagnostic_complete' if diagnostic else 'three_epochs_complete_approval_required')
            expected_sampling_batches = [i for i in range(stop)
                for _ in range(2 if retry_after_gates and i in (31, 62) else 1)]
            self.assertEqual([call.kwargs['model_path'] for call in service.create_sampling_client_async.await_args_list],
                [f'tinker://{"first" if i < boundary else "restored"}/sampler/rl-batch-{i}-step-{applied(i)}'
                 for i in expected_sampling_batches])
            expected = {'0', '2'} if early_stop else {'0', '2', '5', '10'} if diagnostic else {'0', '1', '2', '3'}
            self.assertEqual(set(final['evaluations']), expected)
            self.assertEqual(first.optim_step_async.await_count, applied(boundary))
            self.assertEqual(second.optim_step_async.await_count, applied(stop) - applied(boundary))
            if spindle:
                service.create_training_client_from_state_async.assert_not_awaited()
                service.create_lora_training_client_async.assert_awaited_once_with(rl_run.MODEL, rank=32,
                    train_attn=True, train_mlp=True, train_unembed=True,
                    user_metadata={'orx_run_id': 'run-one', 'condition': config['condition']})
                self.assertEqual(validation_paths[0], 'tinker://first/sampler/rl-batch-0-step-0')
                self.assertTrue(all(call.kwargs['base_url'] == 'https://spindle.invalid' and call.kwargs['api_key'] == 'test-key'
                                    for call in service_factory.call_args_list))
            else:
                service.create_training_client_from_state_async.assert_awaited_once()
            self.assertEqual(service.create_training_client_from_state_with_optimizer_async.await_count,
                             ((3 if interrupt else 2) if epoch_gate else 1) + (2 if retry_after_gates else 0))
            self.assertEqual(service.create_training_client_from_state_with_optimizer_async.await_args_list[0].args[0], prior['state_path'])
            if interrupt:
                self.assertEqual(validation_paths[2], prior['sampler_path'])
            if epoch_gate:
                self.assertEqual(len(validation_paths), 5 if interrupt else 4)
            self.assertEqual(len(observed), stop * 32 * 8)
            self.assertTrue(all(reward == ('sum' if (source_weighting and not curriculum) or index <= 62 else 'macro')
                                for index, _, reward in observed))
            self.assertTrue(all(call.args[0].learning_rate == config['learning_rate']
                                for t in [first, second] for call in t.optim_step_async.await_args_list))

            if source_weighting:
                files = list((root/'study/offline-curriculum/runs').glob('*/source-weighting.jsonl'))
                rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
                self.assertEqual(sorted(row['batch_index'] for row in rows), [i for i in range(1, stop + 1) if i != skip_batch])
                from rl_objective import SOURCE_BALANCING, source_microbatch_coefficients
                self.assertEqual(final['source_hashes'][SOURCE_BALANCING], config['loss_weighting']['source_sha256'])
                for row in rows:
                    self.assertEqual(row['attention_lengths'], [2] * (240 if dapo else 256))
                    self.assertEqual(row['response_mask_counts'], [1] * (240 if dapo else 256))
                    if dapo:
                        self.assertEqual(row['selected_indices'], list(range(16, 256)))
                        self.assertEqual(len(row['task_ids']), 240)
                    self.assertLess(row['negative_weight_sum'], 0)
                    self.assertGreater(row['positive_weight_sum'], 0)
                    self.assertEqual(len(row['effective_float32_weights_sha256']), 64)
                calls = first.forward_backward_async.await_args_list + second.forward_backward_async.await_args_list
                self.assertEqual(len(calls), applied(stop))
                size = 240 if dapo else 256
                coefficients, _ = source_microbatch_coefficients([2] * size, [1] * size)
                advantages = rl_run.group_relative_advantages([i % 2 for i in range(size)],
                                                              [i // 8 for i in range(size)])
                for call in calls:
                    data = call.args[0]
                    self.assertEqual(len(data), 240 if dapo else 256)
                    self.assertEqual(call.kwargs, {'loss_fn': 'cross_entropy'})
                    for datum, advantage, coefficient in zip(data, advantages, coefficients, strict=True):
                        self.assertAlmostEqual(datum.loss_fn_inputs['weights'].data[0],
                                               advantage * coefficient, delta=1e-9)


class SourceWeightingConfigTests(unittest.TestCase):
    def test_stitched_masks_and_effective_weight_evidence(self):
        config = json.loads(Path('rl_config.json').read_text())
        short = {'turns': [{'prompt_tokens': [1], 'response_tokens': [2, 3], 'logprobs': [-.1, -.2]}],
                 'overlong': False}
        longer = {'turns': short['turns'] + [{'prompt_tokens': [1, 2, 3, 4],
                  'response_tokens': [5, 6, 7], 'logprobs': [-.3, -.4, -.5]}], 'overlong': False}
        rows = [longer, {**longer, 'overlong': True}, longer] + [short] * 253
        advantages = [-1., 1., 0.] + [1.] * 253
        data = [rl_run.training_datum(row, advantage) for row, advantage in zip(rows, advantages, strict=True)]
        before = [d.loss_fn_inputs['weights'].data for d in data]
        diagnostics = rl_run.weight_training_data(data, config)
        self.assertEqual(diagnostics['attention_lengths'][:3], [7, 7, 7])
        self.assertEqual(diagnostics['response_mask_counts'][:3], [5, 0, 5])
        self.assertEqual(data[0].loss_fn_inputs['weights'].data[2], 0.)
        self.assertTrue(all(value == 0 for i in (1, 2) for value in data[i].loss_fn_inputs['weights'].data))
        self.assertLess(data[0].loss_fn_inputs['weights'].data[-1], 0.)
        digest = hashlib.sha256()
        for datum in data:
            values = datum.loss_fn_inputs['weights'].data
            digest.update(struct.pack('<Q', len(values)))
            for value in values:
                digest.update(struct.pack('<f', value))
        self.assertEqual(diagnostics['effective_float32_weights_sha256'], digest.hexdigest())
        self.assertGreater(len({m['coefficient'] for m in diagnostics['microbatches']
                                if m['unmasked_tokens']}), 1)
        for micro in diagnostics['microbatches']:
            for i in micro['indices']:
                expected = [struct.unpack('<f', struct.pack('<f', value * micro['coefficient']))[0]
                            for value in before[i]]
                self.assertEqual(data[i].loss_fn_inputs['weights'].data, expected)
            self.assertEqual(micro['unmasked_tokens'],
                             sum(diagnostics['response_mask_counts'][i] for i in micro['indices']))

    def test_source_hash_and_contract_before_training(self):
        config = json.loads(Path('rl_config.json').read_text())
        rl_run.validate_loss_weighting(config)
        for key, value in [('source_sha256', 'changed'), ('virtual_source_dp', 16), ('max_token_len', 4096)]:
            with self.subTest(key=key):
                changed = {**config, 'loss_weighting': {**config['loss_weighting'], key: value}}
                with self.assertRaises(ValueError):
                    rl_run.validate_loss_weighting(changed)
        rl_run.validate_loss_weighting({**config, 'algorithm': 'DAPO'})
        with self.assertRaises(ValueError):
            rl_run.validate_loss_weighting({**config, 'algorithm': 'ARPO'})


if __name__ == '__main__':
    unittest.main()
