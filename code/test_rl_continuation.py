import copy
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import rl_run
from rl_budget import Budget


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    start = 20

    def setUp(self):
        self.actual = Path(__file__).resolve().parent
        self.config = json.loads((self.actual / 'rl_config.json').read_text())
        if self.start == 10:
            checkpoint = Path(self.config['diagnostic']['continuation']['parent_checkpoint'])
            self.config = json.loads(checkpoint.read_text())['config']
        self.end = self.start + 10
        self.reference_name = self.config['diagnostic']['reference_file']
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / 'code'
        self.code.mkdir()
        continuation = self.config['diagnostic']['continuation']
        original = Path(continuation['parent_checkpoint'])
        self.parent = json.loads(original.read_text())
        self.output = self.root / 'study' / self.parent['condition'] / 'runs' / continuation['parent_run_id']
        self.output.mkdir(parents=True)
        self.parent['output'] = str(self.output)
        self.grandparent = self.root / 'study' / 'grandparent' / 'runs' / 'ancestor'
        self.grandparent.mkdir(parents=True)
        self.parent['prior_outputs'] = [str(self.grandparent)] if self.start == 20 else []
        self.parent_path = self.output / 'checkpoint.json'
        self.dump(self.parent_path, self.parent)
        self.dump(self.output.parent.parent / 'frozen-provenance.json',
                  {'config': self.parent['config'], 'hashes': self.parent['source_hashes']})
        for name in ['evaluation-provenance.json', 'dataset-manifest.json']:
            (self.output / name).write_bytes((original.parent / name).read_bytes())
        continuation['parent_checkpoint'] = str(self.parent_path)
        continuation['parent_checkpoint_sha256'] = self.digest(self.parent_path)
        self.dump(self.code / 'rl_config.json', self.config)
        for name in self.parent['source_hashes']:
            target = self.code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(self.actual / name)
        (self.code / self.reference_name).symlink_to(self.actual / self.reference_name)
        self.tasks = [json.loads(line) for line in (self.actual / 'rl_training.jsonl').read_text().splitlines()]
        provenance = json.loads((self.output / 'evaluation-provenance.json').read_text())
        self.cases = [{'case_id': identity, 'query': 'offline', 'days': (3, 5, 7)[i // 20]}
                      for i, identity in enumerate(provenance.pop('diagnostic_case_ids'))]
        provenance.pop('selection')
        self.provenance = provenance
        tokenizer = Mock()
        tokenizer.get_vocab.return_value = {'offline': 1}
        self.trainer = Mock(model_id='offline')
        self.trainer.get_tokenizer.return_value = tokenizer
        def future(value):
            return Mock(result_async=AsyncMock(return_value=value))
        self.trainer.save_state_async = AsyncMock(side_effect=lambda name, **k:
            future(SimpleNamespace(path=f'tinker://offline/state/{name}')))
        self.trainer.save_weights_for_sampler_async = AsyncMock(side_effect=lambda name, **k:
            future(SimpleNamespace(path=f'tinker://offline/sampler/{name}')))
        self.trainer.forward_backward_async = AsyncMock(return_value=future(SimpleNamespace(metrics={})))
        self.trainer.optim_step_async = AsyncMock(return_value=future(SimpleNamespace(metrics={})))
        caps = SimpleNamespace(supported_models=[SimpleNamespace(model_name=rl_run.MODEL,
            trainable=True, sampleable=True, max_context_length=65536)], model_dump_json=lambda **k: '{}')
        self.service = Mock(get_server_capabilities_async=AsyncMock(return_value=caps),
            create_training_client_from_state_async=AsyncMock(side_effect=AssertionError('Weights-only load')),
            create_training_client_from_state_with_optimizer_async=AsyncMock(return_value=self.trainer),
            create_sampling_client_async=AsyncMock(return_value=Mock()))
        self.observed = []
        self.evaluations = []
        self.interrupt = False
        self.eval_successes = 16
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        patches = [
            patch.object(rl_run, 'ROOT', self.code), patch.object(rl_run, 'STUDY', self.root / 'study'),
            patch.object(rl_run, 'load_credentials'), patch.object(rl_run, 'load_tasks', return_value=self.tasks),
            patch.object(rl_run, 'prepare_evaluation', side_effect=lambda _: (self.cases, self.provenance)),
            patch.object(rl_run, 'resource_preflight'),
            patch.object(rl_run, 'Budget', side_effect=lambda **k: Budget(self.root / 'usage.sqlite3', **k)),
            patch.object(rl_run, 'collect_batch', side_effect=self.collect),
            patch.object(rl_run, 'evaluate_checkpoint', side_effect=self.evaluate),
            patch('rl_rollout.initialize_source'),
            patch('rl_run.AutoTokenizer.from_pretrained', return_value=tokenizer), patch('builtins.print')]
        for item in patches:
            self.stack.enter_context(item)
        self.client_factory = self.stack.enter_context(patch('rl_run.tinker.ServiceClient', return_value=self.service))
        self.pointer = self.root / 'study' / self.config['condition'] / 'latest-checkpoint.json'

    @staticmethod
    def dump(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    async def collect(self, pool, sampler, tokenizer, batch, config, output, index, budget):
        self.observed.append((index, [task['id'] for task in batch], config))
        return [[{'turns': [{'prompt_tokens': [1], 'response_tokens': [2], 'logprobs': [-.1]}],
                  'reward': i % 2, 'overlong': False, 'judgement': {'success_pass': i % 2},
                  'reward_error': None, 'termination': 'ENV_DONE', 'task_id': task['id']}
                 for i in range(8)] for task in batch]

    async def evaluate(self, service, sampler, tokenizer, cases, output, config, budget, phase, previous):
        index = int(output.name.rsplit('-', 1)[1])
        self.evaluations.append((index, sampler, cases, config))
        if self.interrupt and index == self.start + 5:
            self.interrupt = False
            raise RuntimeError('offline interrupted validation')
        return {'cases': 60, 'successes': self.eval_successes, 'sampler_path': sampler}

    async def run_main(self, name):
        with patch.dict(os.environ, ORX_RUN_ID=name):
            await rl_run.main()

    async def test_parent_bootstrap_then_restart_has_only_ten_new_updates(self):
        self.interrupt = True
        original_parent = self.parent_path.read_bytes()
        parent_cost = self.output / 'cost-summary.json'
        parent_cost.write_text('{"frozen_parent": true}\n')
        original_cost = parent_cost.read_bytes()
        (self.output / 'cost-events.jsonl').write_text(json.dumps(
            {'reservation_id': 'parent-payment', 'cost_usd': 7.0}) + '\n')
        grandparent_cost = self.grandparent / 'cost-summary.json'
        grandparent_cost.write_text('{"frozen_ancestor": true}\n')
        (self.grandparent / 'cost-events.jsonl').write_text(json.dumps(
            {'reservation_id': 'grandparent-payment', 'cost_usd': 11.0}) + '\n')
        with self.assertRaisesRegex(RuntimeError, 'offline interrupted validation'):
            await self.run_main('first')
        interrupted = json.loads(self.pointer.read_text())
        self.assertEqual(interrupted['batch_index'], self.start + 5)
        self.assertEqual(interrupted['schedule']['completed_window_updates'], 5)
        self.assertEqual(set(interrupted['evaluations']), set(self.parent['evaluations']) | {str(self.start + 2)})
        first_events = Path(interrupted['output']) / 'cost-events.jsonl'
        with first_events.open('a') as stream:
            stream.write(json.dumps({'reservation_id': 'child-payment', 'cost_usd': 3.0}) + '\n')
            stream.write(json.dumps({'reservation_id': 'child-payment', 'cost_usd': 3.0}) + '\n')
        await self.run_main('restart')
        final = json.loads(self.pointer.read_text())
        self.assertEqual(final['batch_index'], self.end)
        self.assertEqual(final['optimizer_step'], self.end)
        self.assertEqual(final['schedule']['planned_batches'], 155)
        self.assertEqual(final['schedule']['max_additional_updates'], 10)
        self.assertEqual(final['schedule']['completed_window_updates'], 10)
        self.assertEqual(final['schedule']['next_epoch'], 1)
        self.assertEqual(final['status'], 'diagnostic_complete')
        self.assertEqual(set(final['evaluations']), set(self.parent['evaluations']) | {str(self.start + i) for i in (2, 5, 10)})
        self.assertEqual(final['lineage']['baseline_successes'], self.parent['evaluations'][str(self.start)]['metrics']['successes'])
        self.assertEqual(final['lineage']['parent_state_path'], self.parent['state_path'])
        for key, evaluation in self.parent['evaluations'].items():
            self.assertEqual(final['evaluations'][key], evaluation)
        self.assertEqual(self.parent_path.read_bytes(), original_parent)
        self.assertEqual(parent_cost.read_bytes(), original_cost)
        summary = json.loads((self.pointer.parent / 'summary.json').read_text())
        self.assertNotIn(str(self.output), summary['run_costs'])
        self.assertAlmostEqual(summary['cumulative_lineage_usage']['combined_usage_estimate_usd'] -
                               summary['attributable_usage']['combined_usage_estimate_usd'], 18.0 if self.start == 20 else 7.0)
        self.assertEqual(summary['attributable_usage']['provider_reported_usage_usd'], 3.0)
        self.assertEqual(grandparent_cost.read_text(), '{"frozen_ancestor": true}\n')
        self.assertNotIn(str(self.grandparent), summary['run_costs'])
        self.assertEqual(final['prior_outputs'], [interrupted['output'], str(self.output)] + self.parent['prior_outputs'])
        self.assertEqual(set(final['lineage']['source_changes']), {'rl_run.py', self.reference_name})
        self.assertEqual([i for i, _, _ in self.observed], list(range(self.start + 1, self.end + 1)))
        reference = json.loads((self.code / self.reference_name).read_text())
        if self.start == 20:
            self.assertEqual(reference['rows'][:20], json.loads(
                (self.actual / 'rl_continuation_reference.json').read_text())['rows'])
        self.assertEqual([ids for _, ids, _ in self.observed], [row['task_ids'] for row in reference['rows'][self.start:self.end]])
        self.assertEqual(self.trainer.optim_step_async.await_count, 10)
        self.assertEqual(self.trainer.forward_backward_async.await_count, 10)
        self.service.create_training_client_from_state_async.assert_not_awaited()
        self.assertEqual([c.args[0] for c in self.service.create_training_client_from_state_with_optimizer_async.await_args_list],
                         [self.parent['state_path'], interrupted['state_path']])
        self.assertEqual([v[0] for v in self.evaluations], [self.start + i for i in (2, 5, 5, 10)])
        self.assertEqual(self.evaluations[2][1], interrupted['sampler_path'])
        for _, _, cases, config in self.evaluations:
            self.assertEqual([case['case_id'] for case in cases], [case['case_id'] for case in self.cases])
            self.assertEqual(config['seed'], self.parent['config']['seed'])
            self.assertEqual(config['eval_temperature'], self.parent['config']['eval_temperature'])
            self.assertEqual(config['eval_concurrency'], 24)
        self.assertTrue(all(c['group_concurrency'] == 8 and c['group_size'] == 8 for _, _, c in self.observed))
        self.assertTrue(all(c.args[0].learning_rate == 2e-5 for c in self.trainer.optim_step_async.await_args_list))
        original_complete = self.pointer.read_bytes()
        client_calls = self.client_factory.call_count
        await self.run_main('already-complete')
        self.assertEqual(self.client_factory.call_count, client_calls)
        self.assertEqual(self.pointer.read_bytes(), original_complete)
        self.assertEqual(final['lineage']['operational_change'], {} if self.start == 20 else
            {'eval_concurrency': {'parent': 12, 'child': 24}})
        self.assertEqual(self.trainer.optim_step_async.await_count, 10)
        self.assertEqual(len(self.evaluations), 4)
        summary_path = self.pointer.parent / 'summary.json'
        original_summary = summary_path.read_bytes()
        for fault in ('missing', 'stale'):
            if fault == 'missing':
                summary_path.unlink()
            else:
                summary_path.write_text('{"checkpoint": {}}')
            with self.assertRaisesRegex(ValueError, 'terminal summary missing or stale'):
                await self.run_main(f'bad-summary-{fault}')
            self.assertEqual(self.client_factory.call_count, client_calls)
            self.assertEqual(self.pointer.read_bytes(), original_complete)
        summary_path.write_bytes(original_summary)
        final = json.loads(self.pointer.read_text())
        final['batch_index'] = final['optimizer_step'] = self.end + 1
        self.dump(self.pointer, final)
        self.client_factory.reset_mock()
        with self.assertRaisesRegex(ValueError, 'child checkpoint/provenance'):
            await self.run_main('out-of-window')
        self.client_factory.assert_not_called()

    async def test_parent_and_runtime_mismatches_fail_before_model_client(self):
        original_config = copy.deepcopy(self.config)
        original_parent = self.parent_path.read_bytes()
        original_provenance = copy.deepcopy(self.provenance)
        frozen_path = self.output.parent.parent / 'frozen-provenance.json'
        original_frozen = frozen_path.read_bytes()
        dataset_path = self.output / 'dataset-manifest.json'
        original_dataset = dataset_path.read_bytes()
        for fault in ['checkpoint_hash', 'training_config', 'database', 'panel', 'dataset', 'frozen', 'source',
                      'parent_cursor', 'parent_sampler', 'parent_evaluations', 'old_reference']:
            with self.subTest(fault=fault):
                self.config = copy.deepcopy(original_config)
                self.parent_path.write_bytes(original_parent)
                frozen_path.write_bytes(original_frozen)
                dataset_path.write_bytes(original_dataset)
                self.provenance = copy.deepcopy(original_provenance)
                source_path = self.code / 'rl_rollout.py'
                if not source_path.is_symlink():
                    source_path.unlink()
                    source_path.symlink_to(self.actual / 'rl_rollout.py')
                old_reference = self.code / ('rl_continuation_reference.json' if self.start == 20 else
                                             'rl_diagnostic_reference.json')
                if not old_reference.is_symlink():
                    old_reference.unlink()
                    old_reference.symlink_to(self.actual / old_reference.name)
                if fault == 'checkpoint_hash':
                    self.parent_path.write_text('{}')
                elif fault == 'training_config':
                    self.config['learning_rate'] = 4e-5
                elif fault == 'database':
                    self.provenance['database_hashes']['attractions.csv'] = 'changed'
                elif fault == 'panel':
                    self.provenance['benchmark_sha256'] = 'changed'
                elif fault == 'dataset':
                    dataset_path.write_text('{}')
                elif fault == 'frozen':
                    frozen_path.write_text('{}')
                elif fault in {'parent_cursor', 'parent_sampler', 'parent_evaluations'}:
                    changed_parent = json.loads(original_parent)
                    if fault == 'parent_cursor':
                        changed_parent['optimizer_step'] -= 1
                    elif fault == 'parent_sampler':
                        changed_parent['sampler_path'] = 'tinker://wrong/sampler'
                    else:
                        changed_parent['evaluations'].pop('2')
                    self.dump(self.parent_path, changed_parent)
                    self.config['diagnostic']['continuation']['parent_checkpoint_sha256'] = self.digest(self.parent_path)
                elif fault == 'old_reference':
                    old_reference.unlink()
                    old_reference.write_text('{}')
                elif fault == 'source':
                    source_path.unlink()
                    source_path.write_text('changed')
                self.dump(self.code / 'rl_config.json', self.config)
                with self.assertRaisesRegex(ValueError, 'Continuation'):
                    await self.run_main(f'bad-{fault}')
                self.client_factory.assert_not_called()

    async def test_early_stop_rerun_preserves_terminal_checkpoint_without_model_calls(self):
        self.eval_successes = self.parent['evaluations'][str(self.start)]['metrics']['successes'] - 10
        await self.run_main('collapse')
        terminal = self.pointer.read_bytes()
        checkpoint = json.loads(terminal)
        self.assertEqual(checkpoint['status'], 'diagnostic_early_stop')
        self.assertEqual(checkpoint['batch_index'], self.start + 2)
        self.assertEqual(self.trainer.optim_step_async.await_count, 2)
        client_calls = self.client_factory.call_count
        await self.run_main('already-stopped')
        self.assertEqual(self.client_factory.call_count, client_calls)
        self.assertEqual(self.trainer.optim_step_async.await_count, 2)
        self.assertEqual(self.pointer.read_bytes(), terminal)
        self.assertEqual(len(self.evaluations), 1)

    async def test_resume_rejects_corrupt_child_before_model_client(self):
        self.interrupt = True
        with self.assertRaisesRegex(RuntimeError, 'offline interrupted validation'):
            await self.run_main('interrupted')
        original = json.loads(self.pointer.read_text())
        for fault in ('low_cursor', 'high_cursor', 'optimizer', 'lineage', 'inherited', 'premature_complete'):
            with self.subTest(fault=fault):
                child = copy.deepcopy(original)
                if fault in {'low_cursor', 'high_cursor'}:
                    index = self.start - 1 if fault == 'low_cursor' else self.end + 1
                    child['batch_index'] = child['optimizer_step'] = index
                    child['schedule'] = rl_run.schedule_cursor(self.config, index)
                elif fault == 'optimizer':
                    child['optimizer_step'] -= 1
                elif fault == 'lineage':
                    child['lineage']['baseline_successes'] += 1
                elif fault == 'inherited':
                    child['evaluations']['0']['metrics']['successes'] += 1
                else:
                    child['status'] = 'diagnostic_complete'
                self.dump(self.pointer, child)
                client_calls = self.client_factory.call_count
                with self.assertRaisesRegex(ValueError, 'Continuation'):
                    await self.run_main(f'bad-child-{fault}')
                self.assertEqual(self.client_factory.call_count, client_calls)

    def test_collapse_guard_uses_parent_terminal_not_sft0(self):
        evaluations = copy.deepcopy(self.parent['evaluations'])
        key = str(self.start + 2)
        baseline = self.parent['evaluations'][str(self.start)]['metrics']['successes']
        evaluations[key] = {'metrics': {'successes': baseline - 9}}
        self.assertIsNone(rl_run.diagnostic_stop_reason(self.config, evaluations))
        evaluations[key]['metrics']['successes'] = baseline - 10
        self.assertIsNotNone(rl_run.diagnostic_stop_reason(self.config, evaluations))
        self.assertEqual(evaluations['0']['metrics']['successes'], 9)


class FirstContinuationTests(ContinuationTests):
    start = 10


if __name__ == '__main__':
    unittest.main()
