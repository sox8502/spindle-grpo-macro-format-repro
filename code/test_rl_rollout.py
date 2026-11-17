"""Offline integration checks against the released agent, cached tools and scorer."""
import ast
import gzip
import json
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from transformers import AutoTokenizer

import rl_rollout
from sft_backend import MODEL, TOKENIZER_REVISION


class FakeBudget:
    def __init__(self):
        self.reservations = []
        self.settlements = []

    def reserve(self, operation, upper):
        self.reservations.append((operation, upper))
        return len(self.reservations)

    def settle(self, reservation, upper=None):
        self.settlements.append((reservation, upper))


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = rl_rollout.initialize_source()
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=TOKENIZER_REVISION, local_files_only=True)
        tree = ast.parse(Path('rl_run.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'stitch_turns')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'rl_run.py', 'exec'), namespace)
        cls.stitch = staticmethod(namespace['stitch_turns'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.budget = FakeBudget()
        self.config = dict(algorithm='grpo', reward_type='sum', total_steps=1,
                           group_size=1, max_steps=60, max_response_length=28 * 1024)
        self.task = {'query': 'Find cities in Colorado.', 'id': 'offline-test'}

    def sample(self, text, stop='stop'):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        # Valid alternate tokenization exposes accidental decode/encode reconstruction.
        if text.endswith('  '):
            tokens = self.tokenizer.encode(text[:-2], add_special_tokens=False)
            tokens += self.tokenizer.encode(' ', add_special_tokens=False) * 2
        seq = SimpleNamespace(tokens=tokens, logprobs=[-.5] * len(tokens), stop_reason=stop)
        return seq, {'sequences': [vars(seq)]}

    def collect(self, samples):
        with patch.object(rl_rollout, 'sample_response', side_effect=samples) as sample:
            result = rl_rollout.collect_group(None, self.tokenizer, self.task, self.config,
                                              self.tmp.name, 1, 0, budget=self.budget)
        return result, sample

    def test_real_tool_and_scorer_preserve_exact_sampled_history(self):
        tool = '<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>  '
        final = '<think>Done.</think><answer>No itinerary.</answer><|im_end|>'
        with patch.object(self.source.reward, 'call_format_model', return_value=[]) as formatter, \
                patch.object(self.source.reward, 'eval_one_sample', wraps=self.source.reward.eval_one_sample) as scorer:
            results, sample = self.collect([self.sample(tool), self.sample(final)])
        result = results[0]
        self.assertEqual(result['termination'], 'ENV_DONE')
        self.assertEqual(len(result['turns']), 2)
        first, second = result['turns']
        prefix = first['prompt_tokens'] + first['response_tokens']
        self.assertEqual(second['prompt_tokens'][:len(prefix)], prefix)
        self.assertIn('Denver', self.tokenizer.decode(second['prompt_tokens']))
        self.assertTrue(self.tokenizer.decode(first['prompt_tokens']).endswith('<|im_start|>assistant\n'))
        tokens, mask, old = self.stitch(result['turns'])
        self.assertEqual(sum(mask), sum(len(turn['response_tokens']) for turn in result['turns']))
        self.assertEqual(len(tokens), len(old))
        self.assertEqual(formatter.call_count, 1)
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(len(self.budget.settlements), 2)

    def test_genuine_format_error_is_zero_without_formatter(self):
        with patch.object(self.source.reward, 'call_format_model') as formatter:
            results, _ = self.collect([self.sample('No required tags')])
        self.assertEqual(results[0]['reward'], 0)
        self.assertEqual(results[0]['reward_error'], 'Format error occurs in the previous turn!')
        formatter.assert_not_called()

    def test_formatter_failure_is_not_silently_a_zero_reward(self):
        with patch('rl_cost.call_format_model', side_effect=RuntimeError('offline failure')):
            with self.assertRaisesRegex(RuntimeError, 'offline failure'):
                self.collect([self.sample('<think>Done.</think><answer>Plan.</answer>')])

    def test_released_scorer_owns_all_caught_error_zero_rewards(self):
        for reward_type in ('sum', 'macro', 'success', 'curriculum'):
            for error in (KeyError(0), TypeError('malformed plan'), RuntimeError('scorer failed')):
                with self.subTest(reward_type=reward_type, error=type(error).__name__), \
                        tempfile.TemporaryDirectory() as output:
                    config = {**self.config, 'reward_type': reward_type}
                    with patch.object(self.source.reward, 'call_format_model', return_value={}), \
                            patch.object(self.source.reward, 'eval_one_sample', side_effect=error), \
                            patch.object(rl_rollout, 'sample_response', return_value=self.sample(
                                '<think>Done.</think><answer>Plan.</answer>')):
                        results = rl_rollout.collect_group(None, self.tokenizer, self.task,
                            config, output, 1, 0, budget=self.budget)
                    self.assertEqual(results[0]['reward'], 0)
                    self.assertEqual(results[0]['termination'], 'ENV_DONE')
                    self.assertEqual(results[0]['reward_error'], f'Error evaluating one sample: {error}')
                    events = rl_rollout._read_journal(results[0]['evidence'])
                    self.assertEqual(events[-2]['info']['metadata']['error'], results[0]['reward_error'])
                    self.assertEqual(events[-1]['event'], 'complete')

    def test_saved_terminal_environment_recovers_without_sampling_or_rescoring(self):
        for stop in ('stop', 'length'):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as original_output:
                config = dict(self.config)
                with patch.object(self.source.reward, 'call_format_model', return_value={}), \
                        patch.object(self.source.reward, 'eval_one_sample', side_effect=KeyError(0)), \
                        patch.object(rl_rollout, 'sample_response', return_value=self.sample(
                            '<think>Done.</think><answer>Plan.</answer>', stop=stop)):
                    original = rl_rollout.collect_group(None, self.tokenizer, self.task,
                        config, original_output, 1, 0, budget=self.budget)
                journal = Path(original[0]['evidence'])
                rows = rl_rollout._read_journal(journal)[:-1]
                self.assertEqual(rows[-1]['event'], 'environment')
                rows[-1].pop('done')
                self.assertFalse(rows[-1]['info']['reach_limit'])
                with gzip.open(journal, 'wt') as stream:
                    stream.write(''.join(json.dumps(row) + '\n' for row in rows))
                config['collection_resume_output'] = original_output
                with tempfile.TemporaryDirectory() as output, \
                        patch.object(rl_rollout, 'sample_response') as sample, \
                        patch.object(self.source.environment, 'step') as environment, \
                        patch('rl_cost.requests.post') as formatter:
                    recovered = rl_rollout.collect_group(None, self.tokenizer, self.task,
                        config, output, 1, 0, budget=self.budget)
                    self.assertEqual({key: value for key, value in recovered[0].items() if key != 'evidence'},
                                     {key: value for key, value in original[0].items() if key != 'evidence'})
                    self.assertEqual(rl_rollout._read_journal(recovered[0]['evidence'])[-1]['event'], 'complete')
                    config['collection_resume_output'] = output
                    with tempfile.TemporaryDirectory() as next_output:
                        replayed = rl_rollout.collect_group(None, self.tokenizer, self.task,
                            config, next_output, 1, 0, budget=self.budget)
                        self.assertEqual(replayed[0]['turns'], original[0]['turns'])
                    sample.assert_not_called()
                    environment.assert_not_called()
                    formatter.assert_not_called()

    def test_budget_exhaustion_escapes_source_reward_catcher(self):
        from rl_budget import BudgetExceeded
        with patch('rl_cost.call_format_model', side_effect=BudgetExceeded('cap reached')):
            with self.assertRaises(BudgetExceeded):
                self.collect([self.sample('<think>Done.</think><answer>Plan.</answer>')])

    def test_exhausted_formatter_content_failure_retains_released_zero(self):
        from rl_cost import FormatterContentError
        for error in (json.JSONDecodeError('Expecting value', 'bad', 0),
                      ValueError('Empty formatter completion'), FormatterContentError('malformed response')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as output:
                with patch('rl_cost.call_format_model', side_effect=error):
                    with patch.object(rl_rollout, 'sample_response', return_value=self.sample(
                            '<think>Done.</think><answer>Plan.</answer>')):
                        results = rl_rollout.collect_group(None, self.tokenizer, self.task,
                            self.config, output, 1, 0, budget=self.budget)
                self.assertEqual(results[0]['reward'], 0)
                self.assertEqual(results[0]['termination'], 'ENV_DONE')
                self.assertIn('Error formatting planning:', results[0]['reward_error'])
                events = rl_rollout._read_journal(results[0]['evidence'])
                self.assertEqual(events[-3]['event'], 'formatter_failure')
                self.assertEqual(events[-1]['event'], 'complete')

    def test_transport_auth_and_harness_errors_escape(self):
        import requests
        for error in (requests.ConnectionError('connection failed'),
                      requests.HTTPError('401 Unauthorized'),
                      ValueError('Unknown formatter profile'), KeyError('choices')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as output:
                with patch('rl_cost.call_format_model', side_effect=error):
                    with patch.object(rl_rollout, 'sample_response', return_value=self.sample(
                            '<think>Done.</think><answer>Plan.</answer>')):
                        with self.assertRaises(type(error)):
                            rl_rollout.collect_group(None, self.tokenizer, self.task,
                                self.config, output, 1, 0, budget=self.budget)

    def test_completed_journal_restores_exact_turns_without_paid_calls(self):
        tool = '<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>  '
        original, _ = self.collect([self.sample(tool), self.sample('invalid')])
        self.config['collection_resume_output'] = self.tmp.name
        with tempfile.TemporaryDirectory() as output:
            with patch.object(rl_rollout, 'sample_response') as sample, \
                    patch('rl_cost.call_format_model') as formatter:
                recovered = rl_rollout.collect_group(None, self.tokenizer, self.task,
                    self.config, output, 1, 0, budget=self.budget)
            self.assertEqual(recovered[0]['turns'], original[0]['turns'])
            self.assertEqual(recovered[0]['reward'], original[0]['reward'])
            sample.assert_not_called()
            formatter.assert_not_called()
            self.assertEqual(len(list(Path(output).iterdir())), 1)
            self.config['collection_resume_output'] = output
            with tempfile.TemporaryDirectory() as next_output, \
                    patch.object(rl_rollout, 'sample_response') as sample:
                replayed = rl_rollout.collect_group(None, self.tokenizer, self.task,
                    self.config, next_output, 1, 0, budget=self.budget)
                self.assertEqual(replayed[0]['turns'], original[0]['turns'])
                sample.assert_not_called()
        self.config['collection_resume_output'] = self.tmp.name
        self.config['temperature'] = .3
        with self.assertRaisesRegex(ValueError, 'scientific control: temperature'):
            self.collect([])

    def test_formatter_interruption_recovers_cached_sample_and_environment(self):
        response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {'choices': [{'message': {'content': 'bad json'}}], 'usage': {'cost': 0}})
        tool = '<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>  '
        final = '<think>Done.</think><answer>Plan.</answer>'
        with patch('rl_cost.requests.post', return_value=response), \
                patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline'}), \
                patch('rl_cost.time.sleep'):
            original, _ = self.collect([self.sample(tool), self.sample(final)])
        journal = Path(original[0]['evidence'])
        events = rl_rollout._read_journal(journal)
        last_sample = max(i for i, row in enumerate(events) if row['event'] == 'sample')
        with gzip.open(journal, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in events[:last_sample + 1]))
        self.config['collection_resume_output'] = self.tmp.name
        reservations = len(self.budget.reservations)
        with tempfile.TemporaryDirectory() as output:
            with patch.object(rl_rollout, 'sample_response') as sample, \
                    patch('rl_cost.requests.post') as formatter:
                recovered = rl_rollout.collect_group(None, self.tokenizer, self.task,
                    self.config, output, 1, 0, budget=self.budget)
            sample.assert_not_called()
            formatter.assert_not_called()
            self.assertEqual(len(self.budget.reservations), reservations)
            self.assertEqual(recovered[0]['turns'], original[0]['turns'])
            self.assertEqual(recovered[0]['reward_error'], original[0]['reward_error'])
            self.assertEqual(recovered[0]['response_token_len'], original[0]['response_token_len'])
            self.assertEqual(recovered[0]['termination'], 'ENV_DONE')
            self.assertEqual(rl_rollout._read_journal(recovered[0]['evidence'])[-1]['event'], 'complete')
            self.config['collection_resume_output'] = output
            with tempfile.TemporaryDirectory() as next_output, \
                    patch.object(rl_rollout, 'sample_response') as sample, \
                    patch('rl_cost.requests.post') as formatter:
                replayed = rl_rollout.collect_group(None, self.tokenizer, self.task,
                    self.config, next_output, 1, 0, budget=self.budget)
                self.assertEqual(replayed[0]['turns'], original[0]['turns'])
                sample.assert_not_called()
                formatter.assert_not_called()

    def test_formatter_network_failure_reuses_model_samples_and_retries_only_formatter(self):
        import requests
        tool = '<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>'
        final = '<think>Done.</think><answer>Plan.</answer>'
        with patch('rl_cost.requests.post', side_effect=requests.ConnectionError('offline DNS failure')), \
                patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline'}), patch('rl_cost.time.sleep'):
            with self.assertRaises(requests.ConnectionError):
                self.collect([self.sample(tool), self.sample(final)])
        journal = next(Path(self.tmp.name).glob('group-*.jsonl.gz'))
        before = rl_rollout._read_journal(journal)
        self.assertEqual(before[-1]['event'], 'sample')
        formatter_path = next(Path(self.tmp.name).glob('*-formatter.jsonl.gz'))
        raw_formatter = formatter_path.read_bytes()
        self.assertIsNone(rl_rollout._cached_formatter(formatter_path))
        self.config['collection_resume_output'] = self.tmp.name
        response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {'choices': [{'message': {'content': '[]'}}], 'usage': {'cost': 0}})
        with tempfile.TemporaryDirectory() as output, patch.object(rl_rollout, 'sample_response') as sample, \
                patch('rl_cost.requests.post', return_value=response) as formatter, \
                patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline'}):
            recovered = rl_rollout.collect_group(None, self.tokenizer, self.task,
                self.config, output, 1, 0, budget=self.budget)
            sample.assert_not_called()
            formatter.assert_called_once()
            self.assertEqual(recovered[0]['termination'], 'ENV_DONE')
            self.assertEqual(len(recovered[0]['turns']), 2)
            self.assertEqual(formatter_path.read_bytes(), raw_formatter)
            self.assertEqual(rl_rollout._read_journal(journal), before)
            with gzip.open(next(Path(output).glob('*-formatter.jsonl.gz')), 'rt') as stream:
                new_rows = [json.loads(line) for line in stream]
            self.assertEqual(sum('request' in row for row in new_rows), 1)
            self.assertEqual(sum('response' in row for row in new_rows), 1)

    def test_cached_formatter_rejects_transport_success_and_request_mismatch(self):
        evidence = Path(self.tmp.name) / 'formatter.jsonl.gz'
        for status, content, error in ((401, 'bad', 'transport failure'),
                                       (200, '[]', 'successful attempt'),
                                       (200, 'bad', 'request differs')):
            with self.subTest(status=status, content=content):
                rows = [{'request': {'messages': []}}] + [
                    {'attempt': attempt, 'status': status,
                     'response': {'choices': [{'message': {'content': content}}]}}
                    for attempt in (1, 2)]
                with gzip.open(evidence, 'wt') as stream:
                    stream.write(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaisesRegex(ValueError, error):
                    rl_rollout._cached_formatter(evidence)([{'role': 'user', 'content': 'changed'}])

    def test_cached_successful_formatter_survives_accounting_failure(self):
        evidence = Path(self.tmp.name) / 'formatter.jsonl.gz'
        rows = [{'request': {'messages': []}},
                {'attempt': 1, 'status': 200,
                 'response': {'choices': [{'message': {'content': '[]'}}]}},
                {'attempt': 1, 'error': 'OperationalError: unable to open database file'}]
        with gzip.open(evidence, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
        with patch('rl_cost.requests.post') as formatter:
            self.assertEqual(rl_rollout._cached_formatter(evidence)([]), [])
        formatter.assert_not_called()

    def test_cached_formatter_accepts_success_after_transport_failure(self):
        evidence = Path(self.tmp.name) / 'formatter.jsonl.gz'
        rows = [{'request': {'messages': []}},
                {'attempt': 1, 'error': 'ConnectionError: read timed out'},
                {'attempt': 2, 'status': 200,
                 'response': {'choices': [{'message': {'content': '[]'}}]}}]
        with gzip.open(evidence, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
        with patch('rl_cost.requests.post') as formatter:
            self.assertEqual(rl_rollout._cached_formatter(evidence)([]), [])
        formatter.assert_not_called()
        rows[1]['error'] = 'UnknownError: missing response'
        with gzip.open(evidence, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            rl_rollout._cached_formatter(evidence)

    def test_cached_formatter_retries_interrupted_response_stream(self):
        evidence = Path(self.tmp.name) / 'formatter.jsonl.gz'
        rows = [{'request': {'messages': []}},
                {'attempt': 1, 'error': 'ChunkedEncodingError: Response ended prematurely'},
                {'attempt': 2, 'error': 'ConnectionError: DNS failed'}]
        with gzip.open(evidence, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
        self.assertIsNone(rl_rollout._cached_formatter(evidence))

    def test_cached_formatter_rate_limits_preserve_success_and_auth_errors(self):
        evidence = Path(self.tmp.name) / 'formatter.jsonl.gz'
        request = {'request': {'messages': []}}
        limited = {'attempt': 1, 'status': 429, 'response': {'error': {'code': 429}}}
        success = {'attempt': 2, 'status': 200,
                   'response': {'choices': [{'message': {'content': '[]'}}]}}
        for tail, expected in [([limited, {**limited, 'attempt': 2}], None),
                               ([limited, success], [])]:
            with gzip.open(evidence, 'wt') as stream:
                stream.write(''.join(json.dumps(row) + '\n' for row in [request] + tail))
            replay = rl_rollout._cached_formatter(evidence)
            self.assertEqual(replay([]) if replay else None, expected)
        with gzip.open(evidence, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in
                                 [request, {**limited, 'status': 401},
                                  {'attempt': 1, 'error': 'HTTPError: 401 Unauthorized'},
                                  {**limited, 'attempt': 2},
                                  {'attempt': 2, 'error': 'HTTPError: 429 Too Many Requests'}]))
        with self.assertRaises(ValueError):
            rl_rollout._cached_formatter(evidence)

    def test_formatter_rate_limit_backoff_is_bounded_and_recoverable(self):
        import requests
        from rl_cost import call_format_model, formatter_context
        for retry_after, expected in [(None, 30), ('300', 120), ('0', 30), ('invalid', 30)]:
            with self.subTest(retry_after=retry_after), tempfile.TemporaryDirectory() as output:
                response = requests.Response()
                response.status_code = 429
                response._content = b'{"error":{"code":429}}'
                if retry_after is not None:
                    response.headers['Retry-After'] = retry_after
                with formatter_context(output, 'offline-rate-limit', self.budget), \
                        patch('rl_cost.requests.post', return_value=response) as post, \
                        patch('rl_cost.time.sleep') as sleep, \
                        patch('rl_cost.random.uniform', return_value=0) as uniform, \
                        patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline'}):
                    with self.assertRaises(requests.HTTPError):
                        call_format_model([], max_retries=2)
                self.assertEqual(post.call_count, 20)
                self.assertEqual(sleep.call_count, 19)
                self.assertEqual(sleep.call_args_list[0].args, (expected,))
                self.assertEqual(sleep.call_args_list[1].args, (120 if retry_after == '300' else 60,))
                uniform.assert_called_with(0, 30)
                self.assertEqual(self.budget.settlements, [])
                evidence = next(Path(output).glob('*-formatter.jsonl.gz'))
                self.assertIsNone(rl_rollout._cached_formatter(evidence))

    def test_rate_limits_do_not_consume_formatter_content_attempts(self):
        import requests
        from rl_cost import call_format_model, formatter_context
        limited = requests.Response()
        limited.status_code = 429
        limited._content = b'{"error":{"code":429}}'
        invalid = requests.Response()
        invalid.status_code = 200
        invalid._content = b'{"choices":[{"message":{"content":"invalid"}}]}'
        valid = requests.Response()
        valid.status_code = 200
        valid._content = b'{"choices":[{"message":{"content":"[]"}}],"usage":{"cost":0.01}}'
        with tempfile.TemporaryDirectory() as output, \
                formatter_context(output, 'offline-recovery', self.budget), \
                patch('rl_cost.requests.post', side_effect=[limited, invalid, limited, valid]) as post, \
                patch('rl_cost.time.sleep'), \
                patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline'}):
            self.assertEqual(call_format_model([]), [])
            self.assertEqual(post.call_count, 4)
            evidence = next(Path(output).glob('*-formatter.jsonl.gz'))
            replay = rl_rollout._cached_formatter(evidence)
            self.assertEqual(replay([]), [])
            self.assertEqual(len(self.budget.reservations), 4)
            self.assertEqual(self.budget.settlements, [(4, 0.01)])

    def test_rate_limited_second_content_attempt_resumes_only_that_attempt(self):
        import requests
        invalid = requests.Response()
        invalid.status_code = 200
        invalid._content = b'{"choices":[{"message":{"content":"invalid"}}]}'
        limited = requests.Response()
        limited.status_code = 429
        limited._content = b'{"error":{"code":429}}'
        valid = requests.Response()
        valid.status_code = 200
        valid._content = b'{"choices":[{"message":{"content":"[]"}}]}'
        tool = '<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>'
        final = '<think>Done.</think><answer>Plan.</answer>'
        with patch('rl_cost.time.sleep'), \
                patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline'}):
            with patch('rl_cost.requests.post', side_effect=[invalid] + [limited] * 20):
                with self.assertRaises(requests.HTTPError):
                    self.collect([self.sample(tool), self.sample(final)])
            original = next(Path(self.tmp.name).glob('*-formatter.jsonl.gz'))
            saved = original.read_bytes()
            self.config['collection_resume_output'] = self.tmp.name
            with tempfile.TemporaryDirectory() as output, \
                    patch.object(rl_rollout, 'sample_response') as sample, \
                    patch('rl_cost.requests.post', return_value=valid) as post:
                recovered = rl_rollout.collect_group(None, self.tokenizer, self.task,
                    self.config, output, 1, 0, budget=self.budget)
                sample.assert_not_called()
                post.assert_called_once()
                self.assertEqual(recovered[0]['termination'], 'ENV_DONE')
                self.assertEqual(len(recovered[0]['turns']), 2)
                self.assertEqual(original.read_bytes(), saved)
                evidence = next(Path(output).glob('*-formatter.jsonl.gz'))
                with gzip.open(evidence, 'rt') as stream:
                    responses = [json.loads(line) for line in stream]
                self.assertEqual([r['attempt'] for r in responses if r.get('status') == 200], [1, 2])
                with patch('rl_cost.requests.post') as post:
                    self.assertEqual(rl_rollout._cached_formatter(evidence)(responses[0]['request']['messages']), [])
                    post.assert_not_called()

    def test_rate_limit_event_does_not_mask_saved_paid_response(self):
        folder = Path(self.tmp.name)
        evidence = folder / 'formatter.jsonl.gz'
        rows = [{'request': {'messages': []}},
                {'attempt': 1, 'status': 429, 'response': {'error': {'code': 429}}},
                {'attempt': 1, 'status': 200,
                 'response': {'choices': [{'message': {'content': '[]'}}], 'usage': {'cost': 0.01}}}]
        with gzip.open(evidence, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
        (folder / 'cost-events.jsonl').write_text(json.dumps({
            'phase': 'reward_formatter', 'identity': 'crash-window', 'attempt': 1,
            'http_status': 429, 'cost_usd': None, 'reservation_id': 'unknown'}) + '\n')
        settlements = rl_rollout._recovery_settlements(folder / 'group.jsonl.gz',
            [{'event': 'sample', 'reservation_id': None}], evidence, 'crash-window')
        self.assertEqual(settlements, [])
        rows[1]['response']['usage'] = {'cost': 0.005}
        with gzip.open(evidence, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
        with (folder / 'cost-events.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase': 'reward_formatter', 'identity': 'crash-window',
                'attempt': 1, 'http_status': 200, 'cost_usd': 0.01, 'reservation_id': 'paid'}) + '\n')
        settlements = rl_rollout._recovery_settlements(folder / 'group.jsonl.gz',
            [{'event': 'sample', 'reservation_id': None}], evidence, 'crash-window')
        self.assertEqual(settlements, [{'reservation_id': 'paid', 'cost': 0.01, 'event': None}])

    def test_max_step_formatter_recovery_restores_original_action_ids(self):
        self.config['max_steps'] = 1
        tool = self.sample('<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>')
        response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {'choices': [{'message': {'content': '[]'}}], 'usage': {'cost': 0}})
        original_reserve, original_settle = self.budget.reserve, self.budget.settle

        def reserve(operation, upper):
            if operation.startswith('formatter:') and operation.endswith(':2'):
                raise sqlite3.OperationalError('unable to open database file')
            return original_reserve(operation, upper)

        def settle(reservation, upper):
            if reservation == 2:
                raise sqlite3.OperationalError('unable to open database file')
            original_settle(reservation, upper)

        with patch.object(self.budget, 'reserve', side_effect=reserve), \
                patch.object(self.budget, 'settle', side_effect=settle), \
                patch('rl_cost.requests.post', return_value=response) as formatter, \
                patch.dict('os.environ', {'OPENROUTER_API_KEY': 'offline'}), patch('rl_cost.time.sleep'):
            with self.assertRaises(sqlite3.OperationalError):
                self.collect([tool])
            self.assertEqual(formatter.call_count, 1)
        evidence = next(Path(self.tmp.name).glob('*-formatter.jsonl.gz'))
        prompt = rl_rollout._read_journal(evidence)[0]['request']['messages'][1]['content']
        original_action = ast.literal_eval(prompt.split('Planning text: ')[-1].strip())
        self.config['collection_resume_output'] = self.tmp.name
        reservations = len(self.budget.reservations)
        with tempfile.TemporaryDirectory() as output, patch.object(rl_rollout, 'sample_response') as sample, \
                patch('rl_cost.requests.post') as formatter:
            result = rl_rollout.collect_group(None, self.tokenizer, self.task,
                self.config, output, 1, 0, budget=self.budget)
            sample.assert_not_called()
            formatter.assert_not_called()
            self.assertEqual(result[0]['termination'], 'ENV_DONE')
            self.assertEqual(result[0]['turns'][0]['response_tokens'], tool[0].tokens)
            events = rl_rollout._read_journal(result[0]['evidence'])
            self.assertEqual(events[-2]['info']['response'], original_action)
            self.assertEqual(len(self.budget.reservations), reservations)
            self.assertEqual([row[0] for row in self.budget.settlements], [1, 2])
        wrong_action = [{**original_action[0], 'function': {'name': 'DifferentTool'}}]
        with self.assertRaisesRegex(ValueError, 'action differs'):
            rl_rollout._restore_formatter_action_ids(wrong_action, self.source, self.task, evidence)

    def test_evaluation_sampling_parameters_are_forwarded(self):
        self.config.update(temperature=0, top_p=.95, collection_phase='validation')
        _, sample = self.collect([self.sample('invalid')])
        self.assertEqual(sample.call_args.args[2], {'temperature': 0, 'top_p': .95})

    def test_failed_sampling_keeps_full_reservation(self):
        with self.assertRaisesRegex(RuntimeError, 'unknown charge'):
            self.collect([RuntimeError('unknown charge')])
        self.assertEqual(len(self.budget.reservations), 1)
        self.assertEqual(self.budget.settlements, [])

    def test_paid_sample_survives_failed_settlement_and_resumes_without_resampling(self):
        tool = self.sample('<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>  ')
        with patch.object(self.budget, 'settle', side_effect=sqlite3.OperationalError('database is locked')):
            with self.assertRaisesRegex(sqlite3.OperationalError, 'locked'):
                self.collect([tool])
        journal = next(Path(self.tmp.name).glob('group-*.gz'))
        rows = rl_rollout._read_journal(journal)
        self.assertEqual([row['event'] for row in rows[-2:]], ['sample', 'sampling_accounting_error'])
        self.assertEqual(rows[-2]['response']['sequences'][0]['tokens'], tool[0].tokens)
        self.assertEqual(rows[-2]['reservation_id'], 1)
        self.assertEqual(self.budget.settlements, [])
        elapsed = rows[-1]['elapsed']
        self.config['collection_resume_output'] = self.tmp.name
        with tempfile.TemporaryDirectory() as output, \
                patch.object(rl_rollout, 'sample_response', return_value=self.sample('invalid')) as sample:
            recovered = rl_rollout.collect_group(None, self.tokenizer, self.task,
                self.config, output, 1, 0, budget=self.budget)
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(sample.call_args.args[-1], 100001)
            self.assertEqual(recovered[0]['turns'][0]['response_tokens'], tool[0].tokens)
            self.assertEqual(recovered[0]['turns'][0]['logprobs'], tool[0].logprobs)
            self.assertEqual(len(recovered[0]['turns']), 2)
            replayed_rows = rl_rollout._read_journal(recovered[0]['evidence'])
            self.assertGreaterEqual(next(row['elapsed'] for row in replayed_rows if row['event'] == 'environment'), elapsed)
            self.assertEqual(len(self.budget.reservations), 2)
            self.assertEqual([row[0] for row in self.budget.settlements], [1, 2])
            cost_events = [json.loads(line) for line in (Path(output) / 'cost-events.jsonl').read_text().splitlines()]
            reconciled = [event for event in cost_events if event.get('reconciled_prior_payment')]
            self.assertEqual(len(reconciled), 1)
            self.assertEqual(reconciled[0]['reservation_id'], 1)
            self.assertEqual(reconciled[0]['sampled_tokens'], len(tool[0].tokens))

    def test_cached_paid_sample_is_kept_when_sampling_already_consumed_timeout(self):
        self.config['trajectory_timeout'] = 1
        tool = self.sample('<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>')
        with patch.object(self.budget, 'settle', side_effect=sqlite3.OperationalError('locked')), \
                patch.object(rl_rollout.time, 'monotonic', side_effect=[0, 2, 3]):
            with self.assertRaises(sqlite3.OperationalError):
                self.collect([tool])
        self.config['collection_resume_output'] = self.tmp.name
        with tempfile.TemporaryDirectory() as output, patch.object(rl_rollout, 'sample_response') as sample:
            recovered = rl_rollout.collect_group(None, self.tokenizer, self.task,
                self.config, output, 1, 0, budget=self.budget)
            sample.assert_not_called()
            self.assertEqual(recovered[0]['termination'], 'TIMEOUT')
            self.assertEqual(recovered[0]['turns'][0]['response_tokens'], tool[0].tokens)

    def test_unrecorded_first_request_keeps_uncertainty_and_uses_new_reservation(self):
        self.collect([self.sample('invalid')])
        journal = next(Path(self.tmp.name).glob('group-*.gz'))
        rows = rl_rollout._read_journal(journal)[:3]
        rows[-1].pop('elapsed')
        with gzip.open(journal, 'wt') as stream:
            stream.write(''.join(json.dumps(row) + '\n' for row in rows))
        with journal.open('rb') as stream:
            created = struct.unpack('<I', stream.read(8)[4:8])[0]
        os.utime(journal, (created + 69, created + 69))
        self.config['collection_resume_output'] = self.tmp.name
        with tempfile.TemporaryDirectory() as output, \
                patch.object(rl_rollout, 'sample_response', return_value=self.sample('invalid')) as sample:
            result = rl_rollout.collect_group(None, self.tokenizer, self.task,
                self.config, output, 1, 0, budget=self.budget)
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(sample.call_args.args[-1], rows[-1]['seed'])
            self.assertEqual(len(self.budget.reservations), 2)
            recovered = rl_rollout._read_journal(result[0]['evidence'])
            note = next(row for row in recovered if row['event'] == 'sampling_recovery')
            self.assertEqual(note['elapsed'], 69)
            self.assertIn('unknown and reserved', note['uncertainty'])
            requests = [row for row in recovered if row['event'] == 'request']
            self.assertEqual([row['reservation_id'] for row in requests], [1, 2])

    def test_network_failure_resumes_exact_prefix_and_excludes_operational_gap(self):
        self.config['trajectory_timeout'] = 10
        tool = self.sample('<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>  ')
        with patch.object(rl_rollout.time, 'monotonic', side_effect=[0, 1, 2, 3, 4, 5, 6, 7206]):
            with self.assertRaisesRegex(RuntimeError, 'No progress'):
                self.collect([tool, tool, RuntimeError('No progress made in 7200s')])
        journal = next(Path(self.tmp.name).glob('group-*.gz'))
        rows = rl_rollout._read_journal(journal)
        self.assertEqual(rows[-1]['operational_gap_seconds'], 7200)
        self.assertEqual(rows[-1]['elapsed'], 4)
        request = rows[-2]
        environments = [row for row in rows if row['event'] == 'environment']
        self.config['collection_resume_output'] = self.tmp.name
        original_update = self.source.agent.update_from_model
        captured = []

        def update(agent, response):
            captured.append((list(agent.trajectory.steps), list(agent.messages), agent.current_observation))
            return original_update(agent, response)

        with tempfile.TemporaryDirectory() as output, \
                patch.object(rl_rollout, 'sample_response', return_value=self.sample('invalid')) as sample, \
                patch.object(self.source.agent, 'update_from_model', autospec=True, side_effect=update), \
                patch.object(self.source.environment, '_execute_tool_calls',
                             side_effect=AssertionError('Recorded tool must not execute')):
            result = rl_rollout.collect_group(None, self.tokenizer, self.task,
                self.config, output, 1, 0, budget=self.budget)
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(sample.call_args.args[-1], request['seed'])
            self.assertEqual(sample.call_args.args[1], result[0]['turns'][2]['prompt_tokens'])
            self.assertEqual(result[0]['termination'], 'ENV_DONE')
            self.assertEqual(len(result[0]['turns']), 3)
            self.assertEqual([turn['response_tokens'] for turn in result[0]['turns'][:2]],
                             [tool[0].tokens, tool[0].tokens])
            self.assertEqual([turn['logprobs'] for turn in result[0]['turns'][:2]],
                             [tool[0].logprobs, tool[0].logprobs])
            steps, messages, observation = captured[0]
            self.assertEqual(len(steps), 2)
            self.assertEqual([step.action for step in steps], [row['info']['response'] for row in environments])
            self.assertEqual(steps[1].observation, environments[0]['observation'])
            self.assertEqual(observation, environments[-1]['observation'])
            self.assertEqual(messages, rows[0]['initial_messages'] +
                             [message for row in environments for message in row['messages']])
            recovery = next(row for row in rl_rollout._read_journal(result[0]['evidence'])
                            if row['event'] == 'sampling_recovery')
            self.assertEqual(recovery['elapsed'], 4)
            self.assertEqual(recovery['previous_reservation_id'], 3)
            self.assertEqual(recovery['operational_gap']['seconds'], 7200)
            self.assertFalse(recovery['operational_gap']['counted_toward_trajectory_timeout'])
            self.assertEqual([entry[0] for entry in self.budget.settlements], [1, 2, 4])

    def test_parallel_paths_preserve_seeds_groups_and_complete_journals(self):
        self.config.update(group_size=8, parallel_rollouts=True, seed=17)
        barrier = threading.Barrier(8, timeout=10)
        tool = self.sample('<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>')
        final = self.sample('invalid')

        def sample(sampler, prompt, parameters, available, seed):
            if (seed - 100017) % 60 == 0:
                barrier.wait()
                return tool
            return final

        for algorithm in ('grpo', 'dapo'):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as output:
                self.config['algorithm'] = algorithm
                with patch.object(rl_rollout, 'sample_response', side_effect=sample) as mock:
                    results = rl_rollout.collect_group(
                        None, self.tokenizer, self.task, self.config, output, 1, 5, budget=self.budget)
                self.assertEqual([row['path_id'] for row in results], list(range(8)))
                self.assertEqual({row['group_id'] for row in results}, {5})
                self.assertEqual(sorted(call.args[-1] for call in mock.call_args_list),
                                 [100017 + path * 60 + turn for path in range(8) for turn in range(2)])
                self.assertEqual(len({row['evidence'] for row in results}), 8)
                for row in results:
                    with gzip.open(row['evidence'], 'rt') as stream:
                        events = [json.loads(line) for line in stream]
                    self.assertEqual(events[0]['group_id'], 5)
                    self.assertEqual(events[-1]['event'], 'complete')
                    self.assertEqual(events[-1]['path_id'], row['path_id'])

    def test_parallel_failure_drains_siblings_and_prioritizes_budget_error(self):
        from rl_budget import BudgetExceeded
        self.config.update(group_size=3, parallel_rollouts=True)
        barrier = threading.Barrier(3, timeout=10)
        final = self.sample('invalid')

        def sample(sampler, prompt, parameters, available, seed):
            barrier.wait()
            if seed == 100000:
                raise RuntimeError('sampling failed')
            if seed == 100060:
                raise BudgetExceeded('cap reached')
            return final

        with patch.object(rl_rollout, 'sample_response', side_effect=sample):
            with self.assertRaisesRegex(BudgetExceeded, 'cap reached'):
                rl_rollout.collect_group(None, self.tokenizer, self.task, self.config,
                                         self.tmp.name, 1, 5, budget=self.budget)
        journals = []
        for path in sorted(Path(self.tmp.name).glob('group-*.gz')):
            with gzip.open(path, 'rt') as stream:
                journals.append([json.loads(line) for line in stream])
        self.assertEqual(len(journals), 3)
        self.assertEqual([events[-1]['event'] for events in journals],
                         ['sampling_error', 'sampling_error', 'complete'])
        self.assertEqual(journals[-1][-1]['path_id'], 2)
        self.assertEqual(journals[-1][-1]['group_id'], 5)
        self.assertEqual(len(self.budget.settlements), 1)

    def test_arpo_forks_and_backfill_reach_exact_group_size(self):
        self.config.update(algorithm='arpo', group_size=3, arpo_initial_rollouts=1, parallel_rollouts=True)
        with patch.object(rl_rollout, 'arpo_branch_score', return_value=1):
            results, _ = self.collect([self.sample('invalid')] * 3)
        self.assertEqual(len(results), 3)
        with gzip.open(next(Path(self.tmp.name).glob('*.gz')), 'rt') as stream:
            events = [json.loads(line) for line in stream]
        self.assertEqual(sum(event['event'] == 'fork' for event in events), 2)
        self.assertEqual(len({row['path_id'] for row in results}), 3)

    def test_arpo_backfills_when_no_fork_is_triggered(self):
        self.config.update(algorithm='arpo', group_size=3, arpo_initial_rollouts=1)
        with patch.object(rl_rollout, 'arpo_branch_score', return_value=0):
            results, _ = self.collect([self.sample('invalid')] * 3)
        self.assertEqual(len(results), 3)
        with gzip.open(next(Path(self.tmp.name).glob('*.gz')), 'rt') as stream:
            events = [json.loads(line) for line in stream]
        self.assertEqual(sum(event['event'] == 'backfill' for event in events), 2)

    def test_length_exhaustion_masks_trajectory(self):
        results, _ = self.collect([self.sample('incomplete', stop='length')])
        self.assertTrue(results[0]['overlong'])
        self.assertEqual(results[0]['termination'], 'TRUNCATION')

    def test_timeout_counts_active_work_and_stops_before_next_request(self):
        tool = '<think>Find cities.</think><tool_call>{"name":"SearchCity","arguments":{"state":"Colorado"}}</tool_call>'
        self.config['trajectory_timeout'] = 10
        with patch.object(rl_rollout.time, 'monotonic', side_effect=[0, 1, 11]):
            results, sample = self.collect([self.sample(tool)])
        self.assertEqual(results[0]['termination'], 'TIMEOUT')
        self.assertTrue(results[0]['overlong'])
        self.assertEqual(sample.call_count, 1)


if __name__ == '__main__':
    unittest.main()
