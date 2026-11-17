import json
import unittest
import os
from pathlib import Path
import tempfile
from concurrent.futures import Future
from unittest.mock import Mock, patch, AsyncMock
import tinker

from transformers import AutoTokenizer

from sft_backend import MODEL, TOKENIZER_REVISION, datum, encode_training, render_messages, train
from sft_data import reasoning_text, supervised_turn
from sft_run import validate_local_labels
from sft_evaluate import split_completion, sample_response, evaluate


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION, local_files_only=True)

    def test_masked_turn_is_context_but_not_target(self):
        messages = [{'role': 'system', 'content': 'Rules', 'train': False},
                    {'role': 'assistant', 'content': '<think>\nbad\n</think>\n\n<tool_call>XML</tool_call>', 'train': False},
                    {'role': 'user', 'content': '<tool_response>\nError calling tool: JSON only</tool_response>', 'train': False},
                    {'role': 'assistant', 'content': '<think>\ngood\n</think>\n\n<answer>done</answer>', 'train': True}]
        tokens, weights, spans = encode_training(self.tokenizer, messages, 65536)
        self.assertIn('XML', self.tokenizer.decode(tokens))
        for span in spans:
            self.assertFalse(any(weights[span['start']:span['body_start']]))
            self.assertEqual(set(weights[span['body_start']:span['end']]), {float(span['train'])})
        item = datum(tokens, weights, sum(weights[1:]))
        self.assertEqual(item.loss_fn_inputs['target_tokens'].data, tokens[1:])
        self.assertAlmostEqual(sum(item.loss_fn_inputs['weights'].data), 1, places=6)
        self.assertEqual(tokens, render_messages(self.tokenizer, messages))
        self.assertIn('</answer>', self.tokenizer.decode(tokens[spans[-1]['body_start']:spans[-1]['end']]))
        with self.assertRaisesRegex(ValueError, 'refusing truncation'):
            encode_training(self.tokenizer, messages, len(tokens) - 1)

    def test_sdk_sampling_future_and_evidence(self):
        sampler = Mock()
        future = Future()
        sequence = tinker.SampledSequence(stop_reason='stop', sequence_id='offline', _tokens_list=[1, 2])
        future.set_result(tinker.SampleResponse(sequences=[sequence]))
        sampler.sample.return_value = future
        result, evidence = sample_response(sampler, [3, 4], {'temperature': .6, 'top_p': .95}, 8192)
        self.assertEqual(result.tokens, [1, 2])
        self.assertEqual(evidence['sequences'][0]['sequence_id'], 'offline')
        json.dumps(evidence)

    def test_original_agent_executes_only_visible_action(self):
        root = Path(__file__).resolve().parent
        previous = Path.cwd()
        import sys
        sys.path.insert(0, str(root / 'upstream/Inference'))
        os.chdir(root / 'upstream/Inference')
        try:
            completions = ['I must not execute <tool_call>fake</tool_call>.\n</think>\n\n<tool_call>{"name":"SearchCity","arguments":{"state":"Texas"}}</tool_call>',
                           'Finish.\n</think>\n\n<answer>Example final</answer>']
            sampler = Mock()
            futures = []
            for index, text in enumerate(completions):
                sequence = tinker.SampledSequence(stop_reason='stop', sequence_id=str(index),
                    _tokens_list=self.tokenizer.encode(text, add_special_tokens=False))
                future = Future()
                future.set_result(tinker.SampleResponse(sequences=[sequence]))
                futures.append(future)
            sampler.sample.side_effect = futures
            config = {'seed': 42, 'eval_steps': 3, 'eval_context': 32768, 'model_context': 65536,
                      'max_turn_tokens': 8192, 'temperature': .6, 'top_p': .95, 'eval_concurrency': 1}
            with tempfile.TemporaryDirectory() as directory, patch('utils.format_planning', return_value=[]):
                output = Path(directory) / 'eval'
                summary = evaluate(sampler, self.tokenizer, [{'case_id': 'offline', 'query_index': 0, 'query': 'Find Texas cities.'}], output, config)
                result = json.loads((output / 'predictions.jsonl').read_text())
                self.assertEqual(summary['generation_errors'], 0)
                self.assertEqual(result['termination'], 'answer')
                self.assertNotIn('fake', result['messages'][2]['content'])
                self.assertIn('Austin', result['messages'][3]['content'])
                self.assertEqual(len(result['messages']), 5)
                evidence = [json.loads(x) for x in (output / 'offline-api.jsonl').read_text().splitlines()]
                self.assertEqual(evidence[2]['request']['seed'], 43)
                self.assertIn('fake', evidence[2]['request']['messages'][2]['content'])
        finally:
            os.chdir(previous)

    def test_unlabeled_test_inputs_fail_before_training(self):
        with self.assertRaisesRegex(ValueError, 'official evaluator'):
            validate_local_labels([{'org': 'Sarasota', 'dest': 'Chicago', 'days': 3}])
        validate_local_labels([dict.fromkeys(['org', 'dest', 'days', 'date', 'query', 'level',
                                              'budget', 'people_number', 'local_constraint', 'visiting_city_number'])])

    def test_connection_recovery_stops_before_dispatching_more_cases(self):
        import httpx
        import sys
        root, previous = Path(__file__).resolve().parent, Path.cwd()
        sys.path.insert(0, str(root / 'upstream/Inference'))
        os.chdir(root / 'upstream/Inference')
        try:
            sampler = Mock()
            future = Future()
            future.set_exception(tinker.APIConnectionError(message='No progress made in7200s',
                request=httpx.Request('POST', 'https://example.test/sample')))
            sampler.sample.return_value = future
            config = {'seed': 42, 'eval_steps': 3, 'eval_context': 32768, 'model_context': 65536,
                      'max_turn_tokens': 8192, 'temperature': .6, 'top_p': .95,
                      'eval_concurrency': 1, 'stop_on_connection_error': True}
            cases = [{'case_id': str(i), 'query_index': i, 'query': 'Test.'} for i in range(3)]
            with tempfile.TemporaryDirectory() as directory, patch('utils.format_planning', return_value=[]):
                output = Path(directory) / 'eval'
                with self.assertRaisesRegex(RuntimeError, 'comparison remains unresolved'):
                    evaluate(sampler, self.tokenizer, cases, output, config)
                sampler.sample.assert_called_once()
                self.assertEqual(len((output / 'predictions.jsonl').read_text().splitlines()), 1)
                self.assertEqual(len(json.loads((output / 'unresolved.json').read_text())['requested_cases']), 3)
                self.assertFalse((output / 'summary.json').exists())
        finally:
            os.chdir(previous)

    def test_official_mode_uses_source_test_formatter_and_never_local_grader(self):
        import sys
        root, previous = Path(__file__).resolve().parent, Path.cwd()
        sys.path.insert(0, str(root / 'upstream/Inference'))
        os.chdir(root / 'upstream/Inference')
        try:
            sequence = tinker.SampledSequence(stop_reason='stop', sequence_id='official-offline',
                _tokens_list=self.tokenizer.encode('Done.\n</think>\n\n<answer>Example final</answer>', add_special_tokens=False))
            future = Future()
            future.set_result(tinker.SampleResponse(sequences=[sequence]))
            sampler = Mock()
            sampler.sample.return_value = future
            config = {'seed': 42, 'eval_steps': 3, 'eval_context': 32768, 'model_context': 65536,
                      'max_turn_tokens': 8192, 'temperature': .6, 'top_p': .95,
                      'eval_concurrency': 1, 'official_test': True}
            with tempfile.TemporaryDirectory() as directory, patch('utils.format_planning_for_official_test', return_value=[]) as formatter, patch('eval.eval_one_sample', side_effect=AssertionError('No test labels')), patch('utils.format_planning', side_effect=AssertionError('Wrong formatter')):
                output = Path(directory) / 'test'
                summary = evaluate(sampler, self.tokenizer, [{'case_id': 'test-0', 'query_index': 0, 'query': 'Test.'}], output, config)
                formatter.assert_called_once()
                self.assertIsNone(summary['task_success'])
                row = json.loads((output / 'predictions.jsonl').read_text())
                self.assertNotIn('success', row)
                self.assertEqual(row['scoring_status'], 'pending official evaluator')
        finally:
            os.chdir(previous)

    def test_disk_reserve_stops_before_sampling(self):
        import sys
        root, previous = Path(__file__).resolve().parent, Path.cwd()
        sys.path.insert(0, str(root / 'upstream/Inference'))
        os.chdir(root / 'upstream/Inference')
        try:
            sampler = Mock()
            config = {'eval_concurrency': 2, 'min_free_disk_bytes': 100}
            cases = [{'case_id': str(i), 'query_index': i, 'query': 'Test.'} for i in range(3)]
            with tempfile.TemporaryDirectory() as directory, patch('sft_evaluate.shutil.disk_usage', return_value=Mock(free=99)):
                output = Path(directory) / 'test'
                with self.assertRaisesRegex(RuntimeError, 'comparison remains unresolved'):
                    evaluate(sampler, self.tokenizer, cases, output, config)
                sampler.sample.assert_not_called()
                self.assertEqual(json.loads((output / 'unresolved.json').read_text())['recorded_cases'], [])
        finally:
            os.chdir(previous)

    def test_actual_reasoning_only(self):
        self.assertEqual(reasoning_text({'reasoning': 'real', 'reasoning_details': [
            {'type': 'reasoning.text', 'text': 'real'}]}), 'real')
        for record in [{}, {'reasoning': 'one', 'reasoning_details': [{'type': 'reasoning.text', 'text': 'two'}]}]:
            with self.assertRaises(ValueError):
                reasoning_text(record)

    def test_rejected_and_multiple_calls_not_supervised(self):
        valid = '<tool_call>{"name":"SearchCity","arguments":{"state":"Texas"}}</tool_call>'
        messages = [{'role': 'assistant', 'content': valid},
                    {'role': 'user', 'content': '<tool_response>\nAustin</tool_response>'},
                    {'role': 'assistant', 'content': '<answer>Done</answer>'}]
        self.assertTrue(supervised_turn(messages, 0))
        self.assertTrue(supervised_turn(messages, 2))
        messages[1]['content'] = '<tool_response>\nError calling tool: bad</tool_response>'
        self.assertFalse(supervised_turn(messages, 0))
        messages[0]['content'] = valid + valid
        self.assertFalse(supervised_turn(messages, 0))

    def test_reasoning_tag_mentions_are_not_actions(self):
        reasoning, content = split_completion('I considered <tool_call>fake</tool_call>.\n</think>\n\n<answer>real</answer><|im_end|>')
        self.assertIn('fake', reasoning)
        self.assertEqual(content, '<answer>real</answer>')
        with self.assertRaisesRegex(ValueError, 'before closing'):
            split_completion('I considered <tool_call>fake</tool_call>')
        prefix = self.tokenizer.decode(render_messages(self.tokenizer, [], generation=True))
        self.assertEqual(prefix, '<|im_start|>assistant\n<think>\n')


class TrainerExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_recipe_covers_all185_four_times_and_saves_cloud_paths(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **kwargs: list(text.encode())
        tokenizer.get_vocab.return_value = {'offline': 1}
        trainer = Mock(model_id='offline-model')
        trainer.get_tokenizer.return_value = tokenizer
        def future(value):
            return Mock(result_async=AsyncMock(return_value=value))
        trainer.forward_backward_async = AsyncMock(side_effect=lambda *a, **k: future(Mock(metrics={'loss': 1.0})))
        trainer.optim_step_async = AsyncMock(side_effect=lambda *a, **k: future(Mock(metrics={})))
        trainer.save_state_async = AsyncMock(side_effect=lambda name, **k: future(Mock(path='tinker://offline/state/' + name)))
        trainer.save_weights_for_sampler_async = AsyncMock(side_effect=lambda name, **k: future(Mock(path='tinker://offline/sampler/' + name)))
        service = Mock(create_lora_training_client_async=AsyncMock(return_value=trainer))
        examples = [{'case_id': str(i), 'messages': [{'role': 'assistant', 'content': '<answer>done</answer>', 'train': True}]} for i in range(185)]
        config = json.loads((Path(__file__).parent / 'sft_config.json').read_text())
        config['phase'] = 'full'
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory)
            checkpoint = await train(service, examples, tokenizer, output, config)
            events = [json.loads(x) for x in (output / 'training-metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 24)
            all_ids = [x for e in events for x in e['case_ids']]
            self.assertTrue(all(all_ids.count(str(i)) == 4 for i in range(185)))
            self.assertEqual(checkpoint['sampler_path'], 'tinker://offline/sampler/sft-final')
            self.assertEqual(trainer.save_state_async.await_count, 4)
            for call in trainer.forward_backward_async.await_args_list:
                total_weight = sum(sum(d.loss_fn_inputs['weights'].data) for d in call.args[0])
                self.assertAlmostEqual(total_weight, 1, places=6)
        service.create_lora_training_client_async.assert_awaited_once()
        self.assertTrue(all(isinstance(v, str) for v in service.create_lora_training_client_async.await_args.kwargs['user_metadata'].values()))


if __name__ == '__main__':
    unittest.main()
