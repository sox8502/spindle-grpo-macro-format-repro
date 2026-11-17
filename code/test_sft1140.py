import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from sft_data import build_dataset, ARTIFACTS
from sft_backend import train
from sft_pipeline import fix_submission_spacing


def offline_service():
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text.encode())
    tokenizer.get_vocab.return_value = {'offline': 1}
    trainer = Mock(model_id='offline-model')
    trainer.get_tokenizer.return_value = tokenizer
    def future(value):
        return Mock(result_async=AsyncMock(return_value=value))
    trainer.forward_backward_async = AsyncMock(side_effect=lambda *a, **k: future(Mock(metrics={'loss': 1})))
    trainer.optim_step_async = AsyncMock(side_effect=lambda *a, **k: future(Mock(metrics={})))
    trainer.save_state_async = AsyncMock(side_effect=lambda name, **k: future(Mock(path='tinker://offline/state/' + name)))
    trainer.save_weights_for_sampler_async = AsyncMock(side_effect=lambda name, **k: future(Mock(path='tinker://offline/sampler/' + name)))
    service = Mock(create_lora_training_client_async=AsyncMock(return_value=trainer),
                   create_training_client_from_state_with_optimizer_async=AsyncMock(return_value=trainer))
    return tokenizer, trainer, service


class SFT1140Tests(unittest.IsolatedAsyncioTestCase):
    async def test_full_and_optimizer_resume_have_identical_order_and_schedule(self):
        examples = [{'case_id': str(i), 'messages': [{'role': 'assistant', 'content': '<answer>done</answer>', 'train': True}]} for i in range(1140)]
        config = json.loads(Path('sft_config.json').read_text())
        config['min_free_disk_bytes'] = 0
        tokenizer, trainer, service = offline_service()
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), patch.dict(os.environ, ORX_RUN_ID='offline-resume'):
            full = Path(directory) / 'full';full.mkdir()
            checkpoint = await train(service, examples, tokenizer, full, config)
            events = [json.loads(line) for line in (full / 'training-metrics.jsonl').read_text().splitlines()]
            self.assertEqual(checkpoint['step'], 144)
            self.assertEqual(len(events), 144)
            ids = [x for row in events for x in row['case_ids']]
            self.assertEqual(len(ids), 4560)
            self.assertTrue(all(ids.count(str(i)) == 4 for i in range(1140)))
            resumed = Path(directory) / 'resumed';resumed.mkdir()
            (resumed / 'checkpoint.json').write_text(json.dumps({'step': 60, 'state_path': 'tinker://offline/state/step-60'}))
            (resumed / 'training-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events[:62]))
            tokenizer, trainer, service = offline_service()
            await train(service, examples, tokenizer, resumed, config)
            recovered = [json.loads(line) for line in (resumed / 'training-metrics.jsonl').read_text().splitlines()]
            self.assertEqual([(r['case_ids'],r['learning_rate']) for r in recovered], [(r['case_ids'],r['learning_rate']) for r in events])
            service.create_training_client_from_state_with_optimizer_async.assert_awaited_once()
            service.create_lora_training_client_async.assert_not_awaited()
            self.assertEqual(trainer.optim_step_async.await_count, 84)
            self.assertEqual(len((resumed / 'metrics-before-resume-offline-resume.jsonl').read_text().splitlines()), 62)

    def test_filtered_export_preserves_exact_source_lines_and_excludes_rl(self):
        original = ARTIFACTS / 'teacher-1200-2026-10-21/collection/accepted.jsonl'
        rl = [json.loads(line) for line in Path('rl_training.jsonl').read_text().splitlines()]
        rl_ids = {row['id'] for row in rl}
        expected = b''.join(line for line in original.read_bytes().splitlines(keepends=True)
                            if json.loads(line)['case_id'] not in rl_ids)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('sft_data.OUTPUT', output):
                examples, manifest = build_dataset(output)
            self.assertEqual((output / 'training.jsonl').read_bytes(), expected)
            self.assertEqual(len(examples), 1140)
            self.assertEqual(len(manifest['excluded_ids']), 60)
            self.assertEqual(manifest['dataset_sha256'], json.loads(Path('sft_config.json').read_text())['training_dataset_sha256'])
            self.assertFalse({row['case_id'] for row in examples} & rl_ids)

    def test_spacing_changes_only_missing_transport_spaces(self):
        rows = [{'idx':0,'query':'Q','plan':[{'day':1,'transportation':'Flight Number:F1, then Flight Number: F2','accommodation':'*Exact* Caf愆'}]}]
        fixed, changes = fix_submission_spacing(rows)
        self.assertEqual(rows[0]['plan'][0]['transportation'],'Flight Number:F1, then Flight Number: F2')
        self.assertEqual(fixed[0]['plan'][0]['transportation'],'Flight Number: F1, then Flight Number: F2')
        self.assertEqual(fixed[0]['plan'][0]['accommodation'],'*Exact* Caf愆')
        self.assertEqual(len(changes),1)


if __name__ == '__main__':
    unittest.main()
