import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from sft_data import append_jsonl
from sft_test import submission_rows, resume_predictions
from sft_evaluate import summarize_results


class OfficialExportTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{'case_id': 'test-' + str(i), 'query_index': i, 'query': f'query{i}'} for i in range(2)]
        self.predictions = [{'case_id': c['case_id'], 'item': copy.deepcopy(c), 'query': c['query'],
                             'plan': [], 'raw_plan': 'answer not found'} for c in reversed(self.cases)]

    def test_benchmark_order_and_only_submission_fields(self):
        rows = submission_rows(self.predictions, self.cases)
        self.assertEqual([r['idx'] for r in rows], [0, 1])
        self.assertEqual([r['query'] for r in rows], ['query0', 'query1'])
        self.assertTrue(all(set(r) == {'idx', 'query', 'plan'} for r in rows))
        self.assertTrue(all(r['plan'] == [] for r in rows))
        summary = summarize_results(self.predictions, self.cases)
        self.assertIsNone(summary['task_success'])
        self.assertIsNone(summary['task_success_rate'])

    def test_missing_duplicate_and_misaligned_predictions_rejected(self):
        for rows in [self.predictions[:1], self.predictions[:1] * 2]:
            with self.assertRaises((ValueError, KeyError)):
                submission_rows(rows, self.cases)
        self.predictions[0]['query'] = 'different'
        with self.assertRaises(ValueError):
            submission_rows(self.predictions, self.cases)

    def test_compressed_resume_preserves_model_failures_and_inherited_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            folder = source / 'sft-test'
            folder.mkdir()
            row = dict(self.predictions[0], generation_error='ValueError: Generation ended before closing its reasoning block')
            append_jsonl(folder / 'predictions.jsonl', row)
            journal = folder / (row['case_id'] + '-api.jsonl.gz')
            append_jsonl(journal, {'request': {'seed': 42}})
            append_jsonl(journal, {'response': {}, 'decoded_completion': 'reasoning'})
            append_jsonl(journal, {'error': row['generation_error']})
            with gzip.open(journal, 'rt') as handle:
                self.assertEqual(len(list(handle)), 3)
            saved, missing, evidence = resume_predictions(source, 'sft', self.cases)
            self.assertEqual(saved, [row])
            self.assertEqual(missing, self.cases[:1])
            external = source / 'external.jsonl.gz'
            journal.rename(external)
            evidence['api_origins'][row['case_id']]['path'] = str(external)
            (source / 'resume-manifest.json').write_text(json.dumps({'variants': {'sft': evidence}}))
            self.assertEqual(resume_predictions(source, 'sft', self.cases)[0], [row])
            append_jsonl(external, {'request': {}})
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                resume_predictions(source, 'sft', self.cases)

    def test_resume_rejects_unfinished_journal_and_operational_error(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            folder = source / 'sft-test'
            folder.mkdir()
            row = self.predictions[0]
            append_jsonl(folder / 'predictions.jsonl', row)
            journal = folder / (row['case_id'] + '-api.jsonl.gz')
            append_jsonl(journal, {'request': {}})
            with self.assertRaisesRegex(ValueError, 'Incomplete API evidence'):
                resume_predictions(source, 'sft', self.cases)
            row['generation_error'] = 'OSError: [Errno 28] No space left on device'
            (folder / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'infrastructure-failed'):
                resume_predictions(source, 'sft', self.cases)


if __name__ == '__main__':
    unittest.main()
