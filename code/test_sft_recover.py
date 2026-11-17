import copy
import unittest

from sft_recover import CONNECTION_ERROR, recovery_cases, merge_recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{'case_id': str(i), 'query': f'query{i}'} for i in range(3)]
        self.original = [dict(case_id=c['case_id'], item=copy.deepcopy(c), query=c['query'],
                              plan=[], success=0) for c in self.cases]
        self.original[1]['generation_error'] = CONNECTION_ERROR
        self.original[2]['generation_error'] = 'ValueError: Generation ended before closing its reasoning block'

    def test_select_only_exact_infrastructure_error_independent_of_score(self):
        self.assertEqual(recovery_cases(self.original, self.cases), [self.cases[1]])
        self.original[1]['success'] = 1
        self.assertEqual(recovery_cases(self.original, self.cases), [self.cases[1]])
        with self.assertRaises(ValueError):
            recovery_cases(self.original[:-1], self.cases)
        self.original[0]['query'] = 'different'
        with self.assertRaises(ValueError):
            recovery_cases(self.original, self.cases)

    def test_merge_preserves_unselected_rows_and_rejects_incomplete_recovery(self):
        selected = recovery_cases(self.original, self.cases)
        recovered = [dict(self.original[1])]
        recovered[0].pop('generation_error')
        recovered[0]['success'] = 1
        merged = merge_recovery(self.original, recovered, selected)
        self.assertIs(merged[0], self.original[0])
        self.assertIs(merged[2], self.original[2])
        self.assertEqual(merged[1]['success'], 1)
        self.assertIn('generation_error', self.original[1])
        for invalid in [[], recovered + recovered, [self.original[1]]]:
            with self.assertRaises(ValueError):
                merge_recovery(self.original, invalid, selected)

    def test_model_failure_is_not_retried_and_operational_failure_stays_unresolved(self):
        selected = [self.cases[1]]
        row = dict(self.original[1], generation_error=self.original[2]['generation_error'])
        self.assertEqual(merge_recovery(self.original, [row], selected)[1]['success'], 0)
        for change in [{'generation_error': 'APITimeoutError: timeout'}, {'scoring_error': 'bad'},
                       {'formatter_error': 'bad'}, {'plan': None}]:
            failed = {**row, **change}
            with self.assertRaises(ValueError):
                merge_recovery(self.original, [failed], selected)


if __name__ == '__main__':
    unittest.main()
