import ast
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rl_run
from rl_budget import BudgetExceeded
from rl_cost import accounting_summary


class SpindleAccountingTests(unittest.TestCase):
    def test_compute_cost_stays_unknown_without_changing_tinker(self):
        value = {'token_rate_estimate_usd': 2.0, 'combined_usage_estimate_usd': 2.5,
                 'provider_reported_usage_usd': 0.5}
        with patch.dict(os.environ, {'STAR_COMPUTE_BACKEND': 'tinker'}):
            self.assertEqual(accounting_summary(dict(value)), value)
        with patch.dict(os.environ, {'STAR_COMPUTE_BACKEND': 'spindle'}):
            result = accounting_summary(dict(value))
        self.assertIsNone(result['combined_usage_estimate_usd'])
        self.assertIsNone(result['spindle_compute_cost_usd'])
        self.assertEqual(result['counterfactual_tinker_token_estimate_usd'], 2.0)
        self.assertEqual(result['provider_reported_usage_usd'], 0.5)


class EpochGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_priority_preserves_original_groups_paths_and_errors(self):
        batch = [{'id': str(i), 'days': days} for i, days in enumerate([3, 7, 5, 7, 3])]
        sampler = object()
        for fail in [False, True]:
            calls = []
            def group(client, tokenizer, task, config, output, step, identity, budget, *, _path_id):
                self.assertIs(client, sampler)
                calls.append((task['id'], _path_id, identity))
                if fail and task['id'] == '1':
                    raise BudgetExceeded('budget') if _path_id else RuntimeError('worker')
                return [(task['id'], _path_id)]
            with patch('rl_rollout.collect_group', side_effect=group), ThreadPoolExecutor(max_workers=1) as pool:
                result = await rl_run.collect_batch(pool, sampler, None, batch,
                    {'group_size': 2, 'rollout_priority': 'longer_trips_first'}, None, 9, None)
            self.assertEqual(calls, [(str(i), p, f'batch-9-{i}-{i}')
                                     for i in [1, 3, 2, 0, 4] for p in range(2)])
            for i, values in enumerate(result):
                if fail and i == 1:
                    self.assertIsInstance(values, BudgetExceeded)
                else:
                    self.assertEqual(values, [(str(i), p) for p in range(2)])


class AttractionEntryTests(unittest.TestCase):
    def test_audited_splitter_and_exact_three_call_sites(self):
        path = Path('upstream/RL/rllm/rewards/travel_utils/eval_commonsense.py')
        tree = ast.parse(path.read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_attraction_entries')
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), namespace)
        for value, expected in [('', []), ('A, City', ['A, City']), ('A, City;', ['A, City']),
                                ('A, City;  ', ['A, City']), (' A, City ;B, City', [' A, City ', 'B, City']),
                                ('A, City;;B, City;', ['A, City', '', 'B, City']), (';', [''])]:
            with self.subTest(value=value):
                self.assertEqual(namespace['_attraction_entries'](value), expected)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == '_attraction_entries']
        self.assertEqual(len(calls), 3)

    def test_all_three_checks_inspect_the_last_attraction(self):
        import pandas as pd
        path = Path('upstream/RL/rllm/rewards/travel_utils/eval_commonsense.py')
        names = {'_attraction_entries', 'get_valid_name_city', 'check_hallucination',
                 'check_attraction_diversity', 'check_valid_day_arrangement'}
        functions = [node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {'re': re, 'RELAX_BREAKFAST': False,
                     'attraction_tool': SimpleNamespace(data=pd.DataFrame({'Name': ['Lake'], 'City': ['City']}))}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        for suffix in ['', ';', ';  ']:
            self.assertFalse(namespace['check_hallucination']({'days': 1},
                [{'attraction': 'Missing, City' + suffix}])[0])
            self.assertTrue(namespace['check_hallucination']({'days': 1},
                [{'attraction': 'Lake, City' + suffix}])[0])
            self.assertFalse(namespace['check_attraction_diversity']({'days': 2},
                [{'attraction': 'Lake, City' + suffix}] * 2)[0])
            self.assertFalse(namespace['check_valid_day_arrangement']({'days': 1},
                [{'current_city': 'City', 'attraction': 'Lake, Other' + suffix}])[0])


if __name__ == '__main__':
    unittest.main()
