import ast
import heapq
import math
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from rl_objective import (
    arpo_branch_score, dapo_keep_indices, group_relative_advantages,
    make_custom_loss, policy_loss, sampled_logprob_entropy,
    SOURCE_BALANCING, source_microbatch_coefficients,
)

try:
    import torch
except ImportError:
    torch = None


def source_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class GroupTests(unittest.TestCase):
    def test_sample_std_epsilon_and_singleton(self):
        values = group_relative_advantages([0, 1, 4, 4, 2], ["a", "a", "b", "b", "c"])
        self.assertAlmostEqual(values[0], -.5 / (math.sqrt(.5) + 1e-6))
        self.assertEqual(values[2:4], [0, 0])
        self.assertAlmostEqual(values[4], 2 / (1 + 1e-6))
        self.assertEqual(group_relative_advantages([2, 4], ["a", "a"], False), [-1, 1])

    def test_dapo_rejects_extremes_but_keeps_equal_intermediate_groups(self):
        self.assertEqual(dapo_keep_indices([1, 1, 0, 1, 0], ["a", "a", "b", "b", "c"]), [0, 1, 2, 3])
        self.assertEqual(dapo_keep_indices([3, 3, 0, -1], ["a", "a", "b", "b"]), [])
        self.assertEqual(dapo_keep_indices([3, 3, 5, 5], ["a", "a", "b", "b"], maximum_reward=5), [0, 1])
        with self.assertRaises(ValueError):
            dapo_keep_indices([float("nan")], ["a"])

    def test_arpo_score_uses_initial_entropy(self):
        self.assertAlmostEqual(arpo_branch_score(3, 2, .3, .5), .8)
        self.assertEqual(sampled_logprob_entropy([]), 0)


@unittest.skipIf(torch is None, "Torch unavailable; Torch/source parity not verified")
class SourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = {"torch": torch, "defaultdict": defaultdict, "heapq": heapq}
        source_functions(SOURCE_BALANCING,
            {"karmarkar_karp", "get_seqlen_balanced_partitions", "ceildiv", "roundup_divisible",
             "rearrange_micro_batches"}, cls.source)
        source_functions(
            "upstream/RL/verl/verl/utils/torch_functional.py",
            {"masked_sum", "masked_mean"}, cls.source,
        )
        cls.source["verl_F"] = SimpleNamespace(masked_mean=cls.source["masked_mean"])
        source_functions(
            "upstream/RL/verl/verl/trainer/ppo/core_algos.py",
            {"compute_grpo_outcome_advantage", "compute_policy_loss", "agg_loss"}, cls.source,
        )
        source_functions(
            "upstream/RL/rllm/engine/agent_execution_engine.py",
            {"_entropy_from_logits"}, cls.source,
        )

    def source_partitions(self, lengths):
        ranks = self.source['get_seqlen_balanced_partitions'](lengths, 8, equal_size=True)
        splits = [min(len(rank), math.ceil(sum(lengths[i] for i in rank) / 30720)) for rank in ranks]
        count = max(splits)
        source_splits = []
        def all_reduce(value, **kwargs):
            source_splits.append(value.item())
            value.fill_(count)
        self.source.update(get_device_name=lambda: 'cpu', dist=SimpleNamespace(
            is_initialized=lambda: True, ReduceOp=SimpleNamespace(MAX='MAX'),
            all_reduce=all_reduce))
        result = []
        class Batch:
            def __init__(self, attention):
                self.attention = attention
            def __getitem__(self, key):
                return self.attention if isinstance(key, str) else self.attention[key]
        for rank in ranks:
            attention = torch.arange(30720)[None, :] < torch.tensor([lengths[i] for i in rank])[:, None]
            _, indices = self.source['rearrange_micro_batches'](Batch(attention), 30720)
            result.extend([[rank[i] for i in partition] for partition in indices])
        self.assertEqual(source_splits, splits)
        return ranks, source_splits, result

    def test_source_partition_max_sync_and_final_target_length(self):
        lengths = [960] * 255 + [961]
        _, diagnostics = source_microbatch_coefficients(lengths, [1] * 256)
        ranks, splits, partitions = self.source_partitions(lengths)
        self.assertEqual(set(splits), {1, 2})
        self.assertEqual(diagnostics['rank_indices'], ranks)
        self.assertEqual(diagnostics['rank_split_counts'], splits)
        self.assertEqual(diagnostics['microbatches_per_rank'], 2)
        self.assertEqual([m['indices'] for m in diagnostics['microbatches']], partitions)
        _, shifted = source_microbatch_coefficients([n - 1 for n in lengths], [1] * 256)
        self.assertEqual(shifted['microbatches_per_rank'], 1)

    def test_source_microbatch_ce_gradient_oracle(self):
        lengths = [9000 + (i % 17) * 1000 for i in range(256)]
        ranks, _, partitions = self.source_partitions(lengths)
        counts = [1 + i % 7 for i in range(256)]
        for i in partitions[0]:
            counts[i] = 0
        self.assertGreater(len(partitions[-1]), 1)
        counts[partitions[-1][0]] = 0
        coefficients, diagnostics = source_microbatch_coefficients(lengths, counts)
        self.assertEqual([m['indices'] for m in diagnostics['microbatches']], partitions)
        self.assertGreater(diagnostics['fully_masked_microbatches'], 0)
        mask = (torch.arange(8)[None, :] < torch.tensor(counts)[:, None]).double()
        advantage = torch.tensor([(i % 3) - 1 for i in range(256)], dtype=torch.float64)[:, None].expand(-1, 8)
        logprob = torch.full((256, 8), -.5, dtype=torch.float64, requires_grad=True)
        source_loss = 0
        for ids in partitions:
            loss, *_ = self.source['compute_policy_loss'](
                logprob[ids].detach(), logprob[ids], advantage[ids], mask[ids],
                cliprange_low=.2, cliprange_high=.28, clip_ratio_c=3.)
            source_loss = source_loss + loss * (len(ids) / len(ranks[0])) / len(ranks)
        weights = mask * advantage * torch.tensor(coefficients, dtype=torch.float64)[:, None]
        actual = torch.autograd.grad(-(logprob * weights).sum(), logprob, retain_graph=True)[0]
        expected = torch.autograd.grad(source_loss, logprob)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-15)
        self.assertTrue(torch.equal(actual[mask == 0], torch.zeros_like(actual[mask == 0])))
        global_weights = mask * advantage / (sum(counts) + 1e-8)
        per_sequence = mask * advantage / (torch.tensor(counts, dtype=torch.float64)[:, None] + 1e-8) / 256
        self.assertFalse(torch.allclose(weights, global_weights))
        self.assertFalse(torch.allclose(weights, per_sequence))
        self.assertTrue(torch.equal(weights[advantage == 0], torch.zeros_like(weights[advantage == 0])))
        for micro in diagnostics['microbatches']:
            tokens = sum(counts[i] for i in micro['indices'])
            expected_coefficient = len(micro['indices']) / (256 * (tokens + 1e-8)) if tokens else 0.
            self.assertEqual(micro['coefficient'], expected_coefficient)

    def test_dapo_retained_batch_source_gradient(self):
        for size in (8, 24, 128, 248):
            with self.subTest(retained=size):
                lengths = [7000 + (i % 15) * 1200 for i in range(size)]
                counts = [i % 7 for i in range(size)]
                coefficients, diagnostics = source_microbatch_coefficients(lengths, counts)
                ranks, _, partitions = self.source_partitions(lengths)
                self.assertEqual([m['indices'] for m in diagnostics['microbatches']], partitions)
                lp = torch.full((size, 8), -.5, dtype=torch.float64, requires_grad=True)
                mask = (torch.arange(8)[None, :] < torch.tensor(counts)[:, None]).double()
                advantage = torch.tensor([(i % 3) - 1 for i in range(size)])[:, None].expand(-1, 8)
                source_loss = 0
                for ids in partitions:
                    loss, *_ = self.source['compute_policy_loss'](
                        lp[ids].detach(), lp[ids], advantage[ids], mask[ids],
                        cliprange_low=.2, cliprange_high=.28, clip_ratio_c=3.)
                    source_loss = source_loss + loss * (len(ids) / 32) / len(ranks)
                actual = -(lp * mask * advantage * torch.tensor(coefficients, dtype=torch.float64)[:, None]).sum()
                torch.testing.assert_close(torch.autograd.grad(actual, lp, retain_graph=True)[0],
                                           torch.autograd.grad(source_loss, lp)[0], rtol=1e-12, atol=1e-15)

    def test_source_weighting_zero_batch_and_fixed_contract(self):
        coefficients, diagnostics = source_microbatch_coefficients([20] * 256, [0] * 256)
        self.assertEqual(coefficients, [0.] * 256)
        self.assertEqual(diagnostics['fully_masked_microbatches'], 8)
        for lengths, counts in [([20] * 255, [1] * 255), ([30721] * 256, [1] * 256),
                                ([20] * 256, [20] * 256)]:
            with self.assertRaises(ValueError):
                source_microbatch_coefficients(lengths, counts)

    def test_advantages_against_released_function(self):
        rewards = [0., 1., 1., 1., 2.]
        groups = ["a", "a", "b", "b", "c"]
        expected, _ = self.source["compute_grpo_outcome_advantage"](
            torch.tensor(rewards)[:, None], torch.ones(5, 1), groups,
        )
        torch.testing.assert_close(torch.tensor(group_relative_advantages(rewards, groups)), expected[:, 0])

    def test_policy_loss_and_gradients_against_released_function(self):
        old = torch.zeros(2, 4)
        new = torch.tensor([[math.log(8), math.log(.5), math.log(1.5), 25.],
                            [math.log(.9), math.log(1.05), -25., 5.]], requires_grad=True)
        advantage = torch.tensor([[-1., -2., 1., -1.], [2., -1., 1., -5.]])
        mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])
        actual, metrics = policy_loss(old, new, advantage, mask, .2, .28, 3.)
        expected, clip, kl, lower = self.source["compute_policy_loss"](
            old, new, advantage, mask, cliprange_low=.2, cliprange_high=.28, clip_ratio_c=3.,
        )
        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(metrics["clip_fraction"], clip.item())
        self.assertAlmostEqual(metrics["ppo_kl"], kl.item())
        self.assertAlmostEqual(metrics["dual_clip_fraction"], lower.item())
        self.assertGreater(lower.item(), 0)
        grad = torch.autograd.grad(actual, new, retain_graph=True)[0]
        torch.testing.assert_close(grad, torch.autograd.grad(expected, new)[0])
        self.assertEqual(grad[0, 0].item(), 0)
        self.assertTrue(torch.equal(grad[mask == 0], torch.zeros(3)))

    def test_extreme_ratios_use_source_clamp(self):
        old = torch.zeros(2)
        new = torch.tensor([-100., 100.], requires_grad=True)
        advantages = torch.tensor([1., -1.])
        mask = torch.ones(2)
        actual, _ = policy_loss(old, new, advantages, mask, .2, .28)
        expected, *_ = self.source["compute_policy_loss"](
            old, new, advantages, mask, cliprange_low=.2, cliprange_high=.28,
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(new.grad, torch.zeros(2))
        empty_loss, _ = policy_loss(old, new, advantages, torch.zeros(2), .2, .28)
        expected_empty, *_ = self.source["compute_policy_loss"](
            old, new, advantages, torch.zeros(2), cliprange_low=.2, cliprange_high=.28,
        )
        torch.testing.assert_close(empty_loss, expected_empty)
        torch.testing.assert_close(torch.autograd.grad(empty_loss, new)[0], torch.zeros(2))

    def test_tinker_callback_ragged_token_mean(self):
        import tinker

        data = [
            tinker.Datum(model_input=tinker.ModelInput.from_ints([1] * n), loss_fn_inputs={
                "target_tokens": [2] * n, "old_logprobs": torch.zeros(n),
                "advantages": torch.full((n,), adv), "response_mask": torch.tensor(mask),
            })
            for n, adv, mask in [(2, 1., [0., 1.]), (3, -1., [1., 1., 0.])]
        ]
        logprobs = [torch.zeros(2, requires_grad=True), torch.zeros(3, requires_grad=True)]
        loss, _ = make_custom_loss(.2, .28)(data, logprobs)
        self.assertAlmostEqual(loss.item(), 1 / 3)
        loss.backward()
        torch.testing.assert_close(logprobs[0].grad, torch.tensor([0., -1 / 3]))
        torch.testing.assert_close(logprobs[1].grad, torch.tensor([1 / 3, 1 / 3, 0.]))

    def test_arpo_sampled_logprob_proxy_matches_source(self):
        values = [-.1, -2., -30.]
        self.assertAlmostEqual(sampled_logprob_entropy(values), self.source["_entropy_from_logits"](torch.tensor(values)), places=6)


if __name__ == "__main__":
    unittest.main()
