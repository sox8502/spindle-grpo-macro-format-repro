"""Objective formulas from upstream/RL/verl and ARPO's agent_execution_engine."""

import ast
import heapq
import math
from functools import cache
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev


SOURCE_BALANCING = 'upstream/RL/verl/verl/utils/seqlen_balancing.py'


@cache
def source_partitioner():
    path = Path(__file__).resolve().parent / SOURCE_BALANCING
    tree = ast.parse(path.read_text())
    names = {'karmarkar_karp', 'get_seqlen_balanced_partitions'}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError('Released sequence partition definitions missing')
    namespace = {'heapq': heapq}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['get_seqlen_balanced_partitions']


def source_microbatch_coefficients(attention_lengths, response_counts, virtual_dp=8, max_token_len=30720):
    size = len(attention_lengths)
    if (not 0 < size <= 256 or len(response_counts) != size
            or virtual_dp not in (8, 16) or size % virtual_dp):
        raise ValueError('Source weighting requires up to256 trajectories divisible by virtual DP8 or DP16')
    if max_token_len != 30720 or any(
            not 2 <= length <= max_token_len or not 0 <= count < length
            for length, count in zip(attention_lengths, response_counts, strict=True)):
        raise ValueError('Invalid source attention lengths or response-mask counts')
    partition = source_partitioner()
    ranks = partition(attention_lengths, virtual_dp, equal_size=True)
    local_size = size // virtual_dp
    split_counts = [min(local_size, math.ceil(sum(attention_lengths[i] for i in rank) / max_token_len))
                    for rank in ranks]
    split_count = max(split_counts)
    coefficients = [0.0] * size
    microbatches = []
    for rank_id, rank in enumerate(ranks):
        lengths = [attention_lengths[i] for i in rank]
        partitions = partition(lengths, split_count, equal_size=False)
        partitions.sort(key=lambda indices: (sum(lengths[i] ** 2 for i in indices), min(indices)), reverse=True)
        for indices in partitions:
            ids = [rank[i] for i in indices]
            tokens = sum(response_counts[i] for i in ids)
            # Retained DAPO batches keep the configured source minibatch denominator.
            coefficient = len(ids) / (256 * (tokens + 1e-8)) if tokens else 0.0
            for i in ids:
                coefficients[i] = coefficient
            microbatches.append({'rank': rank_id, 'indices': ids, 'sequences': len(ids),
                                 'unmasked_tokens': tokens, 'coefficient': coefficient})
    total = sum(response_counts)
    ratios = sorted(coefficients[i] * (total + 1e-8) for i in range(size) if response_counts[i])
    diagnostics = {'mode': 'source_dynamic_microbatch', 'virtual_source_dp': virtual_dp,
        'source_max_token_len': max_token_len, 'attention_lengths': attention_lengths,
        'response_mask_counts': response_counts, 'rank_indices': ranks, 'rank_split_counts': split_counts,
        'microbatches_per_rank': split_count, 'microbatches': microbatches,
        'fully_masked_microbatches': sum(m['unmasked_tokens'] == 0 for m in microbatches),
        'global_unmasked_tokens': total, 'coefficient_ratio_quantiles': {
            str(q): ratios[round(q * (len(ratios) - 1))] if ratios else None for q in (0, .1, .5, .9, 1)}}
    return coefficients, diagnostics


def group_relative_advantages(rewards, group_ids, normalize=True, epsilon=1e-6):
    """GRPO outcome advantage: sample std, epsilon outside std, singleton (0, 1)."""
    groups = defaultdict(list)
    for reward, group in zip(rewards, group_ids, strict=True):
        if not math.isfinite(reward):
            raise ValueError("Rewards must be finite")
        groups[group].append(reward)
    moments = {
        group: (mean(values), stdev(values)) if len(values) > 1 else (0.0, 1.0)
        for group, values in groups.items()
    }
    return [
        (reward - moments[group][0]) / (moments[group][1] + epsilon)
        if normalize else reward - moments[group][0]
        for reward, group in zip(rewards, group_ids, strict=True)
    ]


def dapo_keep_indices(metrics, group_ids, maximum_reward=3.0):
    """Match rllm/trainer/verl/agent_ppo_trainer.py rejection sampling."""
    groups = defaultdict(list)
    for value, group in zip(metrics, group_ids, strict=True):
        if not math.isfinite(value):
            raise ValueError("Filter metrics must be finite")
        groups[group].append(value)
    kept = {
        group for group, values in groups.items()
        if not all(value <= 0 for value in values) and not all(value >= maximum_reward for value in values)
    }
    return [i for i, group in enumerate(group_ids) if group in kept]


def sampled_logprob_entropy(logprobs):
    """Released ARPO 1D pseudo-logit statistic; not full vocabulary entropy."""
    return -sum(min(1.0, max(1e-12, math.exp(lp))) * lp for lp in logprobs)


def arpo_branch_score(entropy, initial_entropy, alpha, beta):
    """Branch when this score > tau and budget > 0; fork BEFORE this action."""
    return alpha + beta * (entropy - initial_entropy)


def policy_loss(old_logprobs, logprobs, advantages, mask, clip_low, clip_high, dual_clip=3.0):
    """Source compute_policy_loss with token-mean aggregation and dual clipping."""
    import torch

    if dual_clip <= 1.0:
        raise ValueError("dual_clip must be greater than one")
    if not (old_logprobs.shape == logprobs.shape == advantages.shape == mask.shape):
        raise ValueError("All policy loss tensors must have the same shape")
    delta = (logprobs - old_logprobs).clamp(-20.0, 20.0)
    ratio = delta.exp()
    raw = -advantages * ratio
    clipped = -advantages * ratio.clamp(1 - clip_low, 1 + clip_high)
    upper = torch.maximum(raw, clipped)
    bound = -advantages * dual_clip
    losses = torch.where(advantages < 0, torch.minimum(bound, upper), upper)

    def masked_mean(values):
        return (torch.where(mask.bool(), values, 0.0) * mask).sum() / (mask.sum() + 1e-8)

    loss = masked_mean(losses)
    return loss, {
        "policy_loss": loss.detach().item(),
        "clip_fraction": masked_mean((clipped > raw).float()).item(),
        "ppo_kl": masked_mean(-delta).detach().item(),
        "dual_clip_fraction": masked_mean(((upper > bound) & (advantages < 0)).float()).item(),
    }


def make_custom_loss(clip_low, clip_high, dual_clip=3.0):
    """Tinker callback; Datum inputs: old_logprobs, advantages, response_mask."""
    import torch

    def custom_loss(data, logprobs):
        fields = {key: [] for key in ("old_logprobs", "advantages", "response_mask")}
        for datum, lp in zip(data, logprobs, strict=True):
            for key in fields:
                value = datum.loss_fn_inputs[key].to_torch().to(device=lp.device, dtype=lp.dtype)
                if value.shape != lp.shape:
                    raise ValueError(f"{key} must match returned token logprobs")
                fields[key].append(value)
        return policy_loss(
            torch.cat(fields["old_logprobs"]), torch.cat(logprobs),
            torch.cat(fields["advantages"]), torch.cat(fields["response_mask"]),
            clip_low, clip_high, dual_clip,
        )

    return custom_loss
