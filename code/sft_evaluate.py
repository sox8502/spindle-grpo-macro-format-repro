"""Tinker serving adapter around the released STAR agent, tools and scorer."""
import copy
import hashlib
import importlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import tinker

from sft_backend import MODEL, render_messages
from sft_data import append_jsonl


def split_completion(text):
    # The generation prefix already contains <think> and its newline.
    if '</think>' not in text:
        raise ValueError('Generation ended before closing its reasoning block')
    reasoning, content = text.split('</think>', 1)
    for token in ('<|im_end|>', '<|im_start|>'):
        content = content.split(token, 1)[0]
    if not content.strip():
        raise ValueError('Generation has no visible action after reasoning')
    return reasoning, content.lstrip('\n')


def sample_response(sampler, tokens, config, max_tokens, seed=None):
    result = sampler.sample(prompt=tinker.ModelInput.from_ints(tokens), num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=max_tokens,
            temperature=config['temperature'], top_p=config['top_p'], seed=seed,
            stop=['<|im_end|>', '<|im_start|>'])).result()
    evidence = {'sequences': [{'tokens': s.tokens, 'logprobs': s.logprobs,
                 'topk_logprobs': s.topk_logprobs, 'stop_reason': s.stop_reason,
                 'sequence_id': s.sequence_id} for s in result.sequences],
                'prompt_cache_hit_tokens': result.prompt_cache_hit_tokens}
    return result.sequences[0], evidence


def evaluate(sampler, tokenizer, cases, output, config):
    from react_agent import TravelReActAgent
    from prompt import TRAVEL_SYSTEM_PROMPT
    from utils import format_planning, format_planning_for_official_test
    from react_agent import TOOL_MAP
    scorer = importlib.import_module('eval')
    output.mkdir(parents=True, exist_ok=True)

    class Agent(TravelReActAgent):
        def __init__(self, evidence, case_index):
            self.system_prompt, self.tools = TRAVEL_SYSTEM_PROMPT, TOOL_MAP
            self.tokenizer, self.model = tokenizer, MODEL
            self.max_turns, self.max_context = config['eval_steps'], config['eval_context']
            self.evidence, self.reasoning, self.partial = evidence, {}, []
            self.case_index = case_index

        def call_server(self, messages, max_retries=10):
            self.partial = copy.deepcopy(messages)
            history = copy.deepcopy(messages)
            for index, reasoning in self.reasoning.items():
                if index < len(history) and history[index]['role'] == 'assistant':
                    history[index]['content'] = '<think>\n' + reasoning + '</think>\n\n' + history[index]['content']
            tokens = render_messages(tokenizer, history, generation=True)
            available = config['model_context'] - len(tokens)
            request = {'messages': history, 'prompt_tokens': len(tokens),
                       'max_tokens': min(config['max_turn_tokens'], available),
                       'seed': config['seed'] + self.case_index * 1000 + len(self.reasoning)}
            if available <= 0:
                append_jsonl(self.evidence, {'request': request, 'error': 'model context exhausted'})
                raise ValueError('Complete evaluation context exceeds model context; refusing truncation')
            append_jsonl(self.evidence, {'request': request})
            try:
                sequence, response = sample_response(sampler, tokens, config, request['max_tokens'], request['seed'])
                text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
                append_jsonl(self.evidence, {'response': response,
                                            'decoded_completion': text})
                reasoning, content = split_completion(text)
                self.reasoning[len(messages)] = reasoning
                self.partial.append({'role': 'assistant', 'content': content})
                return content
            except Exception as error:
                append_jsonl(self.evidence, {'error': f'{type(error).__name__}: {error}'})
                raise

    def one(case):
        case_id = case['case_id']
        suffix = '-api.jsonl.gz' if config.get('compress_evidence') else '-api.jsonl'
        agent = Agent(output / (case_id + suffix), case['query_index'])
        result = {'query': case['query'], 'item': copy.deepcopy(case), 'case_id': case_id}
        try:
            result.update(agent.run(case['query']))
        except Exception as error:
            result.update(plan='answer not found', messages=agent.partial,
                          termination='sampling exception', generation_error=f'{type(error).__name__}: {error}')
        result['raw_plan'] = result.get('plan')
        try:
            formatter = format_planning_for_official_test if config.get('official_test') else format_planning
            result['plan'] = formatter(result['raw_plan'] or 'answer not found', case['query'], 'deepseek-v3.2-exp')
        except Exception as error:
            result.update(plan=[{}] if config.get('official_test') else [], formatter_error=f'{type(error).__name__}: {error}')
        if config.get('official_test'):
            result['scoring_status'] = 'pending official evaluator'
            return result
        try:
            score = scorer.eval_one_sample(copy.deepcopy(case), result['plan'])
            result.update(success=score['success_pass'], scores=score)
        except Exception as error:
            result.update(success=0, scoring_error=f'{type(error).__name__}: {error}')
        return result

    results, stopped = [], False
    todo = iter(cases)
    with ThreadPoolExecutor(max_workers=config['eval_concurrency']) as pool:
        pending = {}
        def submit_next():
            nonlocal stopped
            if shutil.disk_usage(output).free < config.get('min_free_disk_bytes', 0):
                stopped = True
                print('EVALUATION_HALTED disk reserve reached; draining active cases', flush=True)
                return
            case = next(todo, None)
            if case is not None:
                pending[pool.submit(one, case)] = case
        for _ in range(min(config['eval_concurrency'], len(cases))):
            submit_next()
        while pending:
            future = next(as_completed(pending))
            pending.pop(future)
            result = future.result()
            append_jsonl(output / 'predictions.jsonl', result)
            results.append(result)
            print('EVAL_PROGRESS ' + json.dumps({'variant': output.name, 'done': len(results),
                  'total': len(cases), 'success': None if config.get('official_test') else sum(x['success'] for x in results)}), flush=True)
            if config.get('stop_on_connection_error') and result.get('generation_error', '').startswith('APIConnectionError:'):
                stopped = True
                print('RECOVERY_HALTED connection failure; no further cases will be dispatched', flush=True)
            if not stopped:
                submit_next()
    if stopped:
        (output / 'unresolved.json').write_text(json.dumps({
            'requested_cases': [x['case_id'] for x in cases],
            'recorded_cases': [x['case_id'] for x in results],
            'reason': 'Connection failure or disk reserve reached; saved cases retained'}, indent=2))
        raise RuntimeError('Evaluation stopped; comparison remains unresolved')
    assert len(results) == len(cases) and len({r['case_id'] for r in results}) == len(cases)
    summary = summarize_results(results, cases)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('EVAL_SUMMARY ' + json.dumps({'variant': output.name, **summary}), flush=True)
    return summary


def summarize_results(results, cases):
    successes = sum(x['success'] for x in results) if all('success' in x for x in results) else None
    return {'cases': len(results), 'task_success': successes,
               'task_success_rate': successes / len(results) if successes is not None else None,
               'generation_errors': sum('generation_error' in x for x in results),
               'formatter_errors': sum('formatter_error' in x or (x['raw_plan'] not in {None, 'answer not found'} and not isinstance(x['plan'], list)) for x in results),
               'empty_plans': sum(not x['plan'] for x in results),
               'answers_found': sum(x.get('termination') == 'answer' for x in results),
               'scoring_errors': sum('scoring_error' in x for x in results),
               'query_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}
