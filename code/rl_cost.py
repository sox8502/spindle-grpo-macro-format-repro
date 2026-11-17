"""Attempt-level formatter evidence and explicitly estimated token charges."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import random
from pathlib import Path
import threading
import time

import requests

from sft_data import append_jsonl
from rl_budget import Budget, BudgetExceeded, formatter_upper

RATES = {'prefill': 0.33, 'cached_prefill': 0.066, 'sample': 1.005, 'train': 0.737}
_context = threading.local()
_lock = threading.Lock()


class FormatterContentError(ValueError):
    pass


def parse_formatter_content(body):
    try:
        if not isinstance(body, dict):
            raise TypeError('Formatter response must be an object')
        if body.get('usage') is not None and not isinstance(body['usage'], dict):
            raise TypeError('Formatter usage must be an object')
        content = body['choices'][0]['message']['content']
        if not content:
            raise ValueError('Empty formatter completion')
        if not isinstance(content, str):
            raise TypeError('Formatter completion must be a string')
        return json.loads(content.replace('```json', '').replace('```', '').strip())
    except Exception as error:
        raise FormatterContentError(str(error)) from error


@contextmanager
def formatter_context(output, identity, budget=None):
    previous = getattr(_context, 'value', None)
    _context.value = (Path(output), identity, budget if budget is not None else Budget())
    try:
        yield
    finally:
        _context.value = previous


def record(output, event):
    event = {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': os.environ.get('ORX_RUN_ID'), **event}
    with _lock:
        append_jsonl(Path(output) / 'cost-events.jsonl', event)


def call_format_model(messages, format_model='deepseek-chat', max_retries=2, profile='rl', first_attempt=1):
    if getattr(_context, 'value', None) is None:
        raise RuntimeError('Formatter requires an attributed evidence context')
    output, identity, budget = _context.value
    if not 1 <= first_attempt <= max_retries <= 20:
        raise ValueError('Formatter attempts must follow the bounded released limits')
    payload = {'model': 'deepseek/deepseek-v3.2-exp', 'messages': messages,
               'max_tokens': 8192}
    if profile == 'rl':
        payload.update(temperature=0.6, top_p=0.95, reasoning={'enabled': False})
    elif profile != 'inference':
        raise ValueError('Unknown formatter profile')
    evidence = output / (identity + '-formatter.jsonl.gz')
    append_jsonl(evidence, {'request': payload})
    attempt, rate_limits = first_attempt, 0
    while attempt <= max_retries:
        reservation = budget.reserve(f'formatter:{identity}:{attempt}', formatter_upper(messages))
        event = {'phase': 'reward_formatter', 'identity': identity, 'attempt': attempt,
                 'reservation_id': reservation, 'reserved_upper_usd': formatter_upper(messages),
                 'cost_usd': None, 'cost_basis': 'unknown_until_response_or_reconciliation'}
        request_started = time.monotonic()
        try:
            response = requests.post('https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'],
                         'Content-Type': 'application/json'}, json=payload, timeout=180)
            event['request_seconds'] = time.monotonic() - request_started
            try:
                body = response.json()
            except ValueError:
                body = {'non_json_body': response.text}
            append_jsonl(evidence, {'attempt': attempt, 'status': response.status_code, 'response': body})
            response_data = body if isinstance(body, dict) else {}
            usage = response_data.get('usage')
            usage = usage if isinstance(usage, dict) else {}
            event.update(request_id=response_data.get('id'), model=response_data.get('model'),
                         provider=response_data.get('provider'), usage=usage, http_status=response.status_code)
            if usage.get('cost') is not None:
                event.update(cost_usd=usage['cost'], cost_basis='provider_reported_usage')
                budget.settle(reservation, usage['cost'])
            response.raise_for_status()
            parsed = parse_formatter_content(body)
            event['status'] = 'parsed'
            record(output, event)
            return parsed
        except Exception as error:
            event.setdefault('request_seconds', time.monotonic() - request_started)
            event.update(status='error', error=f'{type(error).__name__}: {error}')
            record(output, event)
            append_jsonl(evidence, {'attempt': attempt, 'error': event['error']})
            limited = isinstance(error, requests.HTTPError) and error.response is not None and error.response.status_code == 429
            if limited:
                rate_limits += 1
                if rate_limits == 20:
                    raise
                try:
                    delay = float(error.response.headers['Retry-After'])
                except (ValueError, KeyError):
                    delay = 0
                delay = min(120, max(30 * min(rate_limits, 2), delay))
                time.sleep(delay + random.uniform(0, 30))
                continue
            if isinstance(error, BudgetExceeded) or attempt == max_retries:
                raise
            attempt += 1
            time.sleep(0.2)


def token_estimate(prefill=0, cached_prefill=0, sample=0, train=0):
    counts = dict(prefill=prefill, cached_prefill=cached_prefill, sample=sample, train=train)
    if any(value < 0 for value in counts.values()):
        raise ValueError('Token counts cannot be negative')
    return sum(counts[key] * rate for key, rate in RATES.items()) / 1e6


def accounting_summary(result):
    if os.environ.get('STAR_COMPUTE_BACKEND') == 'spindle':
        result['counterfactual_tinker_token_estimate_usd'] = result.pop('token_rate_estimate_usd')
        result['counterfactual_tinker_plus_formatter_usd'] = result['combined_usage_estimate_usd']
        result.update(combined_usage_estimate_usd=None, spindle_compute_cost_usd=None,
                      warning='Reconcile actual Spindle allocation costs with Modal usage; token estimates are counterfactual, not Spindle spend.',
                      accounting_basis='Formatter usage only; Spindle compute billed separately. Ledger model amounts are counterfactual Tinker estimates.')
    return result


def summarize(output):
    events = [json.loads(line) for path in Path(output).rglob('cost-events.jsonl')
              for line in path.read_text().splitlines()]
    result = {'recorded_events': len(events),
              'provider_reported_usage_usd': sum(x.get('cost_usd') or 0 for x in events),
              'token_rate_estimate_usd': sum(x.get('token_rate_estimate_usd') or 0 for x in events),
              'unknown_cost_events': sum(x.get('cost_usd') is None for x in events),
              'invoice_amount_usd': None,
              'warning': 'Usage and rate estimates are not invoices. Reconcile Tinker billing by session.'}
    result['combined_usage_estimate_usd'] = result['provider_reported_usage_usd'] + result['token_rate_estimate_usd']
    result = accounting_summary(result)
    (Path(output) / 'cost-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
