"""Unchanged previously audited official scoring client."""
import hashlib
import json
import time
from urllib.parse import quote, urljoin, urlparse
import uuid

HOST = "https://osunlp-travelplannerleaderboard.hf.space"


def score_file(session, path, output):
    output.mkdir(parents=True, exist_ok=False)
    response = session.get(HOST + '/config', timeout=30)
    response.raise_for_status()
    config = response.json()
    (output / 'service-config.json').write_text(json.dumps(config, indent=2))
    matches = [i for i, dependency in enumerate(config['dependencies'])
               if dependency.get('api_name') == 'add_new_eval']
    if config.get('protocol') != 'sse_v1' or len(matches) != 1:
        raise ValueError('Official service contract changed; inspect before submitting')
    with path.open('rb') as handle:
        response = session.post(HOST + '/upload', files={'files': (path.name, handle, 'application/json')}, timeout=120)
    response.raise_for_status()
    uploaded = response.json()
    if not isinstance(uploaded, list) or len(uploaded) != 1 or not isinstance(uploaded[0], str):
        raise ValueError('Unexpected upload response')
    session_hash = uuid.uuid4().hex
    payload = {'data': ['test', 'two-stage', {'path': uploaded[0], 'orig_name': path.name}],
               'fn_index': matches[0], 'session_hash': session_hash}
    (output / 'submission.json').write_text(json.dumps(payload, indent=2))
    response = session.post(HOST + '/queue/join', json=payload, timeout=60)
    response.raise_for_status()
    joined = response.json()
    (output / 'queue-join.json').write_text(json.dumps(joined, indent=2))
    event_id = joined['event_id']
    completed = None
    deadline = time.monotonic() + 1800
    with session.get(HOST + '/queue/data', params={'session_hash': session_hash}, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with (output / 'queue-events.jsonl').open('w') as events:
            for line in response.iter_lines():
                if time.monotonic() > deadline:
                    raise TimeoutError('Scoring remains unresolved; do not automatically resubmit')
                if not line.startswith(b'data:'):
                    continue
                event = json.loads(line[5:])
                events.write(json.dumps(event) + '\n')
                events.flush()
                if event.get('event_id') == event_id and event.get('msg') == 'process_completed':
                    completed = event
                    break
    if not completed or not completed.get('success'):
        raise RuntimeError('Official scoring failed or incomplete; inspect saved events')
    data = completed['output']['data']
    (output / 'metrics.md').write_text(data[0])
    report = data[1]
    if isinstance(report, dict) and 'value' in report:
        report = report['value']
    url = urljoin(HOST, report.get('url') or '/file=' + quote(report['path'], safe='/'))
    if urlparse(url).scheme != 'https' or urlparse(url).netloc != urlparse(HOST).netloc:
        raise ValueError('Report download points outside the official evaluator')
    response = session.get(url, timeout=120)
    response.raise_for_status()
    response.json()
    (output / 'official-report.json').write_bytes(response.content)
    (output / 'report-sha256.txt').write_text(hashlib.sha256(response.content).hexdigest() + '\n')

