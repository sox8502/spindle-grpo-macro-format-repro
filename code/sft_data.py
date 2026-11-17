"""Lossless, audited training views; collection artifacts remain immutable."""
import gzip
import hashlib
import json
from pathlib import Path

ARTIFACTS = Path.home() / '.local/share/openresearch/files/modal-blog/agent-star'
SOURCE_RUN = 'ce2561c0-18fa-4c31-9735-f9581a8914ab'
COLLECTION = ARTIFACTS / 'original-model-source/runs' / SOURCE_RUN
AUDIT = ARTIFACTS / 'original-model-source/collection-2000-2026-10-18/accepted-target-audit.json'
OUTPUT = ARTIFACTS / 'sft-1140-spindle-65k-4h200-2026-11-06'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_jsonl(path, row):
    with (gzip.open(path, 'at', encoding='utf-8') if Path(path).suffix == '.gz' else Path(path).open('a')) as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()


def reasoning_text(message):
    reasoning = message.get('reasoning') or message.get('reasoning_content')
    details = ''.join(x.get('text', '') for x in message.get('reasoning_details') or []
                      if x.get('type') == 'reasoning.text')
    if reasoning and details and reasoning != details:
        raise ValueError('Reasoning fields disagree')
    result = reasoning or details
    if not result or '</think>' in result:
        raise ValueError('Missing or ambiguous authentic reasoning')
    return result


def supervised_turn(messages, index):
    content = messages[index]['content']
    if index == len(messages) - 1:
        return (content.count('<answer>') == 1 and content.count('</answer>') == 1
                and '<tool_call>' not in content)
    if content.count('<tool_call>') != 1 or content.count('</tool_call>') != 1:
        return False
    try:
        action = json.loads(content.split('<tool_call>', 1)[1].split('</tool_call>', 1)[0])
    except (ValueError, TypeError):
        return False
    observation = messages[index + 1]
    return (isinstance(action, dict) and set(action) == {'name', 'arguments'}
            and action['name'] in {'SearchFlight', 'SearchAccommodation', 'SearchRestaurant',
                                   'SearchAttraction', 'SearchCity', 'GoogleDistanceMatrix'}
            and isinstance(action['arguments'], dict) and observation['role'] == 'user'
            and observation['content'].startswith('<tool_response>\n')
            and not observation['content'].startswith('<tool_response>\nError calling tool:')
            and 'not found, please rethink your tool call' not in observation['content'])


def build_dataset(output):
    source = ARTIFACTS / 'teacher-1200-2026-10-21/collection/accepted.jsonl'
    expected = 'ef4c114e95614b4d0d483c95cfc4925c04a31c175023bed37faec1c876ae1a6e'
    assert file_hash(source) == expected, 'Audited 1200-example dataset changed'
    rl_file = Path(__file__).resolve().parent / 'rl_training.jsonl'
    rl_hash = '721616fca5835a20c28feb650ba26c012b7a26cd069e460e10aae6fca37d12f2'
    assert file_hash(rl_file) == rl_hash, 'Earlier RL1K dataset changed'
    rl_rows = [json.loads(line) for line in rl_file.read_bytes().splitlines()]
    rl_ids = {row['id'] for row in rl_rows}
    assert len(rl_rows) == len(rl_ids) == 1000
    source_lines = source.read_bytes().splitlines(keepends=True)
    excluded = [json.loads(line)['case_id'] for line in source_lines if json.loads(line)['case_id'] in rl_ids]
    assert len(excluded) == 60
    data = b''.join(line for line in source_lines if json.loads(line)['case_id'] not in rl_ids)
    examples = [json.loads(line) for line in data.splitlines()]
    assert len(examples) == len({r['case_id'] for r in examples}) == 1140
    assert not {row['messages'][1]['content'].strip() for row in examples} & {row['query'].strip() for row in rl_rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'training.jsonl').write_bytes(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'training.jsonl').write_bytes(data)
    manifest = {'source_run': 'da2366da-5c58-4efc-bb40-deacfd41bb92',
                'dataset_sha256': hashlib.sha256(data).hexdigest(), 'examples': 1140,
                'source_dataset_sha256': expected, 'excluded_ids': excluded,
                'rl_training_sha256': rl_hash,
                'case_ids': [r['case_id'] for r in examples],
                'supervised_assistant_turns': sum(m['train'] for r in examples for m in r['messages']),
                'masked_assistant_turns': sum(m['role'] == 'assistant' and not m['train'] for r in examples for m in r['messages']),
                'method': 'Audited1200 export minus60 prior RL1K overlaps; retained lines byte-identical, original185 SFT encoding and rejected-turn masking unchanged.'}
    (output / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2))
    return examples, manifest
