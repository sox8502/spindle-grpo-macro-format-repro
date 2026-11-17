import asyncio
import json
import math
import random
import os
import shutil

import tinker

from sft_data import append_jsonl


MODEL = 'Qwen/Qwen3.5-4B'
TOKENIZER_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def render_messages(tokenizer, messages, generation=False):
    text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if generation:
        text += '<|im_start|>assistant\n<think>\n'
    return tokenizer.encode(text, add_special_tokens=False)


def encode_training(tokenizer, messages, cutoff):
    tokens, weights, spans = [], [], []
    for message in messages:
        header = tokenizer.encode(f"<|im_start|>{message['role']}\n", add_special_tokens=False)
        body = tokenizer.encode(message['content'] + '<|im_end|>\n', add_special_tokens=False)
        start = len(tokens)
        train = message['role'] == 'assistant' and message['train']
        tokens.extend(header + body)
        weights.extend([0.0] * len(header) + [float(train)] * len(body))
        spans.append({'role': message['role'], 'start': start, 'body_start': start + len(header),
                      'end': len(tokens), 'train': train})
    if len(tokens) > cutoff:
        raise ValueError(f'Complete conversation needs {len(tokens)} tokens, limit {cutoff}; refusing truncation')
    assert len(tokens) == len(weights) and sum(weights[1:]) > 0
    assert spans[-1]['train'] and all(weights[spans[-1]['body_start']:])
    return tokens, weights, spans


def datum(tokens, weights, normalization):
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs={
        'target_tokens': tinker.TensorData(data=tokens[1:], dtype='int64', shape=[len(tokens) - 1]),
        'weights': tinker.TensorData(data=[w / normalization for w in weights[1:]],
                                    dtype='float32', shape=[len(tokens) - 1])})


def learning_rate(step, total, base, warmup_ratio):
    warmup = max(1, math.ceil(total * warmup_ratio))
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


async def save_checkpoint(trainer, output, name, step, final=False):
    state = await (await trainer.save_state_async(name, ttl_seconds=None)).result_async()
    record = {'step': step, 'state_path': state.path}
    if final:
        sampler = await (await trainer.save_weights_for_sampler_async(name, ttl_seconds=None)).result_async()
        record['sampler_path'] = sampler.path
    append_jsonl(output / 'checkpoints.jsonl', record)
    temporary = output / 'checkpoint.tmp'
    with temporary.open('w') as handle:
        handle.write(json.dumps(record, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / 'checkpoint.json')
    print('CHECKPOINT ' + json.dumps(record), flush=True)
    return record


async def train(service, examples, tokenizer, output, config):
    encoded = [encode_training(tokenizer, row['messages'], config['cutoff']) for row in examples]
    summary = [{'case_id': row['case_id'], 'tokens': len(data[0]),
                'supervised_tokens': sum(data[1][1:]), 'spans': data[2]}
               for row, data in zip(examples, encoded)]
    (output / 'tokenization.json').write_text(json.dumps(summary))
    checkpoint_path = output / 'checkpoint.json'
    resume = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    start_step = resume['step'] if resume else 0
    metadata = {'phase': config['phase'], 'examples': str(len(examples))}
    if resume:
        trainer = await service.create_training_client_from_state_with_optimizer_async(
            resume['state_path'], base_model=MODEL, user_metadata=metadata)
        metrics = output / 'training-metrics.jsonl'
        if metrics.exists():
            shutil.copyfile(metrics, output / ('metrics-before-resume-' + os.environ['ORX_RUN_ID'] + '.jsonl'))
            rows = [json.loads(line) for line in metrics.read_text().splitlines()]
            metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['step'] <= start_step))
        print('TRAINING_RESUMED ' + json.dumps(resume), flush=True)
    else:
        trainer = await service.create_lora_training_client_async(
            # Miles multi-LoRA rejects per-model seeds; config seed still fixes the shuffle.
            base_model=MODEL, rank=config['lora_rank'], seed=None if config.get('spindle_base_url') else config['seed'],
            train_mlp=True, train_attn=True, train_unembed=True,
            user_metadata=metadata)
    server_tokenizer = await asyncio.to_thread(trainer.get_tokenizer)
    if server_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError('Pinned tokenizer vocabulary differs from Tinker training tokenizer')
    print('TRAINING_MODEL ' + trainer.model_id, flush=True)
    batch_size = min(config['batch_size'], len(encoded))
    total = config['smoke_steps'] if config['phase'] == 'smoke' else math.ceil(len(encoded) / batch_size) * config['epochs']
    order, rng, step = list(range(len(encoded))), random.Random(config['seed']), 0
    for epoch in range(config['epochs']):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            if step < start_step:
                step += 1
                continue
            if shutil.disk_usage(output).free < config.get('min_free_disk_bytes', 0):
                if step:
                    await save_checkpoint(trainer, output, f'disk-stop-{step}', step)
                raise RuntimeError('Training stopped at disk reserve; checkpoint retained')
            indices = order[start:start + batch_size]
            normalizer = sum(sum(encoded[i][1][1:]) for i in indices)
            batch = [datum(encoded[i][0], encoded[i][1], normalizer) for i in indices]
            rate = learning_rate(step, total, config['learning_rate'], config['warmup_ratio'])
            result = await (await trainer.forward_backward_async(batch, loss_fn='cross_entropy')).result_async()
            optim = await (await trainer.optim_step_async(tinker.AdamParams(
                learning_rate=rate, beta1=0.9, beta2=0.999, eps=1e-8,
                weight_decay=0.0, grad_clip_norm=1.0))).result_async()
            step += 1
            event = {'step': step, 'total_steps': total, 'epoch': epoch + 1,
                     'case_ids': [examples[i]['case_id'] for i in indices],
                     'supervised_tokens': normalizer, 'learning_rate': rate,
                     'metrics': result.metrics, 'optimizer_metrics': optim.metrics}
            append_jsonl(output / 'training-metrics.jsonl', event)
            print('TRAIN_STEP ' + json.dumps(event), flush=True)
            if step == total:
                return await save_checkpoint(trainer, output, 'smoke-final' if config['phase'] == 'smoke' else 'sft-final', step, final=True)
            if step % config['save_every'] == 0:
                await save_checkpoint(trainer, output, f'step-{step}', step)
    raise RuntimeError('Training finished without final checkpoint')
