"""Tinker sampling around the released travel agent, environment and reward."""

import ast
import copy
import gzip
import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol, runtime_checkable

from rl_objective import arpo_branch_score, sampled_logprob_entropy
from sft_evaluate import sample_response

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "upstream/RL"
_source = None
_source_lock = threading.Lock()
_formatter_failure = threading.local()


def initialize_source():
    """Load untouched travel modules; skip registries importing unrelated backends."""
    global _source
    with _source_lock:
        if _source is not None:
            return _source
        for name in ("rllm", "rllm.agents", "rllm.tools", "rllm.environments",
                     "rllm.environments.tools", "rllm.environments.base", "rllm.rewards"):
            if name not in sys.modules:
                package = ModuleType(name)
                package.__path__ = [str(SOURCE / name.replace(".", "/"))]
                sys.modules[name] = package
        tools_package = sys.modules["rllm.tools"]
        registry = importlib.import_module("rllm.tools.registry").ToolRegistry()
        tools_package.tool_registry = registry
        paths = {
            "FLIGHT_PATH": "clean_Flights_2022.csv", "ATTRACTION_PATH": "attractions.csv",
            "ACCOMMODATION_PATH": "clean_accommodations_2022.csv",
            "RESTAURANT_PATH": "clean_restaurant_2022.csv", "DISTNACE_PATH": "distance.csv",
            "CITY_PATH": "background/citySet_with_states.txt",
        }
        for variable, filename in paths.items():
            os.environ[variable] = str(ROOT / "upstream/database" / filename)
        travel_tools = importlib.import_module("rllm.tools.travel_tools")
        tool_map = {name: getattr(travel_tools, name) for name in travel_tools.__all__}
        registry.register_all(tool_map)
        reward_types = importlib.import_module("rllm.rewards.reward_types")
        # Load only these unchanged definitions; reward_fn eagerly imports unrelated code execution providers.
        reward_path = SOURCE / "rllm/rewards/reward_fn.py"
        tree = ast.parse(reward_path.read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                       and node.name in {"RewardFunction", "zero_reward"}]
        reward_interface = ModuleType("rllm.rewards.reward_fn")
        reward_interface.__dict__.update(Protocol=Protocol, runtime_checkable=runtime_checkable,
                                         RewardOutput=reward_types.RewardOutput)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(reward_path), "exec"),
             reward_interface.__dict__)
        sys.modules[reward_interface.__name__] = reward_interface
        agent_module = importlib.import_module("rllm.agents.travel_agent")
        agent = agent_module.TravelAgentFormat
        environment_module = importlib.import_module("rllm.environments.tools.travel_tool_env")
        environment_module.TOOL_CALL_ERROR = False
        environment_module.TOOL_CALL_ERROR_PERCENTAGE = 0.0
        environment_module.REWARD_TYPE = ""
        parser = importlib.import_module("rllm.parser").QwenChatTemplateParser
        class TravelParser(parser):
            def _get_generation_prompt(self, tokenizer):
                # Qwen3.5 rejects the base initializer's assistant-only dummy conversation.
                return "<|im_start|>assistant\n"

        converter = importlib.import_module("rllm.agents.utils").convert_messages_to_tokens_and_masks
        previous = Path.cwd()
        try:
            # Released scorer opens its background city file relative to upstream/RL.
            os.chdir(SOURCE)
            reward = importlib.import_module("rllm.rewards.travel_reward")
        finally:
            os.chdir(previous)
        def checked_formatter(*args, **kwargs):
            from rl_cost import call_format_model
            try:
                replay = getattr(_formatter_failure, "replay", None)
                return (replay or call_format_model)(*args, **kwargs)
            except Exception as error:
                _formatter_failure.error = error
                raise

        reward.call_format_model = checked_formatter
        # Complete model/tool output is saved once in evidence, rather than repeated by source debug prints.
        for module in (agent_module, environment_module, reward):
            module.print = lambda *args, **kwargs: None
        _source = SimpleNamespace(agent=agent, environment=environment_module.TravelToolEnvironment,
                                  parser=TravelParser, converter=converter, reward=reward,
                                  step=agent_module.Step, tool_map=tool_map)
        return _source


def _reward_function(source, config, step):
    reward_type = config.get("reward_type", "curriculum").lower()
    if reward_type in {"curriculum", "planner_r1"}:
        return partial(source.reward.planner_r1_reward_fn, current_step=step,
                       total_steps=config["total_steps"])
    return {
        "macro": source.reward.macro_travel_reward_fn,
        "success": source.reward.success_travel_reward_fn,
        "sum": source.reward.sum_travel_reward_fn,
    }[reward_type]


def _clean_response(tokenizer, parser, tokens):
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    previous = None
    while text != previous:
        previous = text
        for ending in (tokenizer.eos_token, parser.eot_token):
            if ending and text.endswith(ending):
                text = text[:-len(ending)]
    return text


def _expected_formatter_error(error):
    from rl_cost import FormatterContentError
    return isinstance(error, (FormatterContentError, json.JSONDecodeError)) or (
        type(error) is ValueError and str(error) == "Empty formatter completion")


def _extend_prompt(tokenizer, parser, source, prompt, tokens, messages):
    env_tokens, _ = source.converter(
        messages[1:], tokenizer, parser, contains_first_msg=False, contains_generation_msg=True)
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    ending = "" if decoded.endswith(parser.eot_token) else (
        "\n" if decoded.endswith(parser.eos_token) else parser.eot_token)
    separator = tokenizer.encode(ending, add_special_tokens=False)
    return prompt + tokens + separator + env_tokens, len(tokens) + len(separator) + len(env_tokens), len(env_tokens)


def _read_journal(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _cached_formatter(path):
    from rl_cost import FormatterContentError, call_format_model, parse_formatter_content
    rows = _read_journal(path)
    request = rows[0]["request"]
    responses = [row for row in rows if "response" in row]
    rate_limited = {row["attempt"] for row in responses if row["status"] == 429}
    attempts = [row for row in responses if row["status"] != 429]
    transport_errors = [row.get("error", "") for row in rows if "error" in row]
    transport_prefixes = ("ConnectionError:", "Timeout:", "ReadTimeout:",
                          "ConnectTimeout:", "ChunkedEncodingError:")
    if not attempts and transport_errors and all(error.startswith(transport_prefixes) for error in transport_errors):
        return None
    if not attempts and rate_limited and all(error.startswith(
            transport_prefixes + ("HTTPError: 429 ",)) for error in transport_errors):
        return None
    numbers = [row["attempt"] for row in attempts]
    transport_attempts = rate_limited | {row["attempt"] for row in rows
                          if row.get("error", "").startswith(transport_prefixes)}
    if (not 1 <= len(attempts) <= 2 or numbers != sorted(set(numbers))
            or any(number not in (1, 2) for number in numbers)
            or set(range(1, max(numbers) + 1)) - set(numbers) - transport_attempts):
        raise ValueError("Cached formatter has no contiguous released attempts")
    errors = []
    parsed = None
    for index, row in enumerate(attempts):
        if row["status"] != 200:
            raise ValueError("Cannot replay a formatter transport failure as zero reward")
        try:
            parsed = parse_formatter_content(row["response"])
        except FormatterContentError as error:
            errors.append(error)
        else:
            if index != len(attempts) - 1:
                raise ValueError("Cached formatter continued after a successful attempt")
            break
    unfinished = numbers == [1] and len(errors) == 1
    if len(errors) == len(attempts) and len(attempts) != 2 and not unfinished:
        raise ValueError("Cached formatter did not exhaust the released two attempts")

    def replay(messages, format_model="deepseek-chat", max_retries=2):
        if messages != request["messages"] or max_retries != 2:
            raise ValueError("Cached formatter request differs from recovered trajectory")
        if unfinished:
            return call_format_model(messages, format_model, first_attempt=2)
        if len(errors) == len(attempts):
            raise errors[-1]
        return parsed

    return replay


def _trajectory_result(path, task, group_id, reward, metadata, termination, evidence):
    return {"turns": path["turns"], "reward": float(reward),
            "judgement": metadata.get("judgement", {}),
            "formatted_plan": metadata.get("formatted_plan", []),
            "reward_error": metadata.get("error"),
            "overlong": termination in {"TRUNCATION", "MAX_STEPS", "TIMEOUT"},
            "termination": termination, "task_id": task.get("id"),
            "group_id": group_id, "path_id": path["path_id"],
            "response_token_len": path["response_len"], "evidence": str(evidence)}


def _restore_formatter_action_ids(action, source, task, evidence):
    messages = _read_journal(evidence)[0]["request"]["messages"]
    prefix, suffix = source.reward.FORMAT_PLANNING.replace(
        "{{{user_query}}}", task["query"]).split("{{{planning_text}}}")
    content = messages[1]["content"]
    if not content.startswith(prefix) or not content.endswith(suffix):
        raise ValueError("Cached formatter prompt differs from released template or task")
    recorded = ast.literal_eval(content[len(prefix):len(content) - len(suffix)])
    if len(recorded) != len(action):
        raise ValueError("Cached formatter action count differs from sampled action")
    for current, saved in zip(action, recorded, strict=True):
        if {**current, "id": saved["id"]} != saved:
            raise ValueError("Cached formatter action differs from sampled action")
        current["id"] = saved["id"]


def _recovery_settlements(journal, rows, formatter_path, identity):
    from rl_cost import token_estimate
    sample = next(row for row in reversed(rows) if row["event"] == "sample")
    reservation = sample.get("reservation_id")
    sampling_recorded, formatter_events = False, {}
    cost_path = journal.parent / "cost-events.jsonl"
    if cost_path.exists():
        with cost_path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if reservation is not None and event.get("reservation_id") == reservation:
                    sampling_recorded = True
                if (event.get("phase") == "reward_formatter" and event.get("identity") == identity
                        and event.get("http_status") != 429):
                    formatter_events[event["attempt"]] = event
    settlements = []
    if reservation is not None and not sampling_recorded:
        request = next(row for row in reversed(rows) if row["event"] == "request")
        if request.get("reservation_id") != reservation:
            raise ValueError("Cached sample reservation differs from its request")
        prompt_length = request["prompt_length"]
        cached = sample["response"].get("prompt_cache_hit_tokens") or 0
        if not 0 <= cached <= prompt_length:
            raise ValueError("Invalid cached sampling cache token count")
        sampled = len(sample["response"]["sequences"][0]["tokens"])
        cost = token_estimate(prefill=prompt_length - cached, cached_prefill=cached, sample=sampled)
        settlements.append({"reservation_id": reservation, "cost": cost, "event": {
            "phase": rows[0]["config"].get("collection_phase", "rollout"), "identity": identity,
            "reservation_id": reservation, "prompt_tokens": prompt_length, "sampled_tokens": sampled,
            "cached_tokens": cached, "token_rate_estimate_usd": cost, "cost_usd": None,
            "reconciled_prior_payment": True, "source_evidence": str(journal)}})
    if formatter_path is not None:
        for row in _read_journal(formatter_path):
            if "response" not in row or row["status"] == 429:
                continue
            event = formatter_events.get(row["attempt"], {})
            body = row["response"]
            usage = body.get("usage") if isinstance(body, dict) else None
            cost = usage.get("cost") if isinstance(usage, dict) else None
            if cost is not None and event.get("reservation_id"):
                if event.get("cost_usd") != cost:
                    raise ValueError("Cached formatter usage differs from its recorded cost")
                settlements.append({"reservation_id": event["reservation_id"], "cost": cost, "event": None})
    return settlements


def _resume_path(source, tokenizer, parser, task, config, step, group_id, path_id,
                 identity, initial_messages, initial_prompt):
    root = config.get("collection_resume_output")
    roots = ([root] if root else []) + config.get("collection_ancestors", [])
    journal = next((Path(folder) / f"group-{identity}.jsonl.gz" for folder in roots
                    if (Path(folder) / f"group-{identity}.jsonl.gz").exists()), None)
    if journal is None or not journal.exists():
        return None
    if config["algorithm"].lower() == "arpo" or path_id is None and config.get("group_size", 8) != 1:
        raise ValueError("Journal replay requires an independent rollout path")
    rows = _read_journal(journal)
    header = rows[0]
    operational = {"resume_checkpoint", "collection_resume_output", "validation_resume_output",
                   "group_concurrency", "eval_concurrency", "min_free_disk_bytes",
                   "execution_order", "parallel_rollouts", "collection_ancestors"}
    for key in (config.keys() | header["config"].keys()) - operational:
        if config.get(key) != header["config"].get(key):
            raise ValueError(f"Cached rollout changes scientific control: {key}")
    if any(header[key] != value for key, value in (
        ("task", task), ("step", step), ("group_id", group_id),
        ("initial_messages", initial_messages), ("initial_prompt_tokens", initial_prompt))):
        raise ValueError("Cached rollout identity or initial prompt differs")
    path_id = 0 if path_id is None else path_id
    path = {"path_id": path_id, "messages": copy.deepcopy(initial_messages), "steps": [],
            "turns": [], "response_len": 0, "initial_entropy": None, "count": 0,
            "prompt_tokens": list(initial_prompt), "elapsed": 0.0, "observation": copy.deepcopy(task)}
    pending = request = sampling_error = None
    for row in rows[1:]:
        if row.get("path_id") != path_id:
            raise ValueError("Cached rollout contains another path")
        if "elapsed" in row:
            path["elapsed"] = row["elapsed"]
        if row["event"] == "request":
            prompt = path["prompt_tokens"]
            if (pending is not None or row["turn"] != path["count"] or
                    row["prompt_length"] != len(prompt) or
                    row["prompt_sha256"] != hashlib.sha256(json.dumps(prompt).encode()).hexdigest()):
                raise ValueError("Cached rollout prompt or turn does not match sampled history")
            request = row
            sampling_error = None
        elif row["event"] == "sample":
            if request is None or row["turn"] != path["count"] or pending is not None:
                raise ValueError("Cached sample has no matching request")
            sequence = row["response"]["sequences"]
            if len(sequence) != 1:
                raise ValueError("Cached rollout must contain exactly one sample")
            pending = sequence[0]
            tokens, logprobs = pending["tokens"], pending["logprobs"]
            if not tokens or logprobs is None or len(tokens) != len(logprobs) or not all(
                    lp is not None and math.isfinite(lp) for lp in logprobs):
                raise ValueError("Cached sampling tokens and logprobs are invalid")
            if path["initial_entropy"] is None:
                path["initial_entropy"] = sampled_logprob_entropy(logprobs)
        elif row["event"] == "environment":
            if pending is None or row["turn"] != path["count"]:
                raise ValueError("Cached environment has no matching sample")
            tokens = pending["tokens"]
            prompt, added, env_token_count = _extend_prompt(
                tokenizer, parser, source, path["prompt_tokens"], tokens, row["messages"])
            path["turns"].append({"prompt_tokens": path["prompt_tokens"],
                                  "response_tokens": tokens, "logprobs": pending["logprobs"]})
            path["prompt_tokens"] = prompt
            path["response_len"] += added
            if path["response_len"] != row["response_token_len"]:
                raise ValueError("Cached environment token count differs")
            if rows[-1]["event"] != "complete":
                action = row["info"]["response"]
                done = row.get("done", path["count"] + 1 >= config.get("max_steps", 60) or
                    isinstance(action, str) or isinstance(action, list) and any(
                        call.get("function", {}).get("name") == "finish" for call in action))
                path["steps"].append(source.step(
                    chat_completions=copy.deepcopy(path["messages"] + row["messages"][:1]),
                    observation=copy.deepcopy(path["observation"]), action=copy.deepcopy(action),
                    model_response=_clean_response(tokenizer, parser, tokens),
                    info=copy.deepcopy(row["info"]), reward=row["reward"], done=done))
            path["messages"].extend(row["messages"])
            path["observation"] = row["observation"]
            path["count"] += 1
            last_sample = pending
            pending = request = None
        elif row["event"] == "sampling_error":
            if request is None or pending is not None or row.get("turn", path["count"]) != path["count"]:
                raise ValueError("Cached sampling failure has no matching unrecorded request")
            sampling_error = row
        elif row["event"] == "complete":
            if pending is not None or row is not rows[-1]:
                raise ValueError("Cached completion has unfinished samples")
            result = {key: value for key, value in row.items() if key != "event"}
            result.update(turns=path["turns"], evidence=str(journal))
            return {"result": result, "events": rows, "journal": str(journal)}
        elif row["event"] not in {"root", "formatter_recovery", "formatter_failure",
                                  "sampling_recovery", "sampling_accounting_error"}:
            raise ValueError(f"Cannot replay cached rollout event: {row['event']}")
    if rows[-1]["event"] == "environment":
        final = rows[-1]
        action = final["info"]["response"]
        max_steps = config.get("max_steps", 60)
        done = final.get("done", path["count"] >= max_steps or isinstance(action, str) or (
            isinstance(action, list) and any(
                call.get("function", {}).get("name") == "finish" for call in action)))
        max_response = config.get("max_response_length", 28 * 1024)
        truncated = path["response_len"] >= max_response or last_sample["stop_reason"] == "length"
        termination = "TRUNCATION" if truncated else ("ENV_DONE" if done else
                      "MAX_STEPS" if path["count"] >= max_steps else None)
        if termination is None:
            raise ValueError("Cached environment did not terminate the trajectory")
        reward = final["reward"]
        if truncated and path["response_len"] - env_token_count > max_response:
            reward = 0.
        result = _trajectory_result(path, task, group_id, reward, final["info"].get("metadata", {}),
                                    termination, journal)
        rows.append({"event": "complete", **{key: value for key, value in result.items() if key != "turns"}})
        return {"result": result, "events": rows, "journal": str(journal)}
    if pending is None and request is not None:
        elapsed_basis = "recorded active time"
        if not any("elapsed" in row for row in rows):
            if path["count"] or sampling_error:
                raise ValueError("Cannot recover active trajectory time from legacy sampling failure")
            with journal.open("rb") as stream:
                created = struct.unpack("<I", stream.read(8)[4:8])[0]
            path["elapsed"] = max(0., journal.stat().st_mtime - created)
            elapsed_basis = "estimate from gzip creation to final file write; includes failure handling"
        return {"path": path, "sample": None, "events": rows, "formatter": None,
                "recovery": {"event": "sampling_recovery", "path_id": path_id,
                    "turn": path["count"], "evidence": str(journal), "elapsed": path["elapsed"],
                    "elapsed_basis": elapsed_basis, "previous_reservation_id": request.get("reservation_id"),
                    "operational_gap": {
                        "seconds": sampling_error.get("operational_gap_seconds") if sampling_error else None,
                        "error": sampling_error.get("error") if sampling_error else None,
                        "counted_toward_trajectory_timeout": False},
                    "uncertainty": "Previous request has no recorded response; charge remains unknown and reserved",
                    "action": "New reservation required before repeating only the unrecorded request"}}
    if pending is None:
        raise ValueError("Cached rollout is incomplete without a recoverable final sample")
    formatter_path = journal.parent / f"{identity}-{path_id}-{path['count']}-formatter.jsonl.gz"
    if not formatter_path.exists():
        return {"path": path, "sample": pending, "events": rows, "formatter": None,
                "settlements": _recovery_settlements(journal, rows, None,
                    f"{identity}-{path_id}-{path['count']}"),
                "recovery": {"event": "sampling_recovery", "path_id": path_id,
                    "turn": path["count"], "evidence": str(journal), "elapsed": path["elapsed"],
                    "action": "Reuse recorded sample; preserve previous reservation without a new sampling charge"}}
    formatter = _cached_formatter(formatter_path)
    if formatter is None:
        return {"path": path, "sample": pending, "events": rows, "formatter": None,
                "settlements": _recovery_settlements(journal, rows, formatter_path,
                    f"{identity}-{path_id}-{path['count']}"),
                "recovery": {"event": "formatter_recovery", "path_id": path_id,
                    "turn": path["count"], "source_formatter_evidence": str(formatter_path),
                    "action": "Reuse recorded model sample; retry only formatter with no recorded response",
                    "uncertainty": "Prior transport-failed request charges remain reserved"}}
    return {"path": path, "sample": pending, "events": rows,
            "formatter": formatter, "formatter_evidence": str(formatter_path),
            "settlements": _recovery_settlements(journal, rows, formatter_path,
                f"{identity}-{path_id}-{path['count']}")}


def collect_group(sampler, tokenizer, task, config, output, step, group_id, budget=None,
                  *, _path_id=None):
    """Collect a GRPO/DAPO group or released ARPO frontier with independent backfill."""
    from rl_cost import formatter_context, record as record_cost, token_estimate
    from rl_budget import Budget, BudgetExceeded, sampling_upper

    budget = budget if budget is not None else Budget()

    source = initialize_source()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    algorithm = config["algorithm"].lower()
    if algorithm not in {"grpo", "dapo", "arpo"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    group_size = config.get("group_size", 8)
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if config.get("parallel_rollouts", False) and algorithm != "arpo" and group_size > 1 and _path_id is None:
        finished, errors = [], []
        with ThreadPoolExecutor(max_workers=group_size) as pool:
            futures = [pool.submit(collect_group, sampler, tokenizer, task, config, output,
                                   step, group_id, budget, _path_id=index)
                       for index in range(group_size)]
            for future in futures:
                try:
                    finished.extend(future.result())
                except Exception as error:
                    errors.append(error)
        if errors:
            raise next((error for error in errors if isinstance(error, BudgetExceeded)), errors[0])
        return finished
    if _path_id is not None:
        group_size = 1
    initial = min(group_size, max(1, config.get("arpo_initial_rollouts", 4))) if algorithm == "arpo" else group_size
    branch_budget = group_size - initial if algorithm == "arpo" else 0
    max_steps = config.get("max_steps", 60)
    max_response = config.get("max_response_length", 28 * 1024)
    context_limit = config.get("model_context", 30 * 1024)
    trajectory_timeout = config.get("trajectory_timeout", 3600 if algorithm == "arpo" else 1200)
    parser = source.parser(tokenizer)
    agent = source.agent(tool_map=source.tool_map)
    env = source.environment(task=copy.deepcopy(task), tool_map=source.tool_map,
                             reward_fn=_reward_function(source, config, step), max_steps=max_steps)
    observation, info = env.reset()
    agent.update_from_env(observation, 0., False, info)
    initial_messages = copy.deepcopy(agent.chat_completions)
    initial_prompt = tokenizer.encode(parser.parse(initial_messages, add_generation_prompt=True,
                                                   is_first_msg=True), add_special_tokens=False)
    if len(initial_prompt) > config.get("max_prompt_length", 2048):
        raise ValueError("Initial prompt exceeds released max_prompt_length; refusing truncation")
    identity = f"{step}-{group_id}"
    if _path_id is not None:
        identity += f"-path-{_path_id}"
    evidence_path = output / f"group-{identity}.jsonl.gz"
    resumed = _resume_path(source, tokenizer, parser, task, config, step, group_id, _path_id,
                           identity, initial_messages, initial_prompt)
    if resumed and "result" in resumed:
        with gzip.open(evidence_path, "xt", encoding="utf-8") as evidence:
            for event in resumed["events"]:
                if event["event"] == "group":
                    event = {**event, "replayed_from": resumed["journal"]}
                evidence.write(json.dumps(event, ensure_ascii=False) + "\n")
        resumed["result"]["evidence"] = str(evidence_path)
        env.close()
        return [resumed["result"]]
    finished = []
    next_id = 0 if _path_id is None else _path_id

    def new_path():
        nonlocal next_id
        path = {"path_id": next_id, "messages": copy.deepcopy(initial_messages), "steps": [],
                "turns": [], "response_len": 0, "initial_entropy": None, "count": 0,
                "prompt_tokens": list(initial_prompt), "elapsed": 0.0,
                "observation": copy.deepcopy(observation)}
        next_id += 1
        return path

    frontier = deque([resumed["path"]] if resumed else (new_path() for _ in range(initial)))
    with gzip.open(evidence_path, "xt", encoding="utf-8") as evidence:
        def record(value):
            evidence.write(json.dumps(value, ensure_ascii=False) + "\n")
            evidence.flush()

        if resumed:
            for event in resumed["events"]:
                record(event)
            if resumed.get("formatter_evidence"):
                prior_formatter = Path(resumed["formatter_evidence"])
                shutil.copyfile(prior_formatter, output / prior_formatter.name)
            record(resumed.get("recovery") or {
                "event": "formatter_recovery", "path_id": resumed["path"]["path_id"],
                "evidence": resumed["formatter_evidence"]})
            for settlement in resumed.get("settlements", []):
                budget.settle(settlement["reservation_id"], settlement["cost"])
                if settlement["event"] is not None:
                    record_cost(output, settlement["event"])
        else:
            record({"event": "group", "task": task, "config": config, "step": step, "group_id": group_id,
                    "initial_messages": initial_messages, "initial_prompt_tokens": initial_prompt,
                    "curriculum": "explicit epoch stages SUM1-2 MACRO3-4 SUCCESS5; reward resolved before dispatch"})
            for path in frontier:
                record({"event": "root", "path_id": path["path_id"]})
        while len(finished) < group_size:
            if not frontier:
                path = new_path()
                frontier.append(path)
                record({"event": "backfill", "path_id": path["path_id"]})
            path = frontier.popleft()
            agent.messages[:] = copy.deepcopy(path["messages"])
            agent._trajectory.steps[:] = copy.deepcopy(path["steps"])
            agent.current_observation = copy.deepcopy(path["observation"])
            env.step_count = path["count"]
            prompt = path["prompt_tokens"]
            turn_started = time.monotonic()
            available = min(max_response - path["response_len"], context_limit - len(prompt))
            cached_sample = resumed.get("sample") if resumed else None
            if path["elapsed"] >= trajectory_timeout and cached_sample is None:
                termination, reward, metadata = "TIMEOUT", 0., {}
            elif available <= 0:
                termination, reward, metadata = "TRUNCATION", 0., {}
            else:
                seed = config.get("seed", 0) + step * 100000 + path["path_id"] * max_steps + path["count"]
                if cached_sample is not None:
                    sequence = SimpleNamespace(**cached_sample)
                    sampling_evidence = {"sequences": [cached_sample]}
                else:
                    reservation = budget.reserve("sampling", sampling_upper(len(prompt), available))
                    record({"event": "request", "path_id": path["path_id"], "turn": path["count"],
                        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                        "prompt_length": len(prompt), "max_tokens": available, "seed": seed,
                        "reservation_id": reservation, "elapsed": path["elapsed"]})
                    try:
                        sequence, sampling_evidence = sample_response(
                            sampler, prompt, {"temperature": config.get("temperature", 1.),
                                              "top_p": config.get("top_p", 1.)}, available, seed)
                    except Exception as error:
                        record({"event": "sampling_error", "path_id": path["path_id"],
                                "turn": path["count"], "reservation_id": reservation,
                                "error": f"{type(error).__name__}: {error}", "elapsed": path["elapsed"],
                                "operational_gap_seconds": time.monotonic() - turn_started})
                        record_cost(output, {"phase": config.get("collection_phase", "rollout"),
                                             "identity": f"{identity}-{path['path_id']}-{path['count']}",
                                             "reservation_id": reservation, "cost_usd": None,
                                             "status": "unknown_charge", "error": str(error)})
                        raise
                    record({"event": "sample", "path_id": path["path_id"], "turn": path["count"],
                            "response": sampling_evidence, "reservation_id": reservation,
                            "elapsed": path["elapsed"] + time.monotonic() - turn_started})
                    cached = sampling_evidence.get("prompt_cache_hit_tokens") or 0
                    if not 0 <= cached <= len(prompt):
                        raise ValueError("Invalid reported prompt cache token count")
                    sampled_cost = token_estimate(prefill=len(prompt) - cached,
                                                 cached_prefill=cached, sample=len(sequence.tokens))
                    try:
                        budget.settle(reservation, sampled_cost)
                    except Exception as error:
                        record({"event": "sampling_accounting_error", "path_id": path["path_id"],
                                "turn": path["count"], "reservation_id": reservation,
                                "error": f"{type(error).__name__}: {error}",
                                "elapsed": path["elapsed"] + time.monotonic() - turn_started})
                        raise
                    record_cost(output, {"phase": config.get("collection_phase", "rollout"),
                                         "identity": f"{identity}-{path['path_id']}-{path['count']}",
                                         "reservation_id": reservation, "prompt_tokens": len(prompt),
                                         "sampled_tokens": len(sequence.tokens),
                                         "cached_tokens": sampling_evidence.get("prompt_cache_hit_tokens", 0),
                                         "token_rate_estimate_usd": sampled_cost, "cost_usd": None})
                tokens, logprobs = list(sequence.tokens), sequence.logprobs
                if not tokens or logprobs is None or len(tokens) != len(logprobs):
                    raise ValueError("Sampling must return nonempty tokens with aligned logprobs")
                if not all(lp is not None and math.isfinite(lp) for lp in logprobs):
                    raise ValueError("Sampling returned missing or nonfinite token logprobs")
                entropy = sampled_logprob_entropy(logprobs)
                if path["initial_entropy"] is None:
                    path["initial_entropy"] = entropy
                score = arpo_branch_score(entropy, path["initial_entropy"],
                                          config.get("arpo_alpha", .5), config.get("arpo_beta", .3))
                if algorithm == "arpo" and score > config.get("arpo_tau", .6) and branch_budget > 0:
                    fork = copy.deepcopy(path)
                    fork["path_id"] = next_id
                    next_id += 1
                    frontier.append(fork)
                    branch_budget -= 1
                    record({"event": "fork", "path_id": fork["path_id"], "parent": path["path_id"],
                            "prefix_turns": path["count"], "entropy": entropy, "branch_score": score})
                response = _clean_response(tokenizer, parser, tokens)
                old_message_count = len(agent.messages)
                _formatter_failure.error = None
                _formatter_failure.replay = resumed["formatter"] if resumed else None
                try:
                    with formatter_context(output, f"{identity}-{path['path_id']}-{path['count']}", budget=budget):
                        action = agent.update_from_model(response).action
                        if (resumed and resumed.get("formatter_evidence") and path["count"] + 1 >= max_steps
                                and not any(call.get("function", {}).get("name") == "finish" for call in action)):
                            # The released max-step formatter prompt includes random tool-call IDs.
                            _restore_formatter_action_ids(action, source, task, resumed["formatter_evidence"])
                        next_observation, reward, done, info = env.step(action)
                        agent.update_from_env(next_observation, reward, done, info)
                finally:
                    _formatter_failure.replay = None
                expected_formatter_error = _expected_formatter_error(_formatter_failure.error)
                if _formatter_failure.error is not None and not expected_formatter_error:
                    raise _formatter_failure.error
                if expected_formatter_error:
                    record({"event": "formatter_failure", "path_id": path["path_id"], "turn": path["count"],
                            "error": f"{type(_formatter_failure.error).__name__}: {_formatter_failure.error}",
                            "handling": "released zero reward"})
                if resumed and resumed["formatter"] and (not done or expected_formatter_error and reward != 0):
                    raise ValueError("Recovered formatter did not preserve terminal reward handling")
                current = agent.get_current_state()
                current.reward, current.done = reward, done
                current.info.update(info)
                new_messages = agent.messages[old_message_count:]
                # Preserve sampled IDs: decoding and re-encoding can change BPE boundaries and whitespace.
                path["prompt_tokens"], added, env_token_count = _extend_prompt(
                    tokenizer, parser, source, prompt, tokens, new_messages)
                path["response_len"] += added
                path["turns"].append({"prompt_tokens": prompt, "response_tokens": tokens,
                                      "logprobs": list(logprobs)})
                path["count"] += 1
                path["elapsed"] += time.monotonic() - turn_started
                record({"event": "environment", "path_id": path["path_id"], "turn": path["count"] - 1,
                        "messages": new_messages, "observation": next_observation, "info": info,
                        "response_token_len": path["response_len"], "reward": reward, "done": done,
                        "elapsed": path["elapsed"]})
                metadata = info.get("metadata", {})
                resumed = None
                truncation = path["response_len"] >= max_response or sequence.stop_reason == "length"
                termination = "TRUNCATION" if truncation else ("ENV_DONE" if done else
                              "MAX_STEPS" if path["count"] >= max_steps else None)
                if termination is None and path["elapsed"] >= trajectory_timeout:
                    termination = "TIMEOUT"
                if truncation and path["response_len"] - env_token_count > max_response:
                    reward = 0.
            if termination is not None:
                result = _trajectory_result(path, task, group_id, reward, metadata, termination, evidence_path)
                finished.append(result)
                record({"event": "complete", **{key: value for key, value in result.items() if key != "turns"}})
            else:
                path["observation"] = copy.deepcopy(next_observation)
                path["messages"] = copy.deepcopy(agent.messages)
                path["steps"] = copy.deepcopy(agent.trajectory.steps)
                frontier.append(path)
    env.close()
    return finished
