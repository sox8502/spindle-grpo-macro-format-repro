import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import urllib.error
import urllib.request

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path.home() / "star-artifacts/agent-star"
MODEL = "deepseek/deepseek-v3.2-exp"
URL = "https://openrouter.ai/api/v1"


def execute(script, arguments):
    previous = sys.argv
    sys.argv = [script, *arguments]
    try:
        runpy.run_path(str(ROOT / "upstream/Inference" / script), run_name="__main__")
    finally:
        sys.argv = previous


def main():
    protocol = json.loads((ROOT / "protocol_config.json").read_text())
    collection = json.loads((ROOT / "collection_config.json").read_text())
    print("GENERATION_CONFIG " + json.dumps({"protocol": protocol, **collection}), flush=True)
    run_id = os.environ["ORX_RUN_ID"]
    output = ARTIFACTS / "original-model-source/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    print("ARTIFACT_DIRECTORY " + str(output), flush=True)
    credential_file = Path.home() / ".openresearch/env"
    if credential_file.exists():
        for line in credential_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                if name.strip().removeprefix("export ") == "OPENROUTER_API_KEY":
                    os.environ.setdefault("OPENROUTER_API_KEY", value.strip().strip("\"'"))
    key = os.environ["OPENROUTER_API_KEY"]
    request = {"model": MODEL, "messages": [{"role": "user", "content": "Reply with OK."}],
               "max_tokens": 8192, "temperature": 0.6, "top_p": 0.95, "reasoning_effort": "high"}
    try:
        with urllib.request.urlopen(urllib.request.Request(URL + "/chat/completions",
                data=json.dumps(request).encode(), headers={"Authorization": "Bearer " + key,
                "Content-Type": "application/json"}), timeout=180) as response:
            probe = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode().replace(key, "[REDACTED]")
        (output / "access-error.json").write_text(json.dumps({"status": error.code, "body": detail}, indent=2))
        raise RuntimeError(f"Original-model access failed: HTTP {error.code}: {detail}") from None
    (output / "access-response.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2))
    print("MODEL_ACCESS " + json.dumps({"model": probe.get("model"), "provider": probe.get("provider"),
          "usage": probe.get("usage"), "finish_reason": probe.get("choices", [{}])[0].get("finish_reason")}), flush=True)
    if probe.get("model") != MODEL:
        raise ValueError("Access check did not report the exact original model; generation stopped")

    query = ROOT / collection["query_file"]
    if hashlib.sha256(query.read_bytes()).hexdigest() != collection["query_file_sha256"]:
        raise ValueError("Frozen collection query bytes changed")
    selected = [json.loads(line) for line in query.read_text().splitlines()]
    if len(selected) != collection["queries"] or len({row["query"] for row in selected}) != len(selected):
        raise ValueError("Collection count or query uniqueness mismatch")
    shutil.copyfile(query, output / "queries.jsonl")
    hashes = {}
    for name in ["attractions.csv", "clean_Flights_2022.csv", "clean_accommodations_2022.csv", "clean_restaurant_2022.csv", "distance.csv"]:
        cached = Path(hf_hub_download("xxwu/Agent-STAR-TravelDatabase", name, repo_type="dataset",
                      revision="985c8c7ba3e6beb986b3b0a55a2b064d18f2e68e"))
        (ROOT / "upstream/database" / name).symlink_to(cached)
        hashes[name] = hashlib.sha256(cached.read_bytes()).hexdigest()
    inference = ROOT / "upstream/Inference"
    (inference / "output").mkdir(exist_ok=True)
    (inference / "output" / f"deepseek-v3.2-exp_validation_{run_id}").symlink_to(output, target_is_directory=True)
    (output / "provenance.json").write_text(json.dumps({"model": MODEL, "endpoint": URL,
        "source_commit": "14aa665d5d0c7256e8de4168c8ab3889d2e6658e", "database_hashes": hashes,
        "source_changes": "Connection adaptation, JSON correction hint, separate teacher API evidence recording",
        "collection": collection,
        "protocol": protocol,
        "teacher_thinking": "Original high reasoning; enable_thinking option remains commented",
        "recipe": f"Released commercial generation settings with {collection['max_workers']} workers; 128K context, 100 turns, temperature0.6, top_p0.95",
        "source_hashes": {str(p.relative_to(inference)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in inference.rglob("*.py")}}, indent=2))
    if collection["record_teacher_api_evidence"]:
        evidence = output / "api-evidence"
        evidence.mkdir()
        os.environ["STAR_API_EVIDENCE_DIR"] = str(evidence)
    os.environ["DASHSCOPE_API_KEY"] = key
    os.chdir(inference)
    sys.path.insert(0, str(inference))
    execute("main.py", ["--model", MODEL, "--server_url", URL, "--server_api_key", key,
            "--use_custom_query", "--input_file", str(output / "queries.jsonl"), "--save_suffix", run_id,
            "--max_workers", str(collection["max_workers"]), "--max_context", "131072", "--max_turns", "100",
            "--temperature", "0.6", "--top_p", "0.95"])
    execute("post_process.py", ["--path", str(output / "predictions.jsonl"), "--use_custom_query",
            "--format_model", "deepseek-v3.2-exp"])
    execute("eval.py", ["--path", str(output / "predictions_deepseek-v3.2-exp_formatted.jsonl"), "--save_score"])
    rows = [json.loads(line) for line in (output / "predictions_deepseek-v3.2-exp_formatted_scored.jsonl").read_text().splitlines()]
    summary = {"run_id": run_id, "model": MODEL, "scored": len(rows),
               "task_success": sum(row["success"] for row in rows), "protocol": protocol, "collection": collection, "artifacts": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("FINAL_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
