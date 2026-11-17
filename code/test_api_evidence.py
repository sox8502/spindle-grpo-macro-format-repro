from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path


spec = importlib.util.spec_from_file_location("api_evidence", Path(__file__).parent / "upstream/Inference/api_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def test_concurrent_evidence_preserves_reasoning_and_request_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("STAR_API_EVIDENCE_DIR", str(tmp_path))

    def collect(i):
        query = f"query-{i}"
        evidence.start_query(query)
        messages = [{"role": "user", "content": query}]
        response = {"choices": [{"message": {"content": "action", "reasoning_content": f"reason-{i}"}}], "usage": {"total_tokens": 30}}
        evidence.record_api(messages, response=SimpleNamespace(model_dump=lambda **kwargs: response))
        messages.append({"role": "assistant", "content": "later"})
        response["usage"]["total_tokens"] = 100
        return query, evidence.finish_query()

    with ThreadPoolExecutor(max_workers=80) as executor:
        results = list(executor.map(collect, range(160)))
    assert len(list(tmp_path.glob("*.json.gz"))) == 160
    for query, metadata in results:
        assert metadata["complete"] and metadata["error"] is None
        path = metadata["path"]
        assert Path(path).stem == hashlib.sha256(query.encode()).hexdigest() + ".json"
        with gzip.open(path, "rt") as handle:
            saved = json.load(handle)
        assert saved["query"] == query
        assert len(saved["records"]) == 1
        assert saved["records"][0]["messages"] == [{"role": "user", "content": query}]
        assert saved["records"][0]["response"]["choices"][0]["message"]["reasoning_content"] == query.replace("query", "reason")
        assert saved["records"][0]["response"]["usage"]["total_tokens"] == 30


def test_disabled_capture_does_not_write(monkeypatch):
    monkeypatch.delenv("STAR_API_EVIDENCE_DIR", raising=False)
    evidence.start_query("no capture")
    evidence.record_api([], response={})
    assert evidence.finish_query() is None


def test_failed_write_marks_evidence_incomplete_without_losing_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("STAR_API_EVIDENCE_DIR", str(tmp_path))
    evidence.start_query("completed prediction")

    def full_disk(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(evidence.gzip, "open", full_disk)
    result = {"plan": "completed prediction", "api_evidence": evidence.finish_query()}
    assert result["plan"] == "completed prediction"
    assert result["api_evidence"]["complete"] is False
    assert result["api_evidence"]["error"] == {"stage": "write", "type": "OSError"}


def test_failed_response_capture_is_explicit_and_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("STAR_API_EVIDENCE_DIR", str(tmp_path))
    evidence.start_query("capture failure")

    def bad_dump(**kwargs):
        raise ValueError("bad response serialization")

    evidence.record_api([], response=SimpleNamespace(model_dump=bad_dump))
    metadata = evidence.finish_query()
    assert metadata["complete"] is False
    with gzip.open(metadata["path"], "rt") as handle:
        assert json.load(handle)["capture_error"] == {"stage": "capture", "type": "ValueError"}
