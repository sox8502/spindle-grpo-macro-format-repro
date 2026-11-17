import io
import json
import urllib.error

import pytest

import collection_launcher as run_original


def test_access_failure_stops_before_generation_and_redacts_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ORX_RUN_ID", "offline")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-private-key")
    monkeypatch.setattr(run_original, "ARTIFACTS", tmp_path)
    calls = []

    def denied(request, **kwargs):
        calls.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 403, "Denied", {}, io.BytesIO(b"Denied test-private-key"))

    monkeypatch.setattr(run_original.urllib.request, "urlopen", denied)
    monkeypatch.setattr(run_original, "hf_hub_download", lambda *a, **k: pytest.fail("Generation/data setup must not start"))
    with pytest.raises(RuntimeError, match="HTTP 403") as caught:
        run_original.main()
    assert len(calls) == 1
    assert calls[0]["model"] == "deepseek/deepseek-v3.2-exp"
    assert calls[0]["max_tokens"] == 8192 and calls[0]["reasoning_effort"] == "high"
    assert "test-private-key" not in str(caught.value)
    saved = (tmp_path / "original-model-source/runs/offline/access-error.json").read_text()
    assert "test-private-key" not in saved


def test_wrong_reported_model_stops_before_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("ORX_RUN_ID", "offline")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-private-key")
    monkeypatch.setattr(run_original, "ARTIFACTS", tmp_path)
    body = {"model": "deepseek/deepseek-v4.1-flash", "provider": "offline", "choices": [{"finish_reason": "stop"}]}
    monkeypatch.setattr(run_original.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(run_original, "hf_hub_download", lambda *a, **k: pytest.fail("No model substitution"))
    with pytest.raises(ValueError, match="exact original model"):
        run_original.main()
