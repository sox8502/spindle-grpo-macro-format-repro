import gzip
import hashlib
import json
import os
from pathlib import Path
import threading


_state = threading.local()


def start_query(query):
    _state.query = query
    _state.records = []
    _state.error = None


def record_api(messages, **result):
    if not os.environ.get("STAR_API_EVIDENCE_DIR"):
        return
    try:
        if "response" in result:
            result["response"] = result["response"].model_dump(mode="json")
        _state.records.append(json.loads(json.dumps({"messages": messages, **result}, ensure_ascii=False)))
    except Exception as error:
        _state.error = {"stage": "capture", "type": type(error).__name__}


def finish_query():
    directory = os.environ.get("STAR_API_EVIDENCE_DIR")
    if not directory:
        return None
    path = Path(directory) / (hashlib.sha256(_state.query.encode()).hexdigest() + ".json.gz")
    try:
        with gzip.open(path, "xt", encoding="utf-8") as handle:
            json.dump({"query": _state.query, "records": _state.records, "capture_error": _state.error}, handle, ensure_ascii=False)
    except OSError as error:
        _state.error = {"stage": "write", "type": type(error).__name__}
    finally:
        _state.records = []
    return {"path": str(path), "complete": _state.error is None, "error": _state.error}
