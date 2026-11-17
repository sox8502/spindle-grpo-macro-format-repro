#!/usr/bin/env python3
"""
Local retrieval server for BrowseComp-Plus (DeepResearch).

Dense retrieval only:
- qwen: Dense index shards at examples/deepresearch/indexes/qwen3-embedding-{0.6b,4b,8b}

Usage:
  python examples/deepresearch/server.py --qwen_variant qwen3-embedding-8b
"""
import argparse
import glob
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from flask import Flask, jsonify, request


def _cuda_if_available(default: str) -> str:
    if default != "auto":
        return default
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_corpus_parquet(corpus_dir: str):
    # Keep only an index mapping to avoid duplicating large text in memory.
    import datasets

    files = sorted(glob.glob(str(Path(corpus_dir) / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {corpus_dir}")
    ds = datasets.load_dataset("parquet", data_files=files, split="train")
    docid_to_row: dict[str, int] = {}
    for i in range(len(ds)):
        docid_to_row[str(ds[i]["docid"])] = i
    return ds, docid_to_row


@dataclass
class QwenDenseIndex:
    index: Any  # faiss index
    docids: list[str]
    device: str
    model_name: str
    dim: int
    model: Any
    tokenizer: Any
    task_prefix: str
    max_length: int
    torch_dtype: str

    def encode_query(self, query: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        text = f"{self.task_prefix}{query}"
        batch = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.torch_dtype == "float16":
            amp_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float32

        with torch.no_grad():
            # Keep it simple: autocast only when on cuda.
            if self.device == "cuda" and amp_dtype != torch.float32:
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                from contextlib import nullcontext

                ctx = nullcontext()
            with ctx:
                out = self.model(**batch, return_dict=True)
                # Recommended pooling for Qwen3 embeddings is EOS/last token pooling
                attn = batch["attention_mask"]
                left_padding = attn[:, -1].sum() == attn.shape[0]
                if left_padding:
                    emb = out.last_hidden_state[:, -1]
                else:
                    seq_lens = attn.sum(dim=1) - 1
                    emb = out.last_hidden_state[torch.arange(out.last_hidden_state.size(0), device=out.last_hidden_state.device), seq_lens]
                emb = F.normalize(emb, p=2, dim=1)

        emb = emb.detach().cpu().numpy().astype(np.float32, order="C")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[1] != self.dim:
            raise ValueError(f"Query embedding dim mismatch: got {emb.shape[1]} expected {self.dim}")
        return emb

    def search(self, query: str, k: int) -> tuple[list[str], list[float]]:
        q = self.encode_query(query)
        scores, idxs = self.index.search(q, k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()
        out_docids = [self.docids[i] for i in idxs if 0 <= i < len(self.docids)]
        out_scores = [float(s) for s in scores[: len(out_docids)]]
        return out_docids, out_scores


class Retriever:
    def __init__(
        self,
        corpus_dir: str,
        index_dir: str,
        qwen_variant: str,
        qwen_model: str | None,
        device: str,
    ):
        self.corpus_dir = corpus_dir
        self.index_dir = Path(index_dir)
        self.qwen_variant = qwen_variant
        self.qwen_model = qwen_model
        self.device = device

        self.corpus_ds, self.docid_to_row = _load_corpus_parquet(self.corpus_dir)

        self._qwen = None
        self._init_qwen()

    @property
    def corpus_size(self) -> int:
        return len(self.corpus_ds)

    def _init_qwen(self):
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        variant_dir = self.index_dir / self.qwen_variant
        if not variant_dir.exists():
            raise FileNotFoundError(f"Qwen index dir not found: {variant_dir}")

        shard_paths = sorted(variant_dir.glob("corpus.shard*_of_4.pkl"))
        if not shard_paths:
            raise FileNotFoundError(f"No Qwen index shards found under: {variant_dir}")

        all_emb = []
        all_docids: list[str] = []
        dim = None
        for p in shard_paths:
            with p.open("rb") as f:
                emb, docids = pickle.load(f)
            emb = np.asarray(emb, dtype=np.float32, order="C")
            if dim is None:
                dim = emb.shape[1]
            if emb.ndim != 2 or emb.shape[1] != dim:
                raise ValueError(f"Bad shard shape in {p}: {emb.shape}, expected (*,{dim})")
            all_emb.append(emb)
            all_docids.extend([str(x) for x in docids])

        mat = np.concatenate(all_emb, axis=0)
        dim = int(mat.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        if self.qwen_model:
            model_name = self.qwen_model
        else:
            model_name = {
                "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
                "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
            }[self.qwen_variant]

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        if self.device == "cuda":
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to("cuda")
            torch_dtype = "float16"
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cpu")
            torch_dtype = "float32"
        model.eval()

        # Default query-side instruction (matches Tevatron/BrowseComp-Plus index scripts)
        task_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
        max_length = 8192

        self._qwen = QwenDenseIndex(
            index=index,
            docids=all_docids,
            device=self.device,
            model_name=model_name,
            dim=dim,
            model=model,
            tokenizer=tokenizer,
            task_prefix=task_prefix,
            max_length=max_length,
            torch_dtype=torch_dtype,
        )

    def _get_doc_by_docid(self, docid: str) -> dict[str, Any] | None:
        i = self.docid_to_row.get(str(docid))
        if i is None:
            return None
        row = self.corpus_ds[i]
        return {"docid": str(row.get("docid")), "text": row.get("text", ""), "url": row.get("url", "")}

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        docids, scores = self._qwen.search(query, k)
        out = []
        for docid, score in zip(docids, scores, strict=False):
            doc = self._get_doc_by_docid(docid)
            if not doc:
                continue
            out.append({"doc": doc, "score": float(score)})
        return out


app = Flask(__name__)
retriever: Retriever | None = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "device": retriever.device if retriever else None,
            "corpus_size": retriever.corpus_size if retriever else 0,
            "index_loaded": retriever is not None,
            "method": "qwen" if retriever else None,
            "qwen_model": retriever._qwen.model_name if retriever else None,
            "qwen_pooling": "last_token_eos" if retriever else None,
            "qwen_task_prefix": retriever._qwen.task_prefix if retriever else None,
        }
    )


@app.route("/retrieve", methods=["POST"])
def retrieve():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = data.get("query")
        if not query:
            return jsonify({"error": "Missing 'query' in request"}), 400
        k = int(data.get("top_k", data.get("k", 10)))
        k = max(1, min(k, 100))

        results = retriever.search(query, k)
        formatted = [
            {
                "id": f"doc_{i}",
                "docid": r["doc"]["docid"],
                "text": r["doc"]["text"],
                "url": r["doc"]["url"],
                "score": r["score"],
            }
            for i, r in enumerate(results, 1)
        ]
        return jsonify(
            {
                "query": query,
                "method": "qwen",
                "results": formatted,
                "num_results": len(formatted),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="DeepResearch local retrieval server")
    parser.add_argument("--corpus_dir", default="./corpus/data", help="Parquet corpus directory")
    parser.add_argument("--index_dir", default="./indexes", help="Index root directory")
    parser.add_argument(
        "--qwen_variant",
        choices=["qwen3-embedding-4b", "qwen3-embedding-8b", "qwen3-embedding-0.6b"],
        default="qwen3-embedding-8b",
        help="Which qwen index under index_dir/ (only for index_type=qwen)",
    )
    parser.add_argument(
        "--qwen_model",
        default=None,
        help="HF model id/path for query encoder (defaults to Qwen/Qwen3-Embedding-*)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Default auto prefers GPU if available",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = _cuda_if_available(args.device)

    global retriever
    start = time.time()
    retriever = Retriever(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        qwen_variant=args.qwen_variant,
        qwen_model=args.qwen_model,
        device=device,
    )
    print(
        f"Loaded corpus={retriever.corpus_size} method=qwen device={retriever.device} in {time.time()-start:.2f}s"
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
