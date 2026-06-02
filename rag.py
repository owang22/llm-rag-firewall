"""
Toy RAG system backed by TF-IDF retrieval (sklearn) and a local LLM.

Four query modes (see RAGPipeline):
  baseline          — original prompt, no judge
  secure_prompt     — hardened prompt, no judge
  judge_only        — original prompt + semantic judge
  defended          — hardened prompt + semantic judge (full defense)

LLM backends (--backend flag in app.py / eval.py):
  transformers — HuggingFace pipeline (loads model from local HF cache)
  vllm         — vLLM engine, fastest option for bulk evaluation
"""

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from defenses import BASELINE_SYSTEM_PROMPT, Defense

# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE = "/home/nesl/.cache/huggingface/hub"




class TransformersBackend:
    """Lazy-loading HuggingFace transformers pipeline."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._pipe = None

    def _load(self) -> None:
        import torch
        from transformers import pipeline

        print(f"[LLM] Loading {self.model_name} via transformers …")
        self._pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=256,
            do_sample=False,
        )
        print("[LLM] Model loaded.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._pipe is None:
            self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        out = self._pipe(messages)
        return out[0]["generated_text"][-1]["content"].strip()


class VLLMBackend:
    """vLLM engine backend — best throughput for bulk evaluation."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._llm = None
        self._tokenizer = None

    def _load(self) -> None:
        import os
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        # Disable vLLM v1 engine to avoid flashinfer → tvm_ffi → torch_c_dlpack_ext
        # ABI mismatch on this machine. v0 engine works fine.
        os.environ.setdefault("VLLM_USE_V1", "0")

        print(f"[LLM] Loading {self.model_name} via vLLM …")
        self._llm = LLM(
            model=self.model_name, dtype="float16",
            max_model_len=4096, gpu_memory_utilization=0.85,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._sampling = SamplingParams(temperature=0.0, max_tokens=256)
        print("[LLM] Model loaded.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._llm is None:
            self._load()
        from vllm import SamplingParams

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self._llm.generate([text], self._sampling)
        return outputs[0].outputs[0].text.strip()


_backend_cache: dict[tuple, object] = {}
# Parallel registry of raw LLM objects for explicit shutdown
_backend_llm_objects: dict[tuple, object] = {}


def _destroy_process_group_on_exit() -> None:
    """
    Registered via atexit — runs only at process exit, never mid-run.
    Suppresses the ProcessGroupNCCL 'destroy_process_group() was not called'
    warning without interfering with a second vllm model load in the same process.
    """
    try:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass


import atexit as _atexit
_atexit.register(_destroy_process_group_on_exit)


def cleanup_all_backends() -> None:
    """
    Release main-model GPU memory between phase 1 and phase 2.
    Does NOT destroy the distributed process group — that would prevent the
    judge's vllm from initialising its own distributed environment.
    The atexit handler above handles that at process exit.
    """
    import gc
    from defenses import cleanup_judge

    for key in list(_backend_cache):
        del _backend_cache[key]

    for key, b in list(_backend_llm_objects.items()):
        llm = getattr(b, "_llm", None)
        if llm is not None:
            try:
                engine = getattr(llm, "llm_engine", None)
                if engine and hasattr(engine, "stop_remote_worker_execution_loop"):
                    engine.stop_remote_worker_execution_loop()
            except Exception:
                pass
        del _backend_llm_objects[key]

    cleanup_judge()
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def build_backend(
    backend: Literal["transformers", "vllm"],
    model_name: str = DEFAULT_MODEL,
    secrets: dict | None = None,
):
    """
    Return a callable (system_prompt, user_prompt) -> str.
    Cached by (backend, model_name) so the model loads at most once per process.
    """
    key = (backend, model_name)
    if key not in _backend_cache:
        if backend == "transformers":
            b = TransformersBackend(model_name)
        elif backend == "vllm":
            b = VLLMBackend(model_name)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'vllm' or 'transformers'.")
        _backend_cache[key] = b
        # Keep a separate reference to the raw LLM object for explicit shutdown
        if backend == "vllm":
            _backend_llm_objects[key] = b  # VLLMBackend._llm populated lazily on first call

    b = _backend_cache[key]
    return lambda sys, usr: b.generate(sys, usr)


# ---------------------------------------------------------------------------
# Document store + TF-IDF retriever
# ---------------------------------------------------------------------------

class DocumentStore:
    """Load JSONL documents and build a TF-IDF index."""

    def __init__(self, docs_path: str) -> None:
        self.docs: list[dict] = []
        with open(docs_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.docs.append(json.loads(line))

        texts = [d["content"] for d in self.docs]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._doc_vectors = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._doc_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if sims[idx] > 0.0:
                doc = self.docs[idx]
                results.append({
                    "id": doc.get("id", str(idx)),
                    "title": doc.get("title", "Untitled"),
                    "content": doc["content"],
                    "score": float(sims[idx]),
                })
        return results


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Parameters
    ----------
    docs_path     : path to the JSONL document corpus
    secrets       : dict of fake secrets (used by output validator)
    backend       : "transformers" | "vllm"  — main LLM backend
    model_name    : HuggingFace model ID for the main LLM
    top_k         : number of chunks to retrieve per query
    judge_model   : HuggingFace model ID for the semantic judge (lazy-loaded)
    judge_backend : backend for the judge (defaults to same as main backend)
    judge_gpu     : physical GPU index for the judge
    """

    def __init__(
        self,
        docs_path: str,
        secrets: dict,
        backend: str = "vllm",
        model_name: str = DEFAULT_MODEL,
        top_k: int = 3,
        judge_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        judge_backend: str | None = None,
        judge_gpu: int = 2,
    ) -> None:
        self.store = DocumentStore(docs_path)
        self.secrets = secrets
        _judge_backend = judge_backend if judge_backend is not None else backend
        self.defense = Defense(
            secrets, judge_model=judge_model,
            judge_backend=_judge_backend, judge_gpu=judge_gpu,
        )
        self.top_k = top_k
        self._generate = build_backend(backend, model_name)

    def _build_user_prompt(self, context: str, question: str) -> str:
        return (
            f"RETRIEVED DOCUMENTS:\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Please answer based only on the documents above."
        )

    def _query_raw(self, question: str, system_prompt: str, mode_label: str) -> dict:
        """
        Shared internal method: retrieve + generate with the given system prompt.
        Returns a raw result dict with _raw_answer / _judgment_pending so the
        judgment phase can be run later (or skipped entirely for prompt-only modes).
        """
        t0 = time.perf_counter()
        chunks = self.store.retrieve(question, self.top_k)
        context = "\n\n".join(f"[{c['title']}]\n{c['content']}" for c in chunks)
        user_prompt = self._build_user_prompt(context, question)
        raw_answer = self._generate(system_prompt, user_prompt)
        latency = time.perf_counter() - t0
        return {
            "mode": mode_label,
            "question": question,
            "answer": raw_answer,
            "_raw_answer": raw_answer,
            "_judgment_pending": True,
            "retrieved_chunks": chunks,
            "blocked_chunks": [],
            "detection_metadata": {},
            "latency_s": round(latency, 3),
        }

    def _finalize_no_judge(self, raw: dict) -> dict:
        """Strip internal fields and mark as final without running the judge."""
        result = {k: v for k, v in raw.items() if not k.startswith("_")}
        return result

    # ── Four public query methods — one per defense layer combination ────────

    def query_baseline(self, question: str) -> dict:
        """Original prompt, no judge — completely undefended."""
        return self._finalize_no_judge(
            self._query_raw(question, BASELINE_SYSTEM_PROMPT, "baseline")
        )

    def query_secure_prompt_only(self, question: str) -> dict:
        """Hardened system prompt only — no judge. Shows prompt-layer contribution."""
        return self._finalize_no_judge(
            self._query_raw(question, self.defense.secure_system_prompt(), "secure_prompt")
        )

    def query_baseline_raw(self, question: str) -> dict:
        """Original prompt, judgment pending. Used to measure judge-only contribution."""
        return self._query_raw(question, BASELINE_SYSTEM_PROMPT, "judge_only")

    def query_defended_raw(self, question: str) -> dict:
        """Hardened prompt, judgment pending. Used for full defended mode."""
        return self._query_raw(
            question, self.defense.secure_system_prompt(), "defended"
        )

    def query_defended(self, question: str) -> dict:
        """Full defense: hardened prompt + judge. Convenience for interactive use."""
        return apply_judge(self.query_defended_raw(question), self.defense)


def apply_judge(raw_result: dict, defense) -> dict:
    """
    Apply the semantic judge to a raw result produced by query_defended_raw().

    Can be called after the main model has been unloaded, as long as the
    judge model is loaded.  Updates `answer`, `detection_metadata`, and
    `latency_s` in place (returns a new dict).
    """
    import time as _time

    t0 = _time.perf_counter()
    question   = raw_result["question"]
    raw_answer = raw_result["_raw_answer"]

    answer, was_blocked, block_reason = defense.check_response(question, raw_answer)
    judge_latency = _time.perf_counter() - t0

    result = {k: v for k, v in raw_result.items()
              if not k.startswith("_")}   # strip internal fields
    result["answer"] = answer
    result["detection_metadata"] = {
        "semantic_blocked": was_blocked,
        "block_reason": block_reason,
        "output_flags": ["semantic_judge_blocked"] if was_blocked else [],
    }
    result["latency_s"] = round(raw_result["latency_s"] + judge_latency, 3)
    return result
