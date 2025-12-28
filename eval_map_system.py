#!/usr/bin/env python3
"""Evaluate a retrieval-constrained misconception diagnosis system on MAP-like data.

This script is intentionally self-contained and reproducible:
- No model fine-tuning.
- Candidate misconception labels are derived only from retrieved evidence.
- A local LLM (via Ollama) is constrained to select/rank among candidates.

It produces:
- Top-1 / Top-k accuracy
- Violation / hallucination rates (schema + label constraints)
- Stability across reruns (agreement rate)

Dataset expected: map-data.csv (columns include QuestionText, StudentExplanation, Misconception).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tantivy
from sentence_transformers import SentenceTransformer


K0 = 60.0  # RRF constant
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIAGNOSER_MODEL = "llama3.2:latest"
DEFAULT_JUDGE_MODEL = "gemma3:1b"


def _clean(s: str) -> str:
    s = (s or "").replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _bm25_text(s: str) -> str:
    """Normalize text into a Tantivy-friendly token query.

    Tantivy's query parser uses a Lucene-like syntax that can fail on LaTeX,
    backslashes, and other punctuation common in math items. We aggressively
    normalize to alphanumeric tokens for robust indexing/search.
    """
    s = _clean(s)
    # Drop LaTeX markers and non-word symbols.
    s = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return _clean(s)


def rrf_fuse(vec_ids: Sequence[int], bm25_ids: Sequence[int]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for rank, idx in enumerate(vec_ids, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (K0 + rank)
    for rank, idx in enumerate(bm25_ids, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (K0 + rank)
    return scores


def _cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> List[int]:
    # query_vec: (d,), mat: (n,d)
    if mat.shape[0] == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sims = (m @ q).astype(np.float32)
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()


def _build_tantivy_index(texts: Sequence[str]) -> Tuple[tantivy.Index, tantivy.Searcher]:
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field("row_idx", stored=True, indexed=True)
    schema_builder.add_text_field("content", stored=False)
    schema = schema_builder.build()

    # In-memory index (no path) keeps evaluation ephemeral.
    index = tantivy.Index(schema)
    writer = index.writer(50_000_000)
    for row_idx, txt in enumerate(texts):
        writer.add_document(tantivy.Document(row_idx=row_idx, content=_bm25_text(txt)))
    writer.commit()
    return index, index.searcher()


def _tantivy_topk(index: tantivy.Index, searcher: tantivy.Searcher, query: str, k: int) -> List[int]:
    if k <= 0:
        return []
    q = index.parse_query(_bm25_text(query))
    res = searcher.search(q, k)
    out: List[int] = []
    for _score, addr in res.hits:
        doc = searcher.doc(addr)
        out.append(int(doc["row_idx"][0]))
    return out


def _ollama_generate(
    model: str,
    prompt: str,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
    keep_alive: str,
) -> str:
    """Call Ollama HTTP API for controlled generation."""
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


@dataclass
class Example:
    question: str
    answer: str
    student: str
    label: str
    qid: int


def _format_query(ex: Example) -> str:
    return _clean(f"Question: {ex.question} Answer key: {ex.answer} Student: {ex.student}")


def _prompt_diagnoser(
    ex: Example,
    candidates: Sequence[str],
    evidence: Sequence[Tuple[int, str, str]],
    top_k: int,
) -> str:
    cand_lines = "\n".join([f"- {c}" for c in candidates])
    ev_lines = []
    for row_idx, ev_student, ev_label in evidence:
        ev_lines.append(f"[e{row_idx}] label={ev_label} student=\"{_clean(ev_student)[:240]}\"")
    ev_block = "\n".join(ev_lines)

    return (
        "You are diagnosing a math misconception label from a CLOSED ontology. "
        "This is NOT grading and you must not invent new labels. "
        "Select and rank only from the candidate labels provided.\n\n"
        f"TASK INPUT\nQuestion: {_clean(ex.question)}\n"
        f"Correct answer (for context only): {_clean(ex.answer)}\n"
        f"Student explanation: {_clean(ex.student)}\n\n"
        "CANDIDATE LABELS (choose ONLY from these):\n"
        f"{cand_lines}\n\n"
        "RETRIEVED EVIDENCE (labeled training exemplars):\n"
        f"{ev_block}\n\n"
        "Return STRICT JSON with keys: ranked_labels, selected_label, evidence_ids.\n"
        f"- ranked_labels: list of up to {top_k} labels from candidates, best first\n"
        "- selected_label: same as ranked_labels[0]\n"
        "- evidence_ids: list of evidence ids like [""e123"", ""e456""] that support the choice\n"
    )


def _prompt_judge(
    ex: Example,
    candidates: Sequence[str],
    evidence: Sequence[Tuple[int, str, str]],
    diagnoser_json: Dict[str, Any],
) -> str:
    cand = ", ".join(candidates)
    ev_lines = []
    for row_idx, ev_student, ev_label in evidence[:10]:
        ev_lines.append(f"[e{row_idx}] label={ev_label} student=\"{_clean(ev_student)[:160]}\"")
    ev_block = "\n".join(ev_lines)

    return (
        "You are a reliability judge for an educational AI system. "
        "Decide if the diagnosis output is valid and grounded in the retrieved evidence.\n\n"
        f"Question: {_clean(ex.question)}\n"
        f"Student: {_clean(ex.student)}\n\n"
        f"Candidates: {cand}\n\n"
        "Evidence:\n"
        f"{ev_block}\n\n"
        "System output (JSON):\n"
        f"{json.dumps(diagnoser_json, ensure_ascii=False)}\n\n"
        "Return STRICT JSON with keys: schema_ok, label_in_candidates, evidence_ids_ok, grounded, notes.\n"
        "- schema_ok: boolean (has required keys and correct types)\n"
        "- label_in_candidates: boolean\n"
        "- evidence_ids_ok: boolean (evidence ids refer to provided evidence)\n"
        "- grounded: boolean (supported by evidence; do not over-assume)\n"
        "- notes: short string\n"
    )


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="map-data.csv")
    ap.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--diagnoser_model", default=DEFAULT_DIAGNOSER_MODEL)
    ap.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--vec_k", type=int, default=12)
    ap.add_argument("--bm25_k", type=int, default=12)
    ap.add_argument("--evidence_k", type=int, default=8)
    ap.add_argument("--candidate_cap", type=int, default=8)
    ap.add_argument("--rank_k", type=int, default=3)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--n_judge", type=int, default=100)
    ap.add_argument("--n_stability", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--keep_alive", default="10m")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    df = pd.read_csv(args.csv)
    df = df[df["Misconception"].notna()].copy()
    df["Misconception"] = df["Misconception"].astype(str)

    examples: List[Example] = []
    for _, row in df.iterrows():
        examples.append(
            Example(
                question=str(row.get("QuestionText", "")),
                answer=str(row.get("MC_Answer", "")),
                student=str(row.get("StudentExplanation", "")),
                label=str(row.get("Misconception", "")),
                qid=int(row.get("QuestionId", 0) or 0),
            )
        )

    ontology = sorted({e.label for e in examples if e.label.strip()})

    # Train/test split: stratify by QuestionId to avoid question leakage patterns in selection.
    by_q: Dict[int, List[int]] = {}
    for i, ex in enumerate(examples):
        by_q.setdefault(ex.qid, []).append(i)

    test_indices: List[int] = []
    for qid, idxs in by_q.items():
        rng.shuffle(idxs)
        take = max(1, int(0.2 * len(idxs)))
        test_indices.extend(idxs[:take])

    rng.shuffle(test_indices)
    test_indices = test_indices[: args.n_test]
    test_set = test_indices
    train_set = [i for i in range(len(examples)) if i not in set(test_set)]

    train_examples = [examples[i] for i in train_set]
    test_examples = [examples[i] for i in test_set]

    # Build retrieval memory
    embedder = SentenceTransformer(args.embed_model)
    train_texts = [_format_query(ex) for ex in train_examples]
    test_texts = [_format_query(ex) for ex in test_examples]

    print(f"[DATA] train={len(train_examples)} test={len(test_examples)} ontology={len(ontology)}")

    print("[EMBED] embedding train...")
    train_emb = np.asarray(embedder.encode(train_texts, show_progress_bar=True), dtype=np.float32)

    print("[BM25] building tantivy index...")
    bm25_index, bm25_searcher = _build_tantivy_index(train_texts)

    def retrieve(ex: Example) -> Tuple[List[int], List[Tuple[int, str, str]]]:
        q_text = _format_query(ex)
        q_emb = np.asarray(embedder.encode([q_text])[0], dtype=np.float32)

        vec_top = _cosine_topk(q_emb, train_emb, args.vec_k)
        bm_top = _tantivy_topk(bm25_index, bm25_searcher, q_text, args.bm25_k)

        fused = rrf_fuse(vec_top, bm_top)
        ranked = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)
        ranked = ranked[: args.evidence_k]

        evidence: List[Tuple[int, str, str]] = []
        for local_idx in ranked:
            ev = train_examples[local_idx]
            evidence.append((train_set[local_idx], ev.student, ev.label))

        # Candidates derived ONLY from evidence
        cand = []
        for _, _, lab in evidence:
            if lab not in cand:
                cand.append(lab)
        cand = cand[: args.candidate_cap]
        return cand, evidence

    def run_diagnoser(ex: Example, candidates: List[str], evidence: List[Tuple[int, str, str]], seed: int) -> Dict[str, Any]:
        prompt = _prompt_diagnoser(ex, candidates, evidence, args.rank_k)
        raw = _ollama_generate(
            args.diagnoser_model,
            prompt,
            temperature=args.temperature,
            seed=seed,
            max_tokens=args.max_tokens,
            keep_alive=args.keep_alive,
        )
        out = _safe_json_loads(raw) or {}
        return out

    def run_judge(ex: Example, candidates: List[str], evidence: List[Tuple[int, str, str]], diag: Dict[str, Any], seed: int) -> Dict[str, Any]:
        prompt = _prompt_judge(ex, candidates, evidence, diag)
        raw = _ollama_generate(
            args.judge_model,
            prompt,
            temperature=0.0,
            seed=seed,
            max_tokens=192,
            keep_alive=args.keep_alive,
        )
        return _safe_json_loads(raw) or {}

    # Baselines: majority, nearest neighbor label, retrieval-vote label.
    label_counts = pd.Series([e.label for e in train_examples]).value_counts()
    majority_label = str(label_counts.idxmax())

    def nn_predict(ex: Example) -> str:
        q_text = _format_query(ex)
        q_emb = np.asarray(embedder.encode([q_text])[0], dtype=np.float32)
        idx = _cosine_topk(q_emb, train_emb, 1)[0]
        return train_examples[idx].label

    def retrieval_vote(ex: Example) -> str:
        cand, evidence = retrieve(ex)
        # Vote by evidence frequency; tie-break by first occurrence.
        freq: Dict[str, int] = {}
        for _, _, lab in evidence:
            freq[lab] = freq.get(lab, 0) + 1
        return max(freq.keys(), key=lambda l: (freq[l], -cand.index(l)))

    # Evaluation
    top1 = 0
    topk = 0
    violations = 0
    judge_schema_ok = 0
    judge_grounded = 0

    base_major = 0
    base_nn = 0
    base_vote = 0

    stability_agree = 0
    stability_total = 0

    print("[EVAL] running diagnoser + judge...")
    t0 = time.time()

    for i, ex in enumerate(test_examples, start=1):
        gold = ex.label

        if majority_label == gold:
            base_major += 1
        if nn_predict(ex) == gold:
            base_nn += 1
        if retrieval_vote(ex) == gold:
            base_vote += 1

        candidates, evidence = retrieve(ex)
        if not candidates:
            violations += 1
            continue

        diag = run_diagnoser(ex, candidates, evidence, seed=args.seed + i)

        ranked = [x for x in _as_list(diag.get("ranked_labels")) if isinstance(x, str)]
        selected = diag.get("selected_label") if isinstance(diag.get("selected_label"), str) else (ranked[0] if ranked else "")

        # Constraint checks
        schema_ok = isinstance(diag, dict) and ("selected_label" in diag or ranked)
        label_ok = isinstance(selected, str) and selected in candidates
        if not (schema_ok and label_ok):
            violations += 1
        else:
            if selected == gold:
                top1 += 1
            if gold in ranked[: args.rank_k]:
                topk += 1

        if i <= args.n_judge:
            judge = run_judge(ex, candidates, evidence, diag, seed=args.seed + 10_000 + i)
            if judge.get("schema_ok") is True and judge.get("label_in_candidates") is True:
                judge_schema_ok += 1
            if judge.get("grounded") is True:
                judge_grounded += 1

        # Stability on a subset
        if stability_total < args.n_stability:
            diag2 = run_diagnoser(ex, candidates, evidence, seed=args.seed + 20_000 + i)
            diag3 = run_diagnoser(ex, candidates, evidence, seed=args.seed + 30_000 + i)

            def _sel(d: Dict[str, Any]) -> str:
                rr = [x for x in _as_list(d.get("ranked_labels")) if isinstance(x, str)]
                s = d.get("selected_label") if isinstance(d.get("selected_label"), str) else (rr[0] if rr else "")
                return s

            s1, s2, s3 = _sel(diag), _sel(diag2), _sel(diag3)
            stability_total += 1
            if s1 and (s1 == s2 == s3):
                stability_agree += 1

        if i % 25 == 0:
            print(f"  {i}/{len(test_examples)} done")

    dt = time.time() - t0

    def pct(x: int, n: int) -> float:
        return 100.0 * x / max(1, n)

    print("\n[RESULTS]")
    print(f"Top-1 accuracy: {top1}/{len(test_examples)} = {pct(top1, len(test_examples)):.2f}%")
    print(f"Top-{args.rank_k} accuracy: {topk}/{len(test_examples)} = {pct(topk, len(test_examples)):.2f}%")
    print(f"Violation rate: {violations}/{len(test_examples)} = {pct(violations, len(test_examples)):.2f}%")
    print(f"Judge schema+label OK rate: {judge_schema_ok}/{min(args.n_judge, len(test_examples))} = {pct(judge_schema_ok, min(args.n_judge, len(test_examples))):.2f}%")
    print(f"Judge grounded rate: {judge_grounded}/{min(args.n_judge, len(test_examples))} = {pct(judge_grounded, min(args.n_judge, len(test_examples))):.2f}%")
    print(f"Stability (3-run exact agreement): {stability_agree}/{stability_total} = {pct(stability_agree, stability_total):.2f}%")

    print("\n[BASELINES]")
    print(f"Majority-class: {base_major}/{len(test_examples)} = {pct(base_major, len(test_examples)):.2f}%")
    print(f"Nearest-neighbor (embed): {base_nn}/{len(test_examples)} = {pct(base_nn, len(test_examples)):.2f}%")
    print(f"Retrieval vote (hybrid): {base_vote}/{len(test_examples)} = {pct(base_vote, len(test_examples)):.2f}%")

    print(f"\n[TIME] {dt:.1f}s for {len(test_examples)} examples")


if __name__ == "__main__":
    main()
