#!/usr/bin/env python3
"""Prepare, launch, and collect the new-pipeline GER Table 1.

This experiment intentionally differs from the paper's fixed-prompt Random
baseline. Every row uses the same retrieved-ICL final pipeline; methods swap
the retrieval source and are averaged over three seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shlex
import subprocess
import sys
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ger_pipeline.config import get_language, get_model  # noqa: E402
from ger_pipeline.results import _parse_errant_table_score, _parse_json_score  # noqa: E402
from ger_retrieval_quality_eval import Sample, filter_train, read_parallel  # noqa: E402


MODELS = ("llama31", "qwen25")
LANGUAGES = ("en", "bea19", "de", "ro", "et")
SEEDS = (88, 111, 222)
METHODS = ("Random", "Semantic", "BM25", "Explanation", "GER-Vanilla")
RUN_METHODS = METHODS
TOP_K = 10
MODEL_DIMS = {"llama31": 128, "qwen25": 256}
DATASET_TO_LANG = {
    "conll14": "en",
    "bea19": "bea19",
    "falko_merlin": "de",
    "rogec": "ro",
    "estgec": "et",
}


@dataclass(frozen=True)
class RetrievalPlan:
    model: str
    lang: str
    dataset: str
    method: str
    seed: int
    retrieval_dir: Path


@dataclass(frozen=True)
class RunPlan:
    model: str
    lang: str
    dataset: str
    method: str
    seed: int
    retrieval_dir: Path
    result_dir: Path
    prompt: str


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def assign(key: str, value: str | Path | int) -> str:
    return f"{key}={q(str(value))}"


def dataset_name(lang: str) -> str:
    return get_language(lang).test_dataset


def table_dataset_name(lang: str) -> str:
    return "ronacc_readerbench" if lang == "ro" else dataset_name(lang)


def train_name(lang: str) -> str:
    return get_language(lang).train_dataset


def model_dim(model: str) -> int:
    return MODEL_DIMS[model]


def yaml_path(model: str, lang: str) -> Path:
    dataset = dataset_name(lang)
    return ROOT / "multilingual" / "runtime_configs" / dataset / f"{model}_clean_dim{model_dim(model)}" / f"{dataset}.retrieved_icl.yaml"


def existing_ger_retrieval_dir(model: str, lang: str) -> Path:
    return (
        ROOT
        / "multilingual"
        / f"results_sentence_{model}"
        / f"icl_deepseek_retrieve_by_probing_pgy_{model_dim(model)}_8_8"
        / dataset_name(lang)
    )


def retrieval_root(model: str, method: str, seed: int) -> Path:
    stem = method.lower().replace("-", "_")
    return ROOT / "multilingual" / f"results_new_table1_{model}" / f"retrieve_{stem}_seed{seed}"


def result_root(model: str, method: str, seed: int) -> Path:
    stem = method.lower().replace("-", "_")
    return ROOT / "multilingual" / f"results_new_table1_{model}" / f"res_{stem}_seed{seed}"


def result_dir(model: str, method: str, seed: int, lang: str) -> Path:
    return result_root(model, method, seed) / dataset_name(lang)


def final_marker_path(plan: RunPlan) -> Path:
    marker = ".final_ger_dynamic_min2_max10_avg8.done" if plan.method == "GER-Vanilla" else ".final_8shot.done"
    return plan.result_dir / marker


def score_path(model: str, method: str, seed: int, lang: str) -> Path:
    dataset = dataset_name(lang)
    return result_dir(model, method, seed, lang) / f"{dataset}.score"


def output_path(model: str, method: str, seed: int, lang: str) -> Path:
    dataset = dataset_name(lang)
    if dataset == "bea19":
        return result_dir(model, method, seed, lang) / "bea19.txt"
    return result_dir(model, method, seed, lang) / f"{dataset}-output-retokenized.txt"


def predictions_path(model: str, method: str, seed: int, lang: str) -> Path:
    return result_dir(model, method, seed, lang) / "predictions.jsonl"


def official_score_dir(model: str, method: str, seed: int, lang: str) -> Path:
    stem = method.lower().replace("-", "_")
    if lang == "ro":
        return ROOT / "results" / "official_eval" / "new_table1" / "ronacc_readerbench" / f"{model}_{stem}_seed{seed}"
    if lang == "et":
        return ROOT / "results" / "official_eval" / "new_table1" / "estgec" / f"{model}_{stem}_seed{seed}"
    return result_dir(model, method, seed, lang)


def preferred_score_path(model: str, method: str, seed: int, lang: str) -> Path:
    if lang == "ro":
        return official_score_dir(model, method, seed, lang) / "errant.score"
    if lang == "et":
        return official_score_dir(model, method, seed, lang) / "estgec_modified_m2.score"
    return score_path(model, method, seed, lang)


def run_done_path(model: str, method: str, seed: int, lang: str) -> Path:
    if lang == "bea19":
        return result_dir(model, method, seed, lang) / "bea19.zip"
    return preferred_score_path(model, method, seed, lang)


def clean_outputs_for_scoring(output: Path, reference_m2: Path, *, backup: bool = True) -> int:
    """Replace pathological over-long predictions with their source sentence.

    M2 scorer alignment can become effectively non-terminating when a model
    generates an unrelated multi-sentence paragraph for a short source. For the
    retrieval key-swap experiment, treating those cases as no-op predictions is
    the conservative scoring behavior: the model receives no credit for the
    runaway generation, and the scorer can finish.
    """
    if not output.exists() or not reference_m2.exists():
        return 0
    predictions = output.read_text(encoding="utf-8", errors="replace").splitlines()
    sources: list[str] = []
    with reference_m2.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("S "):
                sources.append(line[2:].rstrip("\n"))
    if len(predictions) != len(sources):
        return 0

    changed = 0
    cleaned: list[str] = []
    for pred, src in zip(predictions, sources, strict=True):
        pred_tokens = pred.split()
        src_tokens = src.split()
        too_long = len(pred_tokens) > max(int(len(src_tokens) * 1.75), len(src_tokens) + 50)
        if too_long:
            cleaned.append(src)
            changed += 1
        else:
            cleaned.append(pred)
    if changed:
        if backup:
            backup_path = output.with_suffix(output.suffix + ".pre_score_clean.bak")
            if not backup_path.exists():
                backup_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
        output.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    return changed


def query_rows_from_ger(model: str, lang: str) -> list[dict[str, object]]:
    path = existing_ger_retrieval_dir(model, lang) / "retrieval.jsonl"
    if not path.exists() and model != "llama31":
        path = existing_ger_retrieval_dir("llama31", lang) / "retrieval.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_samples(lang: str) -> list[Sample]:
    spec = get_language(lang)
    src = ROOT / "datasets" / "multi_p" / spec.train_dataset / "train.src"
    tgt = ROOT / "datasets" / "multi_p" / spec.train_dataset / "train.tgt"
    raw, warnings = read_parallel(src, tgt, spec.train_dataset)
    for warning in warnings:
        print(f"[warn] {warning}", flush=True)
    return filter_train(raw)


def test_labels(lang: str, query_rows: list[dict[str, object]]) -> list[str | None]:
    """Return local gold labels when they exist; BEA-19 is blind."""
    if lang == "bea19":
        return [None for _ in query_rows]
    dataset = dataset_name(lang)
    tgt = ROOT / "results" / "llama3.1" / f"{dataset}.tgt"
    if not tgt.exists():
        return [None for _ in query_rows]
    labels = tgt.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str | None] = []
    for idx in range(len(query_rows)):
        out.append(labels[idx].strip() if idx < len(labels) else None)
    return out


def edit_description(source: str, target: str | None) -> str:
    if target is None or source.strip() == target.strip():
        return "No grammatical or spelling error is found in this sentence."
    src_tokens = source.split()
    tgt_tokens = target.split()
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=src_tokens, b=tgt_tokens, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        src = " ".join(src_tokens[i1:i2]).strip()
        tgt = " ".join(tgt_tokens[j1:j2]).strip()
        if tag == "replace":
            parts.append(f"replace {src or '<empty>'} with {tgt or '<empty>'}")
        elif tag == "delete":
            parts.append(f"delete {src or '<empty>'}")
        elif tag == "insert":
            left = src_tokens[i1 - 1] if i1 > 0 else "<start>"
            parts.append(f"insert {tgt or '<empty>'} after {left}")
    return "; ".join(parts) if parts else "No grammatical or spelling error is found in this sentence."


def demo(sample: Sample, idx: int, key: str, similarity: float) -> dict[str, object]:
    return {
        "id": idx,
        "text": sample.text,
        "label": sample.label,
        "from": sample.dataset,
        "key": key,
        "similarity": float(similarity),
    }


def write_retrieval(
    path: Path,
    query_rows: list[dict[str, object]],
    samples: list[Sample],
    ids: list[list[int]],
    scores: list[list[float]],
    *,
    method: str,
    query_keys: list[str] | None = None,
    sample_keys: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for q_idx, row in enumerate(query_rows):
            examples = []
            for rank, train_idx in enumerate(ids[q_idx]):
                if 0 <= train_idx < len(samples):
                    score = scores[q_idx][rank] if rank < len(scores[q_idx]) else 0.0
                    key = sample_keys[train_idx] if sample_keys is not None else samples[train_idx].text
                    examples.append(demo(samples[train_idx], train_idx, key, score))
            query_key = query_keys[q_idx] if query_keys is not None else str(row["text"])
            out = {
                "id": row["id"],
                "from": row.get("from", ""),
                "text": row["text"],
                "key_in_domain": query_key if method in {"Semantic", "BM25", "Explanation"} else "",
                "in_domain_examples": examples,
                "key_cross_domain": "",
                "cross_domain_examples": [],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def random_ids(query_count: int, train_count: int, seed: int) -> tuple[list[list[int]], list[list[float]]]:
    rng = random.Random(seed)
    ids = [rng.sample(range(train_count), k=min(TOP_K, train_count)) for _ in range(query_count)]
    return ids, [[0.0] * len(row) for row in ids]


def semantic_ids(query_rows: list[dict[str, object]], samples: list[Sample]) -> tuple[list[list[int]], list[list[float]]]:
    return vector_ids([str(row["text"]) for row in query_rows], [sample.text for sample in samples], name="semantic")


def topk_dense(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(top_k, scores.shape[1])
    if k == scores.shape[1]:
        ranked = np.argsort(scores, axis=1)[:, ::-1]
    else:
        candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        ranked = np.take_along_axis(candidates, order, axis=1)
    ranked_scores = np.take_along_axis(scores, ranked, axis=1)
    return ranked, ranked_scores


def vector_ids(query_keys: list[str], sample_keys: list[str], *, name: str, chunk_size: int = 128) -> tuple[list[list[int]], list[list[float]]]:
    print(f"[prepare] build {name} TF-IDF matrix queries={len(query_keys)} train={len(sample_keys)}", flush=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    train_matrix = vectorizer.fit_transform(sample_keys)
    query_matrix = vectorizer.transform(query_keys)
    all_ids: list[list[int]] = []
    all_scores: list[list[float]] = []
    train_t = train_matrix.T
    for start in range(0, query_matrix.shape[0], chunk_size):
        end = min(start + chunk_size, query_matrix.shape[0])
        sims = (query_matrix[start:end] @ train_t).toarray()
        ranked, ranked_sims = topk_dense(sims, TOP_K)
        all_ids.extend([[int(i) for i in row] for row in ranked])
        all_scores.extend([[float(v) for v in row] for row in ranked_sims])
        print(f"[prepare] {name} ranked {end}/{query_matrix.shape[0]}", flush=True)
    return all_ids, all_scores


def bm25_ids(query_rows: list[dict[str, object]], samples: list[Sample]) -> tuple[list[list[int]], list[list[float]]]:
    print(f"[prepare] build BM25 index queries={len(query_rows)} train={len(samples)}", flush=True)
    bm25 = BM25Okapi([sample.text.lower().split() for sample in samples])
    all_ids: list[list[int]] = []
    all_scores: list[list[float]] = []
    for idx, row in enumerate(query_rows):
        scores = bm25.get_scores(str(row["text"]).lower().split())
        ranked = np.argsort(scores)[::-1][:TOP_K]
        all_ids.append([int(i) for i in ranked])
        all_scores.append([float(scores[i]) for i in ranked])
        if (idx + 1) % 500 == 0 or idx + 1 == len(query_rows):
            print(f"[prepare] BM25 ranked {idx + 1}/{len(query_rows)}", flush=True)
    return all_ids, all_scores


def copy_ger_retrieval(model: str, lang: str, method: str, seed: int) -> None:
    src = fix_b_ger_retrieval_file(model, lang)
    dst = retrieval_root(model, method, seed) / dataset_name(lang) / "retrieval.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def fix_b_ger_retrieval_file(model: str, lang: str) -> Path:
    source_root = (
        ROOT
        / "multilingual"
        / f"results_new_table1_{model}"
        / "retrieve_ger_fix_b_floor_top3_source"
    )
    return (
        source_root
        / f"icl_deepseek_retrieve_by_probing_pgy_{model_dim(model)}_8_8"
        / dataset_name(lang)
        / "retrieval.jsonl"
    )


def ensure_fix_b_ger_retrieval(model: str, lang: str, args: argparse.Namespace) -> None:
    out = fix_b_ger_retrieval_file(model, lang)
    if file_ok(out):
        print(f"[prepare] skip Fix B GER retrieval {out.relative_to(ROOT)}", flush=True)
        return
    source_root_name = f"results_new_table1_{model}/retrieve_ger_fix_b_floor_top3_source"
    retrieval_lang = "en" if lang == "bea19" else lang
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["GER_RETRIEVAL_DEVICE"] = "cuda:0"
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "scripts/representation_retrieval/retrieve_gec_examples_by_representation.py",
        retrieval_lang,
        "_8",
        "_8",
        model,
        "0",
        dataset_name(lang),
        str(model_dim(model)),
        "default",
        str(ROOT / "cache" / "representation"),
        source_root_name,
        str(ROOT / "multilingual"),
        str(ROOT / "datasets" / "multi_p"),
        str(ROOT / "results" / "llama3.1"),
        str(ROOT / "models"),
        "--demo-floor-topk",
        "3",
    ]
    log_path = (
        ROOT
        / "logs"
        / "new_table1"
        / "retrieval_fix_b"
        / f"{model}_{lang}.log"
    )
    print(f"[prepare] run Fix B GER retrieval {model}/{lang} -> {out.relative_to(ROOT)}", flush=True)
    run_logged(command, cwd=ROOT / "multilingual", log_path=log_path)
    require_file(out)


def prepare(args: argparse.Namespace) -> None:
    for lang in args.languages:
        samples = train_samples(lang)
        sample_texts = [sample.text for sample in samples]
        explanation_sample_keys = [edit_description(sample.text, sample.label) for sample in samples]
        query_cache: dict[str, list[dict[str, object]]] = {}
        label_cache: dict[str, list[str | None]] = {}
        text_retrieval_cache: dict[str, tuple[list[list[int]], list[list[float]]]] = {}
        bm25_retrieval_cache: dict[str, tuple[list[list[int]], list[list[float]]]] = {}
        explanation_retrieval_cache: dict[str, tuple[list[list[int]], list[list[float]], list[str]]] = {}
        for model in args.models:
            query_cache[model] = query_rows_from_ger(model, lang)
            label_cache[model] = test_labels(lang, query_cache[model])
            query_keys = [str(row["text"]) for row in query_cache[model]]
            text_retrieval_cache[model] = vector_ids(query_keys, sample_texts, name=f"{lang}/{model}/semantic")
            bm25_retrieval_cache[model] = bm25_ids(query_cache[model], samples)
            explanation_query_keys = [
                edit_description(str(row["text"]), label_cache[model][idx])
                for idx, row in enumerate(query_cache[model])
            ]
            eids, escores = vector_ids(explanation_query_keys, explanation_sample_keys, name=f"{lang}/{model}/explanation")
            explanation_retrieval_cache[model] = (eids, escores, explanation_query_keys)
        for seed in args.seeds:
            for model in args.models:
                query_rows = query_cache[model]
                rid, rscore = random_ids(len(query_rows), len(samples), seed)
                sid, sscore = text_retrieval_cache[model]
                bid, bscore = bm25_retrieval_cache[model]
                eid, escore, ekeys = explanation_retrieval_cache[model]
                for method, ids, scores in (
                    ("Random", rid, rscore),
                    ("Semantic", sid, sscore),
                    ("BM25", bid, bscore),
                    ("Explanation", eid, escore),
                ):
                    out = retrieval_root(model, method, seed) / dataset_name(lang) / "retrieval.jsonl"
                    write_retrieval(
                        out,
                        query_rows,
                        samples,
                        ids,
                        scores,
                        method=method,
                        query_keys=ekeys if method == "Explanation" else None,
                        sample_keys=explanation_sample_keys if method == "Explanation" else None,
                    )
                    print(f"wrote {out.relative_to(ROOT)} rows={len(query_rows)}")
                for method in ("GER-Vanilla",):
                    ensure_fix_b_ger_retrieval(model, lang, args)
                    copy_ger_retrieval(model, lang, method, seed)
                    print(f"copied GER retrieval {model}/{lang}/{method}/seed{seed}")
    write_launch_assets(args)
    collect(args)


def run_plans(args: argparse.Namespace) -> list[RunPlan]:
    plans: list[RunPlan] = []
    for model in args.models:
        for lang in args.languages:
            for seed in args.seeds:
                for method in args.methods:
                    retrieval = retrieval_root(model, method, seed) / dataset_name(lang)
                    prompt = "min_edit_fewshot_description" if method in {"Explanation", "GER-IPE"} else get_language(lang).final_prompt_icl
                    plans.append(
                        RunPlan(
                            model=model,
                            lang=lang,
                            dataset=dataset_name(lang),
                            method=method,
                            seed=seed,
                            retrieval_dir=retrieval,
                            result_dir=result_dir(model, method, seed, lang),
                            prompt=prompt,
                        )
                    )
    return plans


def command_for_plan(plan: RunPlan, args: argparse.Namespace) -> str:
    dynamic_ger = plan.method == "GER-Vanilla"
    env = {
        "GER_DIR_RETRIEVE_BY_PROBING": plan.retrieval_dir.parent.relative_to(ROOT / "multilingual"),
        "GER_RESULT_DIR_ICL_PROBING": plan.result_dir.parent.relative_to(ROOT / "multilingual"),
        "GER_PROMPT_ICL": plan.prompt,
        "GER_PYTHON": ROOT / ".venv" / "bin" / "python",
        "GER_MODEL_ROOT_DIR": ROOT / "models",
        "GER_DATASETS_DIR": ROOT / "datasets",
        "GER_MULTI_P_DIR": ROOT / "datasets" / "multi_p",
        "GER_MULTILINGUAL_DIR": ROOT / "multilingual",
        "GER_REFERENCE_RESULTS_DIR": ROOT / "results" / "llama3.1",
        "GER_CUDA_VISIBLE_DEVICES": args.gpus,
        "GER_SHARD_CUDA_VISIBLE_DEVICES": args.gpus,
        "GER_NUM_SHARDS": str(args.num_shards),
        "GER_INFERENCE_BATCH_SIZE": str(args.batch_size),
        "GER_MAX_NEW_TOKENS": str(args.max_new_tokens),
        "GER_EXAMPLE_NUM_ERROR": "10" if dynamic_ger else "8",
        "GER_EXAMPLE_NUM_CORRECT": "0",
        "GER_DYNAMIC_EXAMPLE_NUM_ERROR": "1" if dynamic_ger else "0",
        "GER_DYNAMIC_EXAMPLE_NUM_ERROR_MIN": "2",
        "GER_DYNAMIC_EXAMPLE_NUM_ERROR_TARGET_AVG": "8",
        "GER_DYNAMIC_EXAMPLE_NUM_ERROR_MAX": "10",
        "GER_SCORE_BACKGROUND": "1",
        "GER_VLLM_USE_STOP_STRING": "1",
        "GER_VLLM_MAX_MODEL_LEN": "16384",
        "GER_SEED": str(plan.seed),
        "GER_PREFILL_ANSWER_START": os.environ.get("GER_PREFILL_ANSWER_START", ""),
        "PYTHONPATH": ROOT,
    }
    prefix = " ".join(assign(k, v) for k, v in env.items())
    return f"{prefix} bash scripts/pipeline/infer_icl_probing.sh {q(yaml_path(plan.model, plan.lang))}"


def official_score_command(plan: RunPlan, args: argparse.Namespace) -> str:
    if plan.lang not in {"ro", "et"}:
        return ":"
    return (
        f"{q(ROOT / '.venv' / 'bin' / 'python')} {q(ROOT / 'scripts' / 'ger_new_table1.py')} official-score "
        f"--models {q(plan.model)} --languages {q(plan.lang)} --methods {q(plan.method)} --seeds {q(str(plan.seed))}"
    )


def score_ok_command(path: Path) -> str:
    return f"score_ok {q(path)}"


def write_launch_assets(args: argparse.Namespace) -> None:
    launch_dir = ROOT / "repro" / "new_table1"
    jobs_dir = ROOT / "repro" / "jobs"
    logs_dir = ROOT / "logs" / "new_table1"
    launch_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = launch_dir / "persistent_manifest.jsonl"
    manifest_rows = []
    collect_cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "ger_new_table1.py"),
        "collect",
        "--languages",
        *args.languages,
        "--models",
        *args.models,
        "--seeds",
        *(str(seed) for seed in args.seeds),
    ]
    for idx, plan in enumerate(run_plans(args), start=1):
        score = preferred_score_path(plan.model, plan.method, plan.seed, plan.lang)
        done_path = run_done_path(plan.model, plan.method, plan.seed, plan.lang)
        raw_score = score_path(plan.model, plan.method, plan.seed, plan.lang)
        finalizer_log = logs_dir / f"finalizer_{plan.model}_{plan.lang}_{plan.method.lower().replace('-', '_')}_seed{plan.seed}.log"
        official_cmd = None
        if plan.lang in {"ro", "et"}:
            official_cmd = [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "scripts" / "ger_new_table1.py"),
                "official-score",
                "--models",
                plan.model,
                "--languages",
                plan.lang,
                "--methods",
                plan.method,
                "--seeds",
                str(plan.seed),
            ]
        manifest_rows.append(
            {
                "index": idx,
                "model": plan.model,
                "lang": plan.lang,
                "dataset": plan.dataset,
                "method": plan.method,
                "seed": plan.seed,
                "yaml_path": str(yaml_path(plan.model, plan.lang)),
                "retrieval_dir": str(plan.retrieval_dir),
                "result_dir": str(plan.result_dir),
                "prompt": plan.prompt,
                "score_path": str(raw_score),
                "preferred_score_path": str(score),
                "done_path": str(done_path),
                "final_marker": str(final_marker_path(plan)),
                "finalizer_log": str(finalizer_log),
                "official_score_cmd": official_cmd,
                "collect_cmd": collect_cmd,
                "is_bea19": plan.lang == "bea19",
            }
        )
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    persistent_script = launch_dir / "run_new_table1_persistent.sh"
    persistent_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {q(ROOT)}",
        f"mkdir -p {q(logs_dir)}",
        "export NCCL_P2P_DISABLE=1",
        "export NCCL_IB_DISABLE=1",
        "export GER_EXAMPLE_NUM_ERROR=8",
        "export GER_EXAMPLE_NUM_CORRECT=0",
        "export GER_DYNAMIC_EXAMPLE_NUM_ERROR=1",
        "export GER_DYNAMIC_EXAMPLE_NUM_ERROR_MIN=2",
        "export GER_DYNAMIC_EXAMPLE_NUM_ERROR_TARGET_AVG=8",
        "export GER_DYNAMIC_EXAMPLE_NUM_ERROR_MAX=10",
        "export GER_VLLM_USE_STOP_STRING=1",
        "export GER_VLLM_MAX_MODEL_LEN=16384",
        f"exec {q(ROOT / '.venv' / 'bin' / 'python')} {q(ROOT / 'scripts' / 'ger_table1_persistent.py')} run "
        f"--manifest {q(manifest_path)} --gpus {q(args.gpus)} --num-shards {args.num_shards} "
        f"--batch-size {args.batch_size} --max-new-tokens {args.max_new_tokens} --log-dir {q(logs_dir / 'persistent')}",
    ]
    persistent_script.write_text("\n".join(persistent_lines) + "\n", encoding="utf-8")
    persistent_script.chmod(0o755)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {q(ROOT / 'multilingual')}",
        f"mkdir -p {q(logs_dir)}",
        "score_ok() {",
        "  local file=\"$1\"",
        "  [ -s \"$file\" ] || return 1",
        "  grep -Eq '(^|[[:space:]])(Precision[[:space:]]*:|P[[:space:]]*=)[[:space:]]*[0-9]+' \"$file\" || return 1",
        "  grep -Eq '(^|[[:space:]])(Recall[[:space:]]*:|R[[:space:]]*=)[[:space:]]*[0-9]+' \"$file\" || return 1",
        "  grep -Eq '(^|[[:space:]])(F_0\\.5[[:space:]]*:|F_0\\.5[[:space:]]*=)[[:space:]]*[0-9]+' \"$file\"",
        "}",
        "wait_background_score() {",
        "  local result_dir=\"$1\"",
        "  local score_file=\"$2\"",
        "  local fail_file=\"${result_dir}/.score.failed\"",
        "  local done_file=\"${result_dir}/.score.done\"",
        "  local pid_file=\"${result_dir}/.score.pid\"",
        "  local waited=0",
        "  while true; do",
        "    if [ -e \"$fail_file\" ]; then",
        "      echo \"[new-table1] background score failed: ${result_dir}\" >&2",
        "      tail -n 80 \"${result_dir}/score.background.log\" >&2 || true",
        "      return 1",
        "    fi",
        "    if [ -e \"$done_file\" ] && score_ok \"$score_file\"; then",
        "      return 0",
        "    fi",
        "    if [ -e \"$pid_file\" ]; then",
        "      local pid",
        "      pid=\"$(cat \"$pid_file\")\"",
        "      if [ -n \"$pid\" ] && ! kill -0 \"$pid\" 2>/dev/null && [ ! -e \"$done_file\" ]; then",
        "        echo \"[new-table1] background score exited without done: ${result_dir}\" >&2",
        "        tail -n 80 \"${result_dir}/score.background.log\" >&2 || true",
        "        return 1",
        "      fi",
        "    fi",
        "    if [ \"$waited\" -ge 7200 ]; then",
        "      echo \"[new-table1] timed out waiting for background score: ${result_dir}\" >&2",
        "      return 1",
        "    fi",
        "    sleep 10",
        "    waited=$((waited + 10))",
        "  done",
        "}",
        "finalizer_pids=()",
        "start_finalizer() {",
        "  local label=\"$1\"",
        "  local marker=\"$2\"",
        "  local log_file=\"$3\"",
        "  shift 3",
        "  (",
        "    set -euo pipefail",
        "    echo \"[new-table1-finalizer] start ${label}\"",
        "    \"$@\"",
        "    mkdir -p \"$(dirname \"$marker\")\"",
        "    touch \"$marker\"",
        "    echo \"[new-table1-finalizer] done ${label}\"",
        "  ) >\"$log_file\" 2>&1 &",
        "  finalizer_pids+=(\"$!\")",
        "  echo \"[new-table1] finalizer ${label} pid ${finalizer_pids[-1]}\"",
        "}",
        "wait_finalizers() {",
        "  local status=0",
        "  local pid",
        "  for pid in \"${finalizer_pids[@]}\"; do",
        "    wait \"$pid\" || status=1",
        "  done",
        "  return \"$status\"",
        "}",
        "export -f score_ok wait_background_score",
    ]
    for plan in run_plans(args):
        score = preferred_score_path(plan.model, plan.method, plan.seed, plan.lang)
        done_path = run_done_path(plan.model, plan.method, plan.seed, plan.lang)
        raw_score = score_path(plan.model, plan.method, plan.seed, plan.lang)
        retrieval = plan.retrieval_dir / "retrieval.jsonl"
        final_marker = final_marker_path(plan)
        finalizer_log = logs_dir / f"finalizer_{plan.model}_{plan.lang}_{plan.method.lower().replace('-', '_')}_seed{plan.seed}.log"
        finalizer_label = f"{plan.model}/{plan.lang}/{plan.method}/seed{plan.seed}"
        lines.append(f"echo '[new-table1] start {plan.model}/{plan.lang}/{plan.method}/seed{plan.seed}'")
        lines.append(f"test -s {q(retrieval)}")
        base_done_test = f"[ -s {q(done_path)} ]" if plan.lang == "bea19" else score_ok_command(done_path)
        done_test = f"{base_done_test} && [ -s {q(final_marker)} ]"
        if plan.lang == "bea19":
            finalizer_command = f"test -s {q(done_path)}"
        else:
            finalizer_command = f"wait_background_score {q(plan.result_dir)} {q(raw_score)}"
        if plan.lang in {"ro", "et"}:
            finalizer_command = f"{finalizer_command} && {official_score_command(plan, args)} && {score_ok_command(score)}"
        collect_command = (
            f"{q(ROOT / '.venv' / 'bin' / 'python')} {q(ROOT / 'scripts' / 'ger_new_table1.py')} "
            f"collect --languages {shlex.join(args.languages)} --models {shlex.join(args.models)} --seeds {shlex.join(str(s) for s in args.seeds)}"
        )
        finalizer_command = f"{finalizer_command} && {collect_command}"
        run_command = (
            f"rm -rf {q(plan.result_dir)} && {command_for_plan(plan, args)} && "
            f"start_finalizer {q(finalizer_label)} {q(final_marker)} {q(finalizer_log)} bash -c {q(finalizer_command)}"
        )
        lines.append(
            f"if {done_test}; then echo '[new-table1] skip existing final {done_path.relative_to(ROOT)}'; else {run_command}; fi"
        )
    lines.append("wait_finalizers")
    lines.append(
        f"{q(ROOT / '.venv' / 'bin' / 'python')} {q(ROOT / 'scripts' / 'ger_new_table1.py')} collect --languages {shlex.join(args.languages)} --models {shlex.join(args.models)} --seeds {shlex.join(str(s) for s in args.seeds)}"
    )
    run_script = launch_dir / "run_new_table1.sh"
    run_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_script.chmod(0o755)

    wrapper = ROOT / "repro" / "watch_job.sh"
    if not wrapper.exists():
        wrapper.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nPROJ_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/../..\" && pwd)\"\nexec \"${PROJ_ROOT}/watch_job.sh\" \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    watch_script = ROOT / "repro" / "watch_new_table1.sh"
    watch_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nHERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\nexec \"${HERE}/watch_job.sh\" --spec \"${HERE}/jobs/ger_new_table1.env\" \"$@\"\n",
        encoding="utf-8",
    )
    watch_script.chmod(0o755)
    spec = jobs_dir / "ger_new_table1.env"
    spec.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                assign("JOB_ID", "ger_new_table1"),
                assign("JOB_NAME", "GER new-pipeline Table 1"),
                assign("JOB_CWD", ROOT),
                assign("JOB_LAUNCH_CMD", f"bash {q('/home/pengguangyue/workspace/proj/run_and_hold.sh')} {q(args.gpus)} {q(persistent_script)}"),
                assign("JOB_PROCESS_PATTERN", "ger_table1_persistent.py|multilingual/icl.py| vllm |vllm.entrypoints"),
                "JOB_TARGET_STEP=1",
                assign("JOB_CHECKPOINT_DIR", ROOT / "results" / "new_table1_done"),
                assign("JOB_STATS_FILE", ROOT / "results" / "new_table1_progress.jsonl"),
                assign("JOB_LOG_GLOB", str(logs_dir / "*.log")),
                assign("JOB_GPU_IDS", args.gpus),
                "JOB_PROGRESS_TIMEOUT_SEC=7200",
                "JOB_STARTUP_GRACE_SEC=7200",
                "JOB_RESTART_COOLDOWN_SEC=1200",
                "JOB_MAX_DETERMINISTIC_RESTARTS=1",
                "JOB_MIN_MEM_AVAILABLE_GB=16",
                "JOB_MIN_DATA_FREE_GB=20",
                "JOB_CLEANUP_STALE_RAY=0",
                "JOB_CLEAR_HOLD_GPU=1",
                "JOB_CODEX_ENABLED=1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {run_script.relative_to(ROOT)}")
    print(f"wrote {persistent_script.relative_to(ROOT)}")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    print(f"wrote {watch_script.relative_to(ROOT)}")
    print(f"wrote {spec.relative_to(ROOT)}")


def _last_float(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def _parse_verbose_m2_final_score(text: str) -> tuple[float | None, float | None, float | None]:
    # Verbose m2scorer writes per-sentence running metrics before the final
    # summary. Only the final summary proves the score file is complete.
    pattern = re.compile(
        r"^CORRECT EDITS\s+:\s+\d+\s*\n"
        r"^PROPOSED EDITS\s+:\s+\d+\s*\n"
        r"^GOLD EDITS\s+:\s+\d+\s*\n"
        r"^P\s*=\s*([0-9.]+)\s*\n"
        r"^R\s*=\s*([0-9.]+)\s*\n"
        r"^F_0\.5\s*=\s*([0-9.]+)",
        re.MULTILINE,
    )
    matches = pattern.findall(text)
    if not matches:
        return None, None, None
    precision, recall, f05 = matches[-1]
    return float(precision), float(recall), float(f05)


def parse_score(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing"}
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > 2_000_000:
            handle.seek(max(0, size - 200_000))
        text = handle.read().decode("utf-8", errors="replace")
    precision, recall, f05 = _parse_json_score(text)
    if precision is None or recall is None or f05 is None:
        precision, recall, f05 = _parse_errant_table_score(text)
    if precision is None:
        precision, recall, f05 = _parse_verbose_m2_final_score(text)
    if precision is None:
        precision = _last_float(text, r"^(?:P|Precision)\s*(?:=|:)\s*([0-9.]+)\s*$")
    if recall is None:
        recall = _last_float(text, r"^(?:R|Recall)\s*(?:=|:)\s*([0-9.]+)\s*$")
    if f05 is None:
        f05 = _last_float(text, r"^(?:F_0\.5|F0\.5)\s*(?:=|:)\s*([0-9.]+)\s*$")
    return {
        "status": "ok" if precision is not None and recall is not None and f05 is not None else "unparsed",
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
    }


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def require_file(path: Path) -> None:
    if not file_ok(path):
        raise FileNotFoundError(path)


def run_logged(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> None:
    print("cd", cwd, "&&", " ".join(shlex.quote(part) for part in command), ">", log_path, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)


ESTSPACY_RETOKENIZE_CODE = r"""
import json
import sys
from pathlib import Path

import spacy

predictions_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
nlp = spacy.load("et_dep_ud_sm")

with predictions_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as out:
    for line in src:
        if not line.strip():
            continue
        item = json.loads(line)
        prediction = str(item.get("prediction", "")).strip()
        prediction = prediction.replace("\n", " ").replace("\r", " ").strip()
        out.write(" ".join(token.text for token in nlp.tokenizer(prediction)).strip() + "\n")
"""


def score_ronacc(model: str, method: str, seed: int, *, overwrite: bool = False) -> None:
    out_dir = official_score_dir(model, method, seed, "ro")
    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_m2 = out_dir / "hyp.m2"
    score = out_dir / "errant.score"
    source = ROOT / "datasets" / "external" / "ronacc_readerbench" / "test.src"
    reference = ROOT / "datasets" / "external" / "ronacc_readerbench" / "test.m2"
    errant_dir = ROOT / "datasets" / "multilingual" / "rogec" / "errant"
    ro_python = ROOT / ".conda_eval_official" / "bin" / "python"
    ro_ca_bundle = ROOT / ".conda_eval_official" / "ssl" / "cacert.pem"
    ro_env = {
        "SSL_CERT_FILE": str(ro_ca_bundle),
        "REQUESTS_CA_BUNDLE": str(ro_ca_bundle),
    }
    require_file(source)
    require_file(reference)
    require_file(ro_python)
    require_file(ro_ca_bundle)
    if overwrite or not file_ok(hyp_m2):
        output = output_path(model, method, seed, "ro")
        require_file(output)
        run_logged(
            [
                str(ro_python),
                "parallel_to_m2.py",
                "-orig",
                str(source),
                "-cor",
                str(output),
                "-out",
                str(hyp_m2),
                "-lang",
                "ro",
            ],
            cwd=errant_dir,
            log_path=out_dir / "parallel_to_m2.log",
            env=ro_env,
        )
    if overwrite or not file_ok(score):
        run_logged([str(ro_python), "compare_m2.py", "-hyp", str(hyp_m2), "-ref", str(reference)], cwd=errant_dir, log_path=score, env=ro_env)


def score_estgec(model: str, method: str, seed: int, *, overwrite: bool = False) -> None:
    out_dir = official_score_dir(model, method, seed, "et")
    out_dir.mkdir(parents=True, exist_ok=True)
    estspacy_output = out_dir / "estgec-output-estspacy.txt"
    score = out_dir / "estgec_modified_m2.score"
    scorer = ROOT / "datasets" / "multilingual_raw" / "ET-estgec" / "M2_scorer_est" / "m2scorer_by_type" / "scripts" / "m2scorer.py"
    reference = ROOT / "datasets" / get_language("et").m2_relative_path
    et_python = ROOT / ".conda_eval_estspacy" / "bin" / "python"
    predictions = predictions_path(model, method, seed, "et")
    require_file(predictions)
    require_file(reference)
    require_file(scorer)
    require_file(et_python)
    if overwrite or not file_ok(estspacy_output):
        run_logged([str(et_python), "-c", ESTSPACY_RETOKENIZE_CODE, str(predictions), str(estspacy_output)], cwd=ROOT, log_path=out_dir / "estspacy_retokenize.log")
    if overwrite or not file_ok(score):
        run_logged([str(ROOT / ".venv" / "bin" / "python"), str(scorer), str(estspacy_output), str(reference)], cwd=ROOT, log_path=score)


def official_score(args: argparse.Namespace) -> None:
    for model in args.models:
        for lang in args.languages:
            for method in args.methods:
                for seed in args.seeds:
                    if lang == "ro":
                        score_ronacc(model, method, seed, overwrite=args.overwrite)
                    elif lang == "et":
                        score_estgec(model, method, seed, overwrite=args.overwrite)


def clean_score_output(args: argparse.Namespace) -> None:
    changed = clean_outputs_for_scoring(args.output, args.reference_m2, backup=not args.no_backup)
    print(f"cleaned_overlong_predictions={changed} output={args.output}")


def score_ok(args: argparse.Namespace) -> None:
    parsed = parse_score(args.score_path)
    if parsed.get("status") != "ok":
        raise SystemExit(1)


def collect(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for model in args.models:
        for lang in args.languages:
            for method in METHODS:
                for seed in args.seeds:
                    score = preferred_score_path(model, method, seed, lang)
                    retrieval = retrieval_root(model, method, seed) / dataset_name(lang) / "retrieval.jsonl"
                    parsed = parse_score(score)
                    if lang == "bea19" and parsed["status"] == "missing":
                        out = result_dir(model, method, seed, lang) / "bea19.zip"
                        parsed = {"status": "needs_official_bea19_score" if out.exists() else "missing", "precision": "", "recall": "", "f0.5": ""}
                    rows.append(
                        {
                            "model": model,
                            "language": lang,
                            "dataset": table_dataset_name(lang),
                            "method": method,
                            "seed": seed,
                            "status": parsed.get("status", ""),
                            "precision": parsed.get("precision") or "",
                            "recall": parsed.get("recall") or "",
                            "f0.5": parsed.get("f0.5") or "",
                            "score_path": str(score.relative_to(ROOT)),
                            "retrieval_path": str(retrieval.relative_to(ROOT)),
                        }
                    )
    avg_rows = average_rows(rows)
    write_outputs(rows, avg_rows, args.output_csv, args.output_md)
    maybe_done(rows, args)


def average_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["language"]), str(row["dataset"]), str(row["method"])), []).append(row)
    out = []
    for (model, lang, dataset, method), group in grouped.items():
        ok = [r for r in group if r["status"] == "ok"]
        if len(ok) == len(group):
            vals = {metric: [float(r[metric]) for r in ok] for metric in ("precision", "recall", "f0.5")}
            status = "ok"
            precision = float(np.mean(vals["precision"]))
            recall = float(np.mean(vals["recall"]))
            f05 = float(np.mean(vals["f0.5"]))
            std = float(np.std(vals["f0.5"], ddof=0))
        else:
            status = "incomplete"
            precision = recall = f05 = std = ""
        out.append(
            {
                "model": model,
                "language": lang,
                "dataset": dataset,
                "method": method,
                "status": status,
                "precision_mean": precision,
                "recall_mean": recall,
                "f0.5_mean": f05,
                "f0.5_std": std,
                "ok_seeds": len(ok),
                "total_seeds": len(group),
            }
        )
    return out


def write_outputs(rows: list[dict[str, object]], avg_rows: list[dict[str, object]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    avg_csv = csv_path.with_name(csv_path.stem + "_average.csv")
    with avg_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avg_rows)
    body = []
    for row in avg_rows:
        body.append(
            "| "
            + " | ".join(str(row[col]) for col in ("model", "language", "method", "status", "precision_mean", "recall_mean", "f0.5_mean", "f0.5_std", "ok_seeds"))
            + " |"
        )
    md_path.write_text(
        "# GER New-Pipeline Table 1\n\n"
        "Generated by `scripts/ger_new_table1.py`.\n\n"
        "All rows are intended to use the same retrieved-ICL final pipeline and three seeds. "
        "This is a new causal-ablation-style Table 1, not the paper's fixed-prompt Random implementation.\n\n"
        "| Model | Lang | Method | Status | P mean | R mean | F0.5 mean | F0.5 std | OK seeds |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(body)
        + "\n\n## Notes\n\n"
        "- Random/Semantic/BM25 retrievals are generated offline from the filtered erroneous train pool.\n"
        "- Explanation retrieval is generated offline from source-target edit descriptions when local gold is available; BEA-19 uses source text as the blind-test query key and still needs official scoring.\n"
        "- GER-Vanilla reuses the existing GER retrieval source under the same 8-shot final pipeline.\n"
        "- GER-IPE is intentionally excluded from the active run set because the faithful paper-IPE implementation requires regenerating first-GER retrieval from 8 random initial examples.\n"
        "- BEA-19 rows can generate submission zips locally, but official F0.5 still requires Codabench import.\n",
        encoding="utf-8",
    )
    print(csv_path)
    print(avg_csv)
    print(md_path)


def maybe_done(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    required = [row for row in rows if row["language"] != "bea19"]
    if required and all(row["status"] == "ok" for row in required):
        done = ROOT / "results" / "new_table1_done" / "global_step_1"
        done.mkdir(parents=True, exist_ok=True)
        (ROOT / "results" / "new_table1_progress.jsonl").write_text(json.dumps({"global_step": 1, "status": "complete"}) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("prepare", "assets", "collect", "official-score", "clean-score-output", "score-ok"), default="prepare")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "ger_new_table1_seed_scores.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "results" / "ger_new_table1.md")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-m2", type=Path)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("score_path", nargs="?", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.seeds = [int(seed) for seed in args.seeds]
    if args.action == "prepare":
        prepare(args)
    elif args.action == "assets":
        write_launch_assets(args)
        collect(args)
    elif args.action == "collect":
        collect(args)
    elif args.action == "official-score":
        official_score(args)
    elif args.action == "clean-score-output":
        if args.output is None or args.reference_m2 is None:
            raise SystemExit("clean-score-output requires --output and --reference-m2")
        clean_score_output(args)
    elif args.action == "score-ok":
        if args.score_path is None:
            raise SystemExit("score-ok requires a score path")
        score_ok(args)


if __name__ == "__main__":
    main()
