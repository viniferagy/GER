"""Retrieve in-context GEC examples by representation-space nearest neighbours.

Pipeline
--------
1. Load the cache produced by ``build_gec_representation_cache.py``.
2. Standard-scale + PCA-fit the cached training diff vectors.
3. Read test.src plus the initial-generation .txt / .label, parse the edit
   tags, build the same truncated pos x neg pairs as the builder did for
   training. Test-side gold .tgt / .label are intentionally not used.
4. Run RepE forward pass on the test pairs in data-parallel shards.
5. Project all (train + test) dedup'd diffs into PCA space.
6. Build a direct NumPy cosine retriever over PCA-projected vectors.
7. For each test sentence:
       - sort its error sites by |PC1|
       - retrieve TOP_K nearest training examples per site
       - add COR_EXP random correct training examples
8. Write retrieval.jsonl.

CLI::

    python retrieve_gec_examples_by_representation.py \\
        --lang en --train-suffix _8 --test-suffix _8 \\
        --model-name llama31 --gpus 0,1,2,3 --test conll14 \\
        --retrieve-dim 128 --cache-dir cache/representation \\
        --output-root multilingual/results_ger_llama31/retrieve_ger_source
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import _icl_examples
from _common import (
    LANG_RETRIEVAL_HP,
    LANG_TRAIN,
    MODEL_LAYER_INDEX,
    TorchPCA,
    build_truncated_pairs,
    collect_prompt_batches,
    dedup_by_line_and_diff,
    extract_pos_neg_hidden_states,
    get_model_name_or_path,
    load_model_and_tokenizer,
    load_tokenizer,
    parse_edit_ops,
    read_lines_unfiltered,
)

random.seed(0)


# =====================================================================
# CLI
# =====================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the retrieval CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lang", required=True)
    parser.add_argument("--train-suffix", required=True)
    parser.add_argument("--test-suffix", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated CUDA devices for retrieval shards.")
    parser.add_argument("--test", required=True)
    parser.add_argument("--retrieve-dim", required=True, type=int)
    parser.add_argument("--initial-result-mode", default="default")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--multilingual-dir", type=Path, default=None)
    parser.add_argument("--train-data-dir", type=Path, default=None)
    parser.add_argument("--test-source-file", type=Path, default=None)
    parser.add_argument("--model-root-dir", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--demo-floor-topk", type=int, default=0)
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in defaults for unset path / name args."""
    here = Path(__file__).resolve()
    ger_root = here.parents[3]
    args.cache_dir = args.cache_dir or here.parent / "cache"
    args.multilingual_dir = args.multilingual_dir or Path(
        os.environ.get("GER_MULTILINGUAL_DIR", ger_root / "multilingual")
    )
    args.output_root = args.output_root or args.multilingual_dir / f"retrieved_icl_{args.model_name}"
    train_data_dir = args.train_data_dir or os.environ.get("GER_TRAIN_DATA_DIR")
    if not train_data_dir:
        raise ValueError("--train-data-dir or GER_TRAIN_DATA_DIR is required")
    args.train_data_dir = Path(train_data_dir)
    args.test_source_file = args.test_source_file or Path(
        os.environ.get(
            "GER_TEST_SOURCE_FILE",
            ger_root / "datasets" / "multilingual" / args.test / "test.src",
        )
    )
    args.model_root_dir = args.model_root_dir or Path(
        os.environ.get("GER_MODEL_ROOT_DIR", ger_root / "models")
    )
    return args


def parse_gpu_list(gpus: str) -> list[str]:
    parsed = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    return parsed or ["0"]


def truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip() in {
        "1", "true", "True", "TRUE", "yes", "Yes", "YES",
    }


def balanced_prompt_ranges(
    rep_tokens_inputs: list[list[int]],
    num_shards: int,
) -> list[tuple[int, int]]:
    n_prompts = len(rep_tokens_inputs)
    if n_prompts == 0:
        return []
    num_shards = max(1, min(num_shards, n_prompts))
    weights = [max(1, len(tokens) // 2) for tokens in rep_tokens_inputs]
    total = sum(weights)
    target = max(1, (total + num_shards - 1) // num_shards)

    ranges: list[tuple[int, int]] = []
    start = 0
    acc = 0
    remaining_shards = num_shards
    for idx, weight in enumerate(weights):
        remaining_prompts = n_prompts - idx
        if (
            idx > start
            and acc >= target
            and remaining_shards > 1
            and remaining_prompts >= remaining_shards - 1
        ):
            ranges.append((start, idx))
            start = idx
            acc = 0
            remaining_shards -= 1
        acc += weight
    ranges.append((start, n_prompts))
    return ranges


def run_hidden_state_worker(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Internal GER retrieval hidden-state worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layer-idx", required=True, type=int)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    args = parser.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.shard_id != 0:
        os.environ["GER_REPE_DISABLE_PROGRESS"] = "1"
    elif not truthy_env("GER_REPE_DISABLE_PROGRESS"):
        os.environ["GER_REPE_PROGRESS_DESC"] = (
            f"retrieval hidden states shard {args.shard_id + 1}/{args.num_shards}"
        )

    with args.input_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    _, _, rep_reading_pipeline = load_model_and_tokenizer(args.model_path)
    hs = extract_pos_neg_hidden_states(
        rep_reading_pipeline,
        payload["data_inputs"],
        payload["rep_tokens_inputs"],
        layer_idx=args.layer_idx,
        hidden_layers=[args.layer_idx],
        desc=os.environ.get("GER_REPE_PROGRESS_DESC"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / f"shard_{args.shard_id:03d}_pos.npy", hs["pos"])
    np.save(args.output_dir / f"shard_{args.shard_id:03d}_neg.npy", hs["neg"])


def extract_hidden_states_parallel(
    *,
    model_path: str,
    data_inputs: list[str],
    rep_tokens_inputs: list[list[int]],
    layer_idx: int,
    gpus: str,
    num_shards: int,
    temp_root: Path,
    temp_prefix: str,
) -> dict[str, np.ndarray]:
    gpu_list = parse_gpu_list(gpus)
    ranges = balanced_prompt_ranges(rep_tokens_inputs, num_shards)
    if len(ranges) <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        _, _, rep_reading_pipeline = load_model_and_tokenizer(model_path)
        return extract_pos_neg_hidden_states(
            rep_reading_pipeline,
            data_inputs,
            rep_tokens_inputs,
            layer_idx=layer_idx,
            hidden_layers=[layer_idx],
            desc="retrieval hidden states",
        )

    print(
        "Run GER retrieval hidden states in data-parallel mode: "
        f"{len(ranges)} shards on GPUs {','.join(gpu_list)}"
    )
    disable_progress = truthy_env("GER_REPE_DISABLE_PROGRESS")
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{temp_prefix}_hidden_shards_", dir=str(temp_root)) as tmp:
        tmp_dir = Path(tmp)
        processes: list[tuple[int, subprocess.Popen]] = []
        try:
            for shard_id, (start, end) in enumerate(ranges):
                shard_input = tmp_dir / f"shard_{shard_id:03d}.json"
                with shard_input.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "data_inputs": data_inputs[start:end],
                            "rep_tokens_inputs": rep_tokens_inputs[start:end],
                        },
                        handle,
                        ensure_ascii=False,
                    )

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_list[shard_id % len(gpu_list)]
                if shard_id != 0 or disable_progress:
                    env["GER_REPE_DISABLE_PROGRESS"] = "1"
                else:
                    env.pop("GER_REPE_DISABLE_PROGRESS", None)
                    env["GER_REPE_PROGRESS_DESC"] = (
                        f"retrieval hidden states shard {shard_id + 1}/{len(ranges)}"
                    )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "__hidden_worker__",
                    "--model-path", model_path,
                    "--input-json", str(shard_input),
                    "--output-dir", str(tmp_dir),
                    "--layer-idx", str(layer_idx),
                    "--gpu", env["CUDA_VISIBLE_DEVICES"],
                    "--shard-id", str(shard_id),
                    "--num-shards", str(len(ranges)),
                ]
                print(
                    f"Launching retrieval hidden-state shard {shard_id + 1}/{len(ranges)} "
                    f"on GPU {env['CUDA_VISIBLE_DEVICES']} ({end - start} prompts)"
                )
                processes.append((shard_id, subprocess.Popen(command, env=env)))

            failed: list[tuple[int, int]] = []
            for shard_id, process in processes:
                status = process.wait()
                if status != 0:
                    failed.append((shard_id, status))
            if failed:
                raise RuntimeError(f"GER retrieval hidden-state shard(s) failed: {failed}")
        except KeyboardInterrupt:
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process in processes:
                if process.poll() is None:
                    process.kill()
            raise

        pos_parts = []
        neg_parts = []
        for shard_id in range(len(ranges)):
            pos_parts.append(np.load(tmp_dir / f"shard_{shard_id:03d}_pos.npy"))
            neg_parts.append(np.load(tmp_dir / f"shard_{shard_id:03d}_neg.npy"))
        return {
            "pos": np.concatenate(pos_parts, axis=0),
            "neg": np.concatenate(neg_parts, axis=0),
        }


# =====================================================================
# Cache loader
# =====================================================================

def load_cache(cache_dir: Path, cache_prefix: str) -> dict[str, Any]:
    """Load the 6 .npy + 2 .json files produced by the builder."""
    p = cache_dir / cache_prefix
    return {
        "pos": np.load(f"{p}_pos.npy"),
        "neg": np.load(f"{p}_neg.npy"),
        "labels": np.load(f"{p}_label.npy"),
        "line_idx": np.load(f"{p}_line_idx.npy"),
        "diff_dedup": np.load(f"{p}_dedup.npy"),
        "diff_dedup_labels": np.load(f"{p}_dedup_labels.npy"),
        "data_dedup": json.load(open(f"{p}_data_dedup.json", encoding="utf-8")),
    }


# =====================================================================
# Test-set processing (mirror of build-time logic, retriever variant)
# =====================================================================

def process_test_set(
    *, args, tokenizer, model_path: str, icl: str, layer_idx: int,
):
    """Read test files, parse edits, build pairs, run forward pass, dedup.

    Returns
    -------
    diff_dedup : np.ndarray
        Mean-pooled diff vectors for the test set (one per error site
        after dedup).
    data_for_retrieval_dedup : list[str]
        Per-site query strings ``"orig\\ttrunc_tgt"`` aligned with
        ``diff_dedup``.
    data_for_ret_dedup_idx : list[int]
        Per-site test-line indices aligned with ``diff_dedup``.
    srcs : list[str]
        The test source sentences (used downstream to seed jsonl rows).
    """
    train_name = LANG_TRAIN[args.lang]
    initial_test_dir = os.environ.get("GER_INITIAL_RESULT_DIR_TEST")
    if initial_test_dir:
        pred_dir = Path(initial_test_dir) / args.test / f"{args.test}-output-retokenized"
    else:
        pred_dir = (
            args.multilingual_dir
            / f"results_{args.model_name}_{args.initial_result_mode}"
            / f"initial_predictions_test{args.test_suffix}"
            / args.test
            / f"{args.test}-output-retokenized"
        )

    srcs = read_lines_unfiltered(args.test_source_file)
    tgts_pred = read_lines_unfiltered(f"{pred_dir}.txt")
    label_pred = read_lines_unfiltered(f"{pred_dir}.label")
    site_targets = tgts_pred
    site_labels = label_pred
    print(len(srcs))
    assert len(srcs) == len(site_targets) == len(site_labels)

    # -- Parse edit tags ---------------------------------------------------
    probe_pred_srcs = {b: [] for b in ("tp", "tpf", "tn", "fp", "fn")}
    probe_pred_diffs: list[list[str]] = []

    # NOTE: empty-tgt lines are silently skipped (matches original).
    # The downstream zip with ``srcs`` truncates at the shorter list,
    # which is the original behaviour. Leaving as-is for parity.
    for src, site_target, tgt_pred, site_label, line_pred in tqdm(
        zip(srcs, site_targets, tgts_pred, site_labels, label_pred), total=len(srcs)
    ):
        if site_target.strip() == "":
            continue
        parsed = parse_edit_ops(site_label, line_pred, site_target, tgt_pred, script="retrieve")
        for k in probe_pred_srcs:
            probe_pred_srcs[k].append(parsed.probe_pred_src[k])
        probe_pred_diffs.append(parsed.probe_pred_diff)

    pos_indices = [
        sorted(
            probe_pred_srcs["tp"][i] + probe_pred_srcs["tpf"][i] + probe_pred_srcs["fp"][i],
            key=lambda x: (x[0], x[1]),
        )
        for i in range(len(probe_pred_srcs["tp"]))
    ]
    neg_indices = [
        sorted(
            probe_pred_srcs["tn"][i] + probe_pred_srcs["fn"][i],
            key=lambda x: (x[0], x[1]),
        )
        for i in range(len(probe_pred_srcs["tp"]))
    ]

    # -- Truncate, pair, batch -------------------------------------------
    line_indices, pairs = build_truncated_pairs(
        srcs,
        tgts_for_embedding=tgts_pred,
        tgts_gold=None,
        probe_pos_indices=pos_indices,
        probe_neg_indices=neg_indices,
        probe_diffs=probe_pred_diffs,
        lang=args.lang,
        keep_tgt_gold=False,
    )
    batched = collect_prompt_batches(
        pairs, tokenizer, icl, args.lang, collect_label=False,
    )
    print(batched.data_inputs[0])

    # -- Forward pass (single layer for speed; data-parallel over GPUs) ----
    shard_count = args.num_shards or len(parse_gpu_list(args.gpus))
    hs = extract_hidden_states_parallel(
        model_path=model_path,
        data_inputs=batched.data_inputs,
        rep_tokens_inputs=batched.rep_tokens_inputs,
        layer_idx=layer_idx,
        gpus=args.gpus,
        num_shards=shard_count,
        temp_root=args.cache_dir,
        temp_prefix=f"{args.model_name}_{args.test}",
    )

    # -- Dedup by (line, error site) -------------------------------------
    diffs = [hs["pos"][i] - hs["neg"][i] for i in range(len(line_indices))]
    diff_dedup, dedup_line_idx, data_for_retrieval_dedup = dedup_by_line_and_diff(
        diffs,
        line_indices,
        batched.diff_idx_per_pair,
        batched.data_for_retrieval_pos,
        data_for_retrieval_label=None,  # retriever has no labels
    )
    assert isinstance(data_for_retrieval_dedup, list)
    return diff_dedup, data_for_retrieval_dedup, dedup_line_idx, srcs


def build_numpy_retriever(
    train_texts: list[str],
    train_embeddings: np.ndarray,
    test_texts: list[str],
    test_embeddings: np.ndarray,
    top_k: int,
) -> tuple[list[int], np.ndarray, dict[str, np.ndarray]]:
    """Build a direct cosine retriever over precomputed vectors.

    This preserves the previous behaviour that filtered training docs longer
    than 200 words, while avoiding per-query framework overhead in the long
    retrieval loop.
    """
    vector_dict = dict(
        zip(train_texts + test_texts, list(train_embeddings) + list(test_embeddings))
    )
    valid_train_ids = [
        idx for idx, text in enumerate(train_texts) if len(text.split()) <= 200
    ]
    valid_train_emb = np.asarray(
        [vector_dict[train_texts[idx]] for idx in valid_train_ids], dtype=np.float32
    )
    valid_train_emb = _l2_normalize(valid_train_emb)
    query_embeddings = {
        text: np.asarray(embedding, dtype=np.float32)
        for text, embedding in zip(test_texts, test_embeddings)
    }
    return valid_train_ids, valid_train_emb, query_embeddings


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


# =====================================================================
# Per-test-line retrieval
# =====================================================================

def retrieve_for_each_line_numpy(
    *,
    srcs: list[str],
    test_name: str,
    train_name: str,
    train_cors: list[list[str]],
    queries_by_line: dict[int, list[tuple[str, float]]],
    valid_train_ids: list[int],
    valid_train_emb: np.ndarray,
    query_embeddings: dict[str, np.ndarray],
    train_texts: list[str],
    hs_data_dedup_label: list[str],
    cor_exp: int,
    top_k: int,
    demo_floor_topk: int,
) -> list[dict]:
    """Assemble retrieval results with direct cosine nearest neighbours."""
    query_top_k = max(top_k, demo_floor_topk)
    save_jsonl = [
        {
            "id": f"{i}_{test_name}",
            "from": test_name,
            "text": src,
            "key_in_domain": "",
            "in_domain_examples": [],
            "key_cross_domain": "",
            "cross_domain_examples": [],
        }
        for i, src in enumerate(srcs)
    ]

    example_num = 0
    cor_example_num = 0
    wrong_lines = set(queries_by_line.keys())

    for line_idx in tqdm(range(len(save_jsonl))):
        row = save_jsonl[line_idx]
        if line_idx in wrong_lines:
            query_list = sorted(
                queries_by_line[line_idx], key=lambda x: x[1], reverse=True
            )
            if line_idx < 5:
                print(query_list)
            for query, _score in query_list:
                query_emb = _l2_normalize(query_embeddings[query].reshape(1, -1))[0]
                similarities = valid_train_emb @ query_emb
                if query_top_k >= len(similarities):
                    local_hit_ids = np.argsort(-similarities)
                else:
                    local_hit_ids = np.argpartition(-similarities, query_top_k - 1)[:query_top_k]
                    local_hit_ids = local_hit_ids[np.argsort(-similarities[local_hit_ids])]
                for local_id in local_hit_ids:
                    idd = valid_train_ids[int(local_id)]
                    hit_text = train_texts[idd].split("\t")[0]
                    text_for_test, label = hs_data_dedup_label[idd].split("\t")
                    assert hit_text == text_for_test
                    row["in_domain_examples"].append({
                        "id": idd,
                        "text": hit_text,
                        "label": label,
                        "from": train_name,
                        "key": train_texts[idd],
                        "similarity": float(similarities[local_id]),
                    })
                    example_num += 1
                    if hit_text == label:
                        cor_example_num += 1

        for text, label in random.sample(train_cors, k=cor_exp):
            row["in_domain_examples"].append({
                "id": -1,
                "text": text,
                "label": label,
                "from": train_name,
                "key": "Cor",
                "similarity": 0,
            })
            example_num += 1
            cor_example_num += 1

    n = len(save_jsonl)
    print("Example Number:", example_num)
    print(f"Avg. {example_num}/{n}={example_num/n} examples for every sentence .")
    print(f"Avg. {cor_example_num}/{n}={cor_example_num/n} correct examples for every sentence .")
    return save_jsonl


# =====================================================================
# Main
# =====================================================================

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "__hidden_worker__":
        run_hidden_state_worker(argv[1:])
        return

    args = resolve_paths(parse_args(argv))
    cache_prefix = (
        f"gec_representation_cache_"
        f"{args.model_name}_{args.initial_result_mode}_{'en' if args.lang == 'bea19' else args.lang}{args.train_suffix}"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(
        f"TEST: {args.test}{args.test_suffix} in MODEL {args.model_name}; "
        f"RETRIEVE_DIM: {args.retrieve_dim}; initial={args.initial_result_mode}; "
        f"cache={args.cache_dir}; output_root={args.output_root}; multilingual={args.multilingual_dir}; "
        f"train_data={args.train_data_dir}; test_source={args.test_source_file}; "
        f"model_root={args.model_root_dir}; gpus={args.gpus}; shards={args.num_shards or len(parse_gpu_list(args.gpus))}; "
        f"demo_floor_topk={args.demo_floor_topk}"
    )

    train_name = LANG_TRAIN[args.lang]
    icl = _icl_examples.get_icl(args.lang)
    cor_exp, top_k = LANG_RETRIEVAL_HP[args.lang]
    layer_idx = MODEL_LAYER_INDEX[args.model_name]

    # -- Load tokenizer & cache ------------------------------------------
    model_path = get_model_name_or_path(args.model_name, args.model_root_dir)
    tokenizer = load_tokenizer(model_path)
    cache = load_cache(args.cache_dir, cache_prefix)

    # -- Fit StandardScaler + TorchPCA on training diffs ------------------
    # Sign-flip via the cached random labels (matches builder's saved file).
    pos, neg, labels = cache["pos"], cache["neg"], cache["labels"]
    line_count = len(cache["line_idx"])
    diff_sign = np.array([
        (pos[i] - neg[i]) if labels[i] else -(pos[i] - neg[i])
        for i in range(line_count)
    ])
    scaler = StandardScaler()
    hs_std = scaler.fit_transform(diff_sign)
    pca_components = min(max(args.retrieve_dim, 5), hs_std.shape[0], hs_std.shape[1])
    pca_device = os.environ.get("GER_RETRIEVAL_PCA_DEVICE", "cpu")
    print(f"PCA device: {pca_device}")
    pca = TorchPCA(n_components=pca_components, device=pca_device).fit(hs_std)
    print(pca.transform(hs_std).shape)
    print(pca.explained_variance_ratio_[:5])
    print(pca.components_[:5])

    # -- Load training corrects (for COR_EXP padding) --------------------
    train_data_dir = args.train_data_dir / train_name / "train"
    train_srcs = read_lines_unfiltered(f"{train_data_dir}.src")
    train_tgts = read_lines_unfiltered(f"{train_data_dir}.tgt")
    train_cors = [[s, t] for s, t in zip(train_srcs, train_tgts) if s == t]
    train_wros = [[s, t] for s, t in zip(train_srcs, train_tgts) if s != t]
    print(f"{len(train_cors)} Cor Examples, {len(train_wros)} Wro Examples")

    # -- Process the test set --------------------------------------------
    test_diff_dedup, test_data_dedup, test_line_idx, srcs = process_test_set(
        args=args, tokenizer=tokenizer, model_path=model_path, icl=icl, layer_idx=layer_idx,
    )

    # -- Project all diffs into PCA space --------------------------------
    train_diff_dedup = cache["diff_dedup"]
    train_texts = cache["data_dedup"]["pos"]
    train_labels_text = cache["data_dedup"]["label"]

    embeddings = np.concatenate((train_diff_dedup, test_diff_dedup), axis=0)
    embeddings_proj = pca.transform(scaler.transform(embeddings))[:, : args.retrieve_dim]
    print("Embeddings:", embeddings_proj.shape)

    n_train = len(train_diff_dedup)
    train_emb = embeddings_proj[:n_train]
    test_emb = embeddings_proj[n_train:]

    # PC1 of each test diff = ranking score for that error site
    test_pc1 = pca.transform(scaler.transform(test_diff_dedup))[:, 0].tolist()

    # -- Group test queries by their source line -------------------------
    queries_by_line: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for line_idx, query, score in zip(test_line_idx, test_data_dedup, test_pc1):
        queries_by_line[line_idx].append((query, abs(score)))

    # -- Build retriever and run -----------------------------------------
    valid_train_ids, valid_train_emb, query_embeddings = build_numpy_retriever(
        train_texts=train_texts,
        train_embeddings=train_emb,
        test_texts=test_data_dedup,
        test_embeddings=test_emb,
        top_k=top_k,
    )
    save_jsonl = retrieve_for_each_line_numpy(
        srcs=srcs,
        test_name=args.test,
        train_name=train_name,
        train_cors=train_cors,
        queries_by_line=queries_by_line,
        valid_train_ids=valid_train_ids,
        valid_train_emb=valid_train_emb,
        query_embeddings=query_embeddings,
        train_texts=train_texts,
        hs_data_dedup_label=train_labels_text,
        cor_exp=cor_exp,
        top_k=top_k,
        demo_floor_topk=args.demo_floor_topk,
    )

    # -- Write output -----------------------------------------------------
    out_dir = (
        args.output_root
        / f"retrieved_examples_dim{args.retrieve_dim}{args.train_suffix}{args.test_suffix}"
        / args.test
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in save_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
