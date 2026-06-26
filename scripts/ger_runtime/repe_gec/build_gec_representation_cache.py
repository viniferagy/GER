"""Build a representation cache for RepE-based GEC retrieval.

Pipeline
--------
1. Load train.src / .tgt / .label + LLM-pred .txt / .label.
2. Parse each line's edit tags into per-token gold/pred indices.
3. At every error site, build (truncated_src, truncated_tgt) pairs on
   both the gold and the predicted target. Skip pairs whose anchor
   token is punctuation / non-content.
4. Pair pos x neg per line (with the original itertools.cycle scheme).
5. Tokenise each pair, record the last-token index as the probe site,
   group pairs that share the same full prompt into one forward pass.
6. Run RepE rep-reading, read layer ``MODEL_LAYER_INDEX[MODEL]``.
7. Split the layer output into pos / neg arrays.
8. Dedup by (line_idx, diff_id) via mean-pooling.
9. Save .npy + .json artefacts.

Saved files (under ``CACHE_DIR``, prefix ``CACHE_PREFIX``)::

    {prefix}_pos.npy            hidden states, positive side  (n_pairs, hidden_dim)
    {prefix}_neg.npy            hidden states, negative side  (n_pairs, hidden_dim)
    {prefix}_label.npy          random  pm1 labels, one per pair  (n_pairs,)
    {prefix}_line_idx.npy       source-line index, one per pair  (n_pairs,)
    {prefix}_data.json          {pos, neg, label} retrieval strings
    {prefix}_dedup.npy          mean-pooled diffs by (line, error site)  (n_unique, hidden_dim)
    {prefix}_dedup_labels.npy   random  pm1 labels for the dedup'd diffs  (n_unique,)
    {prefix}_data_dedup.json    {pos, label} retrieval strings for the dedup'd set

CLI (backwards-compatible positional args)::

    python build_gec_representation_cache.py \\
        LANG SUFFIX TEST_SUFFIX MODEL_NAME GPU \\
        [INITIAL_RESULT_MODE] [CACHE_DIR] \\
        [MULTILINGUAL_DIR] [TRAIN_DATA_DIR] [MODEL_ROOT_DIR]

``TEST_SUFFIX`` is accepted for backwards compatibility but is unused
by the builder (the prefix uses only ``SUFFIX``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

import _icl_examples
from _common import (
    LANG_CACHE_KEY,
    LANG_TRAIN,
    MODEL_LAYER_INDEX,
    build_truncated_pairs,
    collect_prompt_batches,
    dedup_by_line_and_diff,
    extract_pos_neg_hidden_states,
    get_model_name_or_path,
    load_model_and_tokenizer,
    load_tokenizer,
    parse_edit_ops,
    read_lines,
    resolve_label_path,
)

random.seed(0)
np.random.seed(0)


# =====================================================================
# CLI
# =====================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the (legacy positional) CLI. argparse-compatible help."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("lang", help="Language code: en|de|ru|et|zh|ro")
    parser.add_argument("suffix", help="Suffix injected into the cache prefix")
    parser.add_argument(
        "test_suffix", help="Legacy / unused by the builder; kept for CLI parity"
    )
    parser.add_argument("model_name", help="Short model name: llama31|qwen25")
    parser.add_argument("gpu", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("initial_result_mode", nargs="?", default="default")
    parser.add_argument("cache_dir", nargs="?", type=Path, default=None)
    parser.add_argument("multilingual_dir", nargs="?", type=Path, default=None)
    parser.add_argument("train_data_dir", nargs="?", type=Path, default=None)
    parser.add_argument("model_root_dir", nargs="?", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in path defaults from $GER_* env vars or repo layout."""
    here = Path(__file__).resolve()
    ger_root = here.parents[3]  # scripts/ger_runtime/repe_gec/script.py -> repo root
    args.cache_dir = args.cache_dir or here.parent / "cache"
    args.multilingual_dir = args.multilingual_dir or Path(
        os.environ.get("GER_MULTILINGUAL_DIR", ger_root / "multilingual")
    )
    train_data_dir = args.train_data_dir or os.environ.get("GER_TRAIN_DATA_DIR")
    if not train_data_dir:
        raise ValueError("train_data_dir positional argument or GER_TRAIN_DATA_DIR is required")
    args.train_data_dir = Path(train_data_dir)
    args.model_root_dir = args.model_root_dir or Path(
        os.environ.get("GER_MODEL_ROOT_DIR", ger_root / "models")
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    return args


def _parse_gpu_list(gpu_arg: str) -> list[str]:
    gpus = [gpu.strip() for gpu in gpu_arg.split(",") if gpu.strip()]
    return gpus or ["0"]


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip() in {
        "1", "true", "True", "TRUE", "yes", "Yes", "YES",
    }


def _balanced_prompt_ranges(
    rep_tokens_inputs: list[list[int]],
    num_shards: int,
) -> list[tuple[int, int]]:
    """Split prompt indices into contiguous ranges with similar pair counts."""
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


def _run_hidden_state_worker(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Internal RepE cache hidden-state worker")
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
    elif not _truthy_env("GER_REPE_DISABLE_PROGRESS"):
        os.environ["GER_REPE_PROGRESS_DESC"] = (
            f"cache hidden states shard {args.shard_id + 1}/{args.num_shards}"
        )

    with open(args.input_json, encoding="utf-8") as f:
        payload = json.load(f)

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


def _extract_hidden_states_parallel(
    model_path: str,
    data_inputs: list[str],
    rep_tokens_inputs: list[list[int]],
    layer_idx: int,
    gpu_arg: str,
    num_shards: int,
    cache_dir: Path,
    cache_prefix: str,
) -> dict[str, np.ndarray]:
    gpus = _parse_gpu_list(gpu_arg)
    ranges = _balanced_prompt_ranges(rep_tokens_inputs, num_shards)
    if len(ranges) <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg
        _, _, rep_reading_pipeline = load_model_and_tokenizer(model_path)
        return extract_pos_neg_hidden_states(
            rep_reading_pipeline,
            data_inputs,
            rep_tokens_inputs,
            layer_idx=layer_idx,
            hidden_layers=[layer_idx],
            desc="cache hidden states",
        )

    print(
        "Run RepE hidden-state cache in data-parallel mode: "
        f"{len(ranges)} shards on GPUs {','.join(gpus)}"
    )
    disable_progress = _truthy_env("GER_REPE_DISABLE_PROGRESS")

    with tempfile.TemporaryDirectory(
        prefix=f"{cache_prefix}_hidden_shards_",
        dir=str(cache_dir),
    ) as tmp:
        tmp_dir = Path(tmp)
        processes: list[tuple[int, subprocess.Popen]] = []
        try:
            for shard_id, (start, end) in enumerate(ranges):
                shard_input = tmp_dir / f"shard_{shard_id:03d}.json"
                with open(shard_input, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "data_inputs": data_inputs[start:end],
                            "rep_tokens_inputs": rep_tokens_inputs[start:end],
                        },
                        f,
                        ensure_ascii=False,
                    )

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpus[shard_id % len(gpus)]
                if shard_id != 0 or disable_progress:
                    env["GER_REPE_DISABLE_PROGRESS"] = "1"
                else:
                    env.pop("GER_REPE_DISABLE_PROGRESS", None)
                    env["GER_REPE_PROGRESS_DESC"] = (
                        f"cache hidden states shard {shard_id + 1}/{len(ranges)}"
                    )
                cmd = [
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
                    f"Launching cache hidden-state shard {shard_id + 1}/{len(ranges)} "
                    f"on GPU {env['CUDA_VISIBLE_DEVICES']} ({end - start} prompts)"
                )
                processes.append((shard_id, subprocess.Popen(cmd, env=env)))

            failed: list[tuple[int, int]] = []
            for shard_id, proc in processes:
                status = proc.wait()
                if status != 0:
                    failed.append((shard_id, status))
            if failed:
                raise RuntimeError(f"RepE cache hidden-state shard(s) failed: {failed}")
        except KeyboardInterrupt:
            for _, proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for _, proc in processes:
                if proc.poll() is None:
                    proc.kill()
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
# Main
# =====================================================================

def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "__hidden_worker__":
        _run_hidden_state_worker(argv[1:])
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = resolve_paths(parse_args(argv))
    cache_prefix = (
        f"gec_representation_cache_"
        f"{args.model_name}_{args.initial_result_mode}_{LANG_CACHE_KEY[args.lang]}{args.suffix}"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    default_cache_shards = str(max(1, len(_parse_gpu_list(args.gpu))))
    cache_num_shards = max(1, int(os.environ.get("GER_REPE_CACHE_NUM_SHARDS", default_cache_shards) or default_cache_shards))
    cache_gpus = os.environ.get("GER_REPE_CACHE_CUDA_VISIBLE_DEVICES", args.gpu)

    print(
        f"Build representation cache for MODEL={args.model_name}, LANG={args.lang}, "
        f"initial={args.initial_result_mode}, cache={args.cache_dir}, "
        f"multilingual={args.multilingual_dir}, train_data={args.train_data_dir}, "
        f"model_root={args.model_root_dir}, hidden_state_shards={cache_num_shards}, "
        f"hidden_state_gpus={cache_gpus}"
    )

    train_name = LANG_TRAIN[args.lang]
    icl = _icl_examples.get_icl(args.lang)
    layer_idx = MODEL_LAYER_INDEX[args.model_name]
    print(f"Using layer index: {layer_idx}")

    # -- Load tokenizer ----------------------------------------------------
    model_path = get_model_name_or_path(args.model_name, args.model_root_dir)
    tokenizer = load_tokenizer(model_path)

    # -- Read parallel files ----------------------------------------------
    data_dir = args.train_data_dir / train_name / "train"
    initial_train_dir = os.environ.get("GER_INITIAL_RESULT_DIR_TRAIN")
    if initial_train_dir:
        pred_dir = Path(initial_train_dir) / train_name / f"{train_name}-output"
    else:
        pred_dir = (
            args.multilingual_dir
            / f"results_{args.model_name}_{args.initial_result_mode}"
            / f"initial_predictions_train{args.suffix}"
            / train_name
            / f"{train_name}-output"
        )
    srcs = read_lines(f"{data_dir}.src")
    tgts = read_lines(f"{data_dir}.tgt")
    label_gold = read_lines(resolve_label_path(data_dir))
    tgts_pred = read_lines(f"{pred_dir}-retokenized.txt")
    label_pred = read_lines(f"{pred_dir}-retokenized.label")
    assert len(srcs) == len(tgts) == len(label_gold) == len(tgts_pred) == len(label_pred), (
        len(srcs), len(tgts), len(label_gold), len(tgts_pred), len(label_pred)
    )
    print(len(srcs))

    # -- Parse edit tags --------------------------------------------------
    probe_srcs = {"pos": [], "neg": []}
    probe_tgts = {"pos": [], "neg": []}
    probe_pred_srcs = {b: [] for b in ("tp", "tpf", "tn", "fp", "fn")}
    probe_pred_tgts = {b: [] for b in ("tp", "tpf", "tn", "fp", "fn")}
    probe_diffs: list[list[str]] = []
    probe_pred_diffs: list[list[str]] = []

    from tqdm import tqdm
    for src, tgt, tgt_pred, line, line_pred in tqdm(
        zip(srcs, tgts, tgts_pred, label_gold, label_pred), total=len(srcs)
    ):
        if tgt.strip() == "":
            continue
        parsed = parse_edit_ops(line, line_pred, tgt, tgt_pred, script="build")
        for k in ("pos", "neg"):
            probe_srcs[k].append(parsed.probe_src[k])
            probe_tgts[k].append(parsed.probe_tgt[k])
        for k in probe_pred_srcs:
            probe_pred_srcs[k].append(parsed.probe_pred_src[k])
            probe_pred_tgts[k].append(parsed.probe_pred_tgt[k])
        probe_diffs.append(parsed.probe_diff)
        probe_pred_diffs.append(parsed.probe_pred_diff)

    # The original supports two probe_types ('src' and 'src_pred') but
    # only iterates over ['src_pred'] in practice. We do the same.
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
    print("Error Num:", sum(len(i) for i in pos_indices))

    # -- Truncate & pair --------------------------------------------------
    line_indices, pairs = build_truncated_pairs(
        srcs,
        tgts_for_embedding=tgts_pred,
        tgts_gold=tgts,
        probe_pos_indices=pos_indices,
        probe_neg_indices=neg_indices,
        probe_diffs=probe_pred_diffs,
        lang=args.lang,
        keep_tgt_gold=True,
    )

    # -- Tokenise + group prompts ----------------------------------------
    print("Tokenize and group representation prompts...")
    batched = collect_prompt_batches(
        pairs, tokenizer, icl, args.lang, collect_label=True, desc="cache tokenize prompts",
    )
    print(f"Prompt Num: {len(batched.data_inputs)}")
    print(batched.data_inputs[0])

    # -- Forward pass -----------------------------------------------------
    print("Run RepE forward pass for hidden-state cache...")
    hs = _extract_hidden_states_parallel(
        model_path,
        batched.data_inputs,
        batched.rep_tokens_inputs,
        layer_idx,
        cache_gpus,
        cache_num_shards,
        args.cache_dir,
        cache_prefix,
    )
    if hs["pos"].shape[0] != len(line_indices) or hs["neg"].shape[0] != len(line_indices):
        raise RuntimeError(
            "Hidden-state cache row count mismatch: "
            f"pos={hs['pos'].shape[0]}, neg={hs['neg'].shape[0]}, pairs={len(line_indices)}"
        )

    # -- Random sign labels ----------------------------------------------
    train_labels = np.random.choice([1, -1], size=hs["pos"].shape[0])

    # -- Dedup by (line, diff site) --------------------------------------
    diffs = [hs["pos"][i] - hs["neg"][i] for i in range(len(line_indices))]
    deduped, _kept_lines, data_for_retrieval_dedup = dedup_by_line_and_diff(
        diffs,
        line_indices,
        batched.diff_idx_per_pair,
        batched.data_for_retrieval_pos,
        batched.data_for_retrieval_label,
    )
    dedup_labels = np.random.choice([1, -1], size=deduped.shape[0])

    # -- Save -------------------------------------------------------------
    data_for_retrieval = {
        "pos": batched.data_for_retrieval_pos,
        "neg": batched.data_for_retrieval_neg,
        "label": batched.data_for_retrieval_label,
    }
    save_root = args.cache_dir
    np.save(save_root / f"{cache_prefix}_pos.npy", hs["pos"])
    np.save(save_root / f"{cache_prefix}_neg.npy", hs["neg"])
    np.save(save_root / f"{cache_prefix}_label.npy", train_labels)
    np.save(save_root / f"{cache_prefix}_line_idx.npy", np.array(line_indices))
    np.save(save_root / f"{cache_prefix}_dedup.npy", deduped)
    np.save(save_root / f"{cache_prefix}_dedup_labels.npy", dedup_labels)
    with open(save_root / f"{cache_prefix}_data.json", "w", encoding="utf-8") as f:
        json.dump(data_for_retrieval, f, indent=4, ensure_ascii=False)
    with open(save_root / f"{cache_prefix}_data_dedup.json", "w", encoding="utf-8") as f:
        json.dump(data_for_retrieval_dedup, f, indent=4, ensure_ascii=False)
    print(f"Saved representation cache to {save_root} with prefix {cache_prefix}")


if __name__ == "__main__":
    main()
