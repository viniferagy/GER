"""Shared utilities for the RepE-based GEC pipeline.

Two scripts use this module:
- build_gec_representation_cache.py
- retrieve_gec_examples_by_representation.py

The functions below cover everything that used to be duplicated between
them: model loading, prompt formatting, edit-tag parsing, error-site
truncation, probe-token batching, the forward pass, layer selection,
and randomised-sign deduplication.

----------------------------------------------------------------------
Behaviour note: TRANSFORM_SPLIT_HYPHEN
----------------------------------------------------------------------
The original two scripts disagreed in four places on how to advance
the running target-token counter for `$TRANSFORM_SPLIT_HYPHEN`:

  builder gold   : append-in-loop, end at P+k        (range k-1)
  builder pred   : append-in-loop, end at P+k        (range k-1)
  retriever gold : skip then append once, end P+k+1  (range k-1)
  retriever pred : skip then append once, end P+k+1  (range k-2)  *

These look like accidental drift, but they affect the saved cache
arrays. We preserve them through ``parse_edit_ops(script=...)``.
"""
from __future__ import annotations

import itertools
import os
import string
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm


# =====================================================================
# Prompt + system tag
# =====================================================================

SYSTEM_TAG = (
    "You are an language expert who is responsible for grammatical, lexical "
    "and orthographic error corrections given an input sentence. Your job is "
    "to fix grammatical mistakes, awkward phrases, spelling errors, etc. "
    "following standard written usage conventions, but your corrections must "
    "be conservative. Please keep the original sentence (words, phrases, and "
    "structure) as much as possible. The ultimate goal of this task is to "
    "make the given sentence sound natural to native speakers without making "
    "unnecessary changes. Corrections are not required when the sentence is "
    "already grammatical and sounds natural."
)

PROMPT_TEMPLATE = (
    "{system_tag}There is an erroneous sentence between `<erroneous sentence>` "
    "and `</erroneous sentence>`. Then grammatical errors in the erroneous "
    "sentence will be corrected. The corrected version will be between "
    "`<corrected sentence>` and `</corrected sentence>`.\n"
    "{ICL}<erroneous sentence> {src}</erroneous sentence>\n"
    "<corrected sentence> {tgt}"
)


def format_prompt(src: str, tgt: str, icl: str) -> str:
    """Slot src/tgt/ICL into the GEC reading prompt template."""
    return PROMPT_TEMPLATE.format(system_tag=SYSTEM_TAG, ICL=icl, src=src, tgt=tgt)


# =====================================================================
# Per-language config
# =====================================================================

LANG_TRAIN = {
    "en": "wilocness",
    "bea19": "wilocness",
    "de": "falko_merlin_train",
    "et": "estgec_train",
    "ro": "rogec_train",
}

LANG_TEST = {
    "en": "conll14",
    "bea19": "bea19",
    "de": "falko_merlin",
    "et": "estgec",
    "ro": "rogec",
}

# (COR_EXP, TOP_K) per language - retriever-only knobs.
LANG_RETRIEVAL_HP = {
    "en": (2, 4),
    "bea19": (2, 4),
    "de": (2, 4),
    "et": (4, 4),
    "ro": (2, 4),
}


# =====================================================================
# Model registry
# =====================================================================

# Which hidden-layer index to read out per model (negative = from top).
MODEL_LAYER_INDEX = {
    "llama31": -21,
    "qwen25": -12,
}


def get_model_name_or_path(model_name: str, model_root_dir: Path) -> str:
    """Resolve a short model name to a local path, env-var-overridable."""
    mapping = {
        "llama31": os.environ.get(
            "GER_LLAMA31_MODEL_PATH",
            str(model_root_dir / "Meta-Llama-3.1-8B-Instruct"),
        ),
        "qwen25": os.environ.get(
            "GER_QWEN25_MODEL_PATH",
            str(model_root_dir / "Qwen2.5-7B-Instruct"),
        ),
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown MODEL_NAME: {model_name!r}")
    return mapping[model_name]


def load_model_and_tokenizer(model_name_or_path: str):
    """Load HF causal LM + tokenizer + RepE rep-reading pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from repe import repe_pipeline_registry

    repe_pipeline_registry()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    return model, tokenizer, rep_reading_pipeline


def load_tokenizer(model_name_or_path: str):
    """Load the tokenizer with the same settings used by RepE workers."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =====================================================================
# Content-token filter
# =====================================================================

def is_other_symbol(char: str, lang: str = "en") -> bool:
    """True iff ``char`` is "not a content character" for ``lang``.

    Chinese:  anything outside CJK Unified Ideographs (U+4E00-U+9FFF).
    Others :  anything in ``string.punctuation``.
    """
    if lang == "zh":
        return not ("\u4e00" <= char <= "\u9fff")
    return char in string.punctuation


# =====================================================================
# Edit-tag parser
# =====================================================================

GOLD_LABELS = ("pos", "neg")
PRED_LABELS = ("tp", "tpf", "tn", "fp", "fn")
_SEP_OP = "SEPL|||SEPR"
_SEP_EDITS = "SEPL__SEPR"


@dataclass
class ParsedLine:
    """Structured parse of one (gold_edits, pred_edits) pair.

    All four ``probe_*`` dicts mirror the original variable names so
    callers can swap this in without renaming downstream code.
    """

    probe_src: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: {lbl: [] for lbl in GOLD_LABELS}
    )
    probe_tgt: dict[str, list[int]] = field(
        default_factory=lambda: {lbl: [] for lbl in GOLD_LABELS}
    )
    probe_pred_src: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: {lbl: [] for lbl in PRED_LABELS}
    )
    probe_pred_tgt: dict[str, list[int]] = field(
        default_factory=lambda: {lbl: [] for lbl in PRED_LABELS}
    )
    probe_diff: list[str] = field(default_factory=list)
    probe_pred_diff: list[str] = field(default_factory=list)


def parse_edit_ops(
    line: str,
    line_pred: str,
    tgt: str,
    tgt_pred: str,
    *,
    script: str,
) -> ParsedLine:
    """Parse one annotated (gold, pred) line pair.

    Parameters
    ----------
    line, line_pred
        ERRANT-style edit-tag strings, e.g.
        ``"hello SEPL|||SEPR $KEEP world SEPL|||SEPR $REPLACE_universe"``
    tgt, tgt_pred
        The gold and predicted target sentences. Used only for an
        end-of-list guard inside ``$DELETE``.
    script
        Either ``'build'`` or ``'retrieve'``. Selects which variant of
        the TRANSFORM_SPLIT_HYPHEN logic to apply (see module docstring).
    """
    if script not in ("build", "retrieve"):
        raise ValueError(f"script must be 'build' or 'retrieve', got {script!r}")

    out = ParsedLine()
    ops = line.strip().split(" ")
    ops_pred = line_pred.strip().split(" ")
    tgt_list_len = len(tgt.strip().split(" "))
    tgt_pred_list_len = len(tgt_pred.strip().split(" "))

    counter = -1
    counter_pred = -1
    append_suffix = False
    append_suffix_pred = False

    for op_idx, (op, op_pred) in enumerate(zip(ops, ops_pred)):
        token, edits_str = op.split(_SEP_OP)
        token_pred, edits_pred_str = op_pred.split(_SEP_OP)
        assert token_pred == token, (line, line_pred, token, token_pred, op, op_pred)
        edits = edits_str.strip().split(_SEP_EDITS)
        edits_pred = edits_pred_str.strip().split(_SEP_EDITS)
        edit = edits[0]
        edit_pred = edits_pred[0]

        # ---- gold side --------------------------------------------------
        probe_tgt_tmp: list[int] = []

        if edit == "$KEEP":
            if append_suffix:
                probe_tgt_tmp.append(counter)
                append_suffix = False
                out.probe_diff.append(f"{token} -> ADD_SUF {token}")
            else:
                out.probe_src["neg"].append((op_idx - 1, counter))
                out.probe_tgt["neg"].append(counter)
            counter += 1
        else:
            append_suffix = False
            if "$REPLACE" in edit:
                probe_tgt_tmp.append(counter)
                out.probe_diff.append(f"{token} -> {edit.replace('$REPLACE_', '')}")
                counter += 1
            elif "$DELETE" in edit:
                probe_tgt_tmp.append(counter - 1 if counter == tgt_list_len else counter)
                out.probe_diff.append(f"{token} -> ")
            elif "$APPEND" in edit:
                out.probe_tgt["neg"].append(counter)
                counter += 1
                probe_tgt_tmp.append(counter)
                mod_token = edit.replace("$APPEND_", "")
                out.probe_diff.append(f"{token} -> {token} {mod_token}")
                counter += 1
                append_suffix = True
            elif "$MERGE" in edit:
                probe_tgt_tmp.append(counter)
                next_token = ops[op_idx + 1].split(_SEP_OP)[0]
                out.probe_diff.append(f"{token} {next_token} -> {token}{next_token}")
            elif "$TRANSFORM" in edit:
                probe_tgt_tmp.append(counter)
                out.probe_diff.append(f"{token} -> {edit}")
                counter += 1
                if edit == "$TRANSFORM_SPLIT_HYPHEN":
                    parts = token.split("-")
                    if script == "build":
                        # append in loop, end at P+k
                        for _ in range(len(parts) - 1):
                            probe_tgt_tmp.append(counter)
                            counter += 1
                    else:  # retrieve
                        # skip k-1 then append once, end at P+k+1
                        for _ in range(len(parts) - 1):
                            counter += 1
                        probe_tgt_tmp.append(counter)
                        counter += 1
                    out.probe_diff.append(f"{token} -> {' '.join(parts)}")
            else:  # ~= REPLACE fallback (rare; print for visibility)
                print(edit)
                probe_tgt_tmp.append(counter)
                counter += 1

        # secondary APPEND edits (compound edits on the same op)
        if len(edits) > 1:
            for extra in edits[1:]:
                assert "$APPEND" in extra
                probe_tgt_tmp.append(counter)
                mod_token = extra.replace("$APPEND_", "")
                out.probe_diff.append(f"{out.probe_diff[-1]} {mod_token}")
                counter += 1
                append_suffix = True

        out.probe_tgt["pos"].extend(probe_tgt_tmp)
        for tgt_tok in probe_tgt_tmp:
            out.probe_src["pos"].append((op_idx - 1, tgt_tok))

        # ---- pred side --------------------------------------------------
        probe_pred_tgt_tmp: list[int] = []

        if edit_pred == "$KEEP":
            if append_suffix_pred:
                append_suffix_pred = False
                probe_pred_tgt_tmp.append(counter_pred)
                out.probe_pred_diff.append(f"{token} -> ADD_SUF {token}")
            else:
                bucket = "fn" if op_pred != op else "tn"
                out.probe_pred_src[bucket].append((op_idx - 1, counter_pred))
                out.probe_pred_tgt[bucket].append(counter_pred)
            counter_pred += 1
        else:
            append_suffix_pred = False
            if "$REPLACE" in edit_pred:
                probe_pred_tgt_tmp.append(counter_pred)
                out.probe_pred_diff.append(f"{token} -> {edit_pred.replace('$REPLACE_', '')}")
                counter_pred += 1
            elif "$DELETE" in edit_pred:
                probe_pred_tgt_tmp.append(
                    counter_pred - 1 if counter_pred == tgt_pred_list_len else counter_pred
                )
                out.probe_pred_diff.append(f"{token} -> ")
            elif "$APPEND" in edit_pred:
                out.probe_pred_tgt["tn"].append(counter_pred)
                counter_pred += 1
                probe_pred_tgt_tmp.append(counter_pred)
                mod_token = edit_pred.replace("$APPEND_", "")
                out.probe_pred_diff.append(f"{token} -> {token} {mod_token}")
                counter_pred += 1
                append_suffix_pred = True
            elif "$MERGE" in edit_pred:
                probe_pred_tgt_tmp.append(counter_pred)
                next_token = ops[op_idx + 1].split(_SEP_OP)[0]
                out.probe_pred_diff.append(f"{token} {next_token} -> {token}{next_token}")
            elif "$TRANSFORM" in edit_pred:
                probe_pred_tgt_tmp.append(counter_pred)
                out.probe_pred_diff.append(f"{token} -> {edit_pred}")
                counter_pred += 1
                if edit_pred == "$TRANSFORM_SPLIT_HYPHEN":
                    parts = token.split("-")
                    if script == "build":
                        # build pred: same as gold (range k-1, append in loop)
                        for _ in range(len(parts) - 1):
                            probe_pred_tgt_tmp.append(counter_pred)
                            counter_pred += 1
                    else:  # retrieve
                        # NOTE: original uses k-2 here; preserved as-is.
                        for _ in range(len(parts) - 2):
                            counter_pred += 1
                        probe_pred_tgt_tmp.append(counter_pred)
                        counter_pred += 1
                    out.probe_pred_diff.append(f"{token} -> {' '.join(parts)}")
            else:
                print(edit_pred)
                probe_pred_tgt_tmp.append(counter_pred)
                counter_pred += 1

        if len(edits_pred) > 1:
            for extra in edits_pred[1:]:
                assert "$APPEND" in extra
                probe_pred_tgt_tmp.append(counter_pred)
                mod_token = extra.replace("$APPEND_", "")
                out.probe_pred_diff.append(f"{out.probe_pred_diff[-1]} {mod_token}")
                counter_pred += 1
                append_suffix_pred = True

        # ---- bucket pred edits into TP / TPF / FP ----------------------
        if edit == "$KEEP":  # gold says no edit needed -> any pred edit is FP
            out.probe_pred_tgt["fp"].extend(probe_pred_tgt_tmp)
            for tgt_tok in probe_pred_tgt_tmp:
                out.probe_pred_src["fp"].append((op_idx - 1, tgt_tok))
        else:
            same_edits = (
                len(edits_pred) == len(edits)
                and all(edits_pred[i] == edits[i] for i in range(len(edits)))
            )
            bucket = "tp" if same_edits else "tpf"
            out.probe_pred_tgt[bucket].extend(probe_pred_tgt_tmp)
            for tgt_tok in probe_pred_tgt_tmp:
                out.probe_pred_src[bucket].append((op_idx - 1, tgt_tok))

    return out


# =====================================================================
# Per-line truncation + pos x neg pairing
# =====================================================================

@dataclass
class TruncatedExample:
    """One truncated (src, tgt) prefix at a single error-or-keep site.

    ``diff_id`` is set only for positive (errorful) sites; on the
    negative side it is omitted (matches original).
    """
    orig: str
    orig_tgt: str
    trunc_src: str
    trunc_tgt: str
    diff_id: int | None = None
    diff: str | None = None
    tgt_gold: str | None = None  # builder needs this; retriever doesn't


def _build_truncated_side(
    srcs: Sequence[str],
    tgts_for_embedding: Sequence[str],
    tgts_gold: Sequence[str] | None,
    probe_indices: Sequence[Sequence[tuple[int, int]]],
    probe_diffs: Sequence[Sequence[str]] | None,
    lang: str,
    *,
    keep_diff: bool,
    keep_tgt_gold: bool,
) -> list[list[TruncatedExample]]:
    """Construct per-line lists of TruncatedExample.

    Positive side: pass ``probe_diffs`` and set ``keep_diff=True``.
    Negative side: pass ``probe_diffs=None`` and ``keep_diff=False``.

    Iteration uses ``zip``, which truncates at the shortest input -
    matches the original behaviour when ``probe_indices`` is shorter
    than ``srcs`` due to empty-tgt skips.
    """
    out: list[list[TruncatedExample]] = []
    skipped_out_of_bounds = 0
    if keep_diff:
        assert probe_diffs is not None
        zipped = zip(srcs, tgts_for_embedding, probe_indices, probe_diffs)
    else:
        zipped = zip(srcs, tgts_for_embedding, probe_indices)

    for line_idx, items in enumerate(zipped):
        if keep_diff:
            src, tgt, probe_src, line_diffs = items
        else:
            src, tgt, probe_src = items
            line_diffs = None

        src_tokens = src.replace(" ", " ").split(" ")
        tgt_tokens = tgt.replace(" ", " ").split(" ")
        if keep_diff and line_diffs is not None:
            assert len(probe_src) == len(line_diffs), (
                len(probe_src), len(line_diffs)
            )

        truncated: list[TruncatedExample] = []
        diff_idx = 0
        iterator = (
            zip(probe_src, line_diffs) if keep_diff and line_diffs is not None
            else ((pair, None) for pair in probe_src)
        )
        for pair, diff in iterator:
            src_pos, tgt_pos = pair
            if src_pos == -1 or tgt_pos == -1:
                continue
            if src_pos >= len(src_tokens) or tgt_pos >= len(tgt_tokens):
                skipped_out_of_bounds += 1
                continue
            if is_other_symbol(src_tokens[src_pos], lang) or is_other_symbol(
                tgt_tokens[tgt_pos], lang
            ):
                continue
            ex = TruncatedExample(
                orig=" ".join(src_tokens),
                orig_tgt=" ".join(tgt_tokens),
                trunc_src=" ".join(src_tokens[: src_pos + 1]),
                trunc_tgt=" ".join(tgt_tokens[: tgt_pos + 1]),
                diff_id=diff_idx if keep_diff else None,
                diff=diff if keep_diff else None,
                tgt_gold=tgts_gold[line_idx] if keep_tgt_gold and tgts_gold is not None else None,
            )
            truncated.append(ex)
            if keep_diff:
                diff_idx += 1
        out.append(truncated)
    if skipped_out_of_bounds:
        print(
            "Skipped out-of-bounds probe-token alignments: "
            f"{skipped_out_of_bounds}"
        )
    return out


def build_truncated_pairs(
    srcs: Sequence[str],
    tgts_for_embedding: Sequence[str],
    tgts_gold: Sequence[str] | None,
    probe_pos_indices: Sequence[Sequence[tuple[int, int]]],
    probe_neg_indices: Sequence[Sequence[tuple[int, int]]],
    probe_diffs: Sequence[Sequence[str]],
    lang: str,
    *,
    keep_tgt_gold: bool,
) -> tuple[list[int], list[tuple[TruncatedExample, TruncatedExample]]]:
    """Build pairs (pos_example, neg_example) per error site.

    Mirrors the original itertools cycle logic exactly:

        for line in 0..L:
            for (pos, neg) in zip(cycle([line]), cycle(pos_list), neg_list[1:]):
                yield (line, pos, neg)

    Returns
    -------
    line_indices : list[int]
        One entry per pair, indicating the source line.
    pairs : list[(pos, neg)]
        The actual TruncatedExample pairs.
    """
    pos_per_line = _build_truncated_side(
        srcs, tgts_for_embedding, tgts_gold,
        probe_pos_indices, probe_diffs, lang,
        keep_diff=True, keep_tgt_gold=keep_tgt_gold,
    )
    neg_per_line = _build_truncated_side(
        srcs, tgts_for_embedding, tgts_gold,
        probe_neg_indices, None, lang,
        keep_diff=False, keep_tgt_gold=keep_tgt_gold,
    )

    line_indices: list[int] = []
    pairs: list[tuple[TruncatedExample, TruncatedExample]] = []
    for idx, (pos, neg) in enumerate(zip(pos_per_line, neg_per_line)):
        # original: zip(itertools.cycle([idx]), itertools.cycle(pos), neg[1:])
        for _, pos_ex, neg_ex in zip(itertools.cycle([idx]), itertools.cycle(pos), neg[1:]):
            line_indices.append(idx)
            pairs.append((pos_ex, neg_ex))
    return line_indices, pairs


# =====================================================================
# Probe-token batching + forward pass
# =====================================================================

@dataclass
class PromptBatchOutputs:
    """Everything produced by ``collect_prompt_batches`` in one bundle."""
    data_inputs: list[str]
    rep_tokens_inputs: list[list[int]]
    data_for_retrieval_pos: list[str]
    data_for_retrieval_neg: list[str]
    data_for_retrieval_label: list[str] | None  # builder-only
    diff_idx_per_pair: list[int]
    probe_diffs_dup: list[str]


def collect_prompt_batches(
    pairs: Sequence[tuple[TruncatedExample, TruncatedExample]],
    tokenizer,
    icl: str,
    lang: str,
    *,
    collect_label: bool,
    desc: str | None = None,
) -> PromptBatchOutputs:
    """Tokenise each (pos, neg) pair, find probe-token positions, group by prompt.

    The function produces:

    - ``data_inputs`` / ``rep_tokens_inputs``: deduplicated full-context
      prompts and the probe-token indices that share each prompt. RepE
      reads all token positions in one forward pass per prompt.
    - ``data_for_retrieval_pos / _neg``: tab-joined "orig\\ttrunc_tgt"
      strings, one per pair. Used as keys + values for the downstream
      vector index.
    - ``data_for_retrieval_label`` (if ``collect_label``): "orig\\ttgt_gold"
      strings on the neg side (builder cache only).
    - ``diff_idx_per_pair``: per-pair diff_id, used to dedup within a line.
    - ``probe_diffs_dup``: per-pair diff text, mainly for inspection.

    Chinese pre-processing: all string fields of both examples (incl.
    ``diff`` and ``tgt_gold``) have ASCII spaces stripped first.
    """
    out = PromptBatchOutputs(
        data_inputs=[],
        rep_tokens_inputs=[],
        data_for_retrieval_pos=[],
        data_for_retrieval_neg=[],
        data_for_retrieval_label=[] if collect_label else None,
        diff_idx_per_pair=[],
        probe_diffs_dup=[],
    )
    cur_rep_tokens: list[int] = []

    for pos_ex, neg_ex in tqdm(pairs, desc=desc):
        if lang == "zh":
            pos_ex = _strip_zh_spaces(pos_ex)
            neg_ex = _strip_zh_spaces(neg_ex)

        # probe-token = index of the last token of the truncated prompt
        # (pos_ex.orig == neg_ex.orig for any (pos, neg) drawn from the
        # same line, so we use pos_ex.orig for both prompts).
        pos_prompt = format_prompt(pos_ex.orig, pos_ex.trunc_tgt, icl)
        neg_prompt = format_prompt(pos_ex.orig, neg_ex.trunc_tgt, icl)
        pos_token_idx = len(tokenizer(pos_prompt)["input_ids"]) - 1
        neg_token_idx = len(tokenizer(neg_prompt)["input_ids"]) - 1

        out.data_for_retrieval_pos.append(f"{pos_ex.orig}\t{pos_ex.trunc_tgt}")
        out.data_for_retrieval_neg.append(f"{neg_ex.orig}\t{neg_ex.trunc_tgt}")
        if collect_label:
            assert out.data_for_retrieval_label is not None
            out.data_for_retrieval_label.append(f"{neg_ex.orig}\t{neg_ex.tgt_gold}")
        out.diff_idx_per_pair.append(pos_ex.diff_id)  # type: ignore[arg-type]

        # full-context prompt; many probe-tokens can share one forward pass.
        full_prompt = format_prompt(pos_ex.orig, pos_ex.orig_tgt, icl)
        if not out.data_inputs:
            out.data_inputs.append(full_prompt)
        elif out.data_inputs[-1] != full_prompt:
            out.data_inputs.append(full_prompt)
            out.rep_tokens_inputs.append(cur_rep_tokens)
            cur_rep_tokens = []
        cur_rep_tokens.extend([pos_token_idx, neg_token_idx])

        out.probe_diffs_dup.append(pos_ex.diff)  # type: ignore[arg-type]

    out.rep_tokens_inputs.append(cur_rep_tokens)
    return out


def _strip_zh_spaces(ex: TruncatedExample) -> TruncatedExample:
    """Strip ASCII spaces from every string field (Chinese pre-processing)."""
    def s(v):
        return v.replace(" ", "") if isinstance(v, str) else v

    return TruncatedExample(
        orig=s(ex.orig),
        orig_tgt=s(ex.orig_tgt),
        trunc_src=s(ex.trunc_src),
        trunc_tgt=s(ex.trunc_tgt),
        diff_id=ex.diff_id,
        diff=s(ex.diff),
        tgt_gold=s(ex.tgt_gold),
    )


def extract_pos_neg_hidden_states(
    rep_reading_pipeline,
    data_inputs: Sequence[str],
    rep_tokens_inputs: Sequence[Sequence[int]],
    layer_idx: int,
    hidden_layers: Sequence[int] | None = None,
    desc: str | None = None,
) -> dict[str, np.ndarray]:
    """Forward-pass and split alternating positions into pos/neg arrays.

    Each prompt in ``rep_tokens_inputs`` contributes pairs in the order
    (pos_0, neg_0, pos_1, neg_1, ...). We concatenate everything then
    split into two stacks.
    """
    if hidden_layers is None:
        hidden_layers = [layer_idx]
    old_desc = os.environ.get("GER_REPE_PROGRESS_DESC")
    if desc is not None:
        os.environ["GER_REPE_PROGRESS_DESC"] = desc
    try:
        hidden_states_all = rep_reading_pipeline._batched_string_to_hiddens_all(
            list(data_inputs),
            rep_tokens=list(rep_tokens_inputs),
            hidden_layers=list(hidden_layers),
            batch_size=1,
            which_hidden_states=None,
        )
    finally:
        if desc is not None:
            if old_desc is None:
                os.environ.pop("GER_REPE_PROGRESS_DESC", None)
            else:
                os.environ["GER_REPE_PROGRESS_DESC"] = old_desc
    layer = np.concatenate(hidden_states_all[layer_idx], axis=0)
    n_pairs = layer.shape[0] // 2
    pos = np.copy(layer[0::2][:n_pairs])
    neg = np.copy(layer[1::2][:n_pairs])
    return {"pos": pos, "neg": neg}


# =====================================================================
# Dedup by (line, diff-id)
# =====================================================================

def dedup_by_line_and_diff(
    hidden_diff: Sequence[np.ndarray],
    line_indices: Sequence[int],
    diff_indices: Sequence[int],
    data_for_retrieval_pos: Sequence[str],
    data_for_retrieval_label: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[int], dict[str, list[str]] | list[str]]:
    """Mean-pool diff vectors that share (line_idx, diff_idx).

    If ``data_for_retrieval_label`` is provided (builder), returns the
    pos+label dict; otherwise (retriever) returns just the pos list.
    """
    grouped: dict[int, dict[int, list[np.ndarray]]] = defaultdict(dict)
    keep_pos: list[str] = []
    keep_label: list[str] = []
    keep_line_idx: list[int] = []

    for pair_idx, (line_idx, diff_idx) in enumerate(zip(line_indices, diff_indices)):
        if diff_idx not in grouped[line_idx]:
            grouped[line_idx][diff_idx] = []
            keep_pos.append(data_for_retrieval_pos[pair_idx])
            if data_for_retrieval_label is not None:
                keep_label.append(data_for_retrieval_label[pair_idx])
            keep_line_idx.append(line_idx)
        grouped[line_idx][diff_idx].append(hidden_diff[pair_idx])

    means: list[np.ndarray] = []
    for line_idx in grouped:
        for diff_idx in grouped[line_idx]:
            means.append(np.mean(grouped[line_idx][diff_idx], axis=0))
    deduped = np.array(means)

    data_out: dict[str, list[str]] | list[str]
    if data_for_retrieval_label is not None:
        data_out = {"pos": keep_pos, "label": keep_label}
    else:
        data_out = keep_pos
    return deduped, keep_line_idx, data_out


# =====================================================================
# TorchPCA - GPU-friendly drop-in for sklearn / cuml PCA
# =====================================================================

class TorchPCA:
    """A small randomised-SVD PCA that runs on the same device as torch.

    API-compatible subset of sklearn / cuml PCA: ``fit``, ``transform``,
    ``components_``, ``mean_``, ``explained_variance_ratio_``.
    """

    def __init__(self, n_components: int, device: str | None = None, niter: int = 2):
        self.n_components = n_components
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.niter = niter
        self.mean_: torch.Tensor | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, x) -> "TorchPCA":
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.mean_ = x_t.mean(dim=0, keepdim=True)
        x_c = x_t - self.mean_
        q = min(self.n_components, x_c.shape[0], x_c.shape[1])
        _, s, v = torch.pca_lowrank(x_c, q=q, niter=self.niter)
        self.components_ = v.T.detach().cpu().numpy()
        total = x_c.var(dim=0, unbiased=False).sum()
        explained = (s ** 2) / x_c.shape[0]
        self.explained_variance_ratio_ = (explained / total).detach().cpu().numpy()
        return self

    def transform(self, x) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        comps = torch.as_tensor(self.components_.T, dtype=torch.float32, device=self.device)
        return ((x_t - self.mean_) @ comps).detach().cpu().numpy()


# =====================================================================
# I/O helpers
# =====================================================================

def read_lines(path: Path | str) -> list[str]:
    """Read non-blank stripped lines from a text file."""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() != ""]


def read_lines_unfiltered(path: Path | str) -> list[str]:
    """Read all stripped lines (including blanks). Used by retriever."""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f]


def resolve_label_path(data_prefix: Path | str) -> Path:
    """Resolve the gold label file for a src/tgt prefix."""
    prefix = Path(data_prefix)
    candidates = (Path(f"{prefix}.label"),)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No label file found for {prefix}; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )
