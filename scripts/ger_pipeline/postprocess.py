#!/usr/bin/env python3
"""Apply the standard GER postprocess to prediction files and save cleaned outputs."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


STANDARD_CHAR_THRESHOLDS = {
    "conll14": 0.70,
    "bea19": 0.70,
    "falko_merlin": 0.70,
    "ronacc_readerbench": 0.96,
    "estgec": 0.96,
}
NO_CORRECTION_RE = re.compile(
    r"\b(?:no\s+corrections?\s+needed|no\s+correction\s+needed|no\s+correction|"
    r"not\s+grammatically\s+incorrect|not\s+incorrect|no\s+errors?\s+found)\b|ei\s+vaja",
    re.IGNORECASE,
)
COMMENT_RE = re.compile(
    r"\([^)]*(?:see\s+on|küsimus|question|no\s+correction|not\s+grammatically|ei\s+vaja)[^)]*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Variant:
    char_threshold: float


def infer_dataset(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    for dataset in STANDARD_CHAR_THRESHOLDS:
        if dataset in parts:
            return dataset
    name = path.name.lower()
    for dataset in STANDARD_CHAR_THRESHOLDS:
        if dataset in name:
            return dataset
    return ""


def standard_char_threshold(dataset: str) -> float:
    if dataset not in STANDARD_CHAR_THRESHOLDS:
        known = ", ".join(sorted(STANDARD_CHAR_THRESHOLDS))
        raise ValueError(f"unknown dataset for standard GER postprocess: {dataset!r}; known: {known}")
    return STANDARD_CHAR_THRESHOLDS[dataset]


def extract_tagged_answer(response: str, source: str) -> str:
    start = "<corrected sentence>"
    end = "</corrected sentence>"
    end_idx = response.find(end)
    start_idx = response.find(start)
    if end_idx != -1 and (start_idx == -1 or end_idx < start_idx):
        answer = response[:end_idx].strip()
        if answer:
            return answer
    if start_idx == -1:
        return source
    content_start = start_idx + len(start)
    end_idx = response.find(end, content_start)
    if end_idx == -1:
        return source
    answer = response[content_start:end_idx].strip()
    return answer or source


def char_ratio(source: str, prediction: str) -> float:
    a = "".join(source.split())
    b = "".join(prediction.split())
    if not a and not b:
        return 1.0
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def clean_prediction(row: dict[str, object], variant: Variant) -> tuple[str, str]:
    source = str(row.get("text", "")).strip()
    response = str(row.get("response", ""))
    prediction = extract_tagged_answer(response, source)
    low = response.lower()
    first_err = low.find("<erroneous sentence>")
    first_corr = low.find("<corrected sentence>")
    first_no = NO_CORRECTION_RE.search(response)
    if (first_err != -1 and (first_corr == -1 or first_err < first_corr)) or first_no:
        return source, "tagbad_or_no_correction"
    if NO_CORRECTION_RE.search(prediction) or COMMENT_RE.search(prediction):
        return source, "comment_or_no_correction_in_prediction"
    if char_ratio(source, prediction) < variant.char_threshold:
        return source, f"char_lt_{variant.char_threshold:.2f}"
    return prediction.strip() or source, "kept"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-txt", required=True, type=Path)
    parser.add_argument("--dataset", choices=sorted(STANDARD_CHAR_THRESHOLDS))
    parser.add_argument(
        "--char-threshold",
        type=float,
        default=None,
        help="Override the standard dataset-specific threshold.",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    dataset = args.dataset or infer_dataset(args.input) or infer_dataset(args.output_txt)
    char_threshold = args.char_threshold if args.char_threshold is not None else standard_char_threshold(dataset)
    variant = Variant(char_threshold=char_threshold)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)

    cleaned_lines: list[str] = []
    stats: dict[str, int] = {"total": 0}
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            cleaned, reason = clean_prediction(row, variant)
            stats["total"] += 1
            stats[reason] = stats.get(reason, 0) + 1
            item = dict(row)
            item["prediction_raw"] = item.get("prediction", "")
            item["prediction"] = cleaned
            item["postprocess_reason"] = reason
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            cleaned_lines.append(cleaned.replace("\n", " ").replace("\r", " ").strip())

    args.output_txt.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": dataset, "char_threshold": char_threshold, **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
