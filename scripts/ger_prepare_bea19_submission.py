#!/usr/bin/env python3
"""Prepare or validate BEA-19 official submission zip files."""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import spacy

from ger_pipeline.config import DEFAULT_MODELS, get_model
from ger_pipeline.paths import ProjectPaths
from ger_pipeline.validation import file_ok


@dataclass(frozen=True)
class SubmissionRow:
    model: str
    method: str
    status: str
    source_path: Path
    txt_path: Path
    zip_path: Path
    line_count: int | None
    zip_entries: str
    blind_source_lines: int | None
    issue: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--methods", nargs="+", default=["baseline_initial", "ger"])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--train-suffix", default="_8")
    parser.add_argument("--test-suffix", default="_8")
    parser.add_argument("--input", type=Path, default=None, help="Explicit input predictions.jsonl or plain text for a single package")
    parser.add_argument("--model", choices=list(DEFAULT_MODELS), default=None, help="Model for --input")
    parser.add_argument("--method", default=None, help="Method name for --input")
    parser.add_argument("--write", action="store_true", help="Write bea19.txt and bea19.zip when the source exists")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.discover(
        root=args.root,
        cache_dir=args.cache_dir,
        train_suffix=args.train_suffix,
        test_suffix=args.test_suffix,
    )
    rows = build_rows(paths, args)
    markdown_path = args.markdown or paths.root / "results" / "bea19_submission" / "bea19_submission_plan.md"
    csv_path = args.csv or paths.root / "results" / "bea19_submission" / "bea19_submission_plan.csv"
    write_markdown(rows, markdown_path)
    write_csv(rows, csv_path)
    print(markdown_path)
    print(csv_path)


def build_rows(paths: ProjectPaths, args: argparse.Namespace) -> list[SubmissionRow]:
    if args.input:
        if not args.model or not args.method:
            raise ValueError("--input requires --model and --method")
        return [prepare_one(paths, args.model, args.method, args.input.resolve(), args.write)]

    rows: list[SubmissionRow] = []
    for model_key in args.models:
        for method in args.methods:
            rows.append(prepare_one(paths, model_key, method, default_source(paths, model_key, method), args.write))
    return rows


def default_source(paths: ProjectPaths, model_key: str, method: str) -> Path:
    model = get_model(model_key)
    if method == "baseline_initial":
        return paths.baseline_dir(model, "test") / "bea19" / "predictions.jsonl"
    if method == "ger":
        return (
            paths.multilingual_dir
            / paths.sentence_result_dir_name(model)
            / f"icl_deepseek_res_probing_pgy_{model.retrieve_dim}{paths.train_suffix}{paths.test_suffix}"
            / "bea19"
            / "predictions.jsonl"
        )
    return paths.root / "results" / "bea19_submission" / model_key / method / "predictions.jsonl"


def prepare_one(paths: ProjectPaths, model_key: str, method: str, source: Path, write: bool) -> SubmissionRow:
    out_dir = paths.root / "results" / "bea19_submission" / model_key / method
    txt_path = out_dir / "bea19.txt"
    zip_path = out_dir / "bea19.zip"
    issues: list[str] = []
    blind_source_lines = count_lines(paths.bea19_blind_source)
    if blind_source_lines is None:
        issues.append(f"missing blind source {paths.bea19_blind_source}")

    if write and file_ok(source):
        predictions = load_predictions(source)
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(bea_postprocess(prediction) for prediction in predictions) + "\n", encoding="utf-8")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(txt_path, "bea19.txt")

    txt_path = existing_or_output_path(txt_path, source.parent / "bea19.txt")
    zip_path = existing_or_output_path(zip_path, source.parent / "bea19.zip")

    if not file_ok(source):
        issues.append("missing source predictions")
    line_count = count_lines(txt_path)
    if file_ok(txt_path) and blind_source_lines is not None and line_count != blind_source_lines:
        issues.append(f"bea19.txt lines {line_count} != blind source lines {blind_source_lines}")
    elif not file_ok(txt_path):
        issues.append("missing bea19.txt")

    entries = zip_entries(zip_path)
    if not file_ok(zip_path):
        issues.append("missing bea19.zip")
    elif entries != "bea19.txt":
        issues.append(f"unexpected zip entries {entries}")

    status = "ready" if not issues else "pending"
    return SubmissionRow(
        model=model_key,
        method=method,
        status=status,
        source_path=source,
        txt_path=txt_path,
        zip_path=zip_path,
        line_count=line_count,
        zip_entries=entries,
        blind_source_lines=blind_source_lines,
        issue="; ".join(issues),
    )


def existing_or_output_path(output_path: Path, existing_path: Path) -> Path:
    if file_ok(output_path):
        return output_path
    if file_ok(existing_path):
        return existing_path
    return output_path


def load_predictions(source: Path) -> list[str]:
    if source.suffix == ".jsonl":
        predictions: list[str] = []
        with source.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                predictions.append(str(payload.get("predict", payload.get("prediction", payload.get("output", "")))))
        return predictions
    return source.read_text(encoding="utf-8", errors="replace").splitlines()


_NLP = None


def bea_postprocess(text: str) -> str:
    answer = gec_reform_answer(str(text))
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    line = " ".join(token.text for token in _NLP(answer))
    line = re.sub(r"(?<=\d)\s+%", "%", line)
    line = re.sub(r"((?:have)|(?:has)) n't", r"\1n't", line)
    line = re.sub(r"^-", "- ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def gec_reform_answer(text: str) -> str:
    marker_patterns = [
        r"(?is).*?(?:corrected sentence|correction|answer)\s*[:：]\s*(.+)$",
        r"(?is).*?<answer>\s*(.*?)\s*</answer>.*",
    ]
    for pattern in marker_patterns:
        match = re.match(pattern, text.strip())
        if match:
            return match.group(1).strip()
    return text.strip()


def count_lines(path: Path) -> int | None:
    if not file_ok(path):
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def zip_entries(path: Path) -> str:
    if not file_ok(path):
        return ""
    with zipfile.ZipFile(path) as zipf:
        return ";".join(zipf.namelist())


def write_markdown(rows: list[SubmissionRow], path: Path) -> None:
    lines = [
        "# BEA-19 Submission Plan",
        "",
        "Generated by `scripts/ger_prepare_bea19_submission.py`. The official BEA-19 blind test requires submitting `bea19.zip`; this script validates or creates the package from local predictions when available.",
        "",
        "| Model | Method | Status | Blind Source Lines | Submission Lines | Zip Entries | Source | TXT | ZIP | Issue |",
        "|---|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {method} | {status} | {blind_lines} | {lines} | {entries} | `{source}` | `{txt}` | `{zip}` | {issue} |".format(
                model=row.model,
                method=row.method,
                status=row.status,
                blind_lines="" if row.blind_source_lines is None else row.blind_source_lines,
                lines="" if row.line_count is None else row.line_count,
                entries=row.zip_entries,
                source=row.source_path,
                txt=row.txt_path,
                zip=row.zip_path,
                issue=row.issue,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[SubmissionRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "method", "status", "source_path", "txt_path", "zip_path", "blind_source_lines", "line_count", "zip_entries", "issue"])
        for row in rows:
            writer.writerow([row.model, row.method, row.status, row.source_path, row.txt_path, row.zip_path, row.blind_source_lines or "", row.line_count or "", row.zip_entries, row.issue])


if __name__ == "__main__":
    main()
