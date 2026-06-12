#!/usr/bin/env python3
"""Run official-style local evaluators for GER reproduction outputs."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ger_pipeline.config import DEFAULT_MODELS, DEFAULT_PAPER_RANDOM_SEEDS, get_language, get_model
from ger_pipeline.paths import ProjectPaths
from ger_pipeline.results import parse_score_file


@dataclass(frozen=True)
class EvalRun:
    model: str
    method: str
    name: str
    output_path: Path
    retrieval_path: Path | None
    predictions_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--languages", nargs="+", default=["ro", "et"], choices=["ro", "ronacc", "et", "zh", "nlpcc18"])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--train-suffix", default="_8")
    parser.add_argument("--test-suffix", default="_8")
    parser.add_argument("--ro-env-python", type=Path, default=None)
    parser.add_argument("--et-tokenizer-python", type=Path, default=None)
    parser.add_argument("--include-paper-random", action="store_true", help="Also evaluate deferred paper-random seed output directories when present")
    parser.add_argument("--paper-random-seeds", nargs="+", type=int, default=list(DEFAULT_PAPER_RANDOM_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.discover(
        root=args.root,
        cache_dir=args.cache_dir,
        train_suffix=args.train_suffix,
        test_suffix=args.test_suffix,
    )

    if "ro" in args.languages:
        run_rogec_errant(paths, args)
    if "ronacc" in args.languages:
        run_ronacc_readerbench_errant(paths, args)
    if "et" in args.languages:
        run_estgec_modified_m2(paths, args)
    if "zh" in args.languages or "nlpcc18" in args.languages:
        run_nlpcc18_pkunlp_m2(paths, args)


def eval_runs(
    paths: ProjectPaths,
    language: str,
    models: list[str],
    *,
    include_paper_random: bool = False,
    paper_random_seeds: tuple[int, ...] = DEFAULT_PAPER_RANDOM_SEEDS,
) -> list[EvalRun]:
    lang = get_language(language)
    runs: list[EvalRun] = []
    for model_key in models:
        model = get_model(model_key)
        baseline_dir = paths.baseline_dir(model, "test") / lang.test_dataset
        probing_dir = paths.probing_result_dir(model, lang)
        runs.append(
            EvalRun(
                model=model.key,
                method="baseline_initial",
                name=f"{model.key}_baseline",
                output_path=baseline_dir / f"{lang.test_dataset}-output-retokenized.txt",
                retrieval_path=None,
                predictions_path=baseline_dir / "predictions.jsonl",
            )
        )
        runs.append(
            EvalRun(
                model=model.key,
                method="ger",
                name=f"{model.key}_ger",
                output_path=probing_dir / f"{lang.test_dataset}-output-retokenized.txt",
                retrieval_path=paths.retrieval_dir(model, lang) / "retrieval.jsonl",
                predictions_path=probing_dir / "predictions.jsonl",
            )
        )
        if include_paper_random:
            for seed in paper_random_seeds:
                seed_dir = paths.paper_random_dir(model, seed) / lang.test_dataset
                runs.append(
                    EvalRun(
                        model=model.key,
                        method=f"random_8shot_seed{seed}",
                        name=f"{model.key}_random8_seed{seed}",
                        output_path=seed_dir / f"{lang.test_dataset}-output-retokenized.txt",
                        retrieval_path=None,
                        predictions_path=seed_dir / "predictions.jsonl",
                    )
                )
    return runs


def run_rogec_errant(paths: ProjectPaths, args: argparse.Namespace) -> None:
    ro_env_python = args.ro_env_python or paths.root / ".conda_eval_official" / "bin" / "python"
    source = paths.root / "datasets" / "multilingual" / "rogec" / "test.src"
    reference = paths.root / "datasets" / "multilingual" / "rogec" / "test.m2"
    errant_dir = paths.root / "datasets" / "multilingual" / "rogec" / "errant"
    summary_dir = paths.root / "results" / "official_eval" / "rogec"
    rows = []

    for run in eval_runs(paths, "ro", args.models, include_paper_random=args.include_paper_random, paper_random_seeds=tuple(args.paper_random_seeds)):
        out_dir = summary_dir / run.name
        out_dir.mkdir(parents=True, exist_ok=True)
        hyp_m2 = out_dir / "hyp.m2"
        score = out_dir / "errant.score"
        require_file(run.output_path)
        require_file(source)
        require_file(reference)
        require_file(ro_env_python)
        if args.overwrite or not file_ok(hyp_m2):
            command = [
                str(ro_env_python),
                "parallel_to_m2.py",
                "-orig",
                str(source),
                "-cor",
                str(run.output_path),
                "-out",
                str(hyp_m2),
                "-lang",
                "ro",
            ]
            run_command(command, cwd=errant_dir, dry_run=args.dry_run, log_path=out_dir / "parallel_to_m2.log")
        if args.overwrite or not file_ok(score):
            command = [str(ro_env_python), "compare_m2.py", "-hyp", str(hyp_m2), "-ref", str(reference)]
            run_command(command, cwd=errant_dir, dry_run=args.dry_run, log_path=score)
        if not args.dry_run:
            rows.append(
                parse_score_file(
                    score,
                    method=run.method,
                    eval_method="errant_rogec",
                    model=run.model,
                    language="ro",
                    dataset="rogec",
                    retrieval_path=run.retrieval_path,
                    predictions_path=run.predictions_path,
                    output_path=hyp_m2,
                )
            )

    if not args.dry_run:
        write_rogec_summary(rows, summary_dir)


def run_ronacc_readerbench_errant(paths: ProjectPaths, args: argparse.Namespace) -> None:
    ro_env_python = args.ro_env_python or paths.root / ".conda_eval_official" / "bin" / "python"
    source = paths.root / "datasets" / "external" / "ronacc_readerbench" / "test.src"
    reference = paths.root / "datasets" / "external" / "ronacc_readerbench" / "test.m2"
    errant_dir = paths.root / "datasets" / "multilingual" / "rogec" / "errant"
    summary_dir = paths.root / "results" / "official_eval" / "ronacc_readerbench"
    rows = []

    for run in eval_runs(paths, "ro", args.models, include_paper_random=args.include_paper_random, paper_random_seeds=tuple(args.paper_random_seeds)):
        out_dir = summary_dir / run.name
        out_dir.mkdir(parents=True, exist_ok=True)
        hyp_m2 = out_dir / "hyp.m2"
        score = out_dir / "errant.score"
        require_file(run.output_path)
        require_file(source)
        require_file(reference)
        require_file(ro_env_python)
        if args.overwrite or not file_ok(hyp_m2):
            command = [
                str(ro_env_python),
                "parallel_to_m2.py",
                "-orig",
                str(source),
                "-cor",
                str(run.output_path),
                "-out",
                str(hyp_m2),
                "-lang",
                "ro",
            ]
            run_command(command, cwd=errant_dir, dry_run=args.dry_run, log_path=out_dir / "parallel_to_m2.log")
        if args.overwrite or not file_ok(score):
            command = [str(ro_env_python), "compare_m2.py", "-hyp", str(hyp_m2), "-ref", str(reference)]
            run_command(command, cwd=errant_dir, dry_run=args.dry_run, log_path=score)
        if not args.dry_run:
            rows.append(
                parse_score_file(
                    score,
                    method=run.method,
                    eval_method="errant_ronacc_readerbench",
                    model=run.model,
                    language="ro",
                    dataset="ronacc_readerbench",
                    retrieval_path=run.retrieval_path,
                    predictions_path=run.predictions_path,
                    output_path=hyp_m2,
                )
            )

    if not args.dry_run:
        write_errant_summary(rows, summary_dir, "RONACC ReaderBench ERRANT Re-evaluation", "ronacc_readerbench_errant_summary")


def run_estgec_modified_m2(paths: ProjectPaths, args: argparse.Namespace) -> None:
    lang = get_language("et")
    et_tokenizer_python = args.et_tokenizer_python or paths.root / ".conda_eval_estspacy" / "bin" / "python"
    scorer = paths.root / "datasets" / "multilingual_raw" / "ET-estgec" / "M2_scorer_est" / "m2scorer_by_type" / "scripts" / "m2scorer.py"
    reference = paths.root / "datasets" / lang.m2_relative_path
    summary_dir = paths.root / "results" / "official_eval" / "estgec"
    rows = []

    for run in eval_runs(paths, "et", args.models, include_paper_random=args.include_paper_random, paper_random_seeds=tuple(args.paper_random_seeds)):
        out_dir = summary_dir / run.name
        out_dir.mkdir(parents=True, exist_ok=True)
        estspacy_output = out_dir / "estgec-output-estspacy.txt"
        score = out_dir / "estgec_modified_m2.score"
        require_file(run.predictions_path)
        require_file(reference)
        require_file(scorer)
        require_file(et_tokenizer_python)
        if args.overwrite or not file_ok(estspacy_output):
            write_estspacy_retokenized_output(
                et_tokenizer_python,
                run.predictions_path,
                estspacy_output,
                dry_run=args.dry_run,
                log_path=out_dir / "estspacy_retokenize.log",
            )
        if args.overwrite or not file_ok(score):
            command = [str(paths.python), str(scorer), str(estspacy_output), str(reference)]
            run_command(command, cwd=paths.root, dry_run=args.dry_run, log_path=score)
        if not args.dry_run:
            rows.append(
                parse_score_file(
                    score,
                    method=run.method,
                    eval_method="m2scorer_est_modified",
                    model=run.model,
                    language="et",
                    dataset="estgec",
                    retrieval_path=run.retrieval_path,
                    predictions_path=run.predictions_path,
                    output_path=estspacy_output,
                )
            )

    if not args.dry_run:
        write_estgec_summary(rows, summary_dir)


def run_nlpcc18_pkunlp_m2(paths: ProjectPaths, args: argparse.Namespace) -> None:
    lang = get_language("zh")
    segmenter = paths.root / "tools" / "nlpcc18_pkunlp" / "segment_pkunlp.py"
    reference = paths.root / "datasets" / "external" / "nlpcc18_pkunlp" / "Data" / "test" / "nlpcc18.gold.m2.pkunlp"
    scorer = paths.root / "multilingual" / "evaluators" / "m2scorer" / "scripts" / "m2scorer.py"
    summary_dir = paths.root / "results" / "official_eval" / "nlpcc18_pkunlp"
    rows = []

    for run in eval_runs(paths, "zh", args.models, include_paper_random=args.include_paper_random, paper_random_seeds=tuple(args.paper_random_seeds)):
        out_dir = summary_dir / run.name
        out_dir.mkdir(parents=True, exist_ok=True)
        pkunlp_output = out_dir / "nlpcc18-output-pkunlp.txt"
        score = out_dir / "m2scorer.score"
        require_file(run.output_path)
        require_file(reference)
        require_file(scorer)
        require_file(segmenter)
        if args.overwrite or not file_ok(pkunlp_output):
            command = [str(paths.python), str(segmenter), "--output", str(pkunlp_output)]
            run_command_with_stdin(command, input_path=run.output_path, cwd=paths.root, dry_run=args.dry_run, log_path=out_dir / "pkunlp_segment.log")
        if args.overwrite or not file_ok(score):
            command = [str(paths.python), str(scorer), str(pkunlp_output), str(reference)]
            run_command(command, cwd=paths.root, dry_run=args.dry_run, log_path=score)
        if not args.dry_run:
            rows.append(
                parse_score_file(
                    score,
                    method=run.method,
                    eval_method="m2scorer_nlpcc18_pkunlp",
                    model=run.model,
                    language=lang.code,
                    dataset=lang.test_dataset,
                    retrieval_path=run.retrieval_path,
                    predictions_path=run.predictions_path,
                    output_path=pkunlp_output,
                )
            )

    if not args.dry_run:
        write_nlpcc18_pkunlp_summary(rows, summary_dir)


def run_command(command: list[str], *, cwd: Path, dry_run: bool, log_path: Path) -> None:
    print("cd", cwd, "&&", " ".join(command), ">", log_path)
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_command_with_stdin(command: list[str], *, input_path: Path, cwd: Path, dry_run: bool, log_path: Path) -> None:
    print("cd", cwd, "&&", " ".join(command), "<", input_path, ">", log_path)
    if dry_run:
        return
    with input_path.open("r", encoding="utf-8") as stdin, log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdin=stdin, stdout=log, stderr=subprocess.STDOUT, check=True)


def write_estspacy_retokenized_output(
    python: Path,
    predictions_path: Path,
    output_path: Path,
    *,
    dry_run: bool,
    log_path: Path,
) -> None:
    command = [
        str(python),
        "-c",
        ESTSPACY_RETOKENIZE_CODE,
        str(predictions_path),
        str(output_path),
    ]
    run_command(command, cwd=output_path.parents[4], dry_run=dry_run, log_path=log_path)


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


def write_rogec_summary(rows, summary_dir: Path) -> None:
    write_errant_summary(rows, summary_dir, "RoGEC Official ERRANT Re-evaluation", "rogec_errant_summary")


def write_errant_summary(rows, summary_dir: Path, title: str, stem: str) -> None:
    payload = []
    for row in rows:
        tp, fp, fn = parse_errant_counts(row.score_path)
        payload.append((row, tp, fp, fn))

    md = [
        f"# {title}",
        "",
        "| Model | Method | TP | FP | FN | Precision | Recall | F0.5 | Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row, tp, fp, fn in payload:
        md.append(
            f"| {row.model} | {summary_method(row.method)} | {tp} | {fp} | {fn} | {fmt(row.precision)} | {fmt(row.recall)} | {fmt(row.f05)} | `{row.score_path.relative_to(summary_dir.parents[2])}` |"
        )
    (summary_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with (summary_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "method", "tp", "fp", "fn", "precision", "recall", "f0.5", "score"])
        for row, tp, fp, fn in payload:
            writer.writerow([row.model, summary_method(row.method), tp, fp, fn, fmt(row.precision), fmt(row.recall), fmt(row.f05), row.score_path])


def write_estgec_summary(rows, summary_dir: Path) -> None:
    md = [
        "# EstGEC Modified M2 Re-evaluation",
        "",
        "| Model | Method | Precision | Recall | F0.5 | Score |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            f"| {row.model} | {summary_method(row.method)} | {fmt(row.precision)} | {fmt(row.recall)} | {fmt(row.f05)} | `{row.score_path.relative_to(summary_dir.parents[2])}` |"
        )
    (summary_dir / "estgec_modified_m2_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with (summary_dir / "estgec_modified_m2_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "method", "precision", "recall", "f0.5", "score"])
        for row in rows:
            writer.writerow([row.model, summary_method(row.method), fmt(row.precision), fmt(row.recall), fmt(row.f05), row.score_path])


def write_nlpcc18_pkunlp_summary(rows, summary_dir: Path) -> None:
    md = [
        "# NLPCC18 PKUNLP Word-Level M2 Re-evaluation",
        "",
        "| Model | Method | Precision | Recall | F0.5 | Score |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            f"| {row.model} | {summary_method(row.method)} | {fmt(row.precision)} | {fmt(row.recall)} | {fmt(row.f05)} | `{row.score_path.relative_to(summary_dir.parents[2])}` |"
        )
    (summary_dir / "nlpcc18_pkunlp_m2_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with (summary_dir / "nlpcc18_pkunlp_m2_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "method", "precision", "recall", "f0.5", "score"])
        for row in rows:
            writer.writerow([row.model, summary_method(row.method), fmt(row.precision), fmt(row.recall), fmt(row.f05), row.score_path])


def parse_errant_counts(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^TP\s+FP\s+FN\s+Prec\s+Rec\s+F0\.5\s*\n^(\d+)\s+(\d+)\s+(\d+)\s+", text, re.MULTILINE)
    if not match:
        return "", "", ""
    return match.group(1), match.group(2), match.group(3)


def summary_method(method: str) -> str:
    return "baseline" if method == "baseline_initial" else method


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def require_file(path: Path) -> None:
    if not file_ok(path):
        raise FileNotFoundError(path)


if __name__ == "__main__":
    main()
