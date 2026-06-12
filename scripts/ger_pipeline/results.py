"""Score parsing and summary generation for GER reproduction runs."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_LANGUAGES, DEFAULT_MODELS, DEFAULT_PAPER_RANDOM_SEEDS, get_language, get_model
from .paths import ProjectPaths


@dataclass(frozen=True)
class Score:
    method: str
    eval_method: str
    model: str
    language: str
    dataset: str
    precision: float | None
    recall: float | None
    f05: float | None
    retrieval_path: Path | None
    predictions_path: Path
    output_path: Path
    score_path: Path
    status: str


_PATTERNS = {
    "precision": re.compile(r"(?:^P\s*=|^Precision\s*:|^precision\s*:)\s*([0-9.]+)", re.MULTILINE),
    "recall": re.compile(r"(?:^R\s*=|^Recall\s*:|^recall\s*:)\s*([0-9.]+)", re.MULTILINE),
    "f05": re.compile(r"(?:^F_0\.5\s*=|^F_0\.5\s*:|^f_0\.5\s*:)\s*([0-9.]+)", re.MULTILINE),
}
_ERRANT_TABLE_PATTERN = re.compile(
    r"^TP\s+FP\s+FN\s+Prec\s+Rec\s+F0\.5\s*\n"
    r"^\d+\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
    re.MULTILINE,
)


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def parse_score_file(
    path: Path,
    *,
    method: str,
    eval_method: str,
    model: str,
    language: str,
    dataset: str,
    retrieval_path: Path | None,
    predictions_path: Path,
    output_path: Path,
) -> Score:
    if not path.exists():
        return Score(
            method,
            eval_method,
            model,
            language,
            dataset,
            None,
            None,
            None,
            retrieval_path,
            predictions_path,
            output_path,
            path,
            "missing",
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    precision, recall, f05 = _parse_json_score(text)
    if precision is None or recall is None or f05 is None:
        precision, recall, f05 = _parse_errant_table_score(text)
    if precision is None or recall is None or f05 is None:
        precision = _last_float(_PATTERNS["precision"], text)
        recall = _last_float(_PATTERNS["recall"], text)
        f05 = _last_float(_PATTERNS["f05"], text)
    status = "ok" if precision is not None and recall is not None and f05 is not None else "unparsed"
    return Score(
        method,
        eval_method,
        model,
        language,
        dataset,
        precision,
        recall,
        f05,
        retrieval_path,
        predictions_path,
        output_path,
        path,
        status,
    )


def _parse_json_score(text: str) -> tuple[float | None, float | None, float | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, None, None
    if not isinstance(payload, dict):
        return None, None, None

    precision = _as_float(payload.get("precision"))
    recall = _as_float(payload.get("recall"))
    f05 = _as_float(payload.get("f_0.5", payload.get("f0.5", payload.get("f05"))))
    return precision, recall, f05


def _parse_errant_table_score(text: str) -> tuple[float | None, float | None, float | None]:
    matches = _ERRANT_TABLE_PATTERN.findall(text)
    if not matches:
        return None, None, None
    precision, recall, f05 = matches[-1]
    return float(precision), float(recall), float(f05)


def _as_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def collect_scores(
    paths: ProjectPaths,
    languages: list[str] | tuple[str, ...] = DEFAULT_LANGUAGES,
    models: list[str] | tuple[str, ...] = DEFAULT_MODELS,
    paper_random_seeds: tuple[int, ...] = DEFAULT_PAPER_RANDOM_SEEDS,
) -> list[Score]:
    rows: list[Score] = []
    for model_key in models:
        model = get_model(model_key)
        for lang_code in languages:
            lang = get_language(lang_code)
            if not lang.local_scoring:
                rows.extend(_submission_rows(paths, model, lang))
                continue
            baseline_dir = paths.baseline_dir(model, "test") / lang.test_dataset
            baseline_score_name = "nlpcc18.score" if lang.test_dataset == "nlpcc18" else f"{lang.test_dataset}.score"
            rows.append(
                parse_score_file(
                    baseline_dir / baseline_score_name,
                    method="baseline_initial",
                    eval_method=_default_eval_method(lang.test_dataset),
                    model=model.key,
                    language=lang.code,
                    dataset=lang.test_dataset,
                    retrieval_path=None,
                    predictions_path=baseline_dir / "predictions.jsonl",
                    output_path=baseline_dir / f"{lang.test_dataset}-output-retokenized.txt",
                )
            )
            rows.append(
                parse_score_file(
                    paths.score_file(model, lang),
                    method="ger",
                    eval_method=_default_eval_method(lang.test_dataset),
                    model=model.key,
                    language=lang.code,
                    dataset=lang.test_dataset,
                    retrieval_path=paths.retrieval_dir(model, lang) / "retrieval.jsonl",
                    predictions_path=paths.probing_result_dir(model, lang) / "predictions.jsonl",
                    output_path=paths.probing_output_file(model, lang),
                )
            )
    rows.extend(_official_rogec_errant_rows(paths, languages=languages, models=models))
    rows.extend(_official_ronacc_readerbench_errant_rows(paths, languages=languages, models=models, seeds=paper_random_seeds))
    rows.extend(_official_estgec_modified_m2_rows(paths, languages=languages, models=models, seeds=paper_random_seeds))
    rows.extend(_official_nlpcc18_pkunlp_m2_rows(paths, languages=languages, models=models, seeds=paper_random_seeds))
    rows.extend(_official_bea19_rows(paths, languages=languages, models=models, seeds=paper_random_seeds))
    rows.extend(_paper_random_rows(paths, languages=languages, models=models, seeds=paper_random_seeds))
    return rows


def _submission_rows(paths: ProjectPaths, model, lang) -> list[Score]:
    baseline_dir = paths.baseline_dir(model, "test") / lang.test_dataset
    probing_dir = paths.probing_result_dir(model, lang)
    return [
        Score(
            method="baseline_initial",
            eval_method="official_submission_only",
            model=model.key,
            language=lang.code,
            dataset=lang.test_dataset,
            precision=None,
            recall=None,
            f05=None,
            retrieval_path=None,
            predictions_path=baseline_dir / "predictions.jsonl",
            output_path=baseline_dir / f"{lang.test_dataset}.zip",
            score_path=baseline_dir / f"{lang.test_dataset}.score",
            status=_submission_status(baseline_dir, lang.test_dataset),
        ),
        Score(
            method="ger",
            eval_method="official_submission_only",
            model=model.key,
            language=lang.code,
            dataset=lang.test_dataset,
            precision=None,
            recall=None,
            f05=None,
            retrieval_path=paths.retrieval_dir(model, lang) / "retrieval.jsonl",
            predictions_path=probing_dir / "predictions.jsonl",
            output_path=probing_dir / f"{lang.test_dataset}.zip",
            score_path=probing_dir / f"{lang.test_dataset}.score",
            status=_submission_status(probing_dir, lang.test_dataset),
        ),
    ]


def _submission_status(result_dir: Path, dataset: str) -> str:
    required = (
        result_dir / "predictions.jsonl",
        result_dir / f"{dataset}.txt",
        result_dir / f"{dataset}.zip",
    )
    return "ready_for_official_submission" if all(path.exists() and path.stat().st_size > 0 for path in required) else "missing"


def _default_eval_method(dataset: str) -> str:
    if dataset == "nlpcc18":
        return "cherrant_char"
    return "m2scorer"


def _official_rogec_errant_rows(
    paths: ProjectPaths,
    *,
    languages: list[str] | tuple[str, ...],
    models: list[str] | tuple[str, ...],
) -> list[Score]:
    if "ro" not in languages:
        return []
    rows: list[Score] = []
    specs = [
        ("llama31", "baseline_initial", "llama31_baseline"),
        ("llama31", "ger", "llama31_ger"),
        ("qwen25", "baseline_initial", "qwen25_baseline"),
        ("qwen25", "ger", "qwen25_ger"),
    ]
    for model_key, method, score_dir_name in specs:
        if model_key not in models:
            continue
        model = get_model(model_key)
        lang = get_language("ro")
        score_dir = paths.root / "results" / "official_eval" / "rogec" / score_dir_name
        baseline_dir = paths.baseline_dir(model, "test") / lang.test_dataset
        is_baseline = method == "baseline_initial"
        rows.append(
            parse_score_file(
                score_dir / "errant.score",
                method=method,
                eval_method="errant_rogec",
                model=model.key,
                language=lang.code,
                dataset=lang.test_dataset,
                retrieval_path=None if is_baseline else paths.retrieval_dir(model, lang) / "retrieval.jsonl",
                predictions_path=(
                    baseline_dir / "predictions.jsonl"
                    if is_baseline
                    else paths.probing_result_dir(model, lang) / "predictions.jsonl"
                ),
                output_path=score_dir / "hyp.m2",
            )
        )
    return rows


def _official_ronacc_readerbench_errant_rows(
    paths: ProjectPaths,
    *,
    languages: list[str] | tuple[str, ...],
    models: list[str] | tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[Score]:
    if "ro" not in languages:
        return []
    rows: list[Score] = []
    specs = [
        ("llama31", "baseline_initial", "llama31_baseline"),
        ("llama31", "ger", "llama31_ger"),
        ("qwen25", "baseline_initial", "qwen25_baseline"),
        ("qwen25", "ger", "qwen25_ger"),
    ]
    for model_key, method, score_dir_name in specs:
        if model_key not in models:
            continue
        model = get_model(model_key)
        lang = get_language("ro")
        score_dir = paths.root / "results" / "official_eval" / "ronacc_readerbench" / score_dir_name
        baseline_dir = paths.baseline_dir(model, "test") / lang.test_dataset
        is_baseline = method == "baseline_initial"
        rows.append(
            parse_score_file(
                score_dir / "errant.score",
                method=method,
                eval_method="errant_ronacc_readerbench",
                model=model.key,
                language="ro",
                dataset="ronacc_readerbench",
                retrieval_path=None if is_baseline else paths.retrieval_dir(model, lang) / "retrieval.jsonl",
                predictions_path=(
                    baseline_dir / "predictions.jsonl"
                    if is_baseline
                    else paths.probing_result_dir(model, lang) / "predictions.jsonl"
                ),
                output_path=score_dir / "hyp.m2",
            )
        )
    for model_key in models:
        model = get_model(model_key)
        lang = get_language("ro")
        rows.extend(
            _official_seed_rows(
                paths,
                model=model,
                lang=lang,
                eval_method="errant_ronacc_readerbench",
                dataset="ronacc_readerbench",
                score_root=paths.root / "results" / "official_eval" / "ronacc_readerbench",
                score_name="errant.score",
                output_name="hyp.m2",
                seeds=seeds,
            )
        )
    rows.extend(_average_seed_rows(rows, seeds=seeds))
    return rows


def _official_estgec_modified_m2_rows(
    paths: ProjectPaths,
    *,
    languages: list[str] | tuple[str, ...],
    models: list[str] | tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[Score]:
    if "et" not in languages:
        return []
    rows: list[Score] = []
    specs = [
        ("llama31", "baseline_initial", "llama31_baseline"),
        ("llama31", "ger", "llama31_ger"),
        ("qwen25", "baseline_initial", "qwen25_baseline"),
        ("qwen25", "ger", "qwen25_ger"),
    ]
    for model_key, method, score_dir_name in specs:
        if model_key not in models:
            continue
        model = get_model(model_key)
        lang = get_language("et")
        score_dir = paths.root / "results" / "official_eval" / "estgec" / score_dir_name
        baseline_dir = paths.baseline_dir(model, "test") / lang.test_dataset
        is_baseline = method == "baseline_initial"
        official_output = score_dir / "estgec-output-estspacy.txt"
        rows.append(
            parse_score_file(
                score_dir / "estgec_modified_m2.score",
                method=method,
                eval_method="m2scorer_est_modified",
                model=model.key,
                language=lang.code,
                dataset=lang.test_dataset,
                retrieval_path=None if is_baseline else paths.retrieval_dir(model, lang) / "retrieval.jsonl",
                predictions_path=(
                    baseline_dir / "predictions.jsonl"
                    if is_baseline
                    else paths.probing_result_dir(model, lang) / "predictions.jsonl"
                ),
                output_path=official_output,
            )
        )
    for model_key in models:
        model = get_model(model_key)
        lang = get_language("et")
        rows.extend(
            _official_seed_rows(
                paths,
                model=model,
                lang=lang,
                eval_method="m2scorer_est_modified",
                dataset=lang.test_dataset,
                score_root=paths.root / "results" / "official_eval" / "estgec",
                score_name="estgec_modified_m2.score",
                output_name="estgec-output-estspacy.txt",
                seeds=seeds,
            )
        )
    rows.extend(_average_seed_rows(rows, seeds=seeds))
    return rows


def _official_nlpcc18_pkunlp_m2_rows(
    paths: ProjectPaths,
    *,
    languages: list[str] | tuple[str, ...],
    models: list[str] | tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[Score]:
    if "zh" not in languages:
        return []
    rows: list[Score] = []
    specs = [
        ("llama31", "baseline_initial", "llama31_baseline"),
        ("llama31", "ger", "llama31_ger"),
        ("qwen25", "baseline_initial", "qwen25_baseline"),
        ("qwen25", "ger", "qwen25_ger"),
    ]
    for model_key, method, score_dir_name in specs:
        if model_key not in models:
            continue
        model = get_model(model_key)
        lang = get_language("zh")
        score_dir = paths.root / "results" / "official_eval" / "nlpcc18_pkunlp" / score_dir_name
        baseline_dir = paths.baseline_dir(model, "test") / lang.test_dataset
        is_baseline = method == "baseline_initial"
        rows.append(
            parse_score_file(
                score_dir / "m2scorer.score",
                method=method,
                eval_method="m2scorer_nlpcc18_pkunlp",
                model=model.key,
                language=lang.code,
                dataset=lang.test_dataset,
                retrieval_path=None if is_baseline else paths.retrieval_dir(model, lang) / "retrieval.jsonl",
                predictions_path=(
                    baseline_dir / "predictions.jsonl"
                    if is_baseline
                    else paths.probing_result_dir(model, lang) / "predictions.jsonl"
                ),
                output_path=score_dir / "nlpcc18-output-pkunlp.txt",
            )
        )
    for model_key in models:
        model = get_model(model_key)
        lang = get_language("zh")
        rows.extend(
            _official_seed_rows(
                paths,
                model=model,
                lang=lang,
                eval_method="m2scorer_nlpcc18_pkunlp",
                dataset=lang.test_dataset,
                score_root=paths.root / "results" / "official_eval" / "nlpcc18_pkunlp",
                score_name="m2scorer.score",
                output_name="nlpcc18-output-pkunlp.txt",
                seeds=seeds,
            )
        )
    rows.extend(_average_seed_rows(rows, seeds=seeds))
    return rows


def _official_seed_rows(
    paths: ProjectPaths,
    *,
    model,
    lang,
    eval_method: str,
    dataset: str,
    score_root: Path,
    score_name: str,
    output_name: str,
    seeds: tuple[int, ...] = DEFAULT_PAPER_RANDOM_SEEDS,
) -> list[Score]:
    rows: list[Score] = []
    for seed in seeds:
        seed_dir = score_root / f"{model.key}_random8_seed{seed}"
        seed_result_dir = paths.paper_random_dir(model, seed) / lang.test_dataset
        seed_row = parse_score_file(
            seed_dir / score_name,
            method=f"random_8shot_seed{seed}",
            eval_method=eval_method,
            model=model.key,
            language=lang.code,
            dataset=dataset,
            retrieval_path=None,
            predictions_path=seed_result_dir / "predictions.jsonl",
            output_path=seed_dir / output_name,
        )
        if seed_row.status != "missing":
            rows.append(seed_row)
    return rows


def _average_seed_rows(rows: list[Score], seeds: tuple[int, ...] = DEFAULT_PAPER_RANDOM_SEEDS) -> list[Score]:
    if len(seeds) != 3 or len(set(seeds)) != 3:
        return []
    expected_methods = {f"random_8shot_seed{seed}" for seed in seeds}
    seed_rows_by_key: dict[tuple[str, str, str, str], list[Score]] = {}
    for row in rows:
        if row.method not in expected_methods:
            continue
        key = (row.eval_method, row.model, row.language, row.dataset)
        seed_rows_by_key.setdefault(key, []).append(row)

    averages: list[Score] = []
    for (eval_method, model_key, lang_code, dataset), seed_rows in sorted(seed_rows_by_key.items()):
        if {row.method for row in seed_rows} != expected_methods:
            continue
        if any(row.status != "ok" or row.precision is None or row.recall is None or row.f05 is None for row in seed_rows):
            continue
        first = seed_rows[0]
        score_path = average_score_path(first.score_path)
        averages.append(
            Score(
                method="random_8shot_3seed_avg",
                eval_method=eval_method,
                model=model_key,
                language=lang_code,
                dataset=dataset,
                precision=sum(row.precision or 0.0 for row in seed_rows) / len(seed_rows),
                recall=sum(row.recall or 0.0 for row in seed_rows) / len(seed_rows),
                f05=sum(row.f05 or 0.0 for row in seed_rows) / len(seed_rows),
                retrieval_path=None,
                predictions_path=score_path.parent / "seed_average_not_a_prediction.jsonl",
                output_path=score_path.parent / "seed_average_not_an_output.txt",
                score_path=score_path,
                status="ok",
            )
        )
    return averages


def _official_bea19_rows(
    paths: ProjectPaths,
    *,
    languages: list[str] | tuple[str, ...],
    models: list[str] | tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[Score]:
    if "en" not in languages and "bea19" not in languages:
        return []
    rows: list[Score] = []
    specs = [
        ("llama31", "baseline_initial", "llama31_baseline"),
        ("llama31", "ger", "llama31_ger"),
        ("qwen25", "baseline_initial", "qwen25_baseline"),
        ("qwen25", "ger", "qwen25_ger"),
    ]
    bea = get_language("bea19")
    for model_key, method, score_dir_name in specs:
        if model_key not in models:
            continue
        model = get_model(model_key)
        score_dir = paths.root / "results" / "official_eval" / "bea19" / score_dir_name
        score_path = score_dir / "official.score"
        if not score_path.exists():
            continue
        is_baseline = method == "baseline_initial"
        baseline_dir = paths.baseline_dir(model, "test") / bea.test_dataset
        probing_dir = paths.probing_result_dir(model, bea)
        rows.append(
            parse_score_file(
                score_path,
                method=method,
                eval_method="errant_bea19_official",
                model=model.key,
                language="en",
                dataset=bea.test_dataset,
                retrieval_path=None if is_baseline else paths.retrieval_dir(model, bea) / "retrieval.jsonl",
                predictions_path=baseline_dir / "predictions.jsonl" if is_baseline else probing_dir / "predictions.jsonl",
                output_path=baseline_dir / "bea19.zip" if is_baseline else probing_dir / "bea19.zip",
            )
        )
    for model_key in models:
        model = get_model(model_key)
        rows.extend(
            _bea19_seed_rows(
                paths,
                model=model,
                bea=bea,
                seeds=seeds,
            )
        )
    rows.extend(_average_seed_rows(rows, seeds=seeds))
    return rows


def _bea19_seed_rows(paths: ProjectPaths, *, model, bea, seeds: tuple[int, ...]) -> list[Score]:
    rows: list[Score] = []
    for seed in seeds:
        score_dir = paths.root / "results" / "official_eval" / "bea19" / f"{model.key}_random8_seed{seed}"
        result_dir = paths.paper_random_dir(model, seed) / bea.test_dataset
        row = parse_score_file(
            score_dir / "official.score",
            method=f"random_8shot_seed{seed}",
            eval_method="errant_bea19_official",
            model=model.key,
            language="en",
            dataset=bea.test_dataset,
            retrieval_path=None,
            predictions_path=result_dir / "predictions.jsonl",
            output_path=result_dir / "bea19.zip",
        )
        if row.status != "missing":
            rows.append(row)
    return rows


def _paper_random_rows(
    paths: ProjectPaths,
    *,
    languages: list[str] | tuple[str, ...],
    models: list[str] | tuple[str, ...],
    seeds: tuple[int, ...] = DEFAULT_PAPER_RANDOM_SEEDS,
) -> list[Score]:
    rows: list[Score] = []
    for model_key in models:
        model = get_model(model_key)
        for lang_code in languages:
            lang = get_language(lang_code)
            for seed in seeds:
                result_dir = paths.paper_random_dir(model, seed) / lang.test_dataset
                score_name = "nlpcc18.score" if lang.test_dataset == "nlpcc18" else f"{lang.test_dataset}.score"
                row = parse_score_file(
                    result_dir / score_name,
                    method=f"random_8shot_seed{seed}",
                    eval_method=_default_eval_method(lang.test_dataset),
                    model=model.key,
                    language=lang.code,
                    dataset=lang.test_dataset,
                    retrieval_path=None,
                    predictions_path=result_dir / "predictions.jsonl",
                    output_path=result_dir / f"{lang.test_dataset}-output-retokenized.txt",
                )
                if row.status == "missing":
                    continue
                rows.append(row)
    rows.extend(_average_seed_rows(rows, seeds=seeds))
    return rows


def average_score_path(seed_score_path: Path) -> Path:
    """Return the stable 3-seed-average score path for a seed score path."""
    replaced = re.sub(r"random8_seed\d+", "random8_3seed_avg", str(seed_score_path))
    return Path(replaced)


def write_csv(rows: list[Score], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "eval_method",
            "model",
            "language",
            "dataset",
            "standard",
            "precision",
            "recall",
            "f0.5",
            "status",
            "retrieval_path",
            "predictions_path",
            "output_path",
            "score_path",
        ])
        for row in rows:
            writer.writerow([
                row.method,
                row.eval_method,
                row.model,
                row.language,
                row.dataset,
                standard_label(row),
                _fmt(row.precision),
                _fmt(row.recall),
                _fmt(row.f05),
                row.status,
                _path_fmt(row.retrieval_path),
                row.predictions_path,
                row.output_path,
                row.score_path,
            ])


def write_markdown(rows: list[Score], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GER Reproduction Summary",
        "",
        "Note: `baseline_initial` is the existing initial/zero-example baseline artifact, not the paper Table 1 `Random` 8-shot three-seed baseline.",
        "Standard labels: `paper_official_local` matches a paper Table 1 dataset/scorer available locally; `blocked_official_server` requires an external official server or withheld gold; `official_style_local` follows a bundled/official-style scorer for a local non-Table-1 dataset; `local_continuity_only` is retained only for continuity and should not be used as the official number; `extra_not_table1` is outside paper Table 1.",
        "",
        "| Method | Eval | Model | Lang | Dataset | Standard | Precision | Recall | F0.5 | Status | Retrieval | Predictions | Output | Score |",
        "|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {eval_method} | {model} | {language} | {dataset} | {standard} | {precision} | {recall} | {f05} | {status} | {retrieval} | `{predictions}` | `{output}` | `{score}` |".format(
                method=row.method,
                eval_method=row.eval_method,
                model=row.model,
                language=row.language,
                dataset=row.dataset,
                standard=standard_label(row),
                precision=_fmt(row.precision),
                recall=_fmt(row.recall),
                f05=_fmt(row.f05),
                status=row.status,
                retrieval=_markdown_path(row.retrieval_path),
                predictions=row.predictions_path,
                output=row.output_path,
                score=row.score_path,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def standard_label(row: Score) -> str:
    if row.eval_method == "errant_bea19_official" and row.dataset == "bea19":
        return "paper_official_external"
    if row.eval_method == "official_submission_only" and row.dataset == "bea19":
        return "blocked_official_server"
    if row.eval_method == "m2scorer" and row.dataset in {"conll14", "falko_merlin"}:
        return "paper_official_local"
    if row.eval_method == "m2scorer_est_modified" and row.dataset == "estgec":
        return "paper_official_local"
    if row.eval_method == "errant_ronacc_readerbench" and row.dataset == "ronacc_readerbench":
        return "paper_official_local"
    if row.eval_method == "errant_rogec" and row.dataset == "rogec":
        return "official_style_local"
    if row.eval_method == "m2scorer_nlpcc18_pkunlp" and row.dataset == "nlpcc18":
        return "official_style_local"
    if row.dataset in {"rulec", "nlpcc18"}:
        return "extra_not_table1"
    if row.eval_method == "m2scorer" and row.dataset in {"estgec", "rogec"}:
        return "local_continuity_only"
    return "local_continuity_only"


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _path_fmt(path: Path | None) -> str:
    return "" if path is None else str(path)


def _markdown_path(path: Path | None) -> str:
    return "" if path is None else f"`{path}`"
