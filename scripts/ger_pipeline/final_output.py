"""Formal output construction and GER standard postprocessing."""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import LanguageSpec, ModelSpec
from .files import file_ok, require_file
from .postprocess import Variant, clean_prediction, standard_char_threshold


TOKENIZER_MODELS = {
    "de": "de_core_news_sm",
    "ro": "ro_core_news_sm",
    "et": "et_dep_ud_sm",
}


@dataclass(frozen=True)
class FinalRun:
    model: ModelSpec
    lang: LanguageSpec
    method: str
    seed: int
    retrieval_dir: Path
    result_dir: Path
    score_dir: Path
    prompt: str
    dynamic_examples: bool = False
    use_standard_postprocess: bool = False

    @property
    def dataset(self) -> str:
        return self.lang.test_dataset

    @property
    def raw_predictions(self) -> Path:
        return self.result_dir / "predictions.jsonl"

    @property
    def final_predictions(self) -> Path:
        return self.score_dir / "predictions.jsonl"

    @property
    def final_output(self) -> Path:
        if self.lang.code == "en":
            return self.score_dir / "conll14.txt"
        if self.lang.submission_output:
            return self.score_dir / f"{self.dataset}.txt"
        return self.score_dir / f"{self.dataset}-output-retokenized.txt"

    @property
    def final_artifact(self) -> Path:
        if self.lang.code == "en":
            return self.score_dir / "conll14.score"
        if self.lang.code == "de":
            return self.score_dir / "falko_merlin.score"
        if self.lang.code == "ro":
            return self.score_dir / "errant.score"
        if self.lang.code == "et":
            return self.score_dir / "estgec_modified_m2.score"
        if self.lang.submission_output:
            return self.score_dir / f"{self.dataset}.zip"
        return self.score_dir / f"{self.dataset}.score"


def runtime_normalize_line(line: str, dataset: str) -> str:
    normalized = line.replace("\n", " ").replace("\r", " ").strip()
    if dataset in {"ronacc_readerbench", "ronacc_readerbench_train"}:
        normalized = normalized.replace("(", " ( ").replace("  ", " ").replace("( ", "(")
    return normalized


def retokenize_lines_for_runtime(run: FinalRun, lines: list[str]) -> list[str]:
    model_name = TOKENIZER_MODELS.get(run.lang.code)
    if not model_name:
        return lines
    from ger_runtime.inference.evaluators.postprocess import load_spacy_or_blank

    tokenizer_model = load_spacy_or_blank(model_name)
    retokenized: list[str] = []
    for line in lines:
        normalized = runtime_normalize_line(line, run.dataset)
        retokenized.append(" ".join(token.text for token in tokenizer_model.tokenizer(normalized)).strip())
    return retokenized


def write_formal_predictions(run: FinalRun, *, execute: bool, overwrite: bool) -> tuple[Path, Path]:
    print(f"[step] formal-output {run.model.key}/{run.lang.code}/{run.method}/seed{run.seed}", flush=True)
    print(f"       input={run.raw_predictions}", flush=True)
    print(f"       output={run.final_output}", flush=True)
    if not execute:
        return run.final_predictions, run.final_output
    if file_ok(run.final_predictions) and file_ok(run.final_output) and not overwrite:
        if run.lang.submission_output and not file_ok(run.final_artifact):
            with zipfile.ZipFile(run.final_artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(run.final_output, f"{run.dataset}.txt")
        return run.final_predictions, run.final_output
    require_file(run.raw_predictions)
    rows = [json.loads(line) for line in run.raw_predictions.open(encoding="utf-8") if line.strip()]
    run.score_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int | float | str | bool] = {
        "dataset": run.dataset,
        "method": run.method,
        "standard_postprocess": run.use_standard_postprocess,
        "total": 0,
    }
    if run.use_standard_postprocess:
        variant = Variant(char_threshold=standard_char_threshold(run.dataset))
        stats["char_threshold"] = variant.char_threshold
    lines: list[str] = []
    with run.final_predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = dict(row)
            if run.use_standard_postprocess:
                cleaned, reason = clean_prediction(row, variant)
                item["prediction_raw"] = item.get("prediction", "")
                item["prediction"] = cleaned
                item["postprocess_reason"] = reason
            else:
                cleaned = str(item.get("prediction", "")).strip() or str(item.get("text", "")).strip()
                reason = "kept"
            stats["total"] = int(stats["total"]) + 1
            stats[reason] = int(stats.get(reason, 0)) + 1
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            lines.append(cleaned.replace("\n", " ").replace("\r", " ").strip())
    lines = retokenize_lines_for_runtime(run, lines)
    run.final_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if run.lang.code == "en":
        (run.score_dir / "conll14-output-retokenized.txt").write_text(run.final_output.read_text(encoding="utf-8"), encoding="utf-8")
    if run.lang.submission_output:
        with zipfile.ZipFile(run.final_artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(run.final_output, f"{run.dataset}.txt")
    (run.score_dir / "output.stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run.final_predictions, run.final_output
