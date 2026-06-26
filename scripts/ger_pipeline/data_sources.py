"""Standard dataset source material for GER pipeline steps."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import spacy

from .config import LanguageSpec
from .files import require_file
from .paths import ProjectPaths

SPACY_BLANK_MAP = {
    "de_core_news_sm": "de",
    "et_dep_ud_sm": "et",
    "ro_core_news_sm": "ro",
}

TRAIN_TOKENIZER_MODEL = {
    "estgec_train": "et_dep_ud_sm",
    "ronacc_readerbench_train": "ro_core_news_sm",
}


def standard_test_json(paths: ProjectPaths, lang: LanguageSpec) -> Path:
    return paths.datasets_dir / "multilingual" / lang.test_dataset / "test.json"


def test_source_file(paths: ProjectPaths, lang: LanguageSpec) -> Path:
    return paths.multilingual_dir / "runtime_sources" / lang.test_dataset / "test.src"


def train_data_prefix(paths: ProjectPaths, lang: LanguageSpec) -> Path:
    return paths.runtime_train_data_dir / lang.train_dataset / "train"


def standard_train_json(paths: ProjectPaths, lang: LanguageSpec) -> Path:
    return paths.datasets_dir / "multilingual" / lang.train_dataset / "train.json"


def _clean_line(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\n", " ").replace("\r", " ")).strip()


def _load_spacy_or_blank(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError:
        return spacy.blank(SPACY_BLANK_MAP[model_name])


def _retokenize_train_line(line: str, dataset: str):
    model_name = TRAIN_TOKENIZER_MODEL.get(dataset)
    if model_name is None:
        return line
    if not hasattr(_retokenize_train_line, "_models"):
        _retokenize_train_line._models = {}  # type: ignore[attr-defined]
    models = _retokenize_train_line._models  # type: ignore[attr-defined]
    if model_name not in models:
        models[model_name] = _load_spacy_or_blank(model_name)
    text = _clean_line(line)
    if dataset == "ronacc_readerbench_train":
        text = text.replace("(", " ( ").replace("( ", "(")
        text = re.sub(r"\s+", " ", text).strip()
    return " ".join(token.text for token in models[model_name].tokenizer(text)).strip()


def write_standard_test_source(paths: ProjectPaths, lang: LanguageSpec, *, overwrite: bool) -> Path:
    src_path = test_source_file(paths, lang)
    if src_path.exists() and src_path.stat().st_size > 0 and not overwrite:
        return src_path

    json_path = standard_test_json(paths, lang)
    require_file(json_path)
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for index, row in enumerate(rows):
        if "text" not in row:
            raise KeyError(f"{json_path}:{index} has no 'text' field")
        lines.append(_clean_line(row["text"]))

    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    require_file(src_path)
    return src_path


def _write_parallel_train_files(prefix: Path, srcs: list[str], tgts: list[str], *, overwrite: bool) -> None:
    src_path = Path(f"{prefix}.src")
    tgt_path = Path(f"{prefix}.tgt")
    if src_path.exists() and tgt_path.exists() and not overwrite:
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text("\n".join(srcs) + "\n", encoding="utf-8")
    tgt_path.write_text("\n".join(tgts) + "\n", encoding="utf-8")
    require_file(src_path)
    require_file(tgt_path)


def _write_label_file(paths: ProjectPaths, prefix: Path, *, overwrite: bool) -> Path:
    label_path = Path(f"{prefix}.label")
    if label_path.exists() and label_path.stat().st_size > 0 and not overwrite:
        return label_path
    label_path.unlink(missing_ok=True)
    subprocess.run(
        [
            str(paths.python),
            str(paths.preprocess_data),
            "-s",
            str(Path(f"{prefix}.src")),
            "-t",
            str(Path(f"{prefix}.tgt")),
            "-o",
            str(label_path),
        ],
        cwd=paths.root / "scripts" / "ger_runtime",
        check=True,
    )
    require_file(label_path)
    return label_path


def _read_standard_train_json(json_path: Path, dataset: str) -> tuple[list[str], list[str]]:
    require_file(json_path)
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    srcs: list[str] = []
    tgts: list[str] = []
    for index, row in enumerate(rows):
        if "text" not in row:
            raise KeyError(f"{json_path}:{index} has no 'text' field")
        target = row.get("label")
        if target is None:
            labels = row.get("labels")
            if not labels:
                raise KeyError(f"{json_path}:{index} has no 'label' or non-empty 'labels' field")
            target = labels[0]
        srcs.append(_retokenize_train_line(_clean_line(row["text"]), dataset))
        tgts.append(_retokenize_train_line(_clean_line(target), dataset))
    return srcs, tgts


def write_standard_train_data(paths: ProjectPaths, lang: LanguageSpec, *, overwrite: bool) -> Path:
    prefix = train_data_prefix(paths, lang)
    if (
        Path(f"{prefix}.src").exists()
        and Path(f"{prefix}.tgt").exists()
        and Path(f"{prefix}.label").exists()
        and not overwrite
    ):
        return prefix

    srcs, tgts = _read_standard_train_json(standard_train_json(paths, lang), lang.train_dataset)
    _write_parallel_train_files(prefix, srcs, tgts, overwrite=overwrite)
    _write_label_file(paths, prefix, overwrite=overwrite)
    return prefix
