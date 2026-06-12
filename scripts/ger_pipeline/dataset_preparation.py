"""Prepare GER standard JSON datasets from local official/raw sources."""
from __future__ import annotations

import json
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import spacy

from .config import LanguageSpec
from .files import file_ok, require_file
from .paths import ProjectPaths


BLANK_ROW = {
    "id": "-1",
    "text": "Intentionally blank.",
    "labels": ["Intentionally blank."],
    "label": "Intentionally blank.",
}

SPACY_BLANK_MAP = {
    "et_dep_ud_sm": "et",
    "ro_core_web_sm": "ro",
}


def prepare_language_datasets(paths: ProjectPaths, lang: LanguageSpec, *, overwrite: bool) -> None:
    if lang.code == "en":
        prepare_conll14(paths, overwrite=overwrite)
        prepare_wilocness(paths, overwrite=overwrite)
    elif lang.code == "bea19":
        prepare_bea19(paths, overwrite=overwrite)
        prepare_wilocness(paths, overwrite=overwrite)
    elif lang.code == "de":
        prepare_falko_merlin(paths, overwrite=overwrite)
    elif lang.code == "ro":
        prepare_rogec(paths, overwrite=overwrite)
        prepare_ronacc_readerbench(paths, overwrite=overwrite)
    elif lang.code == "et":
        prepare_estgec(paths, overwrite=overwrite)
    else:
        raise ValueError(f"Unsupported GER language for dataset preparation: {lang.code}")


def write_json(path: Path, rows: list[dict[str, object]], *, overwrite: bool) -> Path:
    if file_ok(path) and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    require_file(path)
    return path


def write_blank_splits(dataset_dir: Path, splits: tuple[str, ...], *, overwrite: bool) -> None:
    for split in splits:
        write_json(dataset_dir / f"{split}.json", [dict(BLANK_ROW)], overwrite=overwrite)


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def tokenized_row(dataset: str, index: int, text: str, label: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "id": f"{index}_{dataset}",
        "text": text,
        "src_tokens": text.split(),
    }
    if label is not None:
        row["label"] = label
        row["labels"] = [label]
        row["tgt_tokens"] = label.split()
    return row


def labeled_row(dataset: str, index: int, source: str, target: str) -> dict[str, object]:
    return {
        "id": f"{index}_{dataset}",
        "text": source,
        "labels": [target],
        "label": target,
    }


def read_m2_sources(path: Path) -> list[str]:
    require_file(path)
    return [line[2:].rstrip("\n") for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("S ")]


def read_m2_pairs(path: Path) -> list[tuple[str, str]]:
    require_file(path)
    skip_edits = {"noop", "UNK", "Um"}
    pairs: list[tuple[str, str]] = []
    source_tokens: list[str] | None = None
    target_tokens: list[str] | None = None
    offset = 0
    for raw_line in [*path.read_text(encoding="utf-8", errors="replace").splitlines(), ""]:
        line = raw_line.strip()
        if line:
            prefix = line[0]
            remainder = line[2:]
            if prefix == "S":
                source_tokens = remainder.split(" ")
                target_tokens = deepcopy(source_tokens)
                offset = 0
            elif prefix == "A" and target_tokens is not None:
                fields = remainder.split("|||")
                start, end = map(int, fields[0].split())
                edit_type, edit_text = fields[1], fields[2]
                if edit_type in skip_edits:
                    continue
                if edit_text in {"", "-NONE-"}:
                    for idx in range(start, end):
                        del target_tokens[offset + idx]
                        offset -= 1
                else:
                    edit_tokens = edit_text.split(" ")
                    delta = len(edit_tokens) - (end - start)
                    target_tokens[offset + start : offset + end] = edit_tokens
                    offset += delta
        elif source_tokens is not None and target_tokens is not None:
            pairs.append((" ".join(source_tokens).strip(), " ".join(target_tokens).strip()))
            source_tokens = None
            target_tokens = None
            offset = 0
    return pairs


def prepare_conll14(paths: ProjectPaths, *, overwrite: bool) -> None:
    dataset_dir = paths.datasets_dir / "multilingual" / "conll14"
    m2_path = paths.datasets_dir / "multilingual_raw" / "EN-conll14st-test-data" / "noalt" / "official-2014.combined.m2"
    rows = [tokenized_row("conll14", idx, source) for idx, source in enumerate(read_m2_sources(m2_path))]
    write_blank_splits(dataset_dir, ("train", "valid"), overwrite=overwrite)
    write_json(dataset_dir / "test.json", rows, overwrite=overwrite)


def prepare_bea19(paths: ProjectPaths, *, overwrite: bool) -> None:
    dataset_dir = paths.datasets_dir / "multilingual" / "bea19"
    source_path = paths.datasets_dir / "multilingual_raw" / "EN-wi+locness" / "test" / "ABCN.test.bea19.orig"
    require_file(source_path)
    rows = [
        tokenized_row("bea19", idx, clean_space(line))
        for idx, line in enumerate(source_path.read_text(encoding="utf-8", errors="replace").splitlines())
    ]
    write_blank_splits(dataset_dir, ("train", "valid"), overwrite=overwrite)
    write_json(dataset_dir / "test.json", rows, overwrite=overwrite)


def prepare_wilocness(paths: ProjectPaths, *, overwrite: bool) -> None:
    dataset_dir = paths.datasets_dir / "multilingual" / "wilocness"
    write_wilocness_train_json(paths, overwrite=overwrite)
    write_blank_splits(dataset_dir, ("valid",), overwrite=overwrite)


def wilocness_train_json(paths: ProjectPaths) -> Path:
    return paths.datasets_dir / "multilingual" / "wilocness" / "train.json"


def wilocness_train_m2(paths: ProjectPaths) -> Path:
    return paths.datasets_dir / "multilingual_raw" / "EN-wi+locness" / "m2" / "ABC.train.gold.bea19.m2"


def write_wilocness_train_json(paths: ProjectPaths, *, overwrite: bool) -> Path:
    json_path = wilocness_train_json(paths)
    if file_ok(json_path) and not overwrite:
        return json_path
    rows: list[dict[str, object]] = []
    for source, target in read_m2_pairs(wilocness_train_m2(paths)):
        source_tokens = source.split()
        if len(source_tokens) <= 5:
            continue
        rows.append(
            {
                "id": f"{len(rows)}_wilocness",
                "text": source,
                "src_tokens": source_tokens,
                "label": target,
                "labels": [target],
                "tgt_tokens": target.split(),
            }
        )
    if not rows:
        raise ValueError(f"No WI+LOCNESS train rows generated from {wilocness_train_m2(paths)}")
    return write_json(json_path, rows, overwrite=True)


def prepare_falko_merlin(paths: ProjectPaths, *, overwrite: bool) -> None:
    raw_dir = paths.datasets_dir / "multilingual_raw" / "DE-FALKO-MERLIN"
    dataset_dir = paths.datasets_dir / "multilingual" / "falko_merlin"
    split_map = {"train": "train", "valid": "dev", "test": "test"}
    prepared_rows: dict[str, list[dict[str, object]]] = {}
    for split, raw_split in split_map.items():
        rows = read_parallel_json_rows(
            raw_dir / f"fm-{raw_split}.src",
            raw_dir / f"fm-{raw_split}.trg",
            dataset="falko_merlin",
        )
        prepared_rows[split] = rows
        write_json(dataset_dir / f"{split}.json", rows, overwrite=overwrite)

    train_dir = paths.datasets_dir / "multilingual" / "falko_merlin_train"
    write_json(train_dir / "train.json", prepared_rows["train"], overwrite=overwrite)
    write_json(train_dir / "valid.json", prepared_rows["valid"], overwrite=overwrite)


def _load_spacy_or_blank(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError:
        return spacy.blank(SPACY_BLANK_MAP[model_name])


def tokenize_with_spacy(text: str, model_name: str) -> str:
    if not hasattr(tokenize_with_spacy, "_models"):
        tokenize_with_spacy._models = {}  # type: ignore[attr-defined]
    models = tokenize_with_spacy._models  # type: ignore[attr-defined]
    if model_name not in models:
        models[model_name] = _load_spacy_or_blank(model_name)
    return " ".join(token.text for token in models[model_name].tokenizer(clean_space(text))).strip()


def read_parallel_json_rows(src_path: Path, tgt_path: Path, *, dataset: str) -> list[dict[str, object]]:
    require_file(src_path)
    require_file(tgt_path)
    srcs = src_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tgts = tgt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(srcs) != len(tgts):
        raise ValueError(f"Parallel files have different line counts: {src_path} ({len(srcs)}) vs {tgt_path} ({len(tgts)})")
    return [labeled_row(dataset, idx, clean_space(src), clean_space(tgt)) for idx, (src, tgt) in enumerate(zip(srcs, tgts, strict=True))]


def read_alternating_corrected_erroneous(path: Path, *, dataset: str, tokenizer_model: str | None = None) -> list[dict[str, object]]:
    require_file(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) % 2:
        raise ValueError(f"Expected even number of lines in alternating corrected/erroneous file: {path}")
    rows: list[dict[str, object]] = []
    for idx in range(0, len(lines), 2):
        target = lines[idx]
        source = lines[idx + 1]
        if tokenizer_model:
            source = tokenize_with_spacy(source, tokenizer_model)
            target = tokenize_with_spacy(target, tokenizer_model)
        else:
            source = clean_space(source)
            target = clean_space(target)
        rows.append(labeled_row(dataset, idx // 2, source, target))
    return rows


def prepare_rogec(paths: ProjectPaths, *, overwrite: bool) -> None:
    raw_dir = paths.datasets_dir / "multilingual_raw" / "RO-RoGEC"
    dataset_dir = paths.datasets_dir / "multilingual" / "rogec"
    split_map = {"train": "train", "valid": "dev", "test": "test"}
    prepared_rows: dict[str, list[dict[str, object]]] = {}
    for split, raw_split in split_map.items():
        rows = read_alternating_corrected_erroneous(raw_dir / f"{raw_split}.txt", dataset="rogec", tokenizer_model="ro_core_web_sm")
        prepared_rows[split] = rows
        write_json(dataset_dir / f"{split}.json", rows, overwrite=overwrite)
    write_parallel_text(dataset_dir / "test.src", dataset_dir / "test.tgt", prepared_rows["test"], overwrite=overwrite)

    train_dir = paths.datasets_dir / "multilingual" / "rogec_train"
    write_json(train_dir / "train.json", prepared_rows["train"], overwrite=overwrite)
    write_json(train_dir / "valid.json", prepared_rows["valid"], overwrite=overwrite)


def write_parallel_text(src_path: Path, tgt_path: Path, rows: list[dict[str, object]], *, overwrite: bool) -> None:
    if file_ok(src_path) and file_ok(tgt_path) and not overwrite:
        return
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text("\n".join(str(row["text"]) for row in rows) + "\n", encoding="utf-8")
    tgt_path.write_text("\n".join(str(row["label"]) for row in rows) + "\n", encoding="utf-8")
    require_file(src_path)
    require_file(tgt_path)


def prepare_ronacc_readerbench(paths: ProjectPaths, *, overwrite: bool) -> None:
    out_dir = paths.datasets_dir / "external" / "ronacc_readerbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        raw_path = out_dir / f"{split}.txt"
        if not raw_path.exists():
            continue
        lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) % 2:
            raise ValueError(f"Expected even number of lines in {raw_path}")
        tgts = [clean_space(line) for line in lines[0::2]]
        srcs = [clean_space(line) for line in lines[1::2]]
        src_path = out_dir / f"{split}.src"
        tgt_path = out_dir / f"{split}.tgt"
        if overwrite or not file_ok(src_path):
            src_path.write_text("\n".join(srcs) + "\n", encoding="utf-8")
        if overwrite or not file_ok(tgt_path):
            tgt_path.write_text("\n".join(tgts) + "\n", encoding="utf-8")
    maybe_generate_ronacc_m2(paths, overwrite=overwrite)


def maybe_generate_ronacc_m2(paths: ProjectPaths, *, overwrite: bool) -> None:
    out_dir = paths.datasets_dir / "external" / "ronacc_readerbench"
    m2_path = out_dir / "test.m2"
    if file_ok(m2_path) and not overwrite:
        return
    source = out_dir / "test.src"
    target = out_dir / "test.tgt"
    errant_dir = paths.datasets_dir / "multilingual" / "rogec" / "errant"
    ro_python = paths.root / ".conda_eval_official" / "bin" / "python"
    if not (file_ok(source) and file_ok(target) and ro_python.exists() and (errant_dir / "parallel_to_m2.py").exists()):
        return
    tmp_m2 = m2_path.with_suffix(".m2.tmp")
    tmp_m2.unlink(missing_ok=True)
    ca_bundle = paths.root / ".conda_eval_official" / "ssl" / "cacert.pem"
    env = os.environ.copy()
    if ca_bundle.exists():
        env.update(
            {
                "SSL_CERT_FILE": str(ca_bundle),
                "REQUESTS_CA_BUNDLE": str(ca_bundle),
                "SSL_CERT_DIR": "/etc/ssl/certs",
            }
        )
    subprocess.run(
        [str(ro_python), "parallel_to_m2.py", "-orig", str(source), "-cor", str(target), "-out", str(tmp_m2), "-lang", "ro"],
        cwd=errant_dir,
        env=env,
        check=True,
    )
    require_file(tmp_m2)
    tmp_m2.replace(m2_path)
    require_file(m2_path)


def parse_estgec_train(path: Path) -> list[tuple[str, str]]:
    require_file(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs: list[tuple[str, str]] = []
    pattern = re.compile(r"<mistake>\s*<original>\s*(.*?)\s*</original>\s*<correction>\s*(.*?)\s*</correction>", re.S)
    for match in pattern.finditer(text):
        source = match.group(1).strip()
        target = match.group(2).strip()
        if source and target:
            pairs.append((source, target))
    return pairs


def prepare_estgec(paths: ProjectPaths, *, overwrite: bool) -> None:
    raw_root = paths.datasets_dir / "multilingual_raw" / "ET-estgec"
    dataset_dir = paths.datasets_dir / "multilingual" / "estgec"
    train_pairs = parse_estgec_train(raw_root / "Tartu_L2_corpus" / "Tartu_L2_learner_corpus_parallel.txt")
    train_rows = [
        labeled_row(
            "estgec",
            idx + 1,
            tokenize_with_spacy(source, "et_dep_ud_sm"),
            tokenize_with_spacy(target, "et_dep_ud_sm"),
        )
        for idx, (source, target) in enumerate(train_pairs)
    ]
    test_rows = [
        labeled_row("estgec", idx, source.strip(), target.strip())
        for idx, (source, target) in enumerate(read_m2_pairs(raw_root / "Tartu_L1_corpus" / "test" / "test_m2.txt"))
    ]
    write_json(dataset_dir / "train.json", train_rows, overwrite=overwrite)
    write_blank_splits(dataset_dir, ("valid",), overwrite=overwrite)
    write_json(dataset_dir / "test.json", test_rows, overwrite=overwrite)

    train_dir = paths.datasets_dir / "multilingual" / "estgec_train"
    write_json(train_dir / "train.json", train_rows, overwrite=overwrite)
    write_blank_splits(train_dir, ("valid",), overwrite=overwrite)
