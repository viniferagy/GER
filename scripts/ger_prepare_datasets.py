#!/usr/bin/env python3
"""Prepare GER Table 1 standard datasets from local official/raw sources."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ger_pipeline.config import TABLE1_LANGUAGES, get_language  # noqa: E402
from ger_pipeline.dataset_preparation import prepare_language_datasets  # noqa: E402
from ger_pipeline.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--train-suffix", default="_8")
    parser.add_argument("--test-suffix", default="_8")
    parser.add_argument("--languages", nargs="+", default=list(TABLE1_LANGUAGES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.discover(args.root, args.cache_dir, args.train_suffix, args.test_suffix)
    for code in args.languages:
        lang = get_language(code)
        print(f"[prepare] {code}", flush=True)
        prepare_language_datasets(paths, lang, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
