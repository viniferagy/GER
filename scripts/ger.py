#!/usr/bin/env python3
"""Run the modular GER pipeline.

This script builds the GER-specific retrieval from initial predictions, then
feeds that retrieval into the shared retrieved-ICL final generation and scoring
path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ger_pipeline.config import get_language, get_model  # noqa: E402
from ger_pipeline.paths import ProjectPaths  # noqa: E402
from ger_pipeline.workflow import (  # noqa: E402
    add_common_args,
    default_final_root,
    default_score_root,
    ensure_ger_final_retrieval,
    ensure_ger_source_retrieval,
    ensure_initial_predictions,
    ensure_representation_cache,
    make_final_run,
    run_final_generation,
    score_formal_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--start-at",
        choices=("initial", "cache", "retrieval", "final", "score"),
        default="initial",
        help="Resume point. Earlier steps are skipped.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("initial", "cache", "retrieval", "final", "score"),
        default="score",
        help="Stop after this step.",
    )
    parser.add_argument(
        "--no-standard-postprocess",
        action="store_true",
        help="Disable standard output cleanup before formal scoring. GER enables it by default.",
    )
    return parser.parse_args()


def step_enabled(current: str, start_at: str, stop_after: str) -> bool:
    order = ["initial", "cache", "retrieval", "final", "score"]
    return order.index(start_at) <= order.index(current) <= order.index(stop_after)


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.discover(args.root, args.cache_dir, args.train_suffix, args.test_suffix)
    final_root = (args.final_root or default_final_root(paths, get_model(args.models[0])).parent).resolve()
    score_root = (args.score_root or default_score_root(paths)).resolve()
    use_postprocess = not args.no_standard_postprocess

    print(f"Project root: {paths.root}")
    print(f"Mode: {'execute' if args.execute else 'dry-run'}")
    print(f"Final root parent: {final_root}")
    print(f"Score root: {score_root}")

    for model_key in args.models:
        model = get_model(model_key)
        model_final_root = final_root / f"results_ger_{model.key}"
        for lang_code in args.languages:
            lang = get_language(lang_code)
            if step_enabled("initial", args.start_at, args.stop_after):
                ensure_initial_predictions(
                    paths,
                    model,
                    lang,
                    gpu=args.gpus,
                    execute=args.execute,
                    overwrite=args.overwrite,
                    use_run_and_hold=args.run_and_hold,
                )
            if step_enabled("cache", args.start_at, args.stop_after):
                ensure_representation_cache(
                    paths,
                    model,
                    lang,
                    gpu=args.gpus,
                    execute=args.execute,
                    overwrite=args.overwrite,
                    use_run_and_hold=args.run_and_hold,
                )
            if step_enabled("retrieval", args.start_at, args.stop_after):
                ensure_ger_source_retrieval(
                    paths,
                    model,
                    lang,
                    gpu=args.gpus,
                    execute=args.execute,
                    overwrite=args.overwrite,
                    final_root=model_final_root,
                    use_run_and_hold=args.run_and_hold,
                )
                for seed in args.seeds:
                    ensure_ger_final_retrieval(
                        paths,
                        model,
                        lang,
                        seed=seed,
                        execute=args.execute,
                        overwrite=args.overwrite,
                        final_root=model_final_root,
                    )
            for seed in args.seeds:
                run = make_final_run(
                    paths,
                    model,
                    lang,
                    method="GER-Vanilla",
                    seed=seed,
                    final_root=model_final_root,
                    score_root=score_root,
                    use_standard_postprocess=use_postprocess,
                    dynamic_examples=True,
                )
                if step_enabled("final", args.start_at, args.stop_after):
                    run_final_generation(
                        paths,
                        run,
                        gpu=args.gpus,
                        num_shards=args.num_shards,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        execute=args.execute,
                        overwrite=args.overwrite,
                        use_run_and_hold=args.run_and_hold,
                    )
                if step_enabled("score", args.start_at, args.stop_after):
                    score_formal_output(paths, run, execute=args.execute, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
