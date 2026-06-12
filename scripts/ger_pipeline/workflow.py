"""High-level GER workflow orchestration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import LanguageSpec, ModelSpec
from .data_sources import test_source_file, train_data_prefix, write_standard_test_source, write_standard_train_data
from .dataset_preparation import prepare_language_datasets
from .files import file_ok, require_file
from .final_output import FinalRun
from .paths import ProjectPaths
from .runner import execute_step
from .scoring import score_formal_output
from .steps import (
    Step,
    build_cache_step,
    common_env,
    ensure_initial_prediction_yaml,
    initial_prediction_step,
    runtime_yaml_path,
    write_runtime_yaml,
)


TABLE1_LANGUAGES = ("en", "bea19", "de", "ro", "et")
TABLE1_SEEDS = (88, 111, 222)


def method_stem(method: str) -> str:
    return method.lower().replace("-", "_")


def table_dataset_name(lang: LanguageSpec) -> str:
    return "ronacc_readerbench" if lang.code == "ro" else lang.test_dataset


def default_final_root(paths: ProjectPaths, model: ModelSpec) -> Path:
    return paths.multilingual_dir / f"results_ger_{model.key}"


def default_score_root(paths: ProjectPaths) -> Path:
    return paths.root / "results" / "official_eval" / "ger"


def path_for_runtime_env(paths: ProjectPaths, path: Path) -> str:
    """Return an absolute path for scripts-local runtime commands."""
    return str(path.resolve())


def ger_source_retrieval_root(paths: ProjectPaths, model: ModelSpec, *, final_root: Path | None = None) -> Path:
    root = final_root or default_final_root(paths, model)
    return root / "retrieve_ger_source"


def source_retrieval_subdir(model: ModelSpec, paths: ProjectPaths) -> str:
    return f"retrieved_examples_dim{model.retrieve_dim}{paths.train_suffix}{paths.test_suffix}"


def ger_retrieval_root(paths: ProjectPaths, model: ModelSpec, seed: int, *, final_root: Path | None = None) -> Path:
    root = final_root or default_final_root(paths, model)
    return root / f"retrieve_ger_vanilla_seed{seed}"


def result_root(paths: ProjectPaths, model: ModelSpec, method: str, seed: int, *, final_root: Path | None = None) -> Path:
    root = final_root or default_final_root(paths, model)
    return root / f"res_{method_stem(method)}_seed{seed}"


def score_dir(paths: ProjectPaths, model: ModelSpec, method: str, seed: int, lang: LanguageSpec, *, score_root: Path | None = None) -> Path:
    root = score_root or default_score_root(paths)
    return root / table_dataset_name(lang) / f"{model.key}_{method_stem(method)}_seed{seed}"


def retrieved_icl_yaml_path(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, method: str) -> Path:
    return runtime_yaml_path(paths, model, lang, f"{lang.test_dataset}.{method_stem(method)}.retrieved_icl")


def ensure_retrieved_icl_yaml(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    *,
    method: str,
    gpu: str,
    prompt: str,
    write: bool,
    max_error_examples: int,
    seed: int | None = None,
) -> Path:
    if not write:
        return retrieved_icl_yaml_path(paths, model, lang, method)
    return write_runtime_yaml(
        paths,
        model,
        lang,
        gpu,
        name=f"{lang.test_dataset}.{method_stem(method)}.retrieved_icl",
        dataset_name=lang.test_dataset,
        source_dataset_name=lang.source_dataset,
        dataset=lang.yaml_dataset,
        prompt_icl=prompt,
        example_num_correct=0,
        example_num_error=max_error_examples,
        seed=seed,
    )


def ensure_initial_predictions(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    *,
    gpu: str,
    execute: bool,
    overwrite: bool,
    use_run_and_hold: bool,
) -> None:
    if execute:
        prepare_language_datasets(paths, lang, overwrite=overwrite)
        write_standard_test_source(paths, lang, overwrite=overwrite)
        write_standard_train_data(paths, lang, overwrite=overwrite)
    for split in ("train", "test"):
        yaml_path = ensure_initial_prediction_yaml(paths, model, lang, gpu, split, write=execute)
        step = initial_prediction_step(paths, model, lang, gpu, split, yaml_path, use_run_and_hold=use_run_and_hold)
        if split == "test":
            step.env["GER_TEST_SOURCE_FILE"] = path_for_runtime_env(paths, test_source_file(paths, lang))
        else:
            step.env["GER_TRAIN_SOURCE_FILE"] = path_for_runtime_env(paths, Path(f"{train_data_prefix(paths, lang)}.src"))
        execute_step(step, execute=execute, overwrite=overwrite)


def ensure_representation_cache(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    *,
    gpu: str,
    execute: bool,
    overwrite: bool,
    use_run_and_hold: bool,
) -> None:
    if execute:
        write_standard_train_data(paths, lang, overwrite=overwrite)
    step = build_cache_step(paths, model, lang, gpu, use_run_and_hold=use_run_and_hold)
    execute_step(step, execute=execute, overwrite=overwrite)


def retrieval_command(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    *,
    gpu: str,
    output_root_name: str,
    use_run_and_hold: bool,
    demo_floor_topk: int = 0,
) -> Step:
    output_root_path = Path(output_root_name)
    if not output_root_path.is_absolute():
        output_root_path = paths.multilingual_dir / output_root_path
    output = output_root_path / source_retrieval_subdir(model, paths) / lang.test_dataset / "retrieval.jsonl"
    retrieval_lang = "en" if lang.code == "bea19" else lang.code
    num_shards = max(1, len([item for item in gpu.split(",") if item.strip()]))
    command = [
        str(paths.python),
        "retrieve_gec_examples_by_representation.py",
        "--lang", retrieval_lang,
        "--train-suffix", paths.train_suffix,
        "--test-suffix", paths.test_suffix,
        "--model-name", model.key,
        "--gpus", gpu,
        "--test", lang.test_dataset,
        "--retrieve-dim", str(model.retrieve_dim),
        "--initial-result-mode", model.initial_result_mode,
        "--cache-dir", str(paths.cache_dir),
        "--output-root", output_root_name,
        "--multilingual-dir", str(paths.multilingual_dir),
        "--train-data-dir", str(paths.runtime_train_data_dir),
        "--test-source-file", str(test_source_file(paths, lang)),
        "--model-root-dir", str(paths.root / "models"),
        "--num-shards", str(num_shards),
    ]
    if demo_floor_topk > 0:
        command.extend(["--demo-floor-topk", str(demo_floor_topk)])
    env = common_env(paths, model, gpu)
    env["GER_INITIAL_RESULT_DIR_TEST"] = str(paths.initial_prediction_dir(model, "test"))
    return Step(
        name=f"ger-source-retrieval {model.key}/{lang.code}",
        command=command,
        cwd=paths.repe_gec_dir,
        env=env,
        required_outputs=(output,),
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold if use_run_and_hold else None,
    )


def ensure_ger_source_retrieval(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    *,
    gpu: str,
    execute: bool,
    overwrite: bool,
    final_root: Path | None,
    use_run_and_hold: bool,
) -> Path:
    if execute:
        write_standard_test_source(paths, lang, overwrite=overwrite)
        write_standard_train_data(paths, lang, overwrite=overwrite)
    source_root = ger_source_retrieval_root(paths, model, final_root=final_root)
    output_root_name = path_for_runtime_env(paths, source_root)
    step = retrieval_command(paths, model, lang, gpu=gpu, output_root_name=output_root_name, use_run_and_hold=use_run_and_hold, demo_floor_topk=3)
    execute_step(step, execute=execute, overwrite=overwrite)
    return step.required_outputs[0]


def copy_retrieval(src_file: Path, dst_root: Path, lang: LanguageSpec, *, execute: bool, overwrite: bool) -> Path:
    dst = dst_root / lang.test_dataset / "retrieval.jsonl"
    print(f"[step] copy-retrieval {src_file} -> {dst}", flush=True)
    if not execute:
        return dst
    if file_ok(dst) and not overwrite:
        return dst
    require_file(src_file)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")
    require_file(dst)
    return dst


def ensure_ger_final_retrieval(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    *,
    seed: int,
    execute: bool,
    overwrite: bool,
    final_root: Path | None,
) -> Path:
    source_root = ger_source_retrieval_root(paths, model, final_root=final_root)
    src = source_root / source_retrieval_subdir(model, paths) / lang.test_dataset / "retrieval.jsonl"
    return copy_retrieval(src, ger_retrieval_root(paths, model, seed, final_root=final_root), lang, execute=execute, overwrite=overwrite)


def make_final_run(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    *,
    method: str,
    seed: int,
    final_root: Path | None,
    score_root: Path | None,
    use_standard_postprocess: bool,
    dynamic_examples: bool,
    retrieval_root: Path | None = None,
) -> FinalRun:
    selected_retrieval_root = retrieval_root or ger_retrieval_root(paths, model, seed, final_root=final_root)
    return FinalRun(
        model=model,
        lang=lang,
        method=method,
        seed=seed,
        retrieval_dir=selected_retrieval_root / lang.test_dataset,
        result_dir=result_root(paths, model, method, seed, final_root=final_root) / lang.test_dataset,
        score_dir=score_dir(paths, model, method, seed, lang, score_root=score_root),
        prompt=lang.final_prompt_icl,
        dynamic_examples=dynamic_examples,
        use_standard_postprocess=use_standard_postprocess,
    )


def final_generation_step(
    paths: ProjectPaths,
    run: FinalRun,
    *,
    gpu: str,
    num_shards: int,
    batch_size: int,
    max_new_tokens: int,
    use_run_and_hold: bool,
    execute: bool,
) -> Step:
    yaml_path = ensure_retrieved_icl_yaml(
        paths,
        run.model,
        run.lang,
        method=run.method,
        gpu=gpu,
        prompt=run.prompt,
        write=execute,
        max_error_examples=10 if run.dynamic_examples else 8,
        seed=run.seed,
    )
    env = common_env(paths, run.model, gpu)
    env.update({
        "GER_RETRIEVED_EXAMPLES_DIR": path_for_runtime_env(paths, run.retrieval_dir.parent),
        "GER_FINAL_RESULT_DIR": path_for_runtime_env(paths, run.result_dir.parent),
        "GER_PROMPT_ICL": run.prompt,
        "GER_EXAMPLE_NUM_ERROR": "10" if run.dynamic_examples else "8",
        "GER_EXAMPLE_NUM_CORRECT": "0",
        "GER_DYNAMIC_EXAMPLE_NUM_ERROR": "1" if run.dynamic_examples else "0",
        "GER_DYNAMIC_EXAMPLE_NUM_ERROR_MIN": "2",
        "GER_DYNAMIC_EXAMPLE_NUM_ERROR_TARGET_AVG": "8",
        "GER_DYNAMIC_EXAMPLE_NUM_ERROR_MAX": "10",
        "GER_SEED": str(run.seed),
        "GER_NUM_SHARDS": str(num_shards),
        "GER_SHARD_CUDA_VISIBLE_DEVICES": gpu,
        "GER_INFERENCE_BATCH_SIZE": str(batch_size),
        "GER_MAX_NEW_TOKENS": str(max_new_tokens),
        "GER_VLLM_USE_STOP_STRING": "1",
        "GER_VLLM_MAX_MODEL_LEN": os.environ.get("GER_VLLM_MAX_MODEL_LEN", "16384"),
    })
    required = [run.raw_predictions]
    if run.lang.submission_output:
        required.extend([run.result_dir / f"{run.dataset}.txt", run.result_dir / f"{run.dataset}.zip"])
    else:
        required.append(run.result_dir / f"{run.dataset}-output-retokenized.txt")
    return Step(
        name=f"final-retrieved-icl {run.model.key}/{run.lang.code}/{run.method}/seed{run.seed}",
        command=["bash", "scripts/pipeline/infer_retrieved_icl.sh", str(yaml_path)],
        cwd=paths.inference_runtime_dir,
        env=env,
        required_outputs=tuple(required),
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold if use_run_and_hold else None,
    )


def run_final_generation(
    paths: ProjectPaths,
    run: FinalRun,
    *,
    gpu: str,
    num_shards: int,
    batch_size: int,
    max_new_tokens: int,
    execute: bool,
    overwrite: bool,
    use_run_and_hold: bool,
) -> None:
    step = final_generation_step(
        paths,
        run,
        gpu=gpu,
        num_shards=num_shards,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        use_run_and_hold=use_run_and_hold,
        execute=execute,
    )
    execute_step(step, execute=execute, overwrite=overwrite)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--train-suffix", default="_8")
    parser.add_argument("--test-suffix", default="_8")
    parser.add_argument("--models", nargs="+", default=["llama31", "qwen25"])
    parser.add_argument("--languages", nargs="+", default=list(TABLE1_LANGUAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TABLE1_SEEDS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--final-root", type=Path, default=None)
    parser.add_argument("--score-root", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-and-hold", dest="run_and_hold", action="store_true")
    parser.set_defaults(run_and_hold=False)
