"""Command construction and execution for GER reproduction."""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import LanguageSpec, ModelSpec
from .paths import ProjectPaths


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    cwd: Path
    env: dict[str, str]
    required_outputs: tuple[Path, ...] = ()
    gpu: str | None = None
    use_run_and_hold: bool = False
    hold_script: Path | None = None

    def shell_preview(self) -> str:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items()) if k.startswith("GER_") or k in {"CUDA_VISIBLE_DEVICES", "PYTHONPATH"})
        command_parts = self.command
        if self.use_run_and_hold:
            if self.hold_script is None or self.gpu is None:
                command_parts = ["<run_and_hold-not-configured>", *self.command]
            else:
                command_parts = ["bash", str(self.hold_script), self.gpu, *self.command]
        command = " ".join(shlex.quote(str(part)) for part in command_parts)
        return f"cd {shlex.quote(str(self.cwd))} && {env_prefix + ' ' if env_prefix else ''}{command}"


class Runner:
    def __init__(self, *, dry_run: bool = True, overwrite: bool = False) -> None:
        self.dry_run = dry_run
        self.overwrite = overwrite

    def run(self, step: Step) -> None:
        if step.required_outputs and not self.overwrite and all(p.exists() and p.stat().st_size > 0 for p in step.required_outputs):
            print(f"[skip] {step.name}: outputs already exist")
            for output in step.required_outputs:
                print(f"       {output}")
            return

        print(f"[step] {step.name}")
        print(f"       {step.shell_preview()}")
        if self.dry_run:
            return

        if self.overwrite:
            for output in step.required_outputs:
                if output.exists():
                    output.unlink()

        env = os.environ.copy()
        env.update(step.env)
        command = step.command
        if step.use_run_and_hold:
            if not step.gpu:
                raise ValueError(f"{step.name} requested run_and_hold without gpu")
            if step.hold_script is None:
                raise ValueError(f"{step.name} requested run_and_hold without hold_script")
            command = ["bash", str(step.hold_script), step.gpu, *step.command]
        subprocess.run(command, cwd=step.cwd, env=env, check=True)

        for output in step.required_outputs:
            if not output.exists() or output.stat().st_size == 0:
                raise RuntimeError(f"{step.name} did not create required output: {output}")


def common_env(paths: ProjectPaths, model: ModelSpec, gpu: str) -> dict[str, str]:
    default_shards = str(max(1, len([item for item in gpu.split(",") if item.strip()])))
    env = {
        "CUDA_VISIBLE_DEVICES": gpu,
        "GER_CUDA_VISIBLE_DEVICES": gpu,
        "GER_MODEL_ROOT_DIR": str(paths.root / "models"),
        "GER_DATASETS_DIR": str(paths.datasets_dir),
        "GER_MULTI_P_DIR": str(paths.multi_p_dir),
        "GER_MULTILINGUAL_DIR": str(paths.multilingual_dir),
        "GER_REFERENCE_RESULTS_DIR": str(paths.reference_results_dir),
        "GER_MODEL_KEY": model.key,
        "GER_INFERENCE_BATCH_SIZE": os.environ.get("GER_INFERENCE_BATCH_SIZE", "4"),
        "GER_MAX_NEW_TOKENS": os.environ.get("GER_MAX_NEW_TOKENS", "512"),
        "GER_NUM_SHARDS": os.environ.get("GER_NUM_SHARDS", default_shards),
        "GER_SHARD_CUDA_VISIBLE_DEVICES": os.environ.get("GER_SHARD_CUDA_VISIBLE_DEVICES", gpu),
        "PYTHONPATH": str(paths.root),
        "PATH": f"{paths.root / '.venv' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    for key in (
        "GER_LLM_BACKEND",
        "GER_VLLM_TENSOR_PARALLEL_SIZE",
        "GER_VLLM_GPU_MEMORY_UTILIZATION",
        "GER_VLLM_MAX_MODEL_LEN",
        "GER_VLLM_MAX_NUM_SEQS",
        "GER_VLLM_MAX_NUM_BATCHED_TOKENS",
        "GER_VLLM_ENFORCE_EAGER",
        "GER_VLLM_DTYPE",
        "GER_VLLM_USE_STOP_STRING",
        "GER_PREFILL_ANSWER_START",
        "GER_REPE_CACHE_NUM_SHARDS",
        "GER_REPE_CACHE_CUDA_VISIBLE_DEVICES",
        "GER_REPE_DISABLE_PROGRESS",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def baseline_files(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, split: str) -> tuple[Path, Path]:
    prefix = paths.baseline_prediction_prefix(model, lang, split)
    return Path(f"{prefix}.txt"), Path(f"{prefix}.label")


def missing_baseline_txt_files(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec) -> list[Path]:
    missing: list[Path] = []
    for split in ("train", "test"):
        txt, _ = baseline_files(paths, model, lang, split)
        if not txt.exists() or txt.stat().st_size == 0:
            missing.append(txt)
    return missing


def missing_baseline_label_files(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec) -> list[Path]:
    missing: list[Path] = []
    for split in ("train", "test"):
        _, label = baseline_files(paths, model, lang, split)
        if not label.exists() or label.stat().st_size == 0:
            missing.append(label)
    return missing


def missing_baseline_files(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec) -> list[Path]:
    missing: list[Path] = []
    for split in ("train", "test"):
        for path in baseline_files(paths, model, lang, split):
            if not path.exists() or path.stat().st_size == 0:
                missing.append(path)
    return missing


def baseline_label_step(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, split: str, gpu: str = "0") -> Step:
    txt, label = baseline_files(paths, model, lang, split)
    source = paths.baseline_source_file(lang, split)
    return Step(
        name=f"baseline-label {model.key}/{lang.code}/{split}",
        command=[
            str(paths.python),
            str(paths.preprocess_data),
            "-s",
            str(source),
            "-t",
            str(txt),
            "-o",
            str(label),
        ],
        cwd=paths.root,
        env=common_env(paths, model, gpu),
        required_outputs=(label,),
    )


def build_cache_step(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, gpu: str, *, use_run_and_hold: bool = False) -> Step:
    cache_prefix = paths.cache_prefix(model, lang)
    outputs = (
        paths.cache_dir / f"{cache_prefix}_pos.npy",
        paths.cache_dir / f"{cache_prefix}_neg.npy",
        paths.cache_dir / f"{cache_prefix}_label.npy",
        paths.cache_dir / f"{cache_prefix}_line_idx.npy",
        paths.cache_dir / f"{cache_prefix}_dedup.npy",
        paths.cache_dir / f"{cache_prefix}_dedup_labels.npy",
        paths.cache_dir / f"{cache_prefix}_data.json",
        paths.cache_dir / f"{cache_prefix}_data_dedup.json",
    )
    command = [
        str(paths.python),
        "build_gec_representation_cache.py",
        lang.code,
        paths.train_suffix,
        paths.test_suffix,
        model.key,
        gpu,
        model.baseline_result_mode,
        str(paths.cache_dir),
        str(paths.multilingual_dir),
        str(paths.multi_p_dir),
        str(paths.root / "models"),
    ]
    return Step(
        name=f"build-cache {model.key}/{lang.code}",
        command=command,
        cwd=paths.repe_gec_dir,
        env=common_env(paths, model, gpu),
        required_outputs=outputs,
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold,
    )


def retrieval_step(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, gpu: str, *, use_run_and_hold: bool = False) -> Step:
    output = paths.retrieval_dir(model, lang) / "retrieval.jsonl"
    command = [
        str(paths.python),
        "retrieve_gec_examples_by_representation.py",
        lang.code,
        paths.train_suffix,
        paths.test_suffix,
        model.key,
        gpu,
        lang.test_dataset,
        str(model.retrieve_dim),
        model.baseline_result_mode,
        str(paths.cache_dir),
        paths.sentence_result_dir_name(model),
        str(paths.multilingual_dir),
        str(paths.multi_p_dir),
        str(paths.reference_results_dir),
        str(paths.root / "models"),
    ]
    return Step(
        name=f"retrieve {model.key}/{lang.code}",
        command=command,
        cwd=paths.repe_gec_dir,
        env=common_env(paths, model, gpu),
        required_outputs=(output,),
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold,
    )


def runtime_yaml_path(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, name: str) -> Path:
    return (
        paths.root
        / "multilingual"
        / "runtime_configs"
        / lang.test_dataset
        / f"{model.key}_clean_dim{model.retrieve_dim}"
        / f"{name}.yaml"
    )


def write_runtime_yaml(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    gpu: str,
    *,
    name: str,
    dataset_name: str,
    database_name: str,
    dataset: str,
    prompt_icl: str,
    process_database_text: str,
    example_num_correct: int,
    example_num_error: int,
    seed: int | None = None,
) -> Path:
    yaml_path = runtime_yaml_path(paths, model, lang, name)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = {
        "BASELINE_RESULT_MODE": model.baseline_result_mode,
        "CUDA_VISIBLE_DEVICES": gpu,
        "DATABASE_NAME": database_name,
        "DATASET": dataset,
        "DATASET_NAME": dataset_name,
        "DIALOGUE_FORM": "0",
        "EXAMPLE_NUM_CORRECT": str(example_num_correct),
        "EXAMPLE_NUM_ERROR": str(example_num_error),
        "GEC_LANG": lang.code,
        "MODEL_KEY": model.key,
        "PROCESS_DATABASE_TEXT": process_database_text,
        "PROCESS_KEY": "",
        "PROMPT_ICL": prompt_icl,
        "RETRIEVE_DIM": str(model.retrieve_dim),
        "SENTENCE_RESULT_SUFFIX": "",
        "SUFFIX": paths.train_suffix,
        "TEST_SUFFIX": paths.test_suffix,
        "DO_SAMPLE": "0",
        "TEMPERATURE": "0.0",
        "TOP_P": "1.0",
    }
    if seed is not None:
        lines["SEED"] = str(seed)
    yaml_path.write_text("".join(f"{key}: '{value}'\n" for key, value in lines.items()), encoding="utf-8")
    return yaml_path


def probing_icl_step(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, gpu: str, yaml_path: Path, *, use_run_and_hold: bool = False) -> Step:
    required_outputs = [
        paths.probing_result_dir(model, lang) / "predictions.jsonl",
        paths.probing_output_file(model, lang),
    ]
    if lang.submission_output:
        required_outputs.append(paths.probing_result_dir(model, lang) / f"{lang.test_dataset}.zip")
    if lang.local_scoring:
        required_outputs.append(paths.score_file(model, lang))
    return Step(
        name=f"retrieved-icl {model.key}/{lang.code}",
        command=["bash", "scripts/pipeline/infer_icl_probing.sh", str(yaml_path)],
        cwd=paths.multilingual_dir,
        env=common_env(paths, model, gpu),
        required_outputs=tuple(required_outputs),
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold,
    )


def random_baseline_yaml_path(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, split: str) -> Path:
    return runtime_yaml_path(paths, model, lang, f"{lang.test_dataset}.random_{split}")


def ensure_random_baseline_yaml(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, gpu: str, split: str, *, write: bool) -> Path:
    dataset_name = lang.train_dataset if split == "train" else lang.test_dataset
    dataset = f"{dataset_name}:{lang.database_dataset}"
    if write:
        return write_runtime_yaml(
            paths,
            model,
            lang,
            gpu,
            name=f"{lang.test_dataset}.random_{split}",
            dataset_name=dataset_name,
            database_name=lang.database_dataset,
            dataset=dataset,
            prompt_icl=lang.baseline_prompt_icl,
            process_database_text="",
            example_num_correct=0,
            example_num_error=0,
        )
    return random_baseline_yaml_path(paths, model, lang, split)


def random_baseline_step(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, gpu: str, split: str, yaml_path: Path, *, use_run_and_hold: bool = False) -> Step:
    dataset = lang.train_dataset if split == "train" else lang.test_dataset
    env = common_env(paths, model, gpu)
    env.update({
        "GER_RANDOM_RESULT_SPLIT": split,
        "GER_RANDOM_RESULT_SUFFIX": paths.train_suffix if split == "train" else paths.test_suffix,
        "GER_RESULT_DIR_ICL_RANDOM": str(paths.baseline_dir(model, split).relative_to(paths.multilingual_dir)),
    })
    prefix = paths.baseline_prediction_prefix(model, lang, split)
    return Step(
        name=f"random-baseline {model.key}/{lang.code}/{split}",
        command=["bash", "scripts/pipeline/infer_icl_random.sh", str(yaml_path)],
        cwd=paths.multilingual_dir,
        env=env,
        required_outputs=(Path(f"{prefix}.txt"), Path(f"{prefix}.label")),
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold,
    )


def paper_random_yaml_path(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, seed: int) -> Path:
    return runtime_yaml_path(paths, model, lang, f"{lang.test_dataset}.paper_random8_seed{seed}")


def ensure_paper_random_yaml(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, gpu: str, seed: int, *, write: bool) -> Path:
    if write:
        return write_runtime_yaml(
            paths,
            model,
            lang,
            gpu,
            name=f"{lang.test_dataset}.paper_random8_seed{seed}",
            dataset_name=lang.test_dataset,
            database_name=lang.database_dataset,
            dataset=lang.yaml_dataset,
            prompt_icl=lang.baseline_prompt_icl,
            process_database_text="",
            example_num_correct=4,
            example_num_error=4,
            seed=seed,
        )
    return paper_random_yaml_path(paths, model, lang, seed)


def paper_random_step(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, gpu: str, seed: int, yaml_path: Path, *, use_run_and_hold: bool = False) -> Step:
    env = common_env(paths, model, gpu)
    env.update({
        "GER_RANDOM_RESULT_SPLIT": "test",
        "GER_RANDOM_RESULT_SUFFIX": paths.test_suffix,
        "GER_RESULT_DIR_ICL_RANDOM": str(paths.paper_random_dir(model, seed).relative_to(paths.multilingual_dir)),
        "GER_SEED": str(seed),
    })
    result_dir = paths.paper_random_dir(model, seed) / lang.test_dataset
    score_name = "nlpcc18.score" if lang.test_dataset == "nlpcc18" else f"{lang.test_dataset}.score"
    required_outputs = [
        result_dir / "predictions.jsonl",
        result_dir / (f"{lang.test_dataset}.txt" if lang.submission_output else f"{lang.test_dataset}-output-retokenized.txt"),
    ]
    if lang.submission_output:
        required_outputs.append(result_dir / f"{lang.test_dataset}.zip")
    if lang.local_scoring:
        required_outputs.append(result_dir / score_name)
    return Step(
        name=f"paper-random8 {model.key}/{lang.code}/seed{seed}",
        command=["bash", "scripts/pipeline/infer_icl_random.sh", str(yaml_path)],
        cwd=paths.multilingual_dir,
        env=env,
        required_outputs=tuple(required_outputs),
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold,
    )


def ensure_runtime_yaml(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, gpu: str, *, write: bool) -> Path:
    if write:
        return write_runtime_yaml(
            paths,
            model,
            lang,
            gpu,
            name=f"{lang.test_dataset}.retrieved_icl",
            dataset_name=lang.test_dataset,
            database_name=lang.database_dataset,
            dataset=lang.yaml_dataset,
            prompt_icl=lang.final_prompt_icl,
            process_database_text="prefix",
            example_num_correct=0,
            example_num_error=10,
        )
    return runtime_yaml_path(paths, model, lang, f"{lang.test_dataset}.retrieved_icl")
