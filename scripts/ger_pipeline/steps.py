"""Command construction for the GER reproduction pipeline."""
from __future__ import annotations

import os
import shlex
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
        keys = {"CUDA_VISIBLE_DEVICES", "PYTHONPATH"}
        env_prefix = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in sorted(self.env.items())
            if key.startswith("GER_") or key in keys
        )
        command_parts = self.command
        if self.use_run_and_hold:
            if self.hold_script is None or self.gpu is None:
                command_parts = ["<run-and-hold-not-configured>", *self.command]
            else:
                command_parts = ["bash", str(self.hold_script), self.gpu, *self.command]
        command = " ".join(shlex.quote(str(part)) for part in command_parts)
        return f"cd {shlex.quote(str(self.cwd))} && {env_prefix + ' ' if env_prefix else ''}{command}"


def common_env(paths: ProjectPaths, model: ModelSpec, gpu: str) -> dict[str, str]:
    gpu_count = max(1, len([item for item in gpu.split(",") if item.strip()]))
    default_shards = str(gpu_count)
    env = {
        "CUDA_VISIBLE_DEVICES": gpu,
        "GER_CUDA_VISIBLE_DEVICES": gpu,
        "GER_MODEL_ROOT_DIR": str(paths.root / "models"),
        "GER_PROJECT_ROOT": str(paths.root),
        "GER_DATA_ROOT": str(paths.root),
        "GER_DATASETS_DIR": str(paths.datasets_dir),
        "GER_TRAIN_DATA_DIR": str(paths.runtime_train_data_dir),
        "GER_MULTILINGUAL_DIR": str(paths.multilingual_dir),
        "GER_INFERENCE_RUNTIME_DIR": str(paths.inference_runtime_dir),
        "GER_MODEL_KEY": model.key,
        "GER_INFERENCE_BATCH_SIZE": os.environ.get("GER_INFERENCE_BATCH_SIZE", "4"),
        "GER_MAX_NEW_TOKENS": os.environ.get("GER_MAX_NEW_TOKENS", "512"),
        "GER_NUM_SHARDS": os.environ.get("GER_NUM_SHARDS", default_shards),
        "GER_SHARD_CUDA_VISIBLE_DEVICES": os.environ.get("GER_SHARD_CUDA_VISIBLE_DEVICES", gpu),
        "PYTHONPATH": os.pathsep.join([str(paths.inference_runtime_dir), str(paths.root)]),
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


def runtime_yaml_path(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, name: str) -> Path:
    return (
        paths.multilingual_dir
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
    source_dataset_name: str,
    dataset: str,
    prompt_icl: str,
    example_num_correct: int,
    example_num_error: int,
    seed: int | None = None,
) -> Path:
    yaml_path = runtime_yaml_path(paths, model, lang, name)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = {
        "INITIAL_RESULT_MODE": model.initial_result_mode,
        "CUDA_VISIBLE_DEVICES": gpu,
        "SOURCE_DATASET_NAME": source_dataset_name,
        "DATASET": dataset,
        "DATASET_NAME": dataset_name,
        "DIALOGUE_FORM": "0",
        "EXAMPLE_NUM_CORRECT": str(example_num_correct),
        "EXAMPLE_NUM_ERROR": str(example_num_error),
        "GEC_LANG": lang.code,
        "MODEL_KEY": model.key,
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


def initial_prediction_yaml_path(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec, split: str) -> Path:
    return runtime_yaml_path(paths, model, lang, f"{lang.test_dataset}.initial_{split}")


def ensure_initial_prediction_yaml(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    gpu: str,
    split: str,
    *,
    write: bool,
) -> Path:
    if not write:
        return initial_prediction_yaml_path(paths, model, lang, split)
    dataset_name = lang.train_dataset if split == "train" else lang.test_dataset
    dataset = f"{dataset_name}:{lang.source_dataset}"
    return write_runtime_yaml(
        paths,
        model,
        lang,
        gpu,
        name=f"{lang.test_dataset}.initial_{split}",
        dataset_name=dataset_name,
        source_dataset_name=lang.source_dataset,
        dataset=dataset,
        prompt_icl=lang.initial_prompt_icl,
        example_num_correct=0,
        example_num_error=0,
    )


def initial_prediction_step(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    gpu: str,
    split: str,
    yaml_path: Path,
    *,
    use_run_and_hold: bool = False,
) -> Step:
    env = common_env(paths, model, gpu)
    env.update(
        {
            "GER_INITIAL_RESULT_SPLIT": split,
            "GER_INITIAL_RESULT_SUFFIX": paths.train_suffix if split == "train" else paths.test_suffix,
            "GER_INITIAL_RESULT_DIR": str(paths.initial_prediction_dir(model, split)),
            "GER_PREPROCESS_DATA": str(paths.preprocess_data),
        }
    )
    prefix = paths.initial_prediction_prefix(model, lang, split)
    return Step(
        name=f"initial-prediction {model.key}/{lang.code}/{split}",
        command=["bash", "scripts/pipeline/infer_initial_predictions.sh", str(yaml_path)],
        cwd=paths.inference_runtime_dir,
        env=env,
        required_outputs=(Path(f"{prefix}.txt"), Path(f"{prefix}.label")),
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold if use_run_and_hold else None,
    )


def build_cache_step(
    paths: ProjectPaths,
    model: ModelSpec,
    lang: LanguageSpec,
    gpu: str,
    *,
    use_run_and_hold: bool = False,
) -> Step:
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
        model.initial_result_mode,
        str(paths.cache_dir),
        str(paths.multilingual_dir),
        str(paths.runtime_train_data_dir),
        str(paths.root / "models"),
    ]
    env = common_env(paths, model, gpu)
    env["GER_INITIAL_RESULT_DIR_TRAIN"] = str(paths.initial_prediction_dir(model, "train"))
    env["GER_REPE_CACHE_NUM_SHARDS"] = os.environ.get("GER_REPE_CACHE_NUM_SHARDS", str(max(1, len([item for item in gpu.split(",") if item.strip()]))))
    env["GER_REPE_CACHE_CUDA_VISIBLE_DEVICES"] = os.environ.get("GER_REPE_CACHE_CUDA_VISIBLE_DEVICES", gpu)
    return Step(
        name=f"build-cache {model.key}/{lang.code}",
        command=command,
        cwd=paths.repe_gec_dir,
        env=env,
        required_outputs=outputs,
        gpu=gpu,
        use_run_and_hold=use_run_and_hold,
        hold_script=paths.run_and_hold if use_run_and_hold else None,
    )
