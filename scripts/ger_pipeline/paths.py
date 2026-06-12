"""GER path derivation for reproduction artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import LanguageSpec, ModelSpec, default_model_path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    cache_dir: Path
    train_suffix: str
    test_suffix: str

    @classmethod
    def discover(
        cls,
        root: Path | None = None,
        cache_dir: Path | None = None,
        train_suffix: str = "_8",
        test_suffix: str = "_8",
    ) -> "ProjectPaths":
        resolved_root = (root or Path(__file__).resolve().parents[2]).resolve()
        return cls(
            root=resolved_root,
            cache_dir=(cache_dir or resolved_root / "cache" / "representation").resolve(),
            train_suffix=train_suffix,
            test_suffix=test_suffix,
        )

    @property
    def multilingual_dir(self) -> Path:
        return self.root / "multilingual"

    @property
    def inference_runtime_dir(self) -> Path:
        return self.root / "scripts" / "ger_runtime" / "inference"

    @property
    def runtime_train_data_dir(self) -> Path:
        return self.multilingual_dir / "runtime_train_data"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def repe_gec_dir(self) -> Path:
        return self.root / "scripts" / "ger_runtime" / "repe_gec"

    @property
    def python(self) -> Path:
        venv_python = self.root / ".venv" / "bin" / "python"
        return venv_python if venv_python.exists() else Path("python")

    @property
    def run_and_hold(self) -> Path:
        candidates = (
            self.root / "tools" / "run_and_hold.sh",
            self.root / "run_and_hold.sh",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"run_and_hold.sh not found under project root: {self.root}")

    @property
    def preprocess_data(self) -> Path:
        return self.root / "scripts" / "ger_runtime" / "preprocess_data.py"

    def model_path(self, model: ModelSpec) -> Path:
        return default_model_path(self.root, model)

    def initial_prediction_dir(self, model: ModelSpec, split: str) -> Path:
        suffix = self.train_suffix if split == "train" else self.test_suffix
        return (
            self.multilingual_dir
            / f"results_{model.key}_{model.initial_result_mode}"
            / f"initial_predictions_{split}{suffix}"
        )

    def initial_prediction_prefix(self, model: ModelSpec, lang: LanguageSpec, split: str) -> Path:
        dataset = lang.train_dataset if split == "train" else lang.test_dataset
        return self.initial_prediction_dir(model, split) / dataset / f"{dataset}-output-retokenized"

    def initial_source_file(self, lang: LanguageSpec, split: str) -> Path:
        if split == "train":
            return self.runtime_train_data_dir / lang.train_dataset / "train.src"
        return self.multilingual_dir / "runtime_sources" / lang.test_dataset / "test.src"

    def cache_prefix(self, model: ModelSpec, lang: LanguageSpec) -> str:
        cache_code = "en" if lang.code == "bea19" else lang.code
        return f"gec_representation_cache_{model.key}_{model.initial_result_mode}_{cache_code}{self.train_suffix}"

    def m2_file(self, lang: LanguageSpec) -> Path:
        if not lang.m2_relative_path:
            return self.datasets_dir / "__missing_official_reference__"
        return self.datasets_dir / lang.m2_relative_path
