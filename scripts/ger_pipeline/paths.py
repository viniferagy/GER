"""Path derivation for GER reproduction artifacts."""
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
    def multi_p_dir(self) -> Path:
        return self.root / "datasets" / "multi_p"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def reference_results_dir(self) -> Path:
        return self.root / "results" / "llama3.1"

    @property
    def repe_gec_dir(self) -> Path:
        return self.root / "representation-engineering" / "examples" / "gec"

    @property
    def python(self) -> Path:
        venv_python = self.root / ".venv" / "bin" / "python"
        return venv_python if venv_python.exists() else Path("python")

    @property
    def run_and_hold(self) -> Path:
        candidates = (
            self.root / "tools" / "run_and_hold.sh",
            self.root / "run_and_hold.sh",
            self.root.parent / "run_and_hold.sh",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return self.root.parent / "run_and_hold.sh"

    @property
    def preprocess_data(self) -> Path:
        return self.root / "utils" / "preprocess_data.py"

    @property
    def bea19_blind_source(self) -> Path:
        return self.root / "datasets" / "multilingual" / "wilocness" / "wi+locness" / "test" / "ABCN.test.bea19.orig"

    def model_path(self, model: ModelSpec) -> Path:
        return default_model_path(self.root, model)

    def baseline_dir(self, model: ModelSpec, split: str) -> Path:
        suffix = self.train_suffix if split == "train" else self.test_suffix
        return (
            self.multilingual_dir
            / f"results_{model.key}_{model.baseline_result_mode}"
            / f"icl_{model.key}_res_random_pgy_{split}{suffix}"
        )

    def paper_random_dir(self, model: ModelSpec, seed: int) -> Path:
        return (
            self.multilingual_dir
            / f"results_{model.key}_{model.baseline_result_mode}"
            / f"icl_{model.key}_res_random8_seed{seed}_pgy_test{self.test_suffix}"
        )

    def baseline_prediction_prefix(self, model: ModelSpec, lang: LanguageSpec, split: str) -> Path:
        dataset = lang.train_dataset if split == "train" else lang.test_dataset
        return self.baseline_dir(model, split) / dataset / f"{dataset}-output-retokenized"

    def baseline_source_file(self, lang: LanguageSpec, split: str) -> Path:
        if split == "train":
            return self.multi_p_dir / lang.train_dataset / "train.src"
        return self.reference_results_dir / f"{lang.test_dataset}.src"

    def cache_prefix(self, model: ModelSpec, lang: LanguageSpec) -> str:
        cache_code = "en" if lang.code == "bea19" else lang.code
        return f"gec_representation_cache_{model.key}_{model.baseline_result_mode}_{cache_code}{self.train_suffix}"

    def sentence_result_dir_name(self, model: ModelSpec) -> str:
        return f"results_sentence_{model.key}"

    def retrieval_dir(self, model: ModelSpec, lang: LanguageSpec) -> Path:
        return (
            self.multilingual_dir
            / self.sentence_result_dir_name(model)
            / f"icl_deepseek_retrieve_by_probing_pgy_{model.retrieve_dim}{self.train_suffix}{self.test_suffix}"
            / lang.test_dataset
        )

    def probing_result_dir(self, model: ModelSpec, lang: LanguageSpec) -> Path:
        return (
            self.multilingual_dir
            / self.sentence_result_dir_name(model)
            / f"icl_deepseek_res_probing_pgy_{model.retrieve_dim}{self.train_suffix}{self.test_suffix}"
            / lang.test_dataset
        )

    def score_file(self, model: ModelSpec, lang: LanguageSpec) -> Path:
        if not lang.local_scoring:
            return self.probing_result_dir(model, lang) / f"{lang.test_dataset}.score"
        if lang.test_dataset == "nlpcc18":
            return self.probing_result_dir(model, lang) / "nlpcc18.score"
        return self.probing_result_dir(model, lang) / f"{lang.test_dataset}.score"

    def probing_output_file(self, model: ModelSpec, lang: LanguageSpec) -> Path:
        if lang.submission_output:
            return self.probing_result_dir(model, lang) / f"{lang.test_dataset}.txt"
        return self.probing_result_dir(model, lang) / f"{lang.test_dataset}-output-retokenized.txt"

    def m2_file(self, lang: LanguageSpec) -> Path:
        if not lang.m2_relative_path:
            return self.datasets_dir / "__missing_official_reference__"
        return self.datasets_dir / lang.m2_relative_path
