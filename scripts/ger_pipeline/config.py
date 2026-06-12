"""GER experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    test_dataset: str
    train_dataset: str
    source_dataset: str
    yaml_dataset: str
    initial_prompt_icl: str
    final_prompt_icl: str
    m2_relative_path: str
    retokenized_output: bool = True
    local_scoring: bool = True
    submission_output: bool = False


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_dir_name: str
    initial_result_mode: str = "default"
    retrieve_dim: int = 128
    layer_index: int = -12


TABLE1_LANGUAGES = ("en", "bea19", "de", "ro", "et")
DEFAULT_MODELS = ("llama31", "qwen25")
DEFAULT_TRAIN_SUFFIX = "_8"
DEFAULT_TEST_SUFFIX = "_8"
DEFAULT_SEEDS = (88, 111, 222)


LANGUAGES: dict[str, LanguageSpec] = {
    "en": LanguageSpec(
        code="en",
        test_dataset="conll14",
        train_dataset="wilocness",
        source_dataset="wilocness",
        yaml_dataset="conll14:wilocness",
        initial_prompt_icl="reproduce_space_en_8fix",
        final_prompt_icl="min_edit_fewshot_space",
        m2_relative_path="multilingual_raw/EN-conll14st-test-data/noalt/official-2014.combined.m2",
    ),
    "bea19": LanguageSpec(
        code="bea19",
        test_dataset="bea19",
        train_dataset="wilocness",
        source_dataset="wilocness",
        yaml_dataset="bea19:wilocness",
        initial_prompt_icl="reproduce_space_en_8fix",
        final_prompt_icl="min_edit_fewshot_space",
        m2_relative_path="",
        retokenized_output=False,
        local_scoring=False,
        submission_output=True,
    ),
    "de": LanguageSpec(
        code="de",
        test_dataset="falko_merlin",
        train_dataset="falko_merlin_train",
        source_dataset="falko_merlin",
        yaml_dataset="falko_merlin:falko_merlin",
        initial_prompt_icl="reproduce_space_de_8fix",
        final_prompt_icl="min_edit_fewshot_space",
        m2_relative_path="multilingual_raw/DE-FALKO-MERLIN/fm-test.m2",
    ),
    "ro": LanguageSpec(
        code="ro",
        test_dataset="rogec",
        train_dataset="rogec_train",
        source_dataset="rogec",
        yaml_dataset="rogec:rogec",
        initial_prompt_icl="reproduce_space_ro_8fix",
        final_prompt_icl="min_edit_fewshot_space",
        m2_relative_path="multilingual/rogec/test.m2",
    ),
    "et": LanguageSpec(
        code="et",
        test_dataset="estgec",
        train_dataset="estgec_train",
        source_dataset="estgec",
        yaml_dataset="estgec:estgec",
        initial_prompt_icl="reproduce_space_et_8fix",
        final_prompt_icl="min_edit_fewshot_space",
        m2_relative_path="multilingual_raw/ET-estgec/Tartu_L1_corpus/test/test_m2.txt",
    ),
}


MODELS: dict[str, ModelSpec] = {
    "llama31": ModelSpec(
        key="llama31",
        model_dir_name="Meta-Llama-3.1-8B-Instruct",
        retrieve_dim=128,
        layer_index=-21,
    ),
    "qwen25": ModelSpec(
        key="qwen25",
        model_dir_name="Qwen2.5-7B-Instruct",
        retrieve_dim=256,
        layer_index=-12,
    ),
}


def get_language(code: str) -> LanguageSpec:
    try:
        return LANGUAGES[code]
    except KeyError as exc:
        raise ValueError(f"Unsupported language {code!r}; expected one of {sorted(LANGUAGES)}") from exc


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported model {key!r}; expected one of {sorted(MODELS)}") from exc


def default_model_path(project_root: Path, model: ModelSpec) -> Path:
    return project_root / "models" / model.model_dir_name
