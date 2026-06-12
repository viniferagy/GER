# Shared defaults for ICL pipeline scripts.
# The YAML files may now contain only experiment settings; machine paths,
# model locations, and generated result dirs can come from the task runner.

PIPELINE_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_RUNTIME_DIR_FROM_PIPELINE="$(cd "${PIPELINE_COMMON_DIR}/../.." && pwd)"
PROJECT_ROOT_FROM_PIPELINE="${GER_PROJECT_ROOT:-$(cd "${INFERENCE_RUNTIME_DIR_FROM_PIPELINE}/../../.." && pwd)}"

model_key_to_model_ref() {
    local key="$1"
    case "$key" in
        llama31)
            echo "Meta-Llama-3.1-8B-Instruct"
            ;;
        qwen25)
            echo "Qwen2.5-7B-Instruct"
            ;;
        *)
            echo "$key"
            ;;
    esac
}

infer_model_key_from_ref() {
    local original="$1"
    case "$original" in
        *Meta-Llama-3.1-8B-Instruct)
            echo "llama31"
            ;;
        *Qwen2.5-7B-Instruct)
            echo "qwen25"
            ;;
        *)
            echo ""
            ;;
    esac
}

resolve_project_model_path() {
    local original="$1"
    local model_root="${GER_MODEL_ROOT_DIR:-${PROJECT_ROOT_FROM_PIPELINE}/models}"

    case "$original" in
        *Meta-Llama-3.1-8B-Instruct)
            echo "${GER_LLAMA31_MODEL_PATH:-${model_root}/Meta-Llama-3.1-8B-Instruct}"
            ;;
        *Qwen2.5-7B-Instruct)
            echo "${GER_QWEN25_MODEL_PATH:-${model_root}/Qwen2.5-7B-Instruct}"
            ;;
        *xlm-roberta-large)
            echo "${GER_XLM_ROBERTA_MODEL_PATH:-${model_root}/xlm-roberta-large}"
            ;;
        *)
            echo "$original"
            ;;
    esac
}

resolve_model_alias_or_path() {
    local original="$1"
    local mapped
    mapped="$(model_key_to_model_ref "$original")"
    resolve_project_model_path "$mapped"
}

apply_icl_runtime_defaults() {
    if [ -z "${DATASET_NAME:-}" ] && [ -n "${DATASET:-}" ]; then
        IFS=':' read -r DATASET_NAME _ <<< "$DATASET"
    fi

    if [ -z "${SOURCE_DATASET_NAME:-}" ]; then
        SOURCE_DATASET_NAME="${DATASET_NAME:-}"
    fi

    if [ -z "${DATASET:-}" ] && [ -n "${DATASET_NAME:-}" ]; then
        DATASET="${DATASET_NAME}:${SOURCE_DATASET_NAME}"
    fi

    if [ -z "${DATASET_NAME:-}" ] || [ -z "${DATASET:-}" ]; then
        echo "DATASET_NAME or DATASET must be set in YAML or environment." >&2
        exit 1
    fi

    DATASETS_DIR="${GER_DATASETS_DIR:-${DATASETS_DIR:-${PROJECT_ROOT_FROM_PIPELINE}/datasets}}"
    CUDA_VISIBLE_DEVICES="${GER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

    MODEL_KEY="${MODEL_KEY:-${GER_MODEL_KEY:-}}"
    if [ -z "$MODEL_KEY" ] && [ -n "${MODEL:-}" ]; then
        MODEL_KEY="$(infer_model_key_from_ref "$MODEL")"
    fi

    if [ -z "${MODEL:-}" ]; then
        if [ -z "$MODEL_KEY" ]; then
            echo "MODEL_KEY or MODEL must be set in YAML or environment." >&2
            exit 1
        fi
        MODEL="$(model_key_to_model_ref "$MODEL_KEY")"
    fi
    MODEL="$(resolve_model_alias_or_path "$MODEL")"

    ASSIST_MODEL="${ASSIST_MODEL:-${GER_ASSIST_MODEL:-$MODEL}}"
    EMBED_MODEL="${EMBED_MODEL:-${GER_EMBED_MODEL:-xlm-roberta-large}}"
    ASSIST_MODEL="$(resolve_model_alias_or_path "$ASSIST_MODEL")"
    EMBED_MODEL="$(resolve_model_alias_or_path "$EMBED_MODEL")"

    PROMPT_ICL="${GER_PROMPT_ICL:-${PROMPT_ICL:-min_edit_fewshot_space}}"
    EXAMPLE_NUM_ERROR="${GER_EXAMPLE_NUM_ERROR:-${EXAMPLE_NUM_ERROR:-0}}"
    EXAMPLE_NUM_CORRECT="${GER_EXAMPLE_NUM_CORRECT:-${EXAMPLE_NUM_CORRECT:-0}}"
    DYNAMIC_EXAMPLE_NUM_ERROR="${GER_DYNAMIC_EXAMPLE_NUM_ERROR:-${DYNAMIC_EXAMPLE_NUM_ERROR:-0}}"
    DYNAMIC_EXAMPLE_NUM_ERROR_MIN="${GER_DYNAMIC_EXAMPLE_NUM_ERROR_MIN:-${DYNAMIC_EXAMPLE_NUM_ERROR_MIN:-2}}"
    DYNAMIC_EXAMPLE_NUM_ERROR_TARGET_AVG="${GER_DYNAMIC_EXAMPLE_NUM_ERROR_TARGET_AVG:-${DYNAMIC_EXAMPLE_NUM_ERROR_TARGET_AVG:-8}}"
    DIALOGUE_FORM="${DIALOGUE_FORM:-${GER_DIALOGUE_FORM:-0}}"
}
