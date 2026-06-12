import sys
import yaml
import os
from pathlib import Path


GER_ROOT = Path(os.environ.get("GER_PROJECT_ROOT", Path(__file__).resolve().parents[3]))


def resolve_project_model_path(value):
    if not isinstance(value, str):
        return value

    model_root = os.environ.get("GER_MODEL_ROOT_DIR", str(GER_ROOT / "models"))
    replacements = (
        ("Meta-Llama-3.1-8B-Instruct", os.environ.get("GER_LLAMA31_MODEL_PATH", str(Path(model_root) / "Meta-Llama-3.1-8B-Instruct"))),
        ("Qwen2.5-7B-Instruct", os.environ.get("GER_QWEN25_MODEL_PATH", str(Path(model_root) / "Qwen2.5-7B-Instruct"))),
        ("xlm-roberta-large", os.environ.get("GER_XLM_ROBERTA_MODEL_PATH", str(Path(model_root) / "xlm-roberta-large"))),
    )
    for marker, replacement in replacements:
        if value == f"Qwen/{marker}" or value.endswith(marker):
            return replacement
    return value


def normalize_value(key, value):
    if key == "DATASETS_DIR":
        data_root = os.environ.get("GER_DATA_ROOT", str(GER_ROOT))
        return os.environ.get("GER_DATASETS_DIR", str(Path(data_root) / "datasets"))
    if key in {"MODEL", "ASSIST_MODEL", "EMBED_MODEL"}:
        return resolve_project_model_path(value)
    return value

def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            for key, value in data.items():
                value = normalize_value(key, value)
                yield f"export {key}='{value}'"
    except FileNotFoundError:
        print(f"Error: The file {yaml_file} does not exist.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error in YAML file: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_yaml.py <path-to-yaml-file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    
    # output environments setting command
    for command in load_yaml(yaml_file):
        print(command)
