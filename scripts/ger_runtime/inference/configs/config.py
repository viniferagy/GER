import json
import os

ROOT_WORK_DIR = os.environ.get(
    "GER_DATA_ROOT",
    os.environ.get(
        "GER_ROOT_WORK_DIRECTORY",
        json.load(open(os.path.join(os.path.dirname(__file__), "directory.json")))['root_work_directory'],
    )
)

DATA_ROOT_DIR = os.path.join(ROOT_WORK_DIR, "datasets")

DATA_DIR_NAME = {
    "bea19": "multilingual/bea19",
    "conll14": "multilingual/conll14",
    "estgec": "multilingual/estgec",
    "estgec_train": "multilingual/estgec_train",
    "falko_merlin": "multilingual/falko_merlin",
    "falko_merlin_train": "multilingual/falko_merlin_train",
    "ronacc_readerbench": "multilingual/ronacc_readerbench",
    "ronacc_readerbench_train": "multilingual/ronacc_readerbench_train",
    "wilocness": "multilingual/wilocness",
}


def get_data_dir(single_dataset_name):
    assert single_dataset_name in DATA_DIR_NAME, f"{single_dataset_name} is not in the map in the DATA_DIR_NAME of configs/config.py"
    return os.path.join(DATA_ROOT_DIR, DATA_DIR_NAME[single_dataset_name])
