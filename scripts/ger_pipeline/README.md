# GER Pipeline

This directory contains the formal GER-only reproduction pipeline. It starts
from local official/raw dataset assets, prepares standard JSON splits, builds
runtime train/test text files, runs initial generation, builds GER retrieval,
runs final generation, applies standard postprocessing, and scores the output.

## Entrypoints

- `setup_uv_4090_cuda128.sh`: create the 4090 uv environment. It runs
  `uv sync --extra cuda128`, then applies `requirements.txt` as the post-uv
  runtime override.
- `setup_uv_a800_cuda126.sh`: create the A800 uv environment. It runs
  `uv sync --extra cuda126`, then applies `requirements.txt` as the post-uv
  runtime override.
- `pyproject.toml`: uv resolver baseline for the project environment.
- `requirements.txt`: post-uv override packages for the local vLLM runtime.
- `scripts/ger_prepare_datasets.py`: prepare Table 1 standard datasets from
  local official/raw sources.
- `scripts/ger.py`: run the end-to-end GER pipeline. When executed from
  `--start-at initial`, it calls dataset preparation before runtime files are
  written.

## Dataset Preparation

`dataset_preparation.py` owns the raw-to-standard-JSON conversion.

| Dataset | Standard JSON Output | Local Source |
|---|---|---|
| CoNLL-14 | `datasets/multilingual/conll14/test.json` | `datasets/multilingual_raw/EN-conll14st-test-data/noalt/official-2014.combined.m2` |
| BEA-19 | `datasets/multilingual/bea19/test.json` | `datasets/multilingual_raw/EN-wi+locness/test/ABCN.test.bea19.orig` |
| WI+LOCNESS train | `datasets/multilingual/wilocness/train.json` | `datasets/multilingual_raw/EN-wi+locness/m2/ABC.train.gold.bea19.m2` |
| Falko-Merlin | `datasets/multilingual/falko_merlin/{train,valid,test}.json` and `datasets/multilingual/falko_merlin_train/{train,valid}.json` | `datasets/multilingual_raw/DE-FALKO-MERLIN/fm-{train,dev,test}.{src,trg}` |
| RoGEC | `datasets/multilingual/rogec/{train,valid,test}.json` and `datasets/multilingual/rogec_train/{train,valid}.json` | `datasets/multilingual_raw/RO-RoGEC/{train,dev,test}.txt` |
| RONACC/ReaderBench scoring | `datasets/external/ronacc_readerbench/{train,dev,test}.{src,tgt}` and `test.m2` | `datasets/external/ronacc_readerbench/{train,dev,test}.txt` |
| EstGEC | `datasets/multilingual/estgec/{train,valid,test}.json` and `datasets/multilingual/estgec_train/{train,valid}.json` | `datasets/multilingual_raw/ET-estgec/Tartu_L2_corpus/Tartu_L2_learner_corpus_parallel.txt` and `datasets/multilingual_raw/ET-estgec/Tartu_L1_corpus/test/test_m2.txt` |

The formal pipeline does not generate or read `*_train/test.json` mirrors. If
those files exist in an old workspace, they are legacy experiment artifacts and
are outside the formal GER chain.

## Runtime Flow

1. `dataset_preparation.py` writes standard JSON splits.
2. `data_sources.py` writes:
   - `multilingual/runtime_sources/<test_dataset>/test.src`
   - `multilingual/runtime_train_data/<train_dataset>/train.{src,tgt,label}`
3. `steps.py` writes runtime YAML configs and launches initial generation.
   Train initial generation passes `GER_INITIAL_RESULT_SPLIT=train`, and
   `infer_initial_predictions.sh` forwards this as `--infer_mode=train`.
4. `repe_gec/build_gec_representation_cache.py` builds the GER representation
   cache from train initial predictions and train labels.
5. `repe_gec/retrieve_gec_examples_by_representation.py` retrieves GER examples
   for the test source.
6. `infer_retrieved_icl.sh` runs final generation with dynamic GER examples.
7. `postprocess.py` applies the standard GER output cleanup before scoring.
8. `scoring.py` runs the dataset-specific official or official-style scorer.

## External Dependencies

- Models under `models/`, e.g. `Meta-Llama-3.1-8B-Instruct` and
  `Qwen2.5-7B-Instruct`.
- Python environment `.venv` produced by the setup scripts above.
- Romanian official-style ERRANT environment `.conda_eval_official`.
- Estonian scoring/tokenization environment `.conda_eval_estspacy`.
- Local raw dataset assets listed above.
