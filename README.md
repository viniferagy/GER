# Encode Errors

Code and resources for **[Encode Errors: Representational Retrieval of In-Context Demonstrations for Multilingual Grammatical Error Correction](https://aclanthology.org/2025.findings-acl.1090/)**, ACL 2025 Findings.

GER retrieves in-context demonstrations by encoding grammatical error information from an initial correction pass, then uses the retrieved demonstrations for final multilingual grammatical error correction.

## 1. Environment

Clone the repository and enter the project root.

```bash
cd ger-commit
```

Create the main GER environment.

```bash
bash setup_uv_cuda128.sh
source .venv/bin/activate
```

Create the Romanian scoring environment.

```bash
conda create -y -p ./.conda_eval_official python=3.8
.conda_eval_official/bin/python -m pip install \
  spacy==2.3.9 \
  nltk==3.9.1 \
  regex==2024.11.6 \
  rbpy-rb==0.6.6 \
  certifi==2026.5.20
mkdir -p .conda_eval_official/ssl
.conda_eval_official/bin/python - <<'PY'
import shutil
import certifi
from pathlib import Path
dst = Path(".conda_eval_official/ssl/cacert.pem")
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(certifi.where(), dst)
PY
```

Create the Estonian scoring environment.

```bash
conda create -y -p ./.conda_eval_estspacy python=3.8
.conda_eval_estspacy/bin/python -m pip install \
  spacy==3.0.9 \
  https://github.com/EstSyntax/EstSpaCy/releases/download/v1.0/et_dep_ud_sm-1.0.0.tar.gz
```

## 2. Models

Place the Hugging Face model directories under `models/`.

```text
models/
  Meta-Llama-3.1-8B-Instruct/
  Qwen2.5-7B-Instruct/
```

The pipeline uses the following model keys:

```text
llama31 -> models/Meta-Llama-3.1-8B-Instruct
qwen25  -> models/Qwen2.5-7B-Instruct
```

## 3. Data

Place the dataset sources under `datasets/`.

```text
datasets/
  external/
    ronacc_readerbench/
      train.txt
      dev.txt
      test.txt
  multilingual/
    rogec/
      errant/
  multilingual_raw/
    EN-conll14st-test-data/
    EN-wi+locness/
    DE-FALKO-MERLIN/
    RO-RoGEC/
    ET-estgec/
```

Public source pages for these datasets:

| Dataset files expected here | Public source |
|---|---|
| `multilingual_raw/EN-conll14st-test-data/` | [CoNLL-2014 Shared Task data release](https://www.comp.nus.edu.sg/~nlp/conll14st.html). Use the released annotated test data. |
| `multilingual_raw/EN-wi+locness/` | [BEA-2019 Shared Task data page](https://www.cl.cam.ac.uk/research/nl/bea2019st/). This provides W&I+LOCNESS train/dev/test inputs; official BEA-19 test scores must be obtained through Codabench. |
| `multilingual_raw/DE-FALKO-MERLIN/` | [Falko-MERLIN GEC corpus release](https://github.com/adrianeboyd/boyd-wnut2018) used by Boyd (2018). The original MERLIN corpus is also available from the [MERLIN/PORTA corpus portal](https://www.porta.eurac.edu/lci/merlin/). |
| `multilingual_raw/RO-RoGEC/` and `external/ronacc_readerbench/` | [RoGEC/RONACC repository](https://github.com/teodor-cotet/RoGEC), which links the RONACC corpus and tokenized release used for Romanian GEC. |
| `multilingual_raw/ET-estgec/` | [TartuNLP EstGEC resources](https://github.com/TartuNLP/estgec) for the Tartu L1/L2 corpus files, plus the [EstGEC-L2 corpus](https://github.com/tlu-dt-nlp/EstGEC-L2-Corpus) for the M2-format Estonian L2 dev/test corpus. |

Prepare the standard JSON and runtime files.

```bash
python scripts/ger_prepare_datasets.py --overwrite
```

To prepare selected datasets only:

```bash
python scripts/ger_prepare_datasets.py --languages en de ro --overwrite
```

Language keys:

```text
en    -> CoNLL-14
bea19 -> BEA-19
de    -> Falko-Merlin
ro    -> RoGEC / RONACC ReaderBench
et    -> EstGEC
```

## 4. Run GER

Run the full Table 1 setting: two models, five datasets, three seeds.

```bash
python scripts/ger.py \
  --models llama31 qwen25 \
  --languages en bea19 de ro et \
  --seeds 88 111 222 \
  --gpus 0,1,2,3 \
  --num-shards 4 \
  --batch-size 4 \
  --execute
```

Run one model, one dataset, one seed.

```bash
python scripts/ger.py \
  --models llama31 \
  --languages en \
  --seeds 88 \
  --gpus 0,1,2,3 \
  --num-shards 4 \
  --batch-size 4 \
  --execute
```

Run each pipeline stage separately.

```bash
python scripts/ger.py --models llama31 --languages en --seeds 88 --start-at initial --stop-after initial --gpus 0,1,2,3 --execute
python scripts/ger.py --models llama31 --languages en --seeds 88 --start-at cache --stop-after cache --gpus 0,1,2,3 --execute
python scripts/ger.py --models llama31 --languages en --seeds 88 --start-at retrieval --stop-after retrieval --gpus 0,1,2,3 --execute
python scripts/ger.py --models llama31 --languages en --seeds 88 --start-at final --stop-after final --gpus 0,1,2,3 --execute
python scripts/ger.py --models llama31 --languages en --seeds 88 --start-at score --stop-after score --gpus 0,1,2,3 --execute
```

Resume from a stage and overwrite existing artifacts when needed.

```bash
python scripts/ger.py \
  --models llama31 \
  --languages en \
  --seeds 88 \
  --start-at retrieval \
  --gpus 0,1,2,3 \
  --num-shards 4 \
  --overwrite \
  --execute
```

## 5. Outputs

Initial correction outputs:

```text
multilingual/results_<model>_default/initial_predictions_train_8/<train_dataset>/
multilingual/results_<model>_default/initial_predictions_test_8/<test_dataset>/
```

GER representation cache:

```text
cache/representation/
```

GER retrieved demonstrations:

```text
multilingual/results_ger_<model>/retrieve_ger_source/retrieved_examples_dim<dim>_8_8/<test_dataset>/retrieval.jsonl
multilingual/results_ger_<model>/retrieve_ger_vanilla_seed<seed>/<test_dataset>/retrieval.jsonl
```

Final GER generations:

```text
multilingual/results_ger_<model>/res_ger_vanilla_seed<seed>/<test_dataset>/predictions.jsonl
```

Scores and submission files:

```text
results/official_eval/ger/<dataset>/<model>_ger_vanilla_seed<seed>/
```

BEA-19 submission archives are written to:

```text
results/official_eval/ger/bea19/<model>_ger_vanilla_seed<seed>/bea19.zip
```

BEA-19 test references are not public, so the local pipeline only prepares the
submission archive. Upload `bea19.zip` to the BEA-2019 Codabench competition to
obtain the official test score:
https://www.codabench.org/competitions/10960/

## 6. Citation

```bibtex
@inproceedings{peng-etal-2025-encode,
    title = "Encode Errors: Representational Retrieval of In-Context Demonstrations for Multilingual Grammatical Error Correction",
    author = "Peng, Guangyue  and
      Li, Wei  and
      Luo, Wen  and
      Wang, Houfeng",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1090/",
    doi = "10.18653/v1/2025.findings-acl.1090",
    pages = "21166--21180",
}
```
