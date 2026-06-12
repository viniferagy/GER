# GER

This repository contains the GER pipeline for multilingual grammatical error correction.

## Setup
1. Create the Python environments used by the project.
2. Ensure the model checkpoints are available under `models/`.
3. Ensure the datasets are available under `datasets/`.

## Run Table 1
1. Prepare retrieval and runtime files:
   `python scripts/ger_new_table1.py prepare`
2. Launch the generated Table 1 job script under `repro/new_table1/`.
3. Wait for the persistent run to finish and write the completion markers.
4. Collect scores:
   `python scripts/ger_new_table1.py collect`

## Official scoring
1. Score Romanian outputs with the local ERRANT path.
2. Score Estonian outputs with the bundled modified M2 scorer.
3. Package BEA-19 submissions with:
   `python scripts/ger_prepare_bea19_submission.py --write`
4. Apply the standard fixed postprocess with:
   `python scripts/ger_standard_postprocess.py --input INPUT.jsonl --output-jsonl OUTPUT.jsonl --output-txt OUTPUT.txt --char-threshold 0.96`

## Outputs
- `results/official_eval/new_table1/...` stores official-style scores.
- `results/ger_new_table1*.csv` stores run summaries.
- `results/ger_new_table1.md` stores the Table 1 summary.
