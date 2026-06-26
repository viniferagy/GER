"""Official and official-style scoring for formal GER outputs."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .files import file_ok, require_file
from .final_output import FinalRun, write_formal_predictions
from .paths import ProjectPaths


def clean_outputs_for_scoring(output: Path, reference_m2: Path, *, backup: bool = True) -> int:
    if not output.exists() or not reference_m2.exists():
        return 0
    predictions = output.read_text(encoding="utf-8", errors="replace").splitlines()
    sources: list[str] = []
    with reference_m2.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("S "):
                sources.append(line[2:].rstrip("\n"))
    if len(predictions) != len(sources):
        return 0

    changed = 0
    cleaned: list[str] = []
    for pred, src in zip(predictions, sources, strict=True):
        pred_tokens = pred.split()
        src_tokens = src.split()
        too_long = len(pred_tokens) > max(int(len(src_tokens) * 1.75), len(src_tokens) + 50)
        if too_long:
            cleaned.append(src)
            changed += 1
        else:
            cleaned.append(pred)
    if changed:
        if backup:
            backup_path = output.with_suffix(output.suffix + ".pre_score_clean.bak")
            if not backup_path.exists():
                backup_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
        output.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    return changed


def run_logged(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None, execute: bool) -> None:
    print("       cd", cwd, "&&", " ".join(shlex.quote(str(part)) for part in command), ">", log_path, flush=True)
    if not execute:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)


ESTSPACY_RETOKENIZE_CODE = r"""
import json
import sys
from pathlib import Path

import spacy

predictions_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
nlp = spacy.load("et_dep_ud_sm")

with predictions_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as out:
    for line in src:
        if not line.strip():
            continue
        item = json.loads(line)
        prediction = str(item.get("prediction", "")).strip()
        prediction = prediction.replace("\n", " ").replace("\r", " ").strip()
        out.write(" ".join(token.text for token in nlp.tokenizer(prediction)).strip() + "\n")
"""


def score_formal_output(paths: ProjectPaths, run: FinalRun, *, execute: bool, overwrite: bool) -> None:
    write_formal_predictions(run, execute=execute, overwrite=overwrite)
    if run.lang.submission_output:
        print(f"[step] package {run.final_artifact}", flush=True)
        if execute:
            require_file(run.final_artifact)
        return
    print(f"[step] score {run.final_artifact}", flush=True)
    if file_ok(run.final_artifact) and not overwrite:
        return
    if run.lang.code == "en":
        reference = paths.datasets_dir / "multilingual_raw" / "EN-conll14st-test-data" / "noalt" / "official-2014.combined.m2"
        if execute:
            clean_outputs_for_scoring(run.final_output, reference, backup=True)
        run_logged(
            [str(paths.python), "evaluators/m2scorer/scripts/m2scorer.py", str(run.final_output), str(reference)],
            cwd=paths.inference_runtime_dir,
            log_path=run.final_artifact,
            execute=execute,
        )
    elif run.lang.code == "de":
        reference = paths.datasets_dir / "multilingual_raw" / "DE-FALKO-MERLIN" / "fm-test.m2"
        if execute:
            clean_outputs_for_scoring(run.final_output, reference, backup=True)
        run_logged(
            [str(paths.python), "evaluators/m2scorer/scripts/m2scorer.py", str(run.final_output), str(reference)],
            cwd=paths.inference_runtime_dir,
            log_path=run.final_artifact,
            execute=execute,
        )
    elif run.lang.code == "ro":
        hyp_m2 = run.score_dir / "hyp.m2"
        source = paths.datasets_dir / "external" / "ronacc_readerbench" / "test.src"
        reference = paths.datasets_dir / run.lang.m2_relative_path
        errant_dir = paths.datasets_dir / "external" / "ronacc_readerbench" / "errant"
        ro_python = paths.root / ".conda_eval_official" / "bin" / "python"
        ro_ca_bundle = paths.root / ".conda_eval_official" / "ssl" / "cacert.pem"
        ro_env = {"SSL_CERT_FILE": str(ro_ca_bundle), "REQUESTS_CA_BUNDLE": str(ro_ca_bundle)}
        run_logged(
            [str(ro_python), "parallel_to_m2.py", "-orig", str(source), "-cor", str(run.final_output), "-out", str(hyp_m2), "-lang", "ro"],
            cwd=errant_dir,
            log_path=run.score_dir / "parallel_to_m2.log",
            env=ro_env,
            execute=execute,
        )
        run_logged([str(ro_python), "compare_m2.py", "-hyp", str(hyp_m2), "-ref", str(reference)], cwd=errant_dir, log_path=run.final_artifact, env=ro_env, execute=execute)
    elif run.lang.code == "et":
        estspacy_output = run.score_dir / "estgec-output-estspacy.txt"
        scorer = paths.root / "datasets" / "multilingual_raw" / "ET-estgec" / "M2_scorer_est" / "m2scorer_by_type" / "scripts" / "m2scorer.py"
        reference = paths.datasets_dir / run.lang.m2_relative_path
        et_python = paths.root / ".conda_eval_estspacy" / "bin" / "python"
        run_logged([str(et_python), "-c", ESTSPACY_RETOKENIZE_CODE, str(run.final_predictions), str(estspacy_output)], cwd=paths.root, log_path=run.score_dir / "estspacy_retokenize.log", execute=execute)
        run_logged([str(paths.python), str(scorer), str(estspacy_output), str(reference)], cwd=paths.root, log_path=run.final_artifact, execute=execute)
    else:
        reference = paths.m2_file(run.lang)
        run_logged(
            [str(paths.python), "evaluators/m2scorer/scripts/m2scorer.py", str(run.final_output), str(reference)],
            cwd=paths.inference_runtime_dir,
            log_path=run.final_artifact,
            execute=execute,
        )
    if execute:
        require_file(run.final_artifact)
