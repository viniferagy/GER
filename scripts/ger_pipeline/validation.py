"""Static validation helpers for GER reproduction runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import LanguageSpec, ModelSpec
from .paths import ProjectPaths
from .runner import baseline_files

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class Check:
    name: str
    path: Path
    ok: bool
    required: bool = True

    @property
    def status(self) -> str:
        if self.ok:
            return "ok"
        return "missing" if self.required else "optional-missing"


@dataclass(frozen=True)
class BEAOfficialScoreCheck:
    path: Path
    ok: bool
    issue: str

    @property
    def status(self) -> str:
        return "ok" if self.ok else "check"


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def dir_ok(path: Path) -> bool:
    return path.exists() and path.is_dir()


def bea_official_score_ok(path: Path) -> BEAOfficialScoreCheck:
    if not file_ok(path):
        return BEAOfficialScoreCheck(path, False, "missing official.score")
    text = path.read_text(encoding="utf-8", errors="replace")
    required_labels = ("Precision", "Recall", "F_0.5", "Platform", "Submission", "ZIP_SHA256", "TXT_SHA256")
    values = {label: _score_label_value(text, label) for label in required_labels}
    issues: list[str] = []
    for metric in ("Precision", "Recall", "F_0.5"):
        value = values[metric]
        if value is None:
            issues.append(f"missing {metric}")
            continue
        try:
            number = float(value)
        except ValueError:
            issues.append(f"invalid {metric}")
            continue
        if not 0.0 <= number <= 1.0:
            issues.append(f"{metric} outside [0,1]")
    for label in ("Platform", "Submission"):
        if not values[label]:
            issues.append(f"missing {label}")
    for label in ("ZIP_SHA256", "TXT_SHA256"):
        value = values[label] or ""
        if not SHA256_RE.match(value):
            issues.append(f"invalid {label}")
    return BEAOfficialScoreCheck(path, not issues, "; ".join(issues))


def _score_label_value(text: str, label: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}\s*:\s*(.*?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def checks_for(paths: ProjectPaths, model: ModelSpec, lang: LanguageSpec) -> list[Check]:
    train_prefix = paths.multi_p_dir / lang.train_dataset / "train"
    ref_prefix = paths.reference_results_dir / lang.test_dataset
    reference_label = Path(f"{ref_prefix}.label")
    if not reference_label.exists():
        reference_label = Path(f"{ref_prefix}.label.gold")
    reference_required = lang.local_scoring
    checks = [
        Check("model-dir", paths.model_path(model), dir_ok(paths.model_path(model))),
        Check("repe-cache-script", paths.repe_gec_dir / "build_gec_representation_cache.py", file_ok(paths.repe_gec_dir / "build_gec_representation_cache.py")),
        Check("repe-retrieval-script", paths.repe_gec_dir / "retrieve_gec_examples_by_representation.py", file_ok(paths.repe_gec_dir / "retrieve_gec_examples_by_representation.py")),
        Check("multilingual-icl-script", paths.multilingual_dir / "scripts" / "pipeline" / "infer_icl_probing.sh", file_ok(paths.multilingual_dir / "scripts" / "pipeline" / "infer_icl_probing.sh")),
        Check("train-src", Path(f"{train_prefix}.src"), file_ok(Path(f"{train_prefix}.src"))),
        Check("train-tgt", Path(f"{train_prefix}.tgt"), file_ok(Path(f"{train_prefix}.tgt"))),
        Check("train-label", Path(f"{train_prefix}.label"), file_ok(Path(f"{train_prefix}.label"))),
        Check("reference-src", Path(f"{ref_prefix}.src"), file_ok(Path(f"{ref_prefix}.src"))),
        Check("reference-tgt", Path(f"{ref_prefix}.tgt"), file_ok(Path(f"{ref_prefix}.tgt")), required=reference_required),
        Check("reference-label", reference_label, file_ok(reference_label), required=reference_required),
        Check("reference-m2", paths.m2_file(lang), file_ok(paths.m2_file(lang)), required=reference_required and lang.test_dataset != "nlpcc18"),
    ]
    for split in ("train", "test"):
        txt, label = baseline_files(paths, model, lang, split)
        checks.append(Check(f"baseline-{split}-txt", txt, file_ok(txt), required=False))
        checks.append(Check(f"baseline-{split}-label", label, file_ok(label), required=False))
    return checks


def required_failures(checks: list[Check]) -> list[Check]:
    return [check for check in checks if check.required and not check.ok]
