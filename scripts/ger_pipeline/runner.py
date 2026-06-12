"""Step execution for GER pipeline commands."""
from __future__ import annotations

import os
import subprocess

from .files import file_ok, require_file
from .steps import Step


def command_text(step: Step) -> str:
    return step.shell_preview()


def execute_step(step: Step, *, execute: bool, overwrite: bool = False) -> None:
    if step.required_outputs and not overwrite and all(file_ok(path) for path in step.required_outputs):
        print(f"[skip] {step.name}", flush=True)
        for path in step.required_outputs:
            print(f"       {path}", flush=True)
        return
    print(f"[step] {step.name}", flush=True)
    print(f"       {command_text(step)}", flush=True)
    if not execute:
        return
    if overwrite:
        for path in step.required_outputs:
            path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(step.env)
    command = step.command
    if step.use_run_and_hold:
        if step.hold_script is None or step.gpu is None:
            raise ValueError(f"{step.name} requested run_and_hold without required configuration")
        command = ["bash", str(step.hold_script), step.gpu, *step.command]
    subprocess.run(command, cwd=step.cwd, env=env, check=True)
    for path in step.required_outputs:
        require_file(path)

