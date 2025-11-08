"""Very small unit-test sandbox for RLVR rewards."""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "shutil", "pathlib"}


def is_safe_python(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                return False
    return True


def run_tests(solution: str, tests: str, timeout: float = 3.0) -> bool:
    if not (is_safe_python(solution) and is_safe_python(tests)):
        return False
    cpu_cap = int(timeout)
    mem_cap = 512 * 1024 * 1024
    template = f"""
import resource
import signal
import sys
try:
    cpu_soft, cpu_hard = resource.getrlimit(resource.RLIMIT_CPU)
    desired_cpu = {cpu_cap}
    if cpu_hard > 0:
        cpu_soft = min(desired_cpu, cpu_hard)
    else:
        cpu_soft = desired_cpu
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_hard))
except (ValueError, OSError):
    pass
try:
    as_soft, as_hard = resource.getrlimit(resource.RLIMIT_AS)
    desired_as = {mem_cap}
    if as_hard > 0:
        as_soft = min(desired_as, as_hard)
    else:
        as_soft = desired_as
    resource.setrlimit(resource.RLIMIT_AS, (as_soft, as_hard))
except (ValueError, OSError):
    pass
{solution}
{tests}
"""
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "sandbox_run.py"
        script_path.write_text(template, encoding="utf8")
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                timeout=timeout,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
