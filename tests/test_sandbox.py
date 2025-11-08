from src.utils.sandbox import is_safe_python, run_tests


def test_sandbox_blocks_forbidden_imports():
    code = "import os\nprint('bad')"
    assert not is_safe_python(code)


def test_sandbox_runs_safe_code():
    solution = """
def add(a, b):
    return a + b
"""
    tests = """
assert add(1, 2) == 3
assert add(-1, 1) == 0
"""
    assert run_tests(solution, tests)
