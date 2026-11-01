import pytest
from src.sandbox import PythonSandbox

def test_sandbox_simple_execution():
    sandbox = PythonSandbox(timeout_seconds=2.0)
    success, output = sandbox.execute("print('Hello from sandbox')")
    assert success is True
    assert output == "Hello from sandbox"

def test_sandbox_math_computation():
    sandbox = PythonSandbox(timeout_seconds=2.0)
    code = """
import math
print(math.comb(10, 3))
"""
    success, output = sandbox.execute(code)
    assert success is True
    assert output == "120"

def test_sandbox_syntax_error():
    sandbox = PythonSandbox(timeout_seconds=2.0)
    success, output = sandbox.execute("def broken_code(:")
    assert success is False
    assert "SyntaxError" in output

def test_sandbox_runtime_exception():
    sandbox = PythonSandbox(timeout_seconds=2.0)
    success, output = sandbox.execute("x = 10 / 0")
    assert success is False
    assert "ZeroDivisionError" in output

def test_sandbox_timeout_interruption():
    sandbox = PythonSandbox(timeout_seconds=1.0)
    code = "import time\ntime.sleep(3.0)\nprint('done')"
    success, output = sandbox.execute(code)
    assert success is False
    assert "TimeoutError" in output

def test_sandbox_output_truncation():
    sandbox = PythonSandbox(timeout_seconds=2.0, max_output_chars=50)
    code = "print('A' * 200)"
    success, output = sandbox.execute(code)
    assert success is True
    assert len(output) <= 50
