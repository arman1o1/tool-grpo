import sys
import subprocess
import tempfile
import os
from typing import Tuple

class PythonSandbox:
    """
    Executes Python snippets in an isolated subprocess with strict timeouts.
    Captures stdout and stderr for multi-turn RLVR environment feedback.
    """
    def __init__(self, timeout_seconds: float = 3.0, max_output_chars: int = 1000):
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def execute(self, code: str) -> Tuple[bool, str]:
        """
        Executes code safely in a separate process.
        Returns (success: bool, output: str).
        """
        # Strip dangerous boilerplate if necessary, keep execution lightweight
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                output = stdout if stdout else "[Executed successfully with no stdout]"
                return True, output[:self.max_output_chars]
            else:
                err_msg = stderr.splitlines()[-1] if stderr else "Unknown runtime error"
                return False, f"RuntimeError: {err_msg}"[:self.max_output_chars]

        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Execution exceeded {self.timeout_seconds}s limit."
        except Exception as e:
            return False, f"ExecutionError: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

if __name__ == "__main__":
    sandbox = PythonSandbox()
    
    # Test valid execution
    code1 = "import math\nprint(math.factorial(10))"
    ok1, out1 = sandbox.execute(code1)
    assert ok1 and out1 == "3628800", f"Test 1 Failed: {out1}"
    
    # Test timeout protection
    code2 = "while True: pass"
    ok2, out2 = sandbox.execute(code2)
    assert not ok2 and "TimeoutError" in out2, f"Test 2 Failed: {out2}"
    
    # Test syntax error handling
    code3 = "print(10 / 0)"
    ok3, out3 = sandbox.execute(code3)
    assert not ok3 and "ZeroDivisionError" in out3, f"Test 3 Failed: {out3}"
    
    print("[+] PythonSandbox: All self-tests passed successfully.")
