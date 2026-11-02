import re
import math
from typing import Optional

def extract_boxed(text: str) -> Optional[str]:
    """
    Extracts the contents of the last LaTeX \\boxed{...} environment,
    correctly handling nested curly braces (e.g. \\boxed{\\frac{1}{2}}).
    """
    if not text:
        return None
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None

def latex_to_sympy_str(expr: str) -> str:
    """
    Normalizes common LaTeX math constructs to Python/SymPy expressions.
    """
    cleaned = expr.strip()
    # Unpack nested \boxed if any remains
    boxed = extract_boxed(cleaned)
    if boxed is not None:
        cleaned = boxed
    # \frac{a}{b} -> ((a)/(b))
    # Handle multiple passes for nested fractions
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", cleaned)
    # \sqrt{a} -> sqrt(a)
    cleaned = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", cleaned)
    # \cdot, \times -> *
    cleaned = cleaned.replace(r"\cdot", "*").replace(r"\times", "*")
    cleaned = cleaned.replace(r"\left", "").replace(r"\right", "")
    cleaned = cleaned.replace("$", "").replace(",", "")
    return cleaned.strip()

class MathVerifier:
    """
    Symbolic and numerical verifier for mathematical reasoning benchmarks
    (GSM8K, MATH-500, SVAMP).
    Extracts predicted answers from <answer> tags or \\boxed{} and compares
    against ground truth via exact matching, float equivalence, and SymPy algebra.
    """
    def __init__(self):
        self.answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        self.gsm8k_delim = "####"

    def extract_predicted_answer(self, completion: str) -> Optional[str]:
        """
        Extracts candidate answer from <answer> tags first, falling back to LaTeX \\boxed{}
        or the final numerical token. Unwraps any nested \\boxed{} inside <answer>.
        """
        if not completion:
            return None
            
        candidate = None
        matches = self.answer_tag_pattern.findall(completion)
        if matches:
            candidate = matches[-1].strip()
        else:
            boxed = extract_boxed(completion)
            if boxed is not None:
                candidate = boxed
            else:
                # Fallback: extract last number in the completion text
                numbers = re.findall(r"[-+]?\d*\.?\d+", completion)
                if numbers:
                    candidate = numbers[-1].strip()
                else:
                    lines = [line.strip() for line in completion.strip().splitlines() if line.strip()]
                    candidate = lines[-1] if lines else None

        if candidate:
            # If the candidate still wraps a \boxed{...}, unwrap it
            inner_boxed = extract_boxed(candidate)
            if inner_boxed is not None:
                candidate = inner_boxed
        return candidate

    def extract_ground_truth(self, solution: str) -> str:
        """
        Extracts ground-truth solution from standard benchmark strings.
        Supports <answer>...</answer>, GSM8K '#### <answer>', and MATH '\\boxed{<answer>}'.
        """
        if not solution:
            return ""
        matches = self.answer_tag_pattern.findall(solution)
        if matches:
            gt = matches[-1].strip()
        elif self.gsm8k_delim in solution:
            gt = solution.split(self.gsm8k_delim)[-1].strip()
        else:
            gt = solution.strip()

        boxed = extract_boxed(gt)
        if boxed is not None:
            return boxed
            
        return gt

    def normalize(self, text: str) -> str:
        """
        Cleans mathematical strings: strips currency, commas, percentages, and trailing periods.
        """
        if not text:
            return ""
        # Strip \boxed{} if still present
        b = extract_boxed(text)
        if b is not None:
            text = b
        cleaned = text.replace("$", "").replace(",", "").replace("%", "").strip()
        if cleaned.endswith("."):
            cleaned = cleaned[:-1].strip()
        return cleaned

    def sympy_equivalent(self, pred: str, gt: str) -> bool:
        """
        Tests algebraic and fraction equivalence using SymPy symbolic evaluation.
        """
        try:
            from sympy import simplify
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
            )
            transformations = standard_transformations + (implicit_multiplication_application,)
            p_clean = latex_to_sympy_str(pred)
            g_clean = latex_to_sympy_str(gt)
            p_expr = parse_expr(p_clean, transformations=transformations)
            g_expr = parse_expr(g_clean, transformations=transformations)
            diff = simplify(p_expr - g_expr)
            if diff == 0:
                return True
            try:
                if abs(float(diff)) < 1e-4:
                    return True
            except Exception:
                pass
            return False
        except Exception:
            return False

    def verify(self, completion: str, ground_truth_raw: str) -> bool:
        """
        Compares predicted vs ground truth.
        Hierarchy:
        1. Exact string match after normalization
        2. Float / numeric equivalence (e.g. 42 == 42.0 or 1/2 == 0.5)
        3. SymPy symbolic expression equivalence
        """
        pred = self.extract_predicted_answer(completion)
        if pred is None:
            return False
            
        gt = self.extract_ground_truth(ground_truth_raw)
        norm_pred = self.normalize(pred)
        norm_gt = self.normalize(gt)

        # 1. Exact string match
        if norm_pred == norm_gt:
            return True

        # 2. Float numeric equivalence
        try:
            p_val = float(norm_pred)
            g_val = float(norm_gt)
            if math.isclose(p_val, g_val, rel_tol=1e-5, abs_tol=1e-5):
                return True
        except ValueError:
            pass

        # 3. SymPy symbolic equivalence (fractions, algebraic expressions, radicals)
        if self.sympy_equivalent(norm_pred, norm_gt):
            return True

        return False

if __name__ == "__main__":
    v = MathVerifier()
    # Test cases
    assert v.verify("<think>...</think><answer>42</answer>", "#### 42")
    assert v.verify(r"<answer>\boxed{\frac{1}{2}}</answer>", r"\boxed{\frac{1}{2}}")
    assert v.verify(r"<answer>0.5</answer>", r"\boxed{\frac{1}{2}}")
    assert v.verify(r"<answer>1/2</answer>", r"\boxed{\frac{1}{2}}")
    assert v.verify(r"<answer>2*x + 1</answer>", r"1 + 2*x")
    assert v.verify("<answer>1,250.0</answer>", "#### 1250")
    assert not v.verify("<answer>15</answer>", "#### 16")
    print("[+] MathVerifier: All self-tests passed successfully.")
