import pytest
from src.verifier import MathVerifier, extract_boxed, latex_to_sympy_str

def test_extract_boxed_single():
    assert extract_boxed(r"The answer is \boxed{42}") == "42"

def test_extract_boxed_nested_braces():
    assert extract_boxed(r"Solution: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"Solution: \boxed{\sqrt{\frac{a}{b}}}") == r"\sqrt{\frac{a}{b}}"

def test_latex_to_sympy():
    assert latex_to_sympy_str(r"\frac{1}{2}") == "((1)/(2))"
    assert latex_to_sympy_str(r"\sqrt{16}") == "sqrt(16)"

def test_verifier_gsm8k_exact():
    v = MathVerifier()
    completion = "<think>Compute 10 + 5</think><answer>15</answer>"
    gt = "The sum is 15. #### 15"
    assert v.verify(completion, gt) is True

def test_verifier_float_and_commas():
    v = MathVerifier()
    completion = "<answer>1,500.0</answer>"
    gt = "#### 1500"
    assert v.verify(completion, gt) is True

def test_verifier_symbolic_fractions():
    v = MathVerifier()
    # 0.5 vs 1/2
    assert v.verify("<answer>0.5</answer>", r"\boxed{\frac{1}{2}}") is True
    assert v.verify("<answer>1/2</answer>", r"\boxed{\frac{1}{2}}") is True
    # 3/4 vs 6/8
    assert v.verify("<answer>6/8</answer>", r"\boxed{\frac{3}{4}}") is True

def test_verifier_algebraic_equivalence():
    v = MathVerifier()
    assert v.verify("<answer>2*x + 1</answer>", "1 + 2*x") is True
    assert v.verify("<answer>(x + 1)**2</answer>", "x**2 + 2*x + 1") is True

def test_verifier_negative_mismatch():
    v = MathVerifier()
    assert v.verify("<answer>42</answer>", "#### 43") is False
    assert v.verify("<answer>1/3</answer>", r"\boxed{\frac{1}{4}}") is False
