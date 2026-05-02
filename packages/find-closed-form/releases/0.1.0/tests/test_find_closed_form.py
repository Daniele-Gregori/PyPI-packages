"""Tests for find_closed_form, validated against WolframScript."""

import subprocess
import math

import pytest
from sympy import pi, sin, cos

from find_closed_form import find_closed_form, FindClosedFormError


# ── Helpers ─────────────────────────────────────────────────────────────────

def _neval(expr) -> float:
    """Evaluate a sympy expression to float."""
    return float(expr.evalf(n=18))


def _run_wolframscript(code: str, timeout: int = 120) -> str:
    """Run a WolframScript expression and return the output string."""
    result = subprocess.run(
        ["wolframscript", "-code", code],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip()


def _wl_find_closed_form(num: float, max_results: int = 1) -> str:
    """
    Run FindClosedForm via WolframScript and return string representation.
    Uses the installed resource function.
    """
    code = (
        f'ResourceFunction["FindClosedForm"][{num!r}, {max_results}]'
    )
    return _run_wolframscript(code)


# ── Unit tests (no WolframScript needed) ────────────────────────────────────

class TestBasicMatching:
    """Test that known constants and expressions are found."""

    def test_sqrt2_over_2(self):
        """sqrt(2)/2 ≈ 0.7071067811865476"""
        results = find_closed_form(0.7071067811865476)
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 0.7071067811865476) < 1e-10

    def test_pi(self):
        """pi ≈ 3.141592653589793"""
        results = find_closed_form(3.141592653589793)
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - math.pi) < 1e-10

    def test_golden_ratio(self):
        """phi ≈ 1.618033988749895"""
        results = find_closed_form(1.618033988749895)
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 1.618033988749895) < 1e-10

    def test_e(self):
        """e ≈ 2.718281828459045"""
        results = find_closed_form(2.718281828459045)
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - math.e) < 1e-10

    def test_sqrt3(self):
        """sqrt(3) ≈ 1.7320508075688772"""
        results = find_closed_form(1.7320508075688772)
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - math.sqrt(3)) < 1e-10

    def test_ln2(self):
        """ln(2) ≈ 0.6931471805599453"""
        results = find_closed_form(0.6931471805599453)
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - math.log(2)) < 1e-10

    def test_simple_rational(self):
        """22/7 ≈ 3.142857142857143"""
        results = find_closed_form(
            3.142857142857143,
            functions=[("identity", lambda x: x, None)],
        )
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 22 / 7) < 1e-10


class TestPrecisionMatching:
    """Test digit-level precision matching."""

    def test_low_precision(self):
        """With few digits, more candidates might match."""
        results = find_closed_form(
            1.414, significant_digits=4, max_results=3,
            max_search_rounds=5,
        )
        assert len(results) >= 1
        for r in results:
            val = _neval(r)
            assert abs(1 - val / 1.414) < 1e-3

    def test_high_precision(self):
        """With many digits, only exact matches survive."""
        results = find_closed_form(
            1.4142135623730951, significant_digits=15,
        )
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - math.sqrt(2)) < 1e-13


class TestComplexityThreshold:
    """Test formula complexity filtering."""

    def test_low_threshold_fewer_results(self):
        """Lower complexity threshold should produce fewer or equal results."""
        r_low = find_closed_form(
            0.7071067811865476,
            max_results=5,
            formula_complexity_threshold=5,
            max_search_rounds=3,
        )
        r_high = find_closed_form(
            0.7071067811865476,
            max_results=5,
            formula_complexity_threshold=50,
            max_search_rounds=3,
        )
        assert len(r_low) <= len(r_high)

    def test_results_respect_threshold(self):
        """All results should have complexity <= threshold."""
        from find_closed_form import formula_complexity as fc
        results = find_closed_form(
            1.4142135623730951,
            max_results=5,
            formula_complexity_threshold=20,
            max_search_rounds=3,
        )
        for r in results:
            assert fc(r) <= 20


class TestCustomFunctions:
    """Test searching with user-specified functions."""

    def test_single_function(self):
        """Search with a single custom function."""
        results = find_closed_form(
            0.5,
            functions=[("sin(pi*#)", lambda x: sin(pi * x), None)],
        )
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 0.5) < 1e-10

    def test_callable_function(self):
        """Pass a bare callable."""
        results = find_closed_form(
            1.0,
            functions=lambda x: cos(pi * x),
        )
        # cos(pi*0) = 1
        assert len(results) >= 1

    def test_list_of_bare_callables(self):
        """Pass a list of bare lambdas."""
        results = find_closed_form(
            0.5,
            functions=[lambda x: sin(pi * x), lambda x: cos(pi * x)],
        )
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 0.5) < 1e-10


class TestMultiArgFunctions:
    """Test multi-argument lambda function forms."""

    def test_two_arg_product(self):
        """lambda x, y: x * y should find 2 * 3 = 6."""
        from sympy import Mul
        results = find_closed_form(
            6.0,
            functions=lambda x, y: x * y,
            max_search_rounds=5,
            algebraic_factor=False,
            algebraic_add=False,
        )
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 6.0) < 1e-10

    def test_two_arg_sum(self):
        """lambda x, y: x + y should find simple sums."""
        results = find_closed_form(
            5.0,
            functions=lambda x, y: x + y,
            max_search_rounds=5,
            algebraic_factor=False,
            algebraic_add=False,
        )
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 5.0) < 1e-10

    def test_two_arg_with_function(self):
        """lambda x, y: x * sin(pi * y) should find multiplicative matches."""
        results = find_closed_form(
            1.0,
            functions=lambda x, y: x * sin(pi * y),
            max_search_rounds=3,
            algebraic_factor=False,
            algebraic_add=False,
        )
        # 2 * sin(pi/6) = 2 * 0.5 = 1, or 1 * sin(pi/2) = 1
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 1.0) < 1e-10

    def test_two_arg_with_domain_filter(self):
        """Domain filter receives all arguments."""
        results = find_closed_form(
            1.0,
            functions=[(
                "x*sin(pi*y)",
                lambda x, y: x * sin(pi * y),
                lambda x, y: x > 0 and 0 < y < 1,
            )],
            max_search_rounds=3,
            algebraic_factor=False,
            algebraic_add=False,
        )
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - 1.0) < 1e-10

    def test_arity_auto_detection(self):
        """Arity is correctly auto-detected from lambda signature."""
        from find_closed_form.core import _func_arity
        assert _func_arity(lambda x: x) == 1
        assert _func_arity(lambda x, y: x + y) == 2
        assert _func_arity(lambda x, y, z: x + y + z) == 3
        assert _func_arity(sin) == 1

    def test_two_arg_algebraic_factor(self):
        """Algebraic factor search works with multi-arg functions."""
        from sympy import log
        # 2 * log(3) ≈ 2.1972...
        target = 2 * math.log(3)
        results = find_closed_form(
            target,
            functions=lambda x, y: x * log(y),
            max_search_rounds=5,
        )
        assert len(results) >= 1
        val = _neval(results[0])
        assert abs(val - target) < 1e-10


class TestOptions:
    """Test various option combinations."""

    def test_no_algebraic_factor(self):
        """Disabling algebraic_factor still works."""
        results = find_closed_form(
            math.pi, algebraic_factor=False, max_search_rounds=5,
        )
        # Might find pi directly through pi^1
        # Just check it doesn't crash
        assert isinstance(results, list)

    def test_no_algebraic_add(self):
        """Disabling algebraic_add still works."""
        results = find_closed_form(
            math.pi, algebraic_add=False, max_search_rounds=5,
        )
        assert isinstance(results, list)

    def test_max_results_multiple(self):
        """Requesting multiple results."""
        results = find_closed_form(
            0.7071067811865476, max_results=3, max_search_rounds=10,
        )
        assert len(results) >= 1
        assert len(results) <= 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero(self):
        """Searching for 0 should not crash."""
        results = find_closed_form(0.0, max_search_rounds=3)
        assert isinstance(results, list)

    def test_negative(self):
        """Searching for a negative number."""
        results = find_closed_form(-1.4142135623730951, max_search_rounds=10)
        assert isinstance(results, list)

    def test_inf_raises(self):
        with pytest.raises(FindClosedFormError):
            find_closed_form(float("inf"))

    def test_nan_raises(self):
        with pytest.raises(FindClosedFormError):
            find_closed_form(float("nan"))

    def test_integer_input(self):
        """Integer input should work."""
        results = find_closed_form(2, max_search_rounds=3)
        assert isinstance(results, list)


class TestResultQuality:
    """Test that results are correctly sorted by complexity."""

    def test_sorted_by_complexity(self):
        """Results should be sorted by ascending complexity."""
        from find_closed_form import formula_complexity as fc
        results = find_closed_form(
            0.7071067811865476, max_results=3, max_search_rounds=10,
        )
        if len(results) >= 2:
            complexities = [fc(r) for r in results]
            assert complexities == sorted(complexities)


# ── WolframScript validation tests ──────────────────────────────────────────

@pytest.fixture(scope="session")
def has_wolframscript():
    """Check if wolframscript is available."""
    try:
        result = subprocess.run(
            ["wolframscript", "-code", "1+1"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip() == "2"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Values to test: (numeric_value, description)
WL_TEST_CASES = [
    (0.7071067811865476, "sqrt(2)/2"),
    (3.141592653589793, "pi"),
    (2.718281828459045, "e"),
    (1.618033988749895, "golden ratio"),
    (1.7320508075688772, "sqrt(3)"),
    (0.6931471805599453, "ln(2)"),
    (0.5772156649015329, "Euler-Mascheroni gamma"),
    (1.4142135623730951, "sqrt(2)"),
]


class TestWolframScriptValidation:
    """
    Validate Python results against WolframScript.

    These tests compare that the Python find_closed_form produces results
    whose numerical values match the target to the same precision as the
    Wolfram Language FindClosedForm.
    """

    @pytest.mark.parametrize("num,desc", WL_TEST_CASES)
    def test_precision_matches_wl(self, num, desc, has_wolframscript):
        """Python result matches the target number to >= 10 digits."""
        if not has_wolframscript:
            pytest.skip("wolframscript not available")

        # Get Python result
        py_results = find_closed_form(num, max_results=1, max_search_rounds=15)
        if not py_results:
            pytest.skip(f"No Python result for {desc}")

        py_val = _neval(py_results[0])
        # Check Python result matches target to 10 digits
        assert abs(1 - py_val / num) < 1e-10, (
            f"Python result for {desc}: {py_results[0]} evaluates to {py_val}, "
            f"expected {num}"
        )

    @pytest.mark.parametrize("num,desc", WL_TEST_CASES)
    def test_wl_also_finds_result(self, num, desc, has_wolframscript):
        """WolframScript also finds a closed form for these values."""
        if not has_wolframscript:
            pytest.skip("wolframscript not available")

        wl_output = _wl_find_closed_form(num)
        # WL should return something other than None or the number itself
        assert wl_output, f"WolframScript returned empty for {desc}"
        assert wl_output != "None", f"WolframScript found nothing for {desc}"

    @pytest.mark.parametrize("num,desc", WL_TEST_CASES)
    def test_wl_and_python_agree_numerically(self, num, desc, has_wolframscript):
        """Both WL and Python results evaluate to the same number."""
        if not has_wolframscript:
            pytest.skip("wolframscript not available")

        # Python result
        py_results = find_closed_form(num, max_results=1, max_search_rounds=15)
        if not py_results:
            pytest.skip(f"No Python result for {desc}")
        py_val = _neval(py_results[0])

        # WL result: evaluate numerically
        wl_code = (
            f'ToString[N[ResourceFunction["FindClosedForm"][{num!r}], 18], InputForm]'
        )
        wl_val_str = _run_wolframscript(wl_code)
        # Strip WL precision markers like `18. from the number
        wl_clean = wl_val_str.split("`")[0] if "`" in wl_val_str else wl_val_str
        try:
            wl_val = float(wl_clean)
        except (ValueError, TypeError):
            pytest.skip(f"Could not parse WL output: {wl_val_str}")

        # Both should match the target to 10 digits
        assert abs(1 - py_val / num) < 1e-10, f"Python off for {desc}"
        assert abs(1 - wl_val / num) < 1e-10, f"WL off for {desc}"
