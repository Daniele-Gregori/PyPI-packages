"""
Comprehensive test suite for complex-range.

These tests are a Python translation of the Wolfram Language verification tests,
ensuring feature parity with the original implementation.

Author: Daniele Gregori
"""

import pytest
from fractions import Fraction
from complex_range import complex_range, ComplexRangeError, farey_sequence


class TestBasicRectangularRange:
    """Tests for basic rectangular (two-argument) range functionality."""
    
    def test_two_args_positive_range(self):
        """Test rectangular range with positive complex bounds."""
        result = complex_range(1+1j, 3+3j)
        expected = [1+1j, 1+2j, 1+3j, 2+1j, 2+2j, 2+3j, 3+1j, 3+2j, 3+3j]
        assert result == expected
    
    def test_two_args_from_origin(self):
        """Test rectangular range from origin."""
        result = complex_range(0, 2+2j)
        expected = [0+0j, 0+1j, 0+2j, 1+0j, 1+1j, 1+2j, 2+0j, 2+1j, 2+2j]
        assert result == expected
    
    def test_two_args_negative_to_positive(self):
        """Test rectangular range spanning negative to positive."""
        result = complex_range(-1-1j, 1+1j)
        expected = [-1-1j, -1+0j, -1+1j, 0-1j, 0+0j, 0+1j, 1-1j, 1+0j, 1+1j]
        assert result == expected
    
    def test_single_arg_from_origin(self):
        """Test single argument form (range from 0 to z)."""
        result = complex_range(2+3j)
        # Should be 3 real values (0,1,2) × 4 imag values (0,1,2,3) = 12 elements
        assert len(result) == 12
        assert result[0] == 0+0j
        assert result[-1] == 2+3j


class TestRectangularRangeWithStep:
    """Tests for rectangular range with step size."""
    
    def test_complex_step(self):
        """Test rectangular range with complex step size."""
        result = complex_range(-1-1j, 1+1j, 0.5+1j)
        # Step: real=0.5, imag=1
        # Real: -1, -0.5, 0, 0.5, 1 (5 values)
        # Imag: -1, 0, 1 (3 values)
        assert len(result) == 15
        assert result[0] == -1-1j
        assert result[-1] == 1+1j
        ### add also full result verification

    def test_complex_step_explicit(self):
        """Test rectangular range with complex step size - explicit."""
        result = complex_range(-1-1j, 0+1j, 0.5+1j)
        expected = [-1-1j, -1+0j, -1+1j, -0.5-1j, -0.5+0j, -0.5+1j, 0-1j, 0+0j, 0+1j]
        assert result == expected


    def test_integer_complex_step(self):
        """Test rectangular range with integer complex step."""
        result = complex_range(0, 2+2j, 1+1j)
        expected = [0+0j, 0+1j, 0+2j, 1+0j, 1+1j, 1+2j, 2+0j, 2+1j, 2+2j]
        assert result == expected
    
    def test_list_step(self):
        """Test rectangular range with list step [real_step, imag_step]."""
        result = complex_range(-1-1j, 1+1j, [0.5, 1])
        # Same as complex step test
        assert len(result) == 15

    def test_list_step_alternative(self):
        """Test rectangular range with alternative list step [real_step, imag_step]."""
        result = complex_range(-1-1j, 1+1j, [1, 0.5])
        expected = [-1-1j,-1-0.5j, -1+0j,-1+0.5j, -1+1j, 0-1j, 0-0.5j, 0+0j, 0+0.5j, 0+1j, 1-1j, 1-0.5j, 1+0j, 1+0.5j, 1+1j]
        assert result == expected
     
    def test_different_real_imag_steps(self):
        """Test rectangular range with different real and imaginary steps."""
        result = complex_range(0, 4+6j, [2, 3])
        expected = [0+0j, 0+3j, 0+6j, 2+0j, 2+3j, 2+6j, 4+0j, 4+3j, 4+6j]
        assert result == expected

class TestRectangularRangeFractionalStep: 

    def test_fractional_steps(self):
        """Test rectangular range with fractional steps."""
        result = complex_range(0, 1+1j, [Fraction(1,3), Fraction(1,2)])
        expected = [0+0j, 0+Fraction(1,2)*1j, 0+1j, Fraction(1,3)+0j, Fraction(1,3)+Fraction(1,2)*1j, Fraction(1,3)+1j, Fraction(2,3)+0j, Fraction(2,3)+Fraction(1,2)*1j, Fraction(2,3)+1j, 1+0j, 1+Fraction(1,2)*1j, 1+1j]
        assert result == pytest.approx(expected)
    
    def test_fractional_steps_alternative(self):
        """Test rectangular range with alternative fractional steps."""
        result = complex_range(0, 1+1j, [Fraction(1,2), Fraction(1,5)])
        expected = [0+0j, 0+Fraction(1,5)*1j, 0+Fraction(2,5)*1j, 0+Fraction(3,5)*1j, 0+Fraction(4,5)*1j, 0+1j, Fraction(1,2)+0j, Fraction(1,2)+Fraction(1,5)*1j, Fraction(1,2)+Fraction(2,5)*1j, Fraction(1,2)+Fraction(3,5)*1j, Fraction(1,2)+Fraction(4,5)*1j, Fraction(1,2)+1j, 1+0j, 1+Fraction(1,5)*1j, 1+Fraction(2,5)*1j, 1+Fraction(3,5)*1j, 1+Fraction(4,5)*1j, 1+1j]
        assert result == pytest.approx(expected)

    def test_fractional_steps_with_rest(self):
        """Test rectangular range with fractional steps which does not divide the range."""
        result = complex_range(-1-1j, 1+1j, [Fraction(4,7), Fraction(8,11)])
        expected = [-1-1j, -1-Fraction(3,11)*1j, -1+Fraction(5,11)*1j, Fraction(-3,7)-1j, Fraction(-3,7)-Fraction(3,11)*1j, Fraction(-3,7)+Fraction(5,11)*1j, Fraction(1,7)-1j, Fraction(1,7)-Fraction(3,11)*1j, Fraction(1,7)+Fraction(5,11)*1j, Fraction(5,7)-1j, Fraction(5,7)-Fraction(3,11)*1j, Fraction(5,7)+Fraction(5,11)*1j]
        assert result == pytest.approx(expected)


class TestIncrementFirstOption:
    """Tests for the increment_first option."""
    
    def test_increment_first_im(self):
        """Test with imaginary incrementing first (default)."""
        result = complex_range(-1-1j, 1+1j, 1+1j, increment_first='im')
        expected = [-1-1j, -1+0j, -1+1j, 0-1j, 0+0j, 0+1j, 1-1j, 1+0j, 1+1j]
        assert result == expected
    
    def test_increment_first_re(self):
        """Test with real incrementing first."""
        result = complex_range(-1-1j, 1+1j, 1+1j, increment_first='re')
        expected = [-1-1j, 0-1j, 1-1j, -1+0j, 0+0j, 1+0j, -1+1j, 0+1j, 1+1j]
        assert result == expected
    
    def test_increment_first_different_results(self):
        """Verify that different increment_first values produce different results."""
        result_im = complex_range(-1-1j, 1+1j, 1+1j, increment_first='im')
        result_re = complex_range(-1-1j, 1+1j, 1+1j, increment_first='re')
        assert result_im != result_re
        # But they should have the same elements (just different order)
        assert set(result_im) == set(result_re)


class TestFareyRangeOption:
    """Tests for the farey_range option."""
    
    def test_farey_range_basic(self):
        """Test basic Farey range."""
        result = complex_range(0, 1+1j, [3,3],farey_range=True)
        expected = [
                    0+0j, Fraction(1,3)*1j, Fraction(1,2)*1j, Fraction(2,3)*1j, 1j,
                    Fraction(1,3)+0j, Fraction(1,3)+Fraction(1,3)*1j, Fraction(1,3)+Fraction(1,2)*1j, Fraction(1,3)+Fraction(2,3)*1j, Fraction(1,3)+1j,
                    Fraction(1,2)+0j, Fraction(1,2)+Fraction(1,3)*1j, Fraction(1,2)+Fraction(1,2)*1j, Fraction(1,2)+Fraction(2,3)*1j, Fraction(1,2)+1j,
                    Fraction(2,3)+0j, Fraction(2,3)+Fraction(1,3)*1j, Fraction(2,3)+Fraction(1,2)*1j, Fraction(2,3)+Fraction(2,3)*1j, Fraction(2,3)+1j,
                    1+0j, 1+Fraction(1,3)*1j, 1+Fraction(1,2)*1j, 1+Fraction(2,3)*1j, 1+1j,
                    ]

        assert result == pytest.approx(expected)

    def test_farey_range_larger(self):
        """Test Farey range on larger range."""
        result = complex_range(0, 2+3j, [4,3], farey_range=True)
        expected = [0+0j, Fraction(1,3)*1j, Fraction(1,2)*1j, Fraction(2,3)*1j, 1j, Fraction(4,3)*1j, Fraction(3,2)*1j, Fraction(5,3)*1j, 2j, Fraction(7,3)*1j, Fraction(5,2)*1j, Fraction(8,3)*1j, 3j, Fraction(1,4)+0j, Fraction(1,4)+Fraction(1,3)*1j, Fraction(1,4)+Fraction(1,2)*1j, Fraction(1,4)+Fraction(2,3)*1j, Fraction(1,4)+1j, Fraction(1,4)+Fraction(4,3)*1j, Fraction(1,4)+Fraction(3,2)*1j, Fraction(1,4)+Fraction(5,3)*1j, Fraction(1,4)+2j, Fraction(1,4)+Fraction(7,3)*1j, Fraction(1,4)+Fraction(5,2)*1j, Fraction(1,4)+Fraction(8,3)*1j, Fraction(1,4)+3j, Fraction(1,3)+0j, Fraction(1,3)+Fraction(1,3)*1j, Fraction(1,3)+Fraction(1,2)*1j, Fraction(1,3)+Fraction(2,3)*1j, Fraction(1,3)+1j, Fraction(1,3)+Fraction(4,3)*1j, Fraction(1,3)+Fraction(3,2)*1j, Fraction(1,3)+Fraction(5,3)*1j, Fraction(1,3)+2j, Fraction(1,3)+Fraction(7,3)*1j, Fraction(1,3)+Fraction(5,2)*1j, Fraction(1,3)+Fraction(8,3)*1j, Fraction(1,3)+3j, Fraction(1,2)+0j, Fraction(1,2)+Fraction(1,3)*1j, Fraction(1,2)+Fraction(1,2)*1j, Fraction(1,2)+Fraction(2,3)*1j, Fraction(1,2)+1j, Fraction(1,2)+Fraction(4,3)*1j, Fraction(1,2)+Fraction(3,2)*1j, Fraction(1,2)+Fraction(5,3)*1j, Fraction(1,2)+2j, Fraction(1,2)+Fraction(7,3)*1j, Fraction(1,2)+Fraction(5,2)*1j, Fraction(1,2)+Fraction(8,3)*1j, Fraction(1,2)+3j, Fraction(2,3)+0j, Fraction(2,3)+Fraction(1,3)*1j, Fraction(2,3)+Fraction(1,2)*1j, Fraction(2,3)+Fraction(2,3)*1j, Fraction(2,3)+1j, Fraction(2,3)+Fraction(4,3)*1j, Fraction(2,3)+Fraction(3,2)*1j, Fraction(2,3)+Fraction(5,3)*1j, Fraction(2,3)+2j, Fraction(2,3)+Fraction(7,3)*1j, Fraction(2,3)+Fraction(5,2)*1j, Fraction(2,3)+Fraction(8,3)*1j, Fraction(2,3)+3j, Fraction(3,4)+0j, Fraction(3,4)+Fraction(1,3)*1j, Fraction(3,4)+Fraction(1,2)*1j, Fraction(3,4)+Fraction(2,3)*1j, Fraction(3,4)+1j, Fraction(3,4)+Fraction(4,3)*1j, Fraction(3,4)+Fraction(3,2)*1j, Fraction(3,4)+Fraction(5,3)*1j, Fraction(3,4)+2j, Fraction(3,4)+Fraction(7,3)*1j, Fraction(3,4)+Fraction(5,2)*1j, Fraction(3,4)+Fraction(8,3)*1j, Fraction(3,4)+3j, 1+0j, 1+Fraction(1,3)*1j, 1+Fraction(1,2)*1j, 1+Fraction(2,3)*1j, 1+1j, 1+Fraction(4,3)*1j, 1+Fraction(3,2)*1j, 1+Fraction(5,3)*1j, 1+2j, 1+Fraction(7,3)*1j, 1+Fraction(5,2)*1j, 1+Fraction(8,3)*1j, 1+3j, Fraction(5,4)+0j, Fraction(5,4)+Fraction(1,3)*1j, Fraction(5,4)+Fraction(1,2)*1j, Fraction(5,4)+Fraction(2,3)*1j, Fraction(5,4)+1j, Fraction(5,4)+Fraction(4,3)*1j, Fraction(5,4)+Fraction(3,2)*1j, Fraction(5,4)+Fraction(5,3)*1j, Fraction(5,4)+2j, Fraction(5,4)+Fraction(7,3)*1j, Fraction(5,4)+Fraction(5,2)*1j, Fraction(5,4)+Fraction(8,3)*1j, Fraction(5,4)+3j, Fraction(4,3)+0j, Fraction(4,3)+Fraction(1,3)*1j, Fraction(4,3)+Fraction(1,2)*1j, Fraction(4,3)+Fraction(2,3)*1j, Fraction(4,3)+1j, Fraction(4,3)+Fraction(4,3)*1j, Fraction(4,3)+Fraction(3,2)*1j, Fraction(4,3)+Fraction(5,3)*1j, Fraction(4,3)+2j, Fraction(4,3)+Fraction(7,3)*1j, Fraction(4,3)+Fraction(5,2)*1j, Fraction(4,3)+Fraction(8,3)*1j, Fraction(4,3)+3j, Fraction(3,2)+0j, Fraction(3,2)+Fraction(1,3)*1j, Fraction(3,2)+Fraction(1,2)*1j, Fraction(3,2)+Fraction(2,3)*1j, Fraction(3,2)+1j, Fraction(3,2)+Fraction(4,3)*1j, Fraction(3,2)+Fraction(3,2)*1j, Fraction(3,2)+Fraction(5,3)*1j, Fraction(3,2)+2j, Fraction(3,2)+Fraction(7,3)*1j, Fraction(3,2)+Fraction(5,2)*1j, Fraction(3,2)+Fraction(8,3)*1j, Fraction(3,2)+3j, Fraction(5,3)+0j, Fraction(5,3)+Fraction(1,3)*1j, Fraction(5,3)+Fraction(1,2)*1j, Fraction(5,3)+Fraction(2,3)*1j, Fraction(5,3)+1j, Fraction(5,3)+Fraction(4,3)*1j, Fraction(5,3)+Fraction(3,2)*1j, Fraction(5,3)+Fraction(5,3)*1j, Fraction(5,3)+2j, Fraction(5,3)+Fraction(7,3)*1j, Fraction(5,3)+Fraction(5,2)*1j, Fraction(5,3)+Fraction(8,3)*1j, Fraction(5,3)+3j, Fraction(7,4)+0j, Fraction(7,4)+Fraction(1,3)*1j, Fraction(7,4)+Fraction(1,2)*1j, Fraction(7,4)+Fraction(2,3)*1j, Fraction(7,4)+1j, Fraction(7,4)+Fraction(4,3)*1j, Fraction(7,4)+Fraction(3,2)*1j, Fraction(7,4)+Fraction(5,3)*1j, Fraction(7,4)+2j, Fraction(7,4)+Fraction(7,3)*1j, Fraction(7,4)+Fraction(5,2)*1j, Fraction(7,4)+Fraction(8,3)*1j, Fraction(7,4)+3j, 2+0j, 2+Fraction(1,3)*1j, 2+Fraction(1,2)*1j, 2+Fraction(2,3)*1j, 2+1j, 2+Fraction(4,3)*1j, 2+Fraction(3,2)*1j, 2+Fraction(5,3)*1j, 2+2j, 2+Fraction(7,3)*1j, 2+Fraction(5,2)*1j, 2+Fraction(8,3)*1j, 2+3j]
        assert result == pytest.approx(expected)


    def test_farey_range_negative(self):
        """Test Farey range with negative bounds."""
        result = complex_range(-2-2j, -1-1j, [3,3], farey_range=True)
        expected = [-2-2j, -2-Fraction(5,3)*1j, -2-Fraction(3,2)*1j, -2-Fraction(4,3)*1j, -2-1j, Fraction(-5,3)-2j, Fraction(-5,3)-Fraction(5,3)*1j, Fraction(-5,3)-Fraction(3,2)*1j, Fraction(-5,3)-Fraction(4,3)*1j, Fraction(-5,3)-1j, Fraction(-3,2)-2j, Fraction(-3,2)-Fraction(5,3)*1j, Fraction(-3,2)-Fraction(3,2)*1j, Fraction(-3,2)-Fraction(4,3)*1j, Fraction(-3,2)-1j, Fraction(-4,3)-2j, Fraction(-4,3)-Fraction(5,3)*1j, Fraction(-4,3)-Fraction(3,2)*1j, Fraction(-4,3)-Fraction(4,3)*1j, Fraction(-4,3)-1j, -1-2j, -1-Fraction(5,3)*1j, -1-Fraction(3,2)*1j, -1-Fraction(4,3)*1j, -1-1j]
        assert result == pytest.approx(expected)


    def test_farey_range_creates_finer_grid(self):
        """Test that Farey range with step creates finer grid."""
        farey_result = complex_range(0, 4+4j, [5,7], farey_range=True)
        regular_result = complex_range(0, 4+4j, [Fraction(1,5), Fraction(1,7)], farey_range=False)
        assert len(farey_result) == 2993  
        assert len(regular_result) == 609  
        assert len(farey_result) > len(regular_result)
    
    def test_farey_range_fails_with_non_integer_step(self):
        """Test that Farey range raises error for non-integer step."""
        with pytest.raises(ComplexRangeError):
            complex_range(0, 1+1j, 0.5+0.5j, farey_range=True)


class TestNegativeStep:
    """Tests for negative step scenarios."""
    
    def test_negative_integer_step(self):
        """Test rectangular range with negative integer real and imaginary steps."""
        result = complex_range(2+2j, 0, -1-1j)
        expected = [2+2j,2+1j,2,1+2j,1+ 1j ,1,2j ,1j,0,]
        assert result == expected

    def test_negative_integer_step_bounds(self):
        """Test linear range with negative integer step and bounds."""
        result = complex_range(-1, -2 -2j, -1-1j)
        expected = [-1,-1-1j,-1-2j,-2,-2-1j,-2-2j]
        assert result == expected

    def test_linear_negative_step(self):
        """Test that result is a list."""
        #result = complex_range([1+0.5j, -1-0.5j],[-0.5,-0.5])
        result = complex_range([1+0.5j, -1-0.5j],-0.5-0.5j)
        expected = [1+0.5j, 0.5+0j, -0.5j]
        assert result == expected

    def test_linear_negative_step_bounds(self):
        """Test linear range with negative integer step and bounds."""
        result = complex_range([-1, -2 -2j], -.5-.5j)
        expected = [-1,-1.5-0.5j,-2-1j]
        assert result == expected

    ### add more tests with other step sizes and bounds to verify correct handling of negative steps in various scenarios

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_point(self):
        """Test range where start equals end."""
        result = complex_range(1+1j, 1+1j)
        assert result == [1+1j]
    
    def test_pure_real(self):
        """Test range with only real component."""
        result = complex_range(0, 3+0j)
        expected = [0+0j, 1+0j, 2+0j, 3+0j]
        assert result == expected
    
    def test_pure_imaginary(self):
        """Test range with only imaginary component."""
        result = complex_range(0, 0+3j)
        expected = [0+0j, 0+1j, 0+2j, 0+3j]
        assert result == expected
    
    def test_reversed_direction_empty(self):
        """Test that reversed direction returns empty list."""
        result = complex_range(1+1j, -1-1j)
        assert result == []


class TestNumericTypes:
    """Tests for different numeric types."""
    
    def test_rational_step(self):
        """Test with fractional step sizes."""
        result = complex_range(0, 1+1j, [1/3, 0.5])
        assert len(result) == 12  # 4 real × 3 imag
        assert result[0] == 0+0j
    
    def test_float_precision(self):
        """Test with floating point values."""
        result = complex_range(0.0, 1.0+1.0j, 0.5+0.5j)
        assert len(result) == 9  # 3 × 3
    
    def test_integer_inputs(self):
        """Test with integer inputs (auto-converted to complex)."""
        result = complex_range(0, 2)
        expected = [0+0j, 1+0j, 2+0j]
        assert result == expected


class TestPropertiesAndRelations:
    """Tests for mathematical properties and relations."""
    
    def test_length_is_product(self):
        """Test that result length equals product of component range lengths."""
        result = complex_range(0, 2+3j)
        # Real: 0, 1, 2 (3 values)
        # Imag: 0, 1, 2, 3 (4 values)
        assert len(result) == 3 * 4
    
    def test_all_numeric(self):
        """Test that all elements are complex numbers."""
        result = complex_range(0, 2+2j)
        assert all(isinstance(z, complex) for z in result)
    
    def test_result_is_list(self):
        """Test that result is a list."""
        result = complex_range(0, 2+2j)
        assert isinstance(result, list)
    
    def test_first_is_start(self):
        """Test that first element is the starting corner."""
        result = complex_range(-1-1j, 1+1j)
        assert result[0] == -1-1j
    
    def test_last_is_end(self):
        """Test that last element is the ending corner."""
        result = complex_range(-1-1j, 1+1j)
        assert result[-1] == 1+1j

    ### add more tests to show staring corner is always preserved but ending not if the step does not divide the difference between bounds


class TestCombinedOptions:
    """Tests for combined options."""
    
    def test_both_options_work(self):
        """Test that increment_first and farey_range work together."""
        result = complex_range(0, 4+4j, 2+2j, increment_first='re', farey_range=True)
        assert isinstance(result, list)
        assert len(result) > 0


class TestSpecialValues:
    """Tests for special value cases."""
    
    def test_unit_square(self):
        """Test unit square from origin."""
        result = complex_range(0, 1+1j, 1+1j)
        expected = [0+0j, 0+1j, 1+0j, 1+1j]
        assert result == expected
    
    def test_negative_real_positive_imag(self):
        """Test range with negative real and positive imaginary."""
        result = complex_range(-2+0j, -1+2j)
        expected = [-2+0j, -2+1j, -2+2j, -1+0j, -1+1j, -1+2j]
        assert result == expected


class TestLinearRangeSingleBound:
    """Tests for linear range with single bound list."""
    
    def test_linear_single_element_basic(self):
        """Test linear range from 0 to z."""
        result = complex_range([1+1j])
        expected = [0+0j, 1+1j]
        assert result == expected
    
    def test_linear_single_element_larger_range(self):
        """Test linear range with larger endpoint."""
        result = complex_range([3+3j])
        expected = [0+0j, 1+1j, 2+2j, 3+3j]
        assert result == expected
    
    def test_linear_single_element_pure_real(self):
        """Test linear range with pure real endpoint returns only origin."""
        result = complex_range([5+0j])
        # Linear range needs both components to vary
        assert result == [0+0j]
    
    def test_linear_single_element_pure_imaginary(self):
        """Test linear range with pure imaginary endpoint returns only origin."""
        result = complex_range([0+5j])
        assert result == [0+0j]


class TestLinearRangeTwoBounds:
    """Tests for linear range with two element list."""
    
    def test_linear_two_element_from_origin(self):
        """Test linear range from origin to endpoint."""
        result = complex_range([0, 1+1j])
        expected = [0+0j, 1+1j]
        assert result == expected
    
    def test_linear_two_element_negative_to_positive(self):
        """Test linear range from negative to positive."""
        result = complex_range([-1-1j, 2+2j])
        expected = [-1-1j, 0+0j, 1+1j, 2+2j]
        assert result == expected
    
    def test_linear_two_element_non_origin_start(self):
        """Test linear range not starting at origin."""
        result = complex_range([1+0j, 4+3j])
        expected = [1+0j, 2+1j, 3+2j, 4+3j]
        assert result == expected
    
    def test_linear_two_element_reversed_direction_empty(self):
        """Test that reversed direction returns empty list."""
        result = complex_range([2+2j, 0])
        assert result == []


class TestLinearRangeWithStep:
    """Tests for linear range with step."""
    
    def test_linear_with_step_unit_step(self):
        """Test linear range with unit step."""
        result = complex_range([0, 2+2j], [1, 1])
        expected = [0+0j, 1+1j, 2+2j]
        assert result == expected
    
    def test_linear_with_step_larger_step(self):
        """Test linear range with larger step."""
        result = complex_range([0, 4+4j], [2, 2])
        expected = [0+0j, 2+2j, 4+4j]
        assert result == expected
    
    def test_linear_with_step_float_step(self):
        """Test linear range with float step."""
        result = complex_range([0, 1+1j], [0.5, 0.5])
        expected = [0+0j, 0.5+0.5j, 1+1j]
        assert result == expected

    def test_linear_with_step_fractional_step(self):
        """Test linear range with fractional step."""
        result = complex_range([0, 1+1j], [Fraction(1,3), Fraction(2,7)])
        expected = [0+0j, Fraction(1,3)+Fraction(2,7)*1j, Fraction(2,3)+Fraction(4,7)*1j,1+Fraction(6,7)*1j]
        assert result == pytest.approx(expected)


    def test_linear_with_complex_step(self):
        """Test linear range with complex step."""
        result = complex_range([0, 4+4j], 2+2j)
        expected = [0+0j, 2+2j, 4+4j]
        assert result == expected
    
    def test_linear_with_negative_complex_step(self):
        """Test descending linear range with negative complex step."""
        result = complex_range([2+2j, 0], -1-1j)
        expected = [2+2j, 1+1j, 0+0j]
        assert result == expected


class TestLinearRangeProperties:
    """Tests for linear range properties."""
    
    def test_linear_property_constant_differences(self):
        """Test that linear range has constant differences (collinear points)."""
        result = complex_range([0, 5+5j])
        differences = [result[i+1] - result[i] for i in range(len(result)-1)]
        # All differences should be equal
        assert len(set(differences)) == 1
    
    def test_linear_property_first_is_start(self):
        """Test that first element is start point."""
        result = complex_range([-1-1j, 2+2j])
        assert result[0] == -1-1j
    
    def test_linear_property_last_is_end(self):
        """Test that last element is end point."""
        result = complex_range([-1-1j, 2+2j])
        assert result[-1] == 2+2j
    
    def test_linear_property_result_is_list(self):
        """Test that result is a list."""
        result = complex_range([0, 1+1j])
        assert isinstance(result, list)



class TestFareySequence:
    """Tests for the Farey sequence implementation."""
    
    def test_farey_sequence_order_1(self):
        """Test Farey sequence of order 1."""
        result = farey_sequence(1)
        expected = [Fraction(0, 1), Fraction(1, 1)]
        assert result == expected
    
    def test_farey_sequence_order_2(self):
        """Test Farey sequence of order 2."""
        result = farey_sequence(2)
        expected = [Fraction(0, 1), Fraction(1, 2), Fraction(1, 1)]
        assert result == expected
    
    def test_farey_sequence_order_3(self):
        """Test Farey sequence of order 3."""
        result = farey_sequence(3)
        expected = [Fraction(0, 1), Fraction(1, 3), Fraction(1, 2), 
                   Fraction(2, 3), Fraction(1, 1)]
        assert result == expected
    
    def test_farey_sequence_order_5(self):
        """Test Farey sequence of order 5."""
        result = farey_sequence(5)
        # Should have 11 elements for F_5
        assert len(result) == 11
        assert result[0] == Fraction(0, 1)
        assert result[-1] == Fraction(1, 1)
    
    def test_farey_sequence_invalid_order(self):
        """Test that invalid order raises error."""
        with pytest.raises(ValueError):
            farey_sequence(0)
        with pytest.raises(ValueError):
            farey_sequence(-1)


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_linear_spec_length(self):
        """Test that invalid linear specification raises error."""
        with pytest.raises(ComplexRangeError):
            complex_range([1, 2, 3])  # Too many elements
    
    def test_farey_with_float_step(self):
        """Test that Farey range with float step raises error."""
        with pytest.raises(ComplexRangeError):
            complex_range(0, 1+1j, 0.5+0.5j, farey_range=True)


class TestTupleInput:
    """Tests for tuple input (alternative to list for linear range)."""

    def test_tuple_linear_range(self):
        """Test that tuple works same as list for linear range."""
        list_result = complex_range([0, 2+2j])
        tuple_result = complex_range((0, 2+2j))
        assert list_result == tuple_result


class TestLinearFareyRange:
    """Tests for linear range with farey_range option.

    Expected values verified against Wolfram ResourceFunction["ComplexRange"].
    """

    def test_linear_farey_order_3(self):
        result = complex_range([0+0j, 1+1j], [3, 3], farey_range=True)
        expected = [0+0j, Fraction(1,3)+Fraction(1,3)*1j, Fraction(1,2)+Fraction(1,2)*1j, Fraction(2,3)+Fraction(2,3)*1j, 1+1j]
        assert result == pytest.approx(expected)

    def test_linear_farey_larger(self):

        result = complex_range([0, 3+3j], [5, 7], farey_range=True)
        assert len(result) == 31
        assert result[0] == 0+0j
        assert result[-1] == pytest.approx(3+Fraction(5,3)*1j)
        


    def test_linear_farey_non_zero_start(self):
        result = complex_range([-1+0j, 2+1j], [3, 3], farey_range=True)
        expected = [-1+0j, -Fraction(2,3)+Fraction(1,3)*1j, -Fraction(1,2)+Fraction(1,2)*1j, Fraction(-1,3)+Fraction(2,3)*1j, 0+1j]
        assert result == pytest.approx(expected)
                    

    def test_linear_farey_negative_to_positive(self):
        result = complex_range([-1-1j, 1+1j], [3, 3], farey_range=True)
        expected = [-1-1j, Fraction(-2,3)-Fraction(2,3)*1j, Fraction(-1,2)-Fraction(1,2)*1j, Fraction(-1,3)-Fraction(1,3)*1j, 0+0j, Fraction(1,3)+Fraction(1,3)*1j, Fraction(1,2)+Fraction(1,2)*1j, Fraction(2,3)+Fraction(2,3)*1j, 1+1j]
        assert result == pytest.approx(expected)
        


    def test_linear_farey_fails_with_non_integer_step(self):
        """Test that linear Farey range raises error for non-integer step.

        Wolfram: ResourceFunction["ComplexRange"][{0, 1+I}, {0.5, 0.5}, "FareyRange" -> True]
        Result: "No Farey range with non-integer third argument 0.5 is allowed."
        """
        with pytest.raises(ComplexRangeError):
            complex_range([0, 1+1j], [0.5, 0.5], farey_range=True)

    def test_linear_farey_creates_more_points_than_regular(self):
        """Test that linear Farey range creates more points than regular step."""
        farey_result = complex_range([0, 4+4j], [3, 3], farey_range=True)
        regular_result = complex_range([0, 4+4j], [Fraction(1,3), Fraction(1,3)], farey_range=False)
        assert len(farey_result) > len(regular_result)
        assert len(farey_result) == 17
        assert len(regular_result) == 13

    def test_linear_farey_with_complex_step(self):
        """Test linear Farey range with complex step notation."""
        result = complex_range([0, 2+2j], 2+2j, farey_range=True)
        expected = [0+0j, 0.5+0.5j, 1+1j, 1.5+1.5j, 2+2j]
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
