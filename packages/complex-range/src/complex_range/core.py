"""
Core implementation of the ComplexRange function.

This module provides the complex_range function which generates ranges of
complex numbers in either rectangular (grid) or linear (diagonal) patterns.

Author: Daniele Gregori
"""

from fractions import Fraction
from typing import List, Tuple, Union, Iterator, Optional, Literal
from numbers import Number

from .farey import farey_sequence


# Utilities and internal functions


class ComplexRangeError(Exception):
    """Exception raised for ComplexRange-specific errors."""
    pass


def _arange_inclusive(start: Union[int, float, Fraction], 
                      end: Union[int, float, Fraction], 
                      step: Union[int, float, Fraction]) -> List[Union[int, float, Fraction]]:
    """
    Generate a range from start to end (inclusive, as in Wolfram Language) with given step.
    
    Similar to numpy.arange but includes the endpoint if it's exactly reachable.
    Handles both positive and negative steps.
    """
    if step == 0:
        raise ComplexRangeError("Step cannot be zero")
    
    # Handle mismatched direction (positive step but going down, or negative step but going up)
    if (step > 0 and start > end) or (step < 0 and start < end):
        return []
    
    result = []
    current = start
    
    # Use Fraction for exact arithmetic when possible
    if isinstance(start, (int, Fraction)) and isinstance(end, (int, Fraction)) and isinstance(step, (int, Fraction)):
        start = Fraction(start)
        end = Fraction(end)
        step = Fraction(step)
        current = start
        
        if step > 0:
            while current <= end:
                result.append(current)
                current = current + step
        else:
            # Negative step: go downward
            while current >= end:
                result.append(current)
                current = current + step
    else:
        # Floating point version with tolerance
        eps = abs(step) * 1e-14
        if step > 0:
            while current <= end + eps:
                result.append(current)
                current = current + step
        else:
            # Negative step: go downward
            while current >= end - eps:
                result.append(current)
                current = current + step
            
    return result


def _farey_range_values(start: float, end: float, order: int) -> List[Union[int, float, Fraction]]:
    """
    Generate values for a Farey range.
    
    Should be included as package in later versions
    """
    
    if order < 1:
        raise ComplexRangeError(f"Farey order must be positive, got {order}")
    
    # Get the Farey sequence for this order
    farey = farey_sequence(order)
    
    # Determine the number of unit intervals
    span = end - start
    num_units = int(span)
    
    """to check better this case"""
    # If span is not an integer, we need to handle fractional part
    if span != num_units:
        # For non-integer spans, just scale the Farey sequence
        result = sorted(set(float(start + f * span) for f in farey))
        return result
    
    # For integer spans, apply Farey to each unit interval
    all_values = set()
    
    """ok but what if the step is negative?"""
    for unit in range(num_units):
        unit_start = start + unit
        for f in farey:
            all_values.add(unit_start + float(f))
    
    # Convert to sorted list
    result = sorted(all_values)
    
    # Try to keep as fractions if possible
    if all(isinstance(x, int) or (isinstance(x, float) and x == int(x)) for x in [start, end]):
        frac_result = []
        for val in result:
            try:
                frac = Fraction(val).limit_denominator(1000000) # increased from 1000
                if abs(float(frac) - val) < 1e-14: # decreased from 1e-10
                    frac_result.append(frac)
                else:
                    frac_result.append(val)
            except (ValueError, OverflowError):
                frac_result.append(val)
        return frac_result
    
    return result



# this code is used only in dev.py
def _to_number(z: Union[complex, float, int, Fraction]) -> Union[complex, Fraction, int, float]:

    """Convert input to appropriate numeric type, preserving exactness where possible."""

    if isinstance(z, (int, Fraction)):
        return z
    if isinstance(z, float):
        # Try to convert to Fraction if it's a "nice" number
        try:
            frac = Fraction(z).limit_denominator(1000000)
            if abs(float(frac) - z) < 1e-14:
                return frac
        except (ValueError, OverflowError):
            pass
        return z
    if isinstance(z, complex):
        return z
    return z



def _make_complex(re: Union[int, float, Fraction], im: Union[int, float, Fraction]) -> complex:
    """Create a complex number from real and imaginary parts."""
    # Convert Fractions to float for complex number creation
    re_val = float(re) if isinstance(re, Fraction) else re
    im_val = float(im) if isinstance(im, Fraction) else im
    
    # Simplify representation
    if im_val == 0:
        if isinstance(re_val, float) and re_val.is_integer():
            return complex(int(re_val), 0)
        return complex(re_val, 0)
    if re_val == 0:
        if isinstance(im_val, float) and im_val.is_integer():
            return complex(0, int(im_val))
        return complex(0, im_val)
    
    return complex(re_val, im_val)



# Main range generation subfunctions


def _handle_linear_range(
    z_spec: Union[List, Tuple],
    step_or_none: Optional[Union[complex, float, int, List, Tuple]],
    step: Optional[Union[complex, float, int, List, Tuple]],
    farey_range: bool = False
) -> List[complex]:
    """Handle linear range cases."""
    # Parse the specification
    if len(z_spec) == 1:
        # {z} - linear from 0 to z
        z_start = complex(0, 0)
        z_end = complex(z_spec[0])
    elif len(z_spec) == 2:
        # {z1, z2} - linear from z1 to z2
        z_start = complex(z_spec[0])
        z_end = complex(z_spec[1])
    else:
        raise ComplexRangeError(f"Linear range specification must have 1 or 2 elements, got {len(z_spec)}")

    # Determine step (step_or_none is actually the step for linear ranges)
    if step_or_none is not None:
        if isinstance(step_or_none, (list, tuple)):
            linear_step = tuple(step_or_none)
        elif isinstance(step_or_none, complex):
            # Pass complex step directly - _linear_range will handle it
            linear_step = step_or_none
        else:
            raise ComplexRangeError(f"Invalid step specification: {step_or_none}")
    else:
        linear_step = None

    return _linear_range(z_start, z_end, linear_step, farey_range)


def _handle_rectangular_range(
    z1: Union[complex, float, int],
    z2: Optional[Union[complex, float, int]],
    step: Optional[Union[complex, float, int, List, Tuple]],
    increment_first: Literal['im', 're'],
    farey_range: bool
) -> List[complex]:
    """Handle rectangular range cases."""
    # Convert to complex
    z1_complex = complex(z1)
    
    if z2 is None:
        # Single argument: range from 0 to z1
        z2_complex = z1_complex
        z1_complex = complex(0, 0)
    else:
        z2_complex = complex(z2)
    
    # Handle step
    if step is not None:
        if isinstance(step, (list, tuple)):
            step_tuple = tuple(step)
        else:
            step_val = complex(step) if not isinstance(step, complex) else step
            step_tuple = step_val
    else:
        step_tuple = None
    
    return _rectangular_range(z1_complex, z2_complex, step_tuple, increment_first, farey_range)




# Main functions

def _rectangular_range(
    z1: complex,
    z2: complex,
    step: Union[complex, Tuple[Number, Number], None] = None,
    increment_first: Literal['im', 're'] = 'im',
    farey_range: bool = False
) -> List[complex]:
    """
    Generate a rectangular grid of complex numbers.
    
    Parameters
    ----------
    z1 : complex
        First corner of the rectangle (minimum real and imaginary parts).
    z2 : complex
        Second corner of the rectangle (maximum real and imaginary parts).
    step : complex or tuple, optional
        Step size. Can be a complex number (real part = real step, imag part = imag step)
        or a tuple (real_step, imag_step). Default is 1+1j.
    increment_first : {'im', 're'}
        Which component to increment first in the iteration. Default is 'im'.
    farey_range : bool
        If True, use Farey sequence to generate intermediate points.
        
    Returns
    -------
    List[complex]
        List of complex numbers forming a rectangular grid.
    """

    # Extract real and imaginary bounds
    re_min = z1.real
    re_max = z2.real
    im_min = z1.imag
    im_max = z2.imag
    
    # Determine step sizes
    if step is None:
        re_step = 1
        im_step = 1
    elif isinstance(step, (list, tuple)):
        re_step, im_step = step
    else:
        # Complex number: real part is real step, imag part is imag step
        re_step = step.real if hasattr(step, 'real') else step
        im_step = step.imag if hasattr(step, 'imag') else step

    # Check for reversed range - return empty in case
    if ((z1.real > z2.real and (re_step > 0)) and (z1.imag > z2.imag and (im_step > 0))): ###
        return []
    elif ((z1.real < z2.real and (re_step < 0)) and (z1.imag < z2.imag and (im_step < 0))): ###
        return []


    # Handle Farey range
    if farey_range:
        # Farey range requires integer step
        if not (isinstance(re_step, int) or (isinstance(re_step, float) and re_step.is_integer())):
            raise ComplexRangeError(
                f"No Farey range with non-integer third argument {re_step} is allowed."
            )
        if not (isinstance(im_step, int) or (isinstance(im_step, float) and im_step.is_integer())):
            raise ComplexRangeError(
                f"No Farey range with non-integer third argument {im_step} is allowed."
            )
        
        re_order = int(re_step)
        im_order = int(im_step)
        
        # Generate Farey-based ranges
        # The Farey sequence is applied to each unit interval, then combined
        re_values = _farey_range_values(re_min, re_max, re_order)
        im_values = _farey_range_values(im_min, im_max, im_order)
    else:
        # Regular range
        re_values = _arange_inclusive(re_min, re_max, re_step)
        im_values = _arange_inclusive(im_min, im_max, im_step)
    
    # Handle empty ranges
    if not re_values or not im_values:
        return []
    
    # Generate the grid
    result = []
    if increment_first == 'im':
        # Increment imaginary first (default): for each real, iterate through imaginaries
        for re in re_values:
            for im in im_values:
                result.append(_make_complex(re, im))
    else:
        # Increment real first: for each imaginary, iterate through reals
        for im in im_values:
            for re in re_values:
                result.append(_make_complex(re, im))
    
    return result




def _linear_range(
    z1: complex,
    z2: complex,
    step: Optional[Union[complex, Tuple[Number, Number]]] = None,
    farey_range: bool = False
) -> List[complex]:
    """
    Generate a linear (diagonal) range of complex numbers.

    Points lie on a line from z1 to z2, with both real and imaginary
    components incrementing together.

    Parameters
    ----------
    z1 : complex
        Starting point.
    z2 : complex
        Ending point.
    step : complex or tuple, optional
        Step sizes. Can be a complex number (real part = real step, imag part = imag step)
        or a tuple (real_step, imag_step). Default is (1, 1).
    farey_range : bool
        If True, use Farey sequence to generate intermediate points.

    Returns
    -------
    List[complex]
        List of complex numbers along the line from z1 to z2.
    """
    if step is None:
        re_step = 1
        im_step = 1
    elif isinstance(step, (list, tuple)):
        re_step, im_step = step
    elif isinstance(step, complex):
        re_step = step.real
        im_step = step.imag
    else:
        # Single number: use as both steps
        re_step = step
        im_step = step

    # Get component ranges
    re_start, re_end = z1.real, z2.real
    im_start, im_end = z1.imag, z2.imag

    # Handle Farey range
    if farey_range:
        # Farey range requires integer step
        if not (isinstance(re_step, int) or (isinstance(re_step, float) and re_step.is_integer())):
            raise ComplexRangeError(
                f"No Farey range with non-integer third argument {re_step} is allowed."
            )
        if not (isinstance(im_step, int) or (isinstance(im_step, float) and im_step.is_integer())):
            raise ComplexRangeError(
                f"No Farey range with non-integer third argument {im_step} is allowed."
            )

        re_order = int(re_step)
        im_order = int(im_step)

        # Generate Farey-based ranges
        re_values = _farey_range_values(re_start, re_end, re_order)
        im_values = _farey_range_values(im_start, im_end, im_order)
    else:
        # Generate ranges for each component
        re_values = _arange_inclusive(re_start, re_end, re_step)
        im_values = _arange_inclusive(im_start, im_end, im_step)

    # For linear range, we need to pair up values
    # Take the minimum length (they should increment together)
    n = min(len(re_values), len(im_values))

    if n == 0:
        return []

    return [_make_complex(re_values[i], im_values[i]) for i in range(n)]


# Final main function


def complex_range(
    z1: Union[complex, float, int, List, Tuple],
    z2: Optional[Union[complex, float, int]] = None,
    step: Optional[Union[complex, float, int, List, Tuple]] = None,
    *,
    increment_first: Literal['im', 're'] = 'im',
    farey_range: bool = False
) -> List[complex]:
    """
    Generate a range of complex numbers.
    
    This function can generate either rectangular (grid) or linear (diagonal)
    ranges of complex numbers in the complex plane.
    
    """
    # Check if this is a linear range (first argument is a list/tuple)
    if isinstance(z1, (list, tuple)):
        return _handle_linear_range(z1, z2, step, farey_range)

    # Rectangular range
    return _handle_rectangular_range(z1, z2, step, increment_first, farey_range)



# Future developments

# Convenience function for iterator version
def complex_range_iter(
    z1: Union[complex, float, int, List, Tuple],
    z2: Optional[Union[complex, float, int]] = None,
    step: Optional[Union[complex, float, int, List, Tuple]] = None,
    *,
    increment_first: Literal['im', 're'] = 'im',
    farey_range: bool = False
) -> Iterator[complex]:
    """
    Iterator version of complex_range.
    
    Same as complex_range but returns an iterator instead of a list.
    Useful for memory efficiency with large ranges.
    
    See complex_range for full documentation.
    
    Author: Daniele Gregori
    """
    return iter(complex_range(z1, z2, step, increment_first=increment_first, farey_range=farey_range))
