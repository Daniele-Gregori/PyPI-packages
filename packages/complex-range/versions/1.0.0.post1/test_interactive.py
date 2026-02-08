#!/usr/bin/env python3
"""
Interactive testing script for complex-range.

Run with: python test_interactive.py
Or use in IPython: %run test_interactive.py

Author: Daniele Gregori
"""

import sys
from pathlib import Path

# Add src to path if running from package root
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import everything for testing
from complex_range.core import (
    complex_range,
    ComplexRangeError,
    _rectangular_range,
    _linear_range,
    _arange_inclusive,
    _make_complex,
    _farey_range_values,
)
from complex_range.farey import farey_sequence, scaled_farey_sequence
from fractions import Fraction

# ============================================================
# Test area - modify and run to test specific functionality
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("complex-range Interactive Testing")
    print("Author: Daniele Gregori")
    print("=" * 60)
    
    # --- Test _arange_inclusive ---
    print("\n--- _arange_inclusive ---")
    print(f"_arange_inclusive(0, 5, 1) = {_arange_inclusive(0, 5, 1)}")
    print(f"_arange_inclusive(0, 1, 0.25) = {_arange_inclusive(0, 1, 0.25)}")
    print(f"_arange_inclusive(0, 1, Fraction(1,3)) = {_arange_inclusive(0, 1, Fraction(1,3))}")
    print(f"_arange_inclusive(5, 0, -1) = {_arange_inclusive(5, 0, -1)}")
    
    # --- Test _make_complex ---
    print("\n--- _make_complex ---")
    print(f"_make_complex(1, 2) = {_make_complex(1, 2)}")
    print(f"_make_complex(0, 3) = {_make_complex(0, 3)}")
    print(f"_make_complex(Fraction(1,2), Fraction(1,3)) = {_make_complex(Fraction(1,2), Fraction(1,3))}")
    
    # --- Test farey_sequence ---
    print("\n--- farey_sequence ---")
    for n in range(1, 6):
        print(f"farey_sequence({n}) = {farey_sequence(n)}")
    
    # --- Test _farey_range_values ---
    print("\n--- _farey_range_values ---")
    print(f"_farey_range_values(0, 1, 2) = {_farey_range_values(0, 1, 2)}")
    print(f"_farey_range_values(0, 2, 2) = {_farey_range_values(0, 2, 2)}")
    print(f"_farey_range_values(0, 4, 2) = {_farey_range_values(0, 4, 2)}")
    
    # --- Test _rectangular_range ---
    print("\n--- _rectangular_range ---")
    print(f"_rectangular_range(0j, 2+2j) = {_rectangular_range(0j, 2+2j)}")
    print(f"_rectangular_range(0j, 2+2j, 1+1j, 'im') = {_rectangular_range(0j, 2+2j, 1+1j, 'im')}")
    print(f"_rectangular_range(0j, 2+2j, 1+1j, 're') = {_rectangular_range(0j, 2+2j, 1+1j, 're')}")
    
    # --- Test _linear_range ---
    print("\n--- _linear_range ---")
    print(f"_linear_range(0j, 3+3j) = {_linear_range(0j, 3+3j)}")
    print(f"_linear_range(0j, 4+4j, (2, 2)) = {_linear_range(0j, 4+4j, (2, 2))}")
    print(f"_linear_range(0j, 4+4j, 2+2j) = {_linear_range(0j, 4+4j, 2+2j)}")
    print(f"_linear_range(2+2j, 0j, -1-1j) = {_linear_range(2+2j, 0j, -1-1j)}")
    
    # --- Test complex_range (public API) ---
    print("\n--- complex_range (public API) ---")
    print(f"complex_range(2+2j) = {complex_range(2+2j)}")
    print(f"complex_range([3+3j]) = {complex_range([3+3j])}")
    print(f"complex_range([2+2j, 0], -1-1j) = {complex_range([2+2j, 0], -1-1j)}")
    print(f"complex_range(0, 2+2j, 2+2j, farey_range=True) = {complex_range(0, 2+2j, 2+2j, farey_range=True)}")
    
    print("\n" + "=" * 60)
    print("Testing complete! Modify this script to test other cases.")
    print("=" * 60)

# ============================================================
# Quick test snippets - uncomment and modify as needed
# ============================================================

# # Test a specific case
# result = _rectangular_range(0j, 4+4j, 2+2j, farey_range=True)
# print(f"Length: {len(result)}, Values: {result[:10]}...")

# # Compare Farey vs regular
# farey = complex_range(0, 4+4j, 2+2j, farey_range=True)
# regular = complex_range(0, 4+4j, 2+2j, farey_range=False)
# print(f"Farey: {len(farey)} points, Regular: {len(regular)} points")

# # Test edge cases
# print(complex_range(1+1j, -1-1j))  # Should be []
# print(complex_range([5+0j]))  # Linear with pure real endpoint
