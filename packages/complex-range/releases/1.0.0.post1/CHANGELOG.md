# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-02-03

### Added

- Initial release
- `complex_range()` function for generating complex number ranges
- Rectangular range support (2D grids in complex plane)
- Linear range support (diagonal sequences)
- Custom step sizes (complex or tuple format)
- Support for negative steps in descending ranges
- `increment_first` option to control iteration order ('im' or 're')
- `farey_range` option for Farey sequence subdivision
- `farey_sequence()` function for generating Farey sequences
- `ComplexRangeError` exception for error handling
- Comprehensive test suite
- Full type hints
- Complete documentation
- Development utilities module (`complex_range.dev`)

### Features

- **Rectangular ranges**: Generate 2D grids between two complex corners
- **Linear ranges**: Generate points along diagonals using list syntax
- **Flexible stepping**: Independent control of real and imaginary step sizes
- **Negative steps**: Support for descending ranges
- **Farey subdivision**: Create refined grids using number-theoretic sequences
- **Pure Python**: No external dependencies required
- **Python 3.8+**: Compatible with Python 3.8 and later

## [Unreleased]

### Planned

- NumPy array output option
- Iterator/generator versions for memory efficiency
- Polar coordinate range support
- Additional subdivision methods
