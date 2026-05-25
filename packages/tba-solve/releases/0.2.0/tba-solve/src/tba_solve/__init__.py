"""tba-solve — Numerical solver for Thermodynamic Bethe Ansatz integral equations.

A Python translation of the ThermodynamicBetheAnsatzSolve Wolfram Language
resource function, contributed by Daniele Gregori.
"""

__version__ = "0.2.0"
__author__ = "Daniele Gregori"

from tba_solve.solver import TBASolver, TBASolution
from tba_solve.models import sinh_gordon, liouville, seiberg_witten_su2

__all__ = [
    "TBASolver",
    "TBASolution",
    "sinh_gordon",
    "liouville",
    "seiberg_witten_su2",
]
