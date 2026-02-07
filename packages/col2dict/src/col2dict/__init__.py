"""
col2dict — Create nested dictionaries from columns of tabular data.

A Python translation of the Wolfram Language ResourceFunction "AssociateColumns"
originally contributed by Daniele Gregori to the Wolfram Function Repository.

https://resources.wolframcloud.com/FunctionRepository/resources/AssociateColumns/
"""

__version__ = "0.8.0"
__author__ = "Daniele Gregori"

from col2dict.core import associate_columns

__all__ = ["associate_columns"]
