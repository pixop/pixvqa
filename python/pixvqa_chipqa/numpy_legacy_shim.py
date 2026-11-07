"""NumPy 1.20 alias shim for upstream ChipQA feature code on CPython 3.12+."""
from __future__ import annotations

import numpy as np

for _name, _builtin in (
    ("int", int),
    ("float", float),
    ("bool", bool),
    ("object", object),
    ("str", str),
):
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)
