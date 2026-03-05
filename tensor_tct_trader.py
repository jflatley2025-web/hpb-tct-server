"""
tensor_tct_trader.py — Thin re-export shim
==========================================
All logic lives in 5A_tct_trader.py.

Importing this module is equivalent to importing 5A_tct_trader.
This shim exists so that:
  - server_mexc.py and schematics_5b_trader.py can use standard
    `from tensor_tct_trader import X` syntax without change.
  - unittest.mock.patch("tensor_tct_trader.X") patches the same
    namespace that 5A_tct_trader's code reads from, so all tests
    work correctly.

Why a shim? Python identifiers cannot start with a digit, so
  `import 5A_tct_trader` is a SyntaxError. The shim replaces
  itself in sys.modules with the actual module object so that
  `sys.modules["tensor_tct_trader"]` IS `5A_tct_trader`, making
  attribute patches fully effective.
"""

import importlib
import sys

_mod = importlib.import_module("5A_tct_trader")

# Replace this module in sys.modules so that patches applied to
# "tensor_tct_trader.X" land on the same dict as "5A_tct_trader.X".
sys.modules[__name__] = _mod
