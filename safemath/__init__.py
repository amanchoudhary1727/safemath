"""
SafeMath - Production-ready safe mathematical operations library.

Provides safe mathematical operations that don't crash on invalid inputs
and work seamlessly with Python scalars, NumPy arrays, Pandas Series, and DataFrames.
"""

__version__ = "1.0.0"
__author__ = "SafeMath Team"

from .core import (
    # Arithmetic functions
    safe_add,
    safe_subtract,
    safe_multiply,
    safe_divide,
    safe_mod,
    safe_power,
    safe_abs,
    safe_negate,
    # Math functions
    safe_log,
    safe_log10,
    safe_sqrt,
    safe_sin,
    safe_cos,
    safe_tan,
    # Utilities
    safe_eval,
    safe_function
)

from .config import set_global_fallback, get_global_fallback, enable_trace, disable_trace

from .safenumber import SafeNumber

__all__ = [
    # Arithmetic
    "safe_add", "safe_subtract", "safe_multiply", "safe_divide", "safe_mod",
    "safe_power", "safe_abs", "safe_negate",
    # Math
    "safe_log", "safe_log10", "safe_sqrt", "safe_sin", "safe_cos", "safe_tan",
    # Utils
    "safe_eval", "safe_function",
    # Config
    "set_global_fallback", "get_global_fallback", "enable_trace", "disable_trace",
    # Objects
    "SafeNumber"
]
