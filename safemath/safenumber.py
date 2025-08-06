"""
SafeNumber wrapper class for method chaining.
"""

import numpy as np
from .core import *
from .config import get_global_fallback

class SafeNumber:
    """
    Wrapper class that allows chaining of safe mathematical operations.
    
    Example:
        result = SafeNumber(data).log().divide(2).sqrt().value_or(0)
    """
    
    def __init__(self, value, fallback=None):
        """
        Initialize SafeNumber with a value.
        
        Parameters:
            value: Initial value (scalar, array, Series, DataFrame)
            fallback: Default fallback for this instance
        """
        self._value = value
        self._fallback = fallback if fallback is not None else get_global_fallback()
    
    def _wrap_result(self, result):
        """Wrap result in new SafeNumber instance."""
        return SafeNumber(result, self._fallback)
    
    def _extract_value(self, other):
        """Extract the actual value from SafeNumber objects or return as-is."""
        return other._value if isinstance(other, SafeNumber) else other
    
    # Arithmetic operations
    def add(self, other):
        """Add another value."""
        other_value = self._extract_value(other)
        result = safe_add(self._value, other_value, self._fallback)
        return self._wrap_result(result)
    
    def subtract(self, other):
        """Subtract another value."""
        other_value = self._extract_value(other)
        result = safe_subtract(self._value, other_value, self._fallback)
        return self._wrap_result(result)
    
    def multiply(self, other):
        """Multiply by another value."""
        other_value = self._extract_value(other)
        result = safe_multiply(self._value, other_value, self._fallback)
        return self._wrap_result(result)
    
    def divide(self, other):
        """Divide by another value."""
        other_value = self._extract_value(other)
        result = safe_divide(self._value, other_value, self._fallback)
        return self._wrap_result(result)
    
    def mod(self, other):
        """Modulo operation."""
        other_value = self._extract_value(other)
        result = safe_mod(self._value, other_value, self._fallback)
        return self._wrap_result(result)
    
    def power(self, other):
        """Raise to power."""
        other_value = self._extract_value(other)
        result = safe_power(self._value, other_value, self._fallback)
        return self._wrap_result(result)
    
    def abs(self):
        """Absolute value."""
        result = safe_abs(self._value, self._fallback)
        return self._wrap_result(result)
    
    def negate(self):
        """Negate value."""
        result = safe_negate(self._value, self._fallback)
        return self._wrap_result(result)
    
    # Mathematical operations
    def log(self):
        """Natural logarithm."""
        result = safe_log(self._value, self._fallback)
        return self._wrap_result(result)
    
    def log10(self):
        """Base-10 logarithm."""
        result = safe_log10(self._value, self._fallback)
        return self._wrap_result(result)
    
    def sqrt(self):
        """Square root."""
        result = safe_sqrt(self._value, self._fallback)
        return self._wrap_result(result)
    
    def sin(self):
        """Sine (radians)."""
        result = safe_sin(self._value, self._fallback)
        return self._wrap_result(result)
    
    def cos(self):
        """Cosine (radians)."""
        result = safe_cos(self._value, self._fallback)
        return self._wrap_result(result)
    
    def tan(self):
        """Tangent (radians)."""
        result = safe_tan(self._value, self._fallback)
        return self._wrap_result(result)
    
    # Result extraction
    def value(self):
        """Get the current value."""
        return self._value
    
    def value_or(self, default):
        """Get value or default if current value is NaN/None."""
        if self._value is None:
            return default
        
        if hasattr(self._value, '__iter__') and not isinstance(self._value, str):
            # Handle arrays/series
            import pandas as pd
            if isinstance(self._value, pd.Series):
                return self._value.fillna(default)
            elif isinstance(self._value, pd.DataFrame):
                return self._value.fillna(default)
            elif isinstance(self._value, np.ndarray):
                result = self._value.copy()
                result[~np.isfinite(result)] = default
                return result
        else:
            # Handle scalars
            try:
                if np.isnan(self._value):
                    return default
            except (TypeError, ValueError):
                # Handle cases where np.isnan fails (e.g., strings, complex types)
                pass
        
        return self._value
    
    # Magic methods for direct operations
    def __add__(self, other):
        return self.add(other)
    
    def __radd__(self, other):
        """Reverse addition (other + self)."""
        other_value = self._extract_value(other)
        result = safe_add(other_value, self._value, self._fallback)
        return self._wrap_result(result)
    
    def __sub__(self, other):
        return self.subtract(other)
    
    def __rsub__(self, other):
        """Reverse subtraction (other - self)."""
        other_value = self._extract_value(other)
        result = safe_subtract(other_value, self._value, self._fallback)
        return self._wrap_result(result)
    
    def __mul__(self, other):
        return self.multiply(other)
    
    def __rmul__(self, other):
        """Reverse multiplication (other * self)."""
        other_value = self._extract_value(other)
        result = safe_multiply(other_value, self._value, self._fallback)
        return self._wrap_result(result)
    
    def __truediv__(self, other):
        return self.divide(other)
    
    def __rtruediv__(self, other):
        """Reverse division (other / self)."""
        other_value = self._extract_value(other)
        result = safe_divide(other_value, self._value, self._fallback)
        return self._wrap_result(result)
    
    def __mod__(self, other):
        return self.mod(other)
    
    def __rmod__(self, other):
        """Reverse modulo (other % self)."""
        other_value = self._extract_value(other)
        result = safe_mod(other_value, self._value, self._fallback)
        return self._wrap_result(result)
    
    def __pow__(self, other):
        return self.power(other)
    
    def __rpow__(self, other):
        """Reverse power (other ** self)."""
        other_value = self._extract_value(other)
        result = safe_power(other_value, self._value, self._fallback)
        return self._wrap_result(result)
    
    def __abs__(self):
        return self.abs()
    
    def __neg__(self):
        return self.negate()
    
    def __pos__(self):
        """Unary plus (+self)."""
        return SafeNumber(self._value, self._fallback)
    
    def __repr__(self):
        return f"SafeNumber({self._value})"
    
    def __str__(self):
        return f"SafeNumber({self._value})"
    
    def __eq__(self, other):
        """Equality comparison."""
        other_value = self._extract_value(other)
        try:
            return self._value == other_value
        except Exception:
            return False
    
    def __ne__(self, other):
        """Not equal comparison."""
        return not self.__eq__(other)
    
    def __lt__(self, other):
        """Less than comparison."""
        other_value = self._extract_value(other)
        try:
            return self._value < other_value
        except Exception:
            return False
    
    def __le__(self, other):
        """Less than or equal comparison."""
        other_value = self._extract_value(other)
        try:
            return self._value <= other_value
        except Exception:
            return False
    
    def __gt__(self, other):
        """Greater than comparison."""
        other_value = self._extract_value(other)
        try:
            return self._value > other_value
        except Exception:
            return False
    
    def __ge__(self, other):
        """Greater than or equal comparison."""
        other_value = self._extract_value(other)
        try:
            return self._value >= other_value
        except Exception:
            return False
