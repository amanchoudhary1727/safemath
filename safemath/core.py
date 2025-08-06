"""
Core safe mathematical operations.
"""

import numpy as np
import pandas as pd
import math
from typing import Union, Any, Callable
from .config import get_global_fallback, log_fallback
from .numpy_utils import safe_numpy_operation, safe_numpy_binary_operation
from .pandas_utils import safe_pandas_operation, safe_pandas_binary_operation

# Type definitions
NumericType = Union[int, float, complex, np.number]
ArrayLike = Union[list, tuple, np.ndarray]
PandasType = Union[pd.Series, pd.DataFrame]
InputType = Union[NumericType, ArrayLike, PandasType]

def _handle_input(x):
    """Convert input to appropriate type for processing."""
    if x is None:
        return np.nan
    
    if isinstance(x, (list, tuple)):
        return np.array(x)
    
    return x

def _safe_scalar_operation(func, x, fallback=None, func_name="unknown"):
    """Apply safe operation to scalar values."""
    if fallback is None:
        fallback = get_global_fallback()
    
    try:
        if x is None or (hasattr(x, '__iter__') and not isinstance(x, str)):
            return fallback
        
        result = func(x)
        
        # Allow infinite results, only reject NaN and other invalid results
        if np.isnan(result):
            log_fallback(func_name, x, "NaN result", fallback)
            return fallback
        else:
            return result
            
    except Exception as e:
        log_fallback(func_name, x, e, fallback)
        return fallback


def _safe_binary_scalar_operation(func, a, b, fallback=None, func_name="unknown"):
    """Apply safe binary operation to scalar values."""
    if fallback is None:
        fallback = get_global_fallback()
    
    try:
        if a is None or b is None:
            return fallback
        
        # Special handling for division by zero (return proper infinity)
        if func_name == "safe_divide" and b == 0:
            if a == 0:
                return np.nan  # 0/0 = NaN (indeterminate)
            elif a > 0:
                return np.inf  # positive/0 = +inf
            else:
                return -np.inf  # negative/0 = -inf
        
        # Special handling for modulo by zero
        if func_name == "safe_mod" and b == 0:
            return np.nan  # x % 0 = NaN
        
        result = func(a, b)
        
        # Allow infinite results, only reject NaN from invalid operations
        if np.isnan(result):
            log_fallback(func_name, f"({a}, {b})", "NaN result", fallback)
            return fallback
        else:
            return result
            
    except ZeroDivisionError:
        # This should be handled by the special cases above, but just in case
        if func_name == "safe_divide":
            if a == 0:
                return np.nan
            return np.inf if a > 0 else -np.inf
        elif func_name == "safe_mod":
            return np.nan
        else:
            log_fallback(func_name, f"({a}, {b})", "Division by zero", fallback)
            return fallback
    except Exception as e:
        log_fallback(func_name, f"({a}, {b})", e, fallback)
        return fallback



def _safe_operation_dispatcher(func, x, fallback=None, func_name="unknown", inplace=False):
    """Dispatch safe operation based on input type."""
    x = _handle_input(x)
    
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return safe_pandas_operation(func, x, fallback, func_name, inplace)
    elif isinstance(x, np.ndarray):
        return safe_numpy_operation(func, x, fallback, func_name)
    else:
        return _safe_scalar_operation(func, x, fallback, func_name)

def _safe_binary_operation_dispatcher(func, a, b, fallback=None, func_name="unknown", inplace=False):
    """Dispatch safe binary operation based on input types."""
    a = _handle_input(a)
    b = _handle_input(b)
    
    # Pandas operations
    if isinstance(a, (pd.Series, pd.DataFrame)) or isinstance(b, (pd.Series, pd.DataFrame)):
        return safe_pandas_binary_operation(func, a, b, fallback, func_name, inplace)
    
    # NumPy operations
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return safe_numpy_binary_operation(func, a, b, fallback, func_name)
    
    # Scalar operations
    else:
        return _safe_binary_scalar_operation(func, a, b, fallback, func_name)

# Arithmetic Operations
def safe_add(a: InputType, b: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """
    Safely add two values with fallback on error.
    
    Parameters:
        a, b: Values to add
        fallback: Value to return on error (uses global if None)
        inplace: Modify DataFrame in place
    
    Returns:
        Sum of a and b, or fallback on error
    """
    return _safe_binary_operation_dispatcher(lambda x, y: x + y, a, b, fallback, "safe_add", inplace)

def safe_subtract(a: InputType, b: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely subtract b from a with fallback on error."""
    return _safe_binary_operation_dispatcher(lambda x, y: x - y, a, b, fallback, "safe_subtract", inplace)

def safe_multiply(a: InputType, b: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely multiply two values with fallback on error."""
    return _safe_binary_operation_dispatcher(lambda x, y: x * y, a, b, fallback, "safe_multiply", inplace)

def safe_divide(a: InputType, b: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely divide a by b with fallback on error (including division by zero)."""
    return _safe_binary_operation_dispatcher(lambda x, y: x / y, a, b, fallback, "safe_divide", inplace)

def safe_mod(a: InputType, b: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely compute a modulo b with fallback on error."""
    return _safe_binary_operation_dispatcher(lambda x, y: x % y, a, b, fallback, "safe_mod", inplace)

def safe_power(a: InputType, b: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely raise a to the power of b with fallback on error."""
    return _safe_binary_operation_dispatcher(lambda x, y: x ** y, a, b, fallback, "safe_power", inplace)

def safe_abs(x: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely compute absolute value with fallback on error."""
    return _safe_operation_dispatcher(abs, x, fallback, "safe_abs", inplace)

def safe_negate(x: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely negate value with fallback on error."""
    return _safe_operation_dispatcher(lambda val: -val, x, fallback, "safe_negate", inplace)

# Mathematical Operations
def safe_log(x: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely compute natural logarithm with fallback on error."""
    return _safe_operation_dispatcher(np.log, x, fallback, "safe_log", inplace)

def safe_log10(x: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely compute base-10 logarithm with fallback on error."""
    return _safe_operation_dispatcher(np.log10, x, fallback, "safe_log10", inplace)

def safe_sqrt(x: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely compute square root with fallback on error."""
    return _safe_operation_dispatcher(np.sqrt, x, fallback, "safe_sqrt", inplace)

def safe_sin(x: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely compute sine (in radians) with fallback on error."""
    return _safe_operation_dispatcher(np.sin, x, fallback, "safe_sin", inplace)

def safe_cos(x: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely compute cosine (in radians) with fallback on error."""
    return _safe_operation_dispatcher(np.cos, x, fallback, "safe_cos", inplace)

def safe_tan(x: InputType, fallback: Any = None, inplace: bool = False) -> InputType:
    """Safely compute tangent (in radians) with fallback on error."""
    return _safe_operation_dispatcher(np.tan, x, fallback, "safe_tan", inplace)

# Utility Functions
def safe_function(fallback: Any = None):
    """
    Decorator to make any function safe with automatic fallback.
    
    Parameters:
        fallback: Default fallback value
    
    Returns:
        Decorated function that won't raise exceptions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if hasattr(result, '__iter__') and not isinstance(result, str):
                    # Handle arrays/series
                    if isinstance(result, np.ndarray):
                        mask = ~np.isfinite(result)
                        if np.any(mask):
                            result = result.copy()
                            result[mask] = fallback if fallback is not None else get_global_fallback()
                elif not np.isfinite(result):
                    result = fallback if fallback is not None else get_global_fallback()
                return result
            except Exception as e:
                log_fallback(func.__name__, args, e, fallback)
                return fallback if fallback is not None else get_global_fallback()
        return wrapper
    return decorator

def safe_eval(expression: str, variables: dict = None, fallback: Any = None) -> Any:
    """
    Safely evaluate mathematical expressions with safe functions.
    
    Parameters:
        expression: String expression to evaluate
        variables: Dictionary of variable names and values
        fallback: Fallback value on error
    
    Returns:
        Result of expression or fallback
    """
    if fallback is None:
        fallback = get_global_fallback()
    
    if variables is None:
        variables = {}
    
    # Safe namespace with only safe functions and math constants
    safe_namespace = {
        # Safe functions - use lambdas to pass fallback correctly
        'add': lambda a, b: safe_add(a, b, fallback),
        'subtract': lambda a, b: safe_subtract(a, b, fallback),
        'multiply': lambda a, b: safe_multiply(a, b, fallback),
        'divide': lambda a, b: safe_divide(a, b, fallback),
        'mod': lambda a, b: safe_mod(a, b, fallback),
        'power': lambda a, b: safe_power(a, b, fallback),
        'abs': lambda x: safe_abs(x, fallback),
        'negate': lambda x: safe_negate(x, fallback),
        'log': lambda x: safe_log(x, fallback),
        'log10': lambda x: safe_log10(x, fallback),
        'sqrt': lambda x: safe_sqrt(x, fallback),
        'sin': lambda x: safe_sin(x, fallback),
        'cos': lambda x: safe_cos(x, fallback),
        'tan': lambda x: safe_tan(x, fallback),
        # Constants
        'pi': np.pi, 'e': np.e, 'nan': np.nan, 'inf': np.inf,
        # NumPy functions (safe versions)
        'exp': lambda x: _safe_operation_dispatcher(np.exp, x, fallback, "exp"),
    }
    
    # Add user variables
    safe_namespace.update(variables)
    
    try:
        # Remove dangerous built-ins
        safe_namespace['__builtins__'] = {}
        result = eval(expression, {"__builtins__": {}}, safe_namespace)
        return result
    except Exception as e:
        log_fallback("safe_eval", expression, e, fallback)
        return fallback

