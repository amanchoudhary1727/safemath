"""
NumPy utilities for SafeMath operations.
"""

import numpy as np
from .config import get_global_fallback, log_fallback

def _clean_array_input(arr):
    """Clean array input by converting None to NaN for NumPy operations."""
    if arr is None:
        return np.nan
    
    # Convert to numpy array and handle None values
    try:
        # If it's already a numpy array, check for None values
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:  # May contain None
                return np.where(arr == None, np.nan, arr).astype(float)
            return arr
        
        # For lists/tuples that may contain None
        if hasattr(arr, '__iter__') and not isinstance(arr, str):
            cleaned_list = [np.nan if x is None else x for x in arr]
            return np.array(cleaned_list, dtype=float)
        
        return arr
    except Exception:
        # Fallback: just return as numpy array
        return np.asarray(arr, dtype=float)

def safe_numpy_operation(func, x, fallback=None, func_name="unknown"):
    """
    Apply a safe operation to NumPy arrays with proper fallback handling.
    
    Parameters:
        func: Function to apply
        x: Input array
        fallback: Fallback value (uses global if None)
        func_name: Name for logging
    
    Returns:
        Result array with fallbacks where needed
    """
    if fallback is None:
        fallback = get_global_fallback()
    
    # Clean the input to handle None values
    x = _clean_array_input(x)
    x = np.asarray(x)
    
    try:
        with np.errstate(all='ignore'):
            # Apply the function
            result = func(x)
            
            # Only replace NaN results with fallback (preserve infinity)
            # This preserves mathematically correct results like log(0) = -inf
            nan_mask = np.isnan(result)
            result = np.where(nan_mask, fallback, result)
            
            # Log fallbacks if tracing enabled
            invalid_count = np.sum(nan_mask)
            if invalid_count > 0:
                log_fallback(func_name, f"array[{invalid_count} invalid]", 
                           "NaN results", fallback)
            
            return result.astype(float)
                
    except Exception as e:
        log_fallback(func_name, "array", e, fallback)
        return np.full_like(x, fallback, dtype=float)

def safe_numpy_binary_operation(func, a, b, fallback=None, func_name="unknown"):
    """
    Apply a safe binary operation to NumPy arrays.
    
    Parameters:
        func: Binary function to apply
        a, b: Input arrays
        fallback: Fallback value
        func_name: Name for logging
    
    Returns:
        Result array with fallbacks where needed
    """
    if fallback is None:
        fallback = get_global_fallback()
    
    # Clean both inputs to handle None values
    a = _clean_array_input(a)
    b = _clean_array_input(b)
    
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Broadcast to common shape
    try:
        a, b = np.broadcast_arrays(a, b)
    except ValueError as e:
        log_fallback(func_name, f"arrays with shapes {a.shape}, {b.shape}", e, fallback)
        return np.full_like(a, fallback, dtype=float)
    
    try:
        with np.errstate(all='ignore'):
            # Compute the basic operation
            result = func(a, b)
            
            # Handle special cases by operation type
            if func_name == "safe_mod":
                # Modulo: x % 0 should be NaN, others should be preserved
                div_by_zero_mask = (b == 0)
                result = np.where(div_by_zero_mask, np.nan, result)
                
            elif func_name == "safe_divide":
                # Division: handle x/0 cases properly
                div_by_zero_mask = (b == 0)
                zero_div_zero_mask = (a == 0) & (b == 0)
                pos_div_zero_mask = (a > 0) & (b == 0)
                neg_div_zero_mask = (a < 0) & (b == 0)
                
                # Apply division by zero rules
                result = np.where(zero_div_zero_mask, np.nan, result)      # 0/0 = NaN
                result = np.where(pos_div_zero_mask, np.inf, result)       # pos/0 = +inf
                result = np.where(neg_div_zero_mask, -np.inf, result)      # neg/0 = -inf
                
            elif func_name == "safe_add":
                # Addition: ONLY inf + (-inf) = NaN should use fallback
                # inf + finite_number = inf should be preserved!
                invalid_inf_add = (np.isposinf(a) & np.isneginf(b)) | (np.isneginf(a) & np.isposinf(b))
                result = np.where(invalid_inf_add, fallback, result)
                # DO NOT replace other cases - preserve inf + 1 = inf
                
            elif func_name == "safe_subtract":
                # Subtraction: ONLY inf - inf = NaN should use fallback
                invalid_inf_sub = (np.isposinf(a) & np.isposinf(b)) | (np.isneginf(a) & np.isneginf(b))
                result = np.where(invalid_inf_sub, fallback, result)
                
            elif func_name == "safe_multiply":
                # Multiplication: handle 0 * inf = NaN
                zero_times_inf = ((a == 0) & np.isinf(b)) | ((b == 0) & np.isinf(a))
                result = np.where(zero_times_inf, fallback, result)
                
            elif func_name == "safe_power":
                # Power: handle special cases like 0^0, inf^0, etc.
                invalid_power = np.isnan(result)
                result = np.where(invalid_power, fallback, result)
            
            return result.astype(float)
            
    except Exception as e:
        log_fallback(func_name, "arrays", e, fallback)
        return np.full_like(a, fallback, dtype=float)
