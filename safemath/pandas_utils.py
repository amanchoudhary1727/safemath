"""
Pandas utilities for SafeMath operations.
"""

import pandas as pd
import numpy as np
from .config import get_global_fallback, log_fallback
from .numpy_utils import safe_numpy_operation, safe_numpy_binary_operation

def safe_pandas_operation(func, data, fallback=None, func_name="unknown", inplace=False):
    """
    Apply safe operation to Pandas Series or DataFrame.
    
    Parameters:
        func: Function to apply
        data: Pandas Series or DataFrame
        fallback: Fallback value
        func_name: Function name for logging
        inplace: Modify data in place (DataFrame only)
    
    Returns:
        Result Series/DataFrame or None if inplace=True
    """
    if fallback is None:
        fallback = get_global_fallback()
    
    if isinstance(data, pd.Series):
        try:
            # Apply to numeric Series
            if pd.api.types.is_numeric_dtype(data):
                result_values = safe_numpy_operation(func, data.values, fallback, func_name)
                return pd.Series(result_values, index=data.index, name=data.name)
            else:
                log_fallback(func_name, "non-numeric Series", "TypeError", fallback)
                return pd.Series([fallback] * len(data), index=data.index, name=data.name)
        except Exception as e:
            log_fallback(func_name, "Series", e, fallback)
            return pd.Series([fallback] * len(data), index=data.index, name=data.name)
    
    elif isinstance(data, pd.DataFrame):
        if inplace:
            result_df = data
        else:
            result_df = data.copy()
        
        for col in data.columns:
            try:
                if pd.api.types.is_numeric_dtype(data[col]):
                    result_values = safe_numpy_operation(func, data[col].values, fallback, func_name)
                    result_df[col] = result_values
                else:
                    # Skip non-numeric columns
                    if not inplace:
                        result_df[col] = data[col]
            except Exception as e:
                log_fallback(func_name, f"DataFrame column '{col}'", e, fallback)
                result_df[col] = fallback
        
        return None if inplace else result_df
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def safe_pandas_binary_operation(func, a, b, fallback=None, func_name="unknown", inplace=False):
    """
    Apply safe binary operation to Pandas data structures.
    """
    if fallback is None:
        fallback = get_global_fallback()
    
    # Handle Series operations
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        try:
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                result_values = safe_numpy_binary_operation(func, a.values, b.values, fallback, func_name)
                return pd.Series(result_values, index=a.index)
        except Exception as e:
            log_fallback(func_name, "Series operation", e, fallback)
            return pd.Series([fallback] * len(a), index=a.index)
    
    # Handle DataFrame operations
    elif isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        if inplace:
            result_df = a
        else:
            result_df = a.copy()
        
        common_cols = set(a.columns) & set(b.columns)
        for col in common_cols:
            try:
                if (pd.api.types.is_numeric_dtype(a[col]) and 
                    pd.api.types.is_numeric_dtype(b[col])):
                    result_values = safe_numpy_binary_operation(
                        func, a[col].values, b[col].values, fallback, func_name
                    )
                    result_df[col] = result_values
            except Exception as e:
                log_fallback(func_name, f"DataFrame column '{col}'", e, fallback)
                result_df[col] = fallback
        
        return None if inplace else result_df
    
    # Handle mixed operations (Series with scalar, etc.)
    else:
        try:
            if isinstance(a, (pd.Series, pd.DataFrame)):
                return safe_pandas_operation(lambda x: func(x, b), a, fallback, func_name, inplace)
            elif isinstance(b, (pd.Series, pd.DataFrame)):
                return safe_pandas_operation(lambda x: func(a, x), b, fallback, func_name, inplace)
        except Exception as e:
            log_fallback(func_name, "mixed pandas operation", e, fallback)
    
    raise TypeError(f"Unsupported operation between {type(a)} and {type(b)}")
