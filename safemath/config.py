"""
Configuration module for SafeMath library.
"""

import numpy as np
import logging

# Global configuration
_CONFIG = {
    'global_fallback': np.nan,
    'trace_enabled': False,
    'logger': None
}

def set_global_fallback(value):
    """
    Set the global fallback value for all safe operations.
    
    Parameters:
        value: The fallback value to use (default: np.nan)
    """
    _CONFIG['global_fallback'] = value

def get_global_fallback():
    """Get the current global fallback value."""
    return _CONFIG['global_fallback']

def enable_trace(log_level=logging.INFO):
    """
    Enable trace logging for safe operations.
    
    Parameters:
        log_level: Logging level (default: logging.INFO)
    """
    _CONFIG['trace_enabled'] = True
    if _CONFIG['logger'] is None:
        _CONFIG['logger'] = logging.getLogger('safemath')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        _CONFIG['logger'].addHandler(handler)
    _CONFIG['logger'].setLevel(log_level)

def disable_trace():
    """Disable trace logging."""
    _CONFIG['trace_enabled'] = False

def log_fallback(func_name, input_val, error, output):
    """Log fallback usage if tracing is enabled."""
    if _CONFIG['trace_enabled'] and _CONFIG['logger']:
        _CONFIG['logger'].info(
            f"Fallback used in {func_name}: input={input_val}, "
            f"error={type(error).__name__}, output={output}"
        )
