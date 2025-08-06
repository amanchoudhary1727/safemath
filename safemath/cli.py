"""
Command-line interface for SafeMath.
"""

import argparse
import sys
from .core import safe_eval
from .config import set_global_fallback
import numpy as np

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='SafeMath CLI - Evaluate mathematical expressions safely')
    parser.add_argument('expression', help='Mathematical expression to evaluate')
    parser.add_argument('--fallback', type=float, default=np.nan, 
                       help='Fallback value for errors (default: nan)')
    parser.add_argument('--variables', '-v', action='append', 
                       help='Variables in format name=value (can be used multiple times)')
    
    args = parser.parse_args()
    
    # Set global fallback
    set_global_fallback(args.fallback)
    
    # Parse variables
    variables = {}
    if args.variables:
        for var_def in args.variables:
            try:
                name, value = var_def.split('=', 1)
                variables[name.strip()] = float(value)
            except ValueError:
                print(f"Warning: Invalid variable definition '{var_def}', skipping")
    
    # Evaluate expression
    result = safe_eval(args.expression, variables, args.fallback)
    print(result)

if __name__ == '__main__':
    main()
