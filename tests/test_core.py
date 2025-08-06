"""
Tests for core SafeMath functionality.
"""

import unittest
import numpy as np
import pandas as pd
from safemath import *

class TestCoreFunctions(unittest.TestCase):
    
    def setUp(self):
        """Reset global configuration before each test."""
        set_global_fallback(np.nan)
        disable_trace()
    
    def test_safe_add_scalars(self):
        """Test safe addition with scalars."""
        self.assertEqual(safe_add(2, 3), 5)
        self.assertEqual(safe_add(-1, 1), 0)
        self.assertTrue(np.isnan(safe_add(None, 5)))
        self.assertEqual(safe_add(None, 5, fallback=0), 0)
    
    def test_safe_divide_scalars(self):
        """Test safe division with scalars."""
        self.assertEqual(safe_divide(10, 2), 5)
        self.assertTrue(np.isnan(safe_divide(10, 0)))
        self.assertEqual(safe_divide(10, 0, fallback=0), 0)
    
    def test_safe_log_scalars(self):
        """Test safe logarithm with scalars."""
        self.assertAlmostEqual(safe_log(np.e), 1.0)
        self.assertTrue(np.isnan(safe_log(-1)))
        self.assertTrue(np.isneginf(safe_log(0)))
        self.assertEqual(safe_log(-1, fallback=0), 0)
    
    def test_safe_sqrt_scalars(self):
        """Test safe square root with scalars."""
        self.assertEqual(safe_sqrt(4), 2)
        self.assertTrue(np.isnan(safe_sqrt(-1)))
        self.assertEqual(safe_sqrt(-1, fallback=0), 0)
    
    def test_safe_trig_scalars(self):
        """Test safe trigonometric functions with scalars."""
        self.assertAlmostEqual(safe_sin(0), 0)
        self.assertAlmostEqual(safe_cos(0), 1)
        self.assertAlmostEqual(safe_tan(0), 0)
        
        # Test problematic values
        self.assertTrue(np.isfinite(safe_tan(np.pi/2)))  # Should handle gracefully
    
    def test_arrays(self):
        """Test with NumPy arrays."""
        arr = np.array([1, -1, 0, 4])
        
        # Test sqrt
        result = safe_sqrt(arr)
        self.assertAlmostEqual(result[0], 1)
        self.assertTrue(np.isnan(result[1]))  # sqrt(-1)
        self.assertAlmostEqual(result[2], 0)
        self.assertAlmostEqual(result[3], 2)
        
        # Test with fallback
        result = safe_sqrt(arr, fallback=0)
        self.assertEqual(result[1], 0)  # sqrt(-1) -> fallback
    
    def test_global_fallback(self):
        """Test global fallback configuration."""
        set_global_fallback(-999)
        self.assertEqual(safe_divide(1, 0), -999)
        self.assertEqual(safe_log(-1), -999)
        
        # Reset
        set_global_fallback(np.nan)
    
    def test_safe_eval(self):
        """Test safe expression evaluation."""
        self.assertEqual(safe_eval("2 + 3"), 5)
        self.assertAlmostEqual(safe_eval("sin(0)"), 0)
        self.assertTrue(np.isnan(safe_eval("log(-1)")))
        self.assertEqual(safe_eval("log(-1)", fallback=0), 0)
        
        # With variables
        result = safe_eval("x + y", {"x": 2, "y": 3})
        self.assertEqual(result, 5)

class TestEdgeCases(unittest.TestCase):
    
    def test_none_inputs(self):
        """Test handling of None inputs."""
        self.assertTrue(np.isnan(safe_add(None, 5)))
        self.assertTrue(np.isnan(safe_add(5, None)))
        self.assertTrue(np.isnan(safe_log(None)))
    
    def test_inf_inputs(self):
        """Test handling of infinity."""
        self.assertTrue(np.isinf(safe_add(np.inf, 1)))
        self.assertTrue(np.isnan(safe_add(np.inf, -np.inf)))
    
    def test_empty_arrays(self):
        """Test with empty arrays."""
        empty = np.array([])
        result = safe_sqrt(empty)
        self.assertEqual(len(result), 0)
    
    def test_mixed_types(self):
        """Test with mixed input types."""
        # List input
        result = safe_sqrt([1, 4, -1])
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 2)
        self.assertTrue(np.isnan(result[2]))

if __name__ == '__main__':
    unittest.main()
