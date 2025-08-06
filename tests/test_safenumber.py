"""
Tests for SafeNumber wrapper class.
"""

import unittest
import numpy as np
import pandas as pd
from safemath import SafeNumber, set_global_fallback

class TestSafeNumber(unittest.TestCase):
    
    def setUp(self):
        set_global_fallback(np.nan)
    
    def test_scalar_chaining(self):
        """Test method chaining with scalars."""
        result = SafeNumber(16).sqrt().log().value()
        expected = np.log(4)  # sqrt(16) = 4, log(4)
        self.assertAlmostEqual(result, expected)
        
        # Test with fallback in chain
        result = SafeNumber(-1).sqrt().value_or(0)
        self.assertEqual(result, 0)  # sqrt(-1) -> nan -> 0
    
    def test_array_chaining(self):
        """Test chaining with arrays."""
        arr = np.array([16, 25, -1, 0])
        result = SafeNumber(arr).sqrt().value()
        
        self.assertEqual(result[0], 4)   # sqrt(16)
        self.assertEqual(result[1], 5)   # sqrt(25)
        self.assertTrue(np.isnan(result[2]))  # sqrt(-1)
        self.assertEqual(result[3], 0)   # sqrt(0)
    
    def test_pandas_chaining(self):
        """Test chaining with Pandas objects."""
        series = pd.Series([1, 4, -1, 9])
        result = SafeNumber(series).sqrt().multiply(2).value()
        
        self.assertEqual(result.iloc[0], 2)   # sqrt(1) * 2 = 2
        self.assertEqual(result.iloc[1], 4)   # sqrt(4) * 2 = 4
        self.assertTrue(np.isnan(result.iloc[2]))  # sqrt(-1) * 2 = nan
        self.assertEqual(result.iloc[3], 6)   # sqrt(9) * 2 = 6
    
    def test_magic_methods(self):
        """Test magic method operations."""
        sn1 = SafeNumber(10)
        sn2 = SafeNumber(2)
        
        self.assertEqual((sn1 + sn2).value(), 12)
        self.assertEqual((sn1 - sn2).value(), 8)
        self.assertEqual((sn1 * sn2).value(), 20)
        self.assertEqual((sn1 / sn2).value(), 5)
        self.assertEqual((sn1 % sn2).value(), 0)
        self.assertEqual((sn1 ** sn2).value(), 100)
    
    def test_fallback_handling(self):
        """Test fallback value handling."""
        # Custom fallback
        sn = SafeNumber(-1, fallback=-999)
        result = sn.sqrt().value()
        self.assertEqual(result, -999)
        
        # value_or with different types
        sn_scalar = SafeNumber(np.nan)
        self.assertEqual(sn_scalar.value_or(42), 42)
        
        sn_array = SafeNumber(np.array([1, np.nan, 3]))
        result = sn_array.value_or(0)
        np.testing.assert_array_equal(result, [1, 0, 3])
    
    def test_complex_chain(self):
        """Test complex operation chains."""
        # Mathematical expression: sqrt(log(exp(x))) for x = [1, 2, 3]
        arr = np.array([1, 2, 3])
        result = SafeNumber(arr).multiply(2).sqrt().log().value()
        
        # sqrt(2) ≈ 1.414, log(1.414) ≈ 0.346
        # sqrt(4) = 2, log(2) ≈ 0.693
        # sqrt(6) ≈ 2.449, log(2.449) ≈ 0.895
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(len(result), 3)

if __name__ == '__main__':
    unittest.main()
