"""
Tests for NumPy integration and array operations.
"""

import unittest
import numpy as np
from safemath import *

class TestNumpyIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        set_global_fallback(np.nan)
        disable_trace()
        
        # Various test arrays
        self.simple_array = np.array([1, 2, 3, 4, 5])
        self.mixed_array = np.array([1, -1, 0, 4, np.nan])
        self.negative_array = np.array([-5, -3, -1, 0, 2])
        self.zero_div_array = np.array([10, 20, 0, 5, 8])
        self.divisor_array = np.array([2, 0, 1, 0, 4])  # Contains zeros for division
        
        # 2D arrays
        self.matrix_2d = np.array([[1, 4, -1], [0, 9, 25]])
        self.complex_array = np.array([1+2j, 3-1j, 0+0j])
        
        # Edge case arrays
        self.inf_array = np.array([1, np.inf, -np.inf, 0])
        self.large_array = np.random.randn(1000)  # Large array for performance
        self.large_array[::100] = -1  # Add some negative values
        
    def test_basic_arithmetic_arrays(self):
        """Test basic arithmetic operations on NumPy arrays."""
        # Addition
        result = safe_add(self.simple_array, 5)
        expected = np.array([6, 7, 8, 9, 10])
        np.testing.assert_array_equal(result, expected)
        
        # Subtraction
        result = safe_subtract(self.simple_array, 2)
        expected = np.array([-1, 0, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)
        
        # Multiplication
        result = safe_multiply(self.simple_array, 3)
        expected = np.array([3, 6, 9, 12, 15])
        np.testing.assert_array_equal(result, expected)
    
    def test_array_division_with_zeros(self):
        """Test division operations that include division by zero."""
        result = safe_divide(self.zero_div_array, self.divisor_array)
        
        # Expected: [10/2, 20/0, 0/1, 5/0, 8/4] = [5, +inf, 0, +inf, 2]
        self.assertEqual(result[0], 5.0)         # 10/2 = 5
        self.assertTrue(np.isposinf(result[1]))  # 20/0 = +inf (mathematically correct!)
        self.assertEqual(result[2], 0.0)         # 0/1 = 0  
        self.assertTrue(np.isposinf(result[3]))  # 5/0 = +inf (mathematically correct!)
        self.assertEqual(result[4], 2.0)         # 8/4 = 2
        
        # Test the special case: 0/0 = NaN
        zero_div_zero = safe_divide(np.array([0]), np.array([0]))
        self.assertTrue(np.isnan(zero_div_zero[0]))  # 0/0 = NaN

    
    def test_array_square_root(self):
        """Test square root operations on arrays with negative values."""
        test_array = np.array([0, 1, 4, 9, -1, -4])
        result = safe_sqrt(test_array)
        
        self.assertEqual(result[0], 0.0)      # sqrt(0)
        self.assertEqual(result[1], 1.0)      # sqrt(1)
        self.assertEqual(result[2], 2.0)      # sqrt(4)
        self.assertEqual(result[3], 3.0)      # sqrt(9)
        self.assertTrue(np.isnan(result[4]))  # sqrt(-1)
        self.assertTrue(np.isnan(result[5]))  # sqrt(-4)
    
    def test_array_logarithms(self):
        """Test logarithmic operations on arrays."""
        test_array = np.array([1, np.e, 10, 0, -1])
        
        # Natural log
        result = safe_log(test_array)
        self.assertAlmostEqual(result[0], 0.0)      # log(1)
        self.assertAlmostEqual(result[1], 1.0)      # log(e)
        self.assertTrue(np.isneginf(result[3]))     # log(0) = -inf
        self.assertTrue(np.isnan(result[4]))        # log(-1) = nan
        
        # Base-10 log
        result = safe_log10(test_array)
        self.assertAlmostEqual(result[0], 0.0)      # log10(1)
        self.assertAlmostEqual(result[2], 1.0)      # log10(10)
        self.assertTrue(np.isneginf(result[3]))     # log10(0) = -inf
        self.assertTrue(np.isnan(result[4]))        # log10(-1) = nan
    
    def test_trigonometric_arrays(self):
        """Test trigonometric functions on arrays."""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        
        # Sine
        result = safe_sin(angles)
        self.assertAlmostEqual(result[0], 0.0, places=10)   # sin(0)
        self.assertAlmostEqual(result[1], 1.0, places=10)   # sin(π/2)
        self.assertAlmostEqual(result[2], 0.0, places=10)   # sin(π)
        self.assertAlmostEqual(result[3], -1.0, places=10)  # sin(3π/2)
        self.assertAlmostEqual(result[4], 0.0, places=10)   # sin(2π)
        
        # Cosine
        result = safe_cos(angles)
        self.assertAlmostEqual(result[0], 1.0, places=10)   # cos(0)
        self.assertAlmostEqual(result[1], 0.0, places=10)   # cos(π/2)
        self.assertAlmostEqual(result[2], -1.0, places=10)  # cos(π)
        self.assertAlmostEqual(result[3], 0.0, places=10)   # cos(3π/2)
        
        # Tangent (including problematic values)
        result = safe_tan(angles)
        self.assertAlmostEqual(result[0], 0.0, places=10)   # tan(0)
        self.assertTrue(np.isfinite(result[1]))             # tan(π/2) - should be finite
        self.assertAlmostEqual(result[2], 0.0, places=10)   # tan(π)
    
    def test_2d_arrays(self):
        """Test operations on 2D arrays."""
        result = safe_sqrt(self.matrix_2d)
        
        # First row: sqrt([1, 4, -1]) = [1, 2, nan]
        self.assertEqual(result[0][0], 1.0)
        self.assertEqual(result[0][1], 2.0)
        self.assertTrue(np.isnan(result[0][2]))
        
        # Second row: sqrt([0, 9, 25]) = [0, 3, 5]
        self.assertEqual(result[1][0], 0.0)
        self.assertEqual(result[1][1], 3.0)
        self.assertEqual(result[1][2], 5.0)
        
        # Check shape preservation
        self.assertEqual(result.shape, self.matrix_2d.shape)
    
    def test_array_with_fallback(self):
        """Test array operations with custom fallback values."""
        test_array = np.array([1, -1, 0, 4])
        
        # Test with fallback = -999
        result = safe_sqrt(test_array, fallback=-999)
        self.assertEqual(result[0], 1.0)      # sqrt(1)
        self.assertEqual(result[1], -999)     # sqrt(-1) -> fallback
        self.assertEqual(result[2], 0.0)      # sqrt(0)
        self.assertEqual(result[3], 2.0)      # sqrt(4)
    
    def test_infinity_handling(self):
        """Test handling of infinite values in arrays."""
        result = safe_add(self.inf_array, 1)
        
        self.assertEqual(result[0], 2.0)           # 1 + 1
        self.assertTrue(np.isposinf(result[1]))    # inf + 1 = inf
        self.assertTrue(np.isneginf(result[2]))    # -inf + 1 = -inf
        self.assertEqual(result[3], 1.0)           # 0 + 1
        
        # Test operations that result in NaN
        result = safe_add(np.array([np.inf]), np.array([-np.inf]))
        self.assertTrue(np.isnan(result[0]))       # inf + (-inf) = nan
    
    def test_mixed_data_types(self):
        """Test arrays with mixed or unusual data types."""
        # Integer array
        int_array = np.array([1, 2, 3, 4], dtype=int)
        result = safe_sqrt(int_array)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], np.sqrt(2))
        
        # Float32 array
        float32_array = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        result = safe_sqrt(float32_array)
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 2.0)
        self.assertEqual(result[2], 3.0)
    
    def test_empty_and_single_element_arrays(self):
        """Test edge cases with empty and single-element arrays."""
        # Empty array
        empty = np.array([])
        result = safe_sqrt(empty)
        self.assertEqual(len(result), 0)
        
        # Single element
        single = np.array([16])
        result = safe_sqrt(single)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 4.0)
        
        # Single negative element
        single_neg = np.array([-1])
        result = safe_sqrt(single_neg)
        self.assertTrue(np.isnan(result[0]))
    
    def test_broadcasting(self):
        """Test broadcasting behavior with different array shapes."""
        # 1D array with scalar
        result = safe_multiply(self.simple_array, 2)
        expected = self.simple_array * 2
        np.testing.assert_array_equal(result, expected)
        
        # 2D array with 1D array (should broadcast)
        arr_2d = np.array([[1, 2], [3, 4]])
        arr_1d = np.array([10, 20])
        
        result = safe_add(arr_2d, arr_1d)
        expected = np.array([[11, 22], [13, 24]])
        np.testing.assert_array_equal(result, expected)
    
    def test_large_array_performance(self):
        """Test performance and correctness with large arrays."""
        # This test ensures the vectorized operations work efficiently
        result = safe_sqrt(self.large_array)
        
        # Check that array length is preserved
        self.assertEqual(len(result), len(self.large_array))
        
        # Check that negative values became NaN
        negative_indices = self.large_array < 0
        self.assertTrue(np.all(np.isnan(result[negative_indices])))
        
        # Check that positive values are correctly computed
        positive_indices = self.large_array > 0
        positive_results = result[positive_indices]
        expected_positive = np.sqrt(self.large_array[positive_indices])
        np.testing.assert_array_almost_equal(positive_results, expected_positive)
    
    def test_array_modulo_operations(self):
        """Test modulo operations on arrays."""
        dividend = np.array([10, 15, 7, 20])
        divisor = np.array([3, 0, 2, 6])  # Include zero divisor
        
        result = safe_mod(dividend, divisor)
        
        self.assertEqual(result[0], 1.0)      # 10 % 3 = 1
        self.assertTrue(np.isnan(result[1]))  # 15 % 0 -> nan
        self.assertEqual(result[2], 1.0)      # 7 % 2 = 1
        self.assertEqual(result[3], 2.0)      # 20 % 6 = 2
    
    def test_array_power_operations(self):
        """Test power operations on arrays."""
        base = np.array([2, 3, -2, 0])
        exponent = np.array([3, 2, 2, 5])
        
        result = safe_power(base, exponent)
        
        self.assertEqual(result[0], 8.0)      # 2^3 = 8
        self.assertEqual(result[1], 9.0)      # 3^2 = 9
        self.assertEqual(result[2], 4.0)      # (-2)^2 = 4
        self.assertEqual(result[3], 0.0)      # 0^5 = 0
        
        # Test with problematic exponents
        problematic = safe_power(np.array([0]), np.array([0]))  # 0^0
        # 0^0 is mathematically undefined, should handle gracefully
        self.assertTrue(len(problematic) == 1)  # Just ensure it doesn't crash
    
    def test_array_absolute_values(self):
        """Test absolute value operations on arrays."""
        test_array = np.array([-5, -2.5, 0, 3.7, 10])
        result = safe_abs(test_array)
        
        expected = np.array([5, 2.5, 0, 3.7, 10])
        np.testing.assert_array_equal(result, expected)
    
    def test_array_negation(self):
        """Test negation operations on arrays."""
        test_array = np.array([-5, 0, 3, -2.5])
        result = safe_negate(test_array)
        
        expected = np.array([5, 0, -3, 2.5])
        np.testing.assert_array_equal(result, expected)

class TestNumpyEdgeCases(unittest.TestCase):
    
    def setUp(self):
        set_global_fallback(np.nan)
        disable_trace()
    
    def test_nan_input_arrays(self):
        """Test arrays that already contain NaN values."""
        nan_array = np.array([1, np.nan, 3, np.nan, 5])
        result = safe_sqrt(nan_array)
        
        self.assertEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))
        self.assertAlmostEqual(result[2], np.sqrt(3))
        self.assertTrue(np.isnan(result[3]))
        self.assertAlmostEqual(result[4], np.sqrt(5))
    
    def test_complex_numbers(self):
        """Test operations on complex number arrays."""
        complex_array = np.array([1+0j, 0+1j, -1+0j])
        
        # Test absolute value (should work with complex)
        result = safe_abs(complex_array)
        self.assertEqual(result[0], 1.0)  # |1+0j| = 1
        self.assertEqual(result[1], 1.0)  # |0+1j| = 1
        self.assertEqual(result[2], 1.0)  # |-1+0j| = 1
    
    def test_very_large_and_small_numbers(self):
        """Test with very large and very small numbers."""
        large_numbers = np.array([1e100, 1e-100, 1e308, 1e-308])
        
        # Test that operations don't crash
        result = safe_sqrt(large_numbers)
        self.assertEqual(len(result), 4)
        self.assertTrue(np.all(np.isfinite(result)))  # All should be finite
        
        # Test very large power operations
        result = safe_power(np.array([10]), np.array([100]))
        self.assertTrue(len(result) == 1)  # Should not crash
    
    def test_dtype_preservation(self):
        """Test that appropriate dtypes are preserved when possible."""
        int_array = np.array([1, 4, 9, 16], dtype=np.int32)
        result = safe_sqrt(int_array)
        
        # Result should be float (since sqrt can produce non-integers)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        
        # Check values
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
