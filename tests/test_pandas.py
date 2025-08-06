"""
Tests for Pandas integration.
"""

import unittest
import numpy as np
import pandas as pd
from safemath import *

class TestPandasIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        set_global_fallback(np.nan)
        disable_trace()
        
        self.series = pd.Series([1, -1, 0, 4, np.nan], name='test_series')
        self.df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 0, -1, 1, 2],
            'c': [1, -1, 0, 9, 25],
            'text': ['a', 'b', 'c', 'd', 'e']  # Non-numeric column
        })
    
    def test_series_operations(self):
        """Test operations on Pandas Series."""
        result = safe_sqrt(self.series)
        
        self.assertAlmostEqual(result.iloc[0], 1)  # sqrt(1)
        self.assertTrue(np.isnan(result.iloc[1]))   # sqrt(-1)
        self.assertAlmostEqual(result.iloc[2], 0)   # sqrt(0)
        self.assertAlmostEqual(result.iloc[3], 2)   # sqrt(4)
        self.assertTrue(np.isnan(result.iloc[4]))   # sqrt(nan)
        
        # Check index preservation
        pd.testing.assert_index_equal(result.index, self.series.index)
        self.assertEqual(result.name, 'test_series')
    
    def test_series_binary_operations(self):
        """Test binary operations on Series."""
        s1 = pd.Series([10, 20, 30])
        s2 = pd.Series([2, 0, 5])
        
        result = safe_divide(s1, s2)
        self.assertEqual(result.iloc[0], 5)          # 10/2 = 5
        self.assertTrue(np.isposinf(result.iloc[1])) # 20/0 = +inf (mathematically correct!)
        self.assertEqual(result.iloc[2], 6)          # 30/5 = 6
    
    def test_dataframe_operations(self):
        """Test operations on DataFrames."""
        result = safe_sqrt(self.df)
        
        # Check numeric columns were processed
        self.assertAlmostEqual(result['a'].iloc[0], 1)     # sqrt(1)
        self.assertAlmostEqual(result['b'].iloc[1], 0)     # sqrt(0) = 0
        self.assertTrue(np.isnan(result['b'].iloc[2]))     # sqrt(-1) = NaN
        self.assertAlmostEqual(result['c'].iloc[3], 3)     # sqrt(9)
        
        # Check non-numeric column preserved
        pd.testing.assert_series_equal(result['text'], self.df['text'])
        
        # Check structure preservation
        self.assertEqual(result.shape, self.df.shape)
        pd.testing.assert_index_equal(result.index, self.df.index)
        pd.testing.assert_index_equal(result.columns, self.df.columns)
    
    def test_dataframe_binary_operations(self):
        """Test binary operations on DataFrames."""
        df1 = pd.DataFrame({'x': [10, 20], 'y': [30, 40]})
        df2 = pd.DataFrame({'x': [2, 0], 'y': [5, 8]})
        
        result = safe_divide(df1, df2)
        self.assertEqual(result['x'].iloc[0], 5)         # 10/2 = 5
        self.assertTrue(np.isposinf(result['x'].iloc[1])) # 20/0 = +inf (mathematically correct!)
        self.assertEqual(result['y'].iloc[0], 6)         # 30/5 = 6
        self.assertEqual(result['y'].iloc[1], 5)         # 40/8 = 5
    
    def test_inplace_operations(self):
        """Test in-place DataFrame operations."""
        df_copy = self.df.copy()
        
        result = safe_sqrt(df_copy, inplace=True)
        self.assertIsNone(result)  # Should return None for inplace
        
        # Check that original DataFrame was modified
        self.assertAlmostEqual(df_copy['c'].iloc[3], 3)  # sqrt(9)
        self.assertTrue(np.isnan(df_copy['b'].iloc[2]))   # sqrt(-1)
    
    def test_series_with_fallback(self):
        """Test Series operations with custom fallback."""
        result = safe_sqrt(self.series, fallback=-1)
        
        self.assertEqual(result.iloc[1], -1)  # sqrt(-1) -> fallback
        self.assertEqual(result.iloc[4], -1)  # sqrt(nan) -> fallback

if __name__ == '__main__':
    unittest.main()
