import numpy as np
import pandas as pd
from safemath import safe_sqrt

# Recreate the test data
df = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [2, 0, -1, 1, 2],
    'c': [1, -1, 0, 9, 25],
    'text': ['a', 'b', 'c', 'd', 'e']
})

print("Original DataFrame:")
print(df)
print(f"\ndf['b'] column: {df['b'].tolist()}")
print(f"df['b'].iloc[1] = {df['b'].iloc[1]} (should be 0)")

result = safe_sqrt(df)
print(f"\nAfter safe_sqrt:")
print(result)
print(f"\nresult['b'] column: {result['b'].tolist()}")
print(f"result['b'].iloc[1] = {result['b'].iloc[1]}")
print(f"Is NaN? {np.isnan(result['b'].iloc[1])}")
print(f"Type: {type(result['b'].iloc[1])}")

# Test just the problematic value
single_result = safe_sqrt(-1)
print(f"\nsafe_sqrt(-1) = {single_result}")
print(f"Is NaN? {np.isnan(single_result)}")
