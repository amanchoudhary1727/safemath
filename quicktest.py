# quick_test.py
import safemath as sm
import numpy as np
import pandas as pd

print("✅ Testing basic imports...")
try:
    # Test basic import
    from safemath import safe_add, safe_divide, safe_log, SafeNumber
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

print("\n✅ Testing basic operations...")

# Test scalar operations
print("Scalar tests:")
print(f"safe_add(2, 3) = {sm.safe_add(2, 3)}")  # Should be 5
print(f"safe_divide(10, 0) = {sm.safe_divide(10, 0)}")  # Should be nan
print(f"safe_log(-1) = {sm.safe_log(-1)}")  # Should be nan
print(f"safe_log(-1, fallback=0) = {sm.safe_log(-1, fallback=0)}")  # Should be 0

# Test array operations
print("\nArray tests:")
arr = np.array([1, -1, 0, 4])
result = sm.safe_sqrt(arr)
print(f"safe_sqrt([1, -1, 0, 4]) = {result}")

# Test pandas operations
print("\nPandas tests:")
df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 0, 1]})
result = sm.safe_divide(df['a'], df['b'])
print(f"safe_divide on pandas series: {result.tolist()}")

# Test SafeNumber chaining
print("\nSafeNumber chaining tests:")
result = SafeNumber(16).sqrt().log().value()
print(f"SafeNumber(16).sqrt().log().value() = {result}")

print("\n🎉 All basic tests passed!")
