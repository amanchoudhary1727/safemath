import numpy as np
from safemath import safe_mod, safe_add

# Test modulo by zero
dividend = np.array([10, 15, 7, 20])
divisor = np.array([3, 0, 2, 6])
result = safe_mod(dividend, divisor)

print("Modulo test:")
print(f"15 % 0 = {result[1]}")
print(f"Is NaN? {np.isnan(result[1])}")
print(f"Type: {type(result[1])}")
print(f"All results: {result}")

print("\nInfinity test:")
inf_array = np.array([1, np.inf, -np.inf, 0])
result = safe_add(inf_array, 1)

print(f"inf + 1 = {result[1]}")
print(f"Is positive inf? {np.isposinf(result[1])}")
print(f"Type: {type(result[1])}")
print(f"All results: {result}")
