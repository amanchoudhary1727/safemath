<p align="center">
  <img src="logo.png" alt="SafeMath Logo" width="120"/>
</p>

# 🛡️ SafeMath

[![PyPI version](https://badge.fury.io/py/safemath.svg)](https://pypi.org/project/safemath/)
[![Python Support](https://img.shields.io/pypi/pyversions/safemath.svg)](https://pypi.org/project/safemath/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Safe, production-ready mathematical operations for Python, NumPy, and Pandas. Never crash on edge cases—get reliable results every time.**

---

## 🚀 Quick Start

```bash
pip install safemath
```

```python
import safemath as sm
import numpy as np
import pandas as pd

# Never crashes, always returns a fallback
sm.safe_divide(10, 0)                # nan
sm.safe_log(-1, fallback=0)          # 0
sm.safe_sqrt([-1, 4, 9])             # [nan, 2.0, 3.0]
```

---

## 📈 Use Cases

SafeMath is designed for any scenario where mathematical reliability and safety are critical. Here are some ideal use cases:

- **Data Science & Analytics:**
  - Prevents crashes in data pipelines due to NaNs, infinities, or invalid operations.
  - Seamlessly handles missing, zero, or out-of-domain values in NumPy and Pandas workflows.

- **Financial & Business Applications:**
  - Ensures robust calculations for KPIs, ratios, and financial metrics, even with incomplete or edge-case data.
  - Avoids propagation of errors in chained calculations (e.g., division by zero, log of negative numbers).

- **Machine Learning Pipelines:**
  - Guarantees safe preprocessing and feature engineering, so models never break on bad input.
  - Makes batch transformations and aggregations safe for production.

- **APIs & Web Services:**
  - Protects backend services from runtime math errors, returning fallback values instead of 500 errors.
  - Enables safe mathematical evaluation of user-provided expressions.

- **Scientific Computing:**
  - Handles edge cases in simulations, experiments, and research code without manual error checks.

- **General Python Development:**
  - Drop-in replacement for standard math operations to make any codebase more robust and production-ready.

---

## ✨ Features

- 🛡️ **Never Crashes:** Handles all edge cases with configurable fallbacks
- 📊 **Universal Compatibility:** Works with scalars, NumPy arrays, Pandas Series/DataFrames
- ⚡ **High Performance:** Vectorized operations using NumPy
- 🔧 **Flexible Fallbacks:** Global and per-function fallback strategies
- 🔗 **Method Chaining:** Fluent API with `SafeNumber` wrapper
- 📝 **Expression Evaluation:** Safe evaluation of mathematical expressions
- 🧪 **Comprehensive Logging:** Optional tracing for debugging and auditing

---

## 📚 Function Reference

### Arithmetic Operations

```python
sm.safe_add(a, b)
sm.safe_subtract(a, b)
sm.safe_multiply(a, b)
sm.safe_divide(a, b)
sm.safe_mod(a, b)
sm.safe_power(a, b)
sm.safe_abs(x)
sm.safe_negate(x)
```

### Mathematical Functions

```python
sm.safe_log(x)
sm.safe_log10(x)
sm.safe_sqrt(x)
sm.safe_sin(x)
sm.safe_cos(x)
sm.safe_tan(x)
```

---

## 💡 Usage Examples

### Scalars & Arrays

```python
sm.safe_divide(10, 0)                # nan (no ZeroDivisionError)
sm.safe_log(-5, fallback=0)          # 0 (no ValueError)
sm.safe_sqrt(np.array([1, -1, 0]))   # [1.0, nan, 0.0]
```

### Pandas Integration

```python
import pandas as pd

df = pd.DataFrame({
    'revenue': [100, 200, 0],
    'costs': [50, 250, 0],
    'temperature': [25, -10, 0]
})

# Safe operations on columns
df['profit_margin'] = sm.safe_divide(df['revenue'] - df['costs'], df['revenue'])
df['log_temp'] = sm.safe_log(df['temperature'] + 273.15)

# In-place operation
sm.safe_sqrt(df[['revenue', 'costs']], inplace=True)
```

### Global Fallback Configuration

```python
sm.set_global_fallback(0)
sm.safe_log(-1)  # Returns 0 instead of nan

sm.enable_trace()
sm.safe_divide(10, 0)  # Logs fallback usage
```

### Method Chaining with SafeNumber

```python
from safemath import SafeNumber

data = [16, -4, 0, 25]
result = SafeNumber(data).abs().sqrt().log().value_or(0)
# Equivalent to: log(sqrt(abs(data))) with fallback 0
```

### Safe Expression Evaluation

```python
sm.safe_eval("log(x) + sqrt(y)", {"x": 10, "y": 25})
sm.safe_eval("log(-1) + tan(pi/2)", fallback=0)
```

---

## 🧪 Error Handling Examples

All of these return sensible fallbacks instead of crashing:

```python
sm.safe_divide(1, 0)         # nan
sm.safe_log(-1)              # nan
sm.safe_sqrt([-1, 4])        # [nan, 2.0]
sm.safe_tan(np.pi/2)         # finite value
sm.safe_power(10, 1000)      # handles overflow
sm.safe_add(None, 5)         # nan
sm.safe_log(0, fallback=-np.inf)  # -inf
```

---

## 🏗️ Advanced Usage

### CLI

```bash
safemath "log(10) + sqrt(25)" --fallback=0
safemath "x^2 + y" --variables="x=3" --variables="y=4"
```

### Integration

```python
import numpy as np
import safemath as sm

# Instead of:
result = np.log(data) / np.sqrt(data - 1)  # Can crash

# Use:
result = sm.safe_divide(sm.safe_log(data), sm.safe_sqrt(data - 1))
```

---

## 📊 Supported Data Types

| Input Type         | Example                  | Output Type         |
|--------------------|-------------------------|---------------------|
| Scalar             | `5`, `3.14`, `1+2j`     | Same type           |
| List/Tuple         | `[1, 2, 3]`             | NumPy array         |
| NumPy Array        | `np.array([1, 2, 3])`   | NumPy array         |
| Pandas Series      | `pd.Series([1, 2, 3])`  | Pandas Series       |
| Pandas DataFrame   | Numeric columns only     | Pandas DataFrame    |

---

## 🔧 Configuration

```python
sm.set_global_fallback(0)      # Use 0 for errors
sm.set_global_fallback(None)   # Use None for errors
sm.enable_trace()              # Enable logging
sm.disable_trace()             # Disable logging
current_fallback = sm.get_global_fallback()
```

---

## 🧪 Testing

```bash
python -m pytest tests/
python -m pytest tests/ --cov=safemath
python -m pytest tests/test_core.py
python -m pytest tests/test_pandas.py
```

---

## 📋 Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- Pandas >= 1.3.0

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🔗 Links

- [PyPI Package](https://pypi.org/project/safemath/)
- [GitHub Repository](https://github.com/amanchoudhary1727/safemath)
- [Documentation](https://github.com/amanchoudhary1727/safemath#readme)

---

**SafeMath** – Making mathematical operations safe and reliable for production Python applications.