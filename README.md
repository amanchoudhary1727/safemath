# SafeMath 🛡️

[![PyPI version](https://badge.fury.io/py/safemath.svg)](https://badge.fury.io/py/safemath)
[![Python Support](https://img.shields.io/pypi/pyversions/safemath.svg)](https://pypi.org/project/safemath/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready safe mathematical operations that never crash.**

SafeMath provides robust mathematical functions that handle edge cases gracefully across Python scalars, NumPy arrays, Pandas Series, and DataFrames. Perfect for data science, financial calculations, and any application where mathematical reliability is crucial.

## 🚀 Quick Start

pip install safemath


undefined
import safemath as sm
import numpy as np
import pandas as pd

Never crashes, returns fallback values instead
result = sm.safe_divide(10, 0) # Returns: nan
result = sm.safe_log(-1, fallback=0) # Returns: 0
result = sm.safe_sqrt([-1, 4, 9]) # Returns: [nan, 2.0, 3.0]

Works seamlessly with Pandas
df = pd.DataFrame({'a': [1, -1, 0], 'b': })
df['safe_ratio'] = sm.safe_divide(df['a'], df['b']) # No crashes on division by zero



## ✨ Key Features

- **🛡️ Never Crashes**: All functions handle edge cases with configurable fallbacks
- **📊 Universal Compatibility**: Works with scalars, NumPy arrays, Pandas Series/DataFrames
- **⚡ High Performance**: Vectorized operations using NumPy under the hood  
- **🎯 Production Ready**: Comprehensive error handling and logging
- **🔧 Flexible Fallbacks**: Global and per-function fallback strategies
- **🔗 Method Chaining**: Fluent API with SafeNumber wrapper
- **📝 Expression Evaluation**: Safe evaluation of mathematical expressions

## 📚 Complete Function Reference

### Arithmetic Operations
safe_add(a, b) # Addition with overflow protection
safe_subtract(a, b) # Subtraction
safe_multiply(a, b) # Multiplication
safe_divide(a, b) # Division (handles division by zero)
safe_mod(a, b) # Modulo operation
safe_power(a, b) # Exponentiation
safe_abs(x) # Absolute value
safe_negate(x) # Negation



### Mathematical Functions  
safe_log(x) # Natural logarithm
safe_log10(x) # Base-10 logarithm
safe_sqrt(x) # Square root
safe_sin(x) # Sine (radians)
safe_cos(x) # Cosine (radians)
safe_tan(x) # Tangent (radians)



## 💡 Usage Examples

### Basic Operations
import safemath as sm

Scalars
result = sm.safe_divide(10, 0) # nan instead of ZeroDivisionError
result = sm.safe_log(-5, fallback=0) # 0 instead of ValueError

Arrays
import numpy as np
data = np.array([1, -1, 0, 4])
result = sm.safe_sqrt(data) # [1.0, nan, 0.0, 2.0]



### Pandas Integration
import pandas as pd

df = pd.DataFrame({
'revenue': ,
'costs': ,
'temperature': [25, -10, 0, 35]
})

Safe operations on columns
df['profit_margin'] = sm.safe_divide(df['revenue'] - df['costs'], df['revenue'])
df['log_temp'] = sm.safe_log(df['temperature'] + 273.15) # Convert to Kelvin first

In-place operations
sm.safe_sqrt(df[['revenue', 'costs']], inplace=True)



### Global Fallback Configuration
Set global fallback for all operations
sm.set_global_fallback(0)
result = sm.safe_log(-1) # Returns 0 instead of nan

Enable logging to track fallback usage
sm.enable_trace()
result = sm.safe_divide(10, 0) # Logs the fallback usage



### Method Chaining with SafeNumber
from safemath import SafeNumber

Fluent API for complex calculations
data = [16, -4, 0, 25]
result = SafeNumber(data).abs().sqrt().log().value_or(0)

Equivalent to: log(sqrt(abs(data))) with fallback 0


### Safe Expression Evaluation
Evaluate expressions safely
result = sm.safe_eval("log(x) + sqrt(y)", {"x": 10, "y": 25})

With fallback for invalid operations
result = sm.safe_eval("log(-1) + tan(pi/2)", fallback=0)

Works with arrays and DataFrames too
df = pd.DataFrame({'x': [1, -1, 0], 'y': })
result = sm.safe_eval("sqrt(x) + log(y)", df.to_dict('series'))



### Custom Fallback Strategies  
Per-function fallbacks
profit_margin = sm.safe_divide(profit, revenue, fallback=0.0)
log_values = sm.safe_log(data, fallback=-999)

Function decorator for custom functions
@sm.safe_function(fallback=0)
def complex_calculation(x):
return np.log(x) / np.sqrt(x - 1)

result = complex_calculation() # Handles all edge cases



## 🧪 Error Handling Examples

All of these return sensible fallbacks instead of crashing:
sm.safe_divide(1, 0) # nan (division by zero)
sm.safe_log(-1) # nan (negative logarithm)
sm.safe_sqrt([-1, 4]) # [nan, 2.0] (negative sqrt)
sm.safe_tan(np.pi/2) # finite value (domain error)
sm.safe_power(10, 1000) # handles overflow
sm.safe_add(None, 5) # nan (None input)

Custom fallbacks
sm.safe_log(0, fallback=-np.inf) # -inf instead of nan
sm.safe_divide(, 0, fallback=999) #



## 🏗️ Advanced Usage

### Command Line Interface
Evaluate expressions from command line
safemath "log(10) + sqrt(25)" --fallback=0
safemath "x^2 + y" --variables="x=3" --variables="y=4"



### Integration with Existing Code
Replace standard operations
import numpy as np
import safemath as sm

Instead of:
result = np.log(data) / np.sqrt(data - 1) # Can crash
Use:
result = sm.safe_divide(sm.safe_log(data), sm.safe_sqrt(data - 1))

Or with method chaining:
result = SafeNumber(data).log().divide(SafeNumber(data - 1).sqrt()).value()



### Performance Considerations
SafeMath is optimized for production use:
- Vectorized NumPy operations for arrays
- Minimal overhead for scalar operations  
- Efficient fallback handling
- Optional tracing that can be disabled in production

## 📊 Supported Data Types

| Input Type | Example | Output Type |
|------------|---------|-------------|  
| Scalar | `5`, `3.14`, `1+2j` | Same type |
| List/Tuple | `[1, 2, 3]` | NumPy array |
| NumPy Array | `np.array([1, 2, 3])` | NumPy array |
| Pandas Series | `pd.Series([1, 2, 3])` | Pandas Series |
| Pandas DataFrame | Numeric columns only | Pandas DataFrame |

## 🔧 Configuration Options

Global fallback value
sm.set_global_fallback(0) # Use 0 instead of nan
sm.set_global_fallback(None) # Use None for errors
sm.set_global_fallback(-999) # Custom error code

Logging and tracing
sm.enable_trace() # Log all fallback usage
sm.disable_trace() # Disable logging

Check current settings
current_fallback = sm.get_global_fallback()



## 🧪 Testing

Run full test suite
python -m pytest tests/

With coverage
python -m pytest tests/ --cov=safemath

Run specific test modules
python -m pytest tests/test_core.py
python -m pytest tests/test_pandas.py



## 📋 Requirements

- Python >= 3.7
- NumPy >= 1.19.0  
- Pandas >= 1.3.0

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔗 Links

- [PyPI Package](https://pypi.org/project/safemath/)
- [GitHub Repository](https://github.com/username/safemath)
- [Documentation](https://github.com/username/safemath#readme)
- [Issue Tracker](https://github.com/username/safemath/issues)

## ⭐ Show your support  

Give a ⭐️ if this project helped you!

---

**SafeMath** - Making mathematical operations safe and reliable for production Python applications.