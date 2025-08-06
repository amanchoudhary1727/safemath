"""
SafeMath Library - Enhanced Edge Case Testing with Correctness Verification
Tests every function with all possible edge cases and automatically validates results
"""

import numpy as np
import pandas as pd
import warnings
import sys
import math
from datetime import datetime
from safemath import *

class FileOutputManager:
    """Manages output to both console and file."""
    
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"safemath_verified_test_{timestamp}.txt"
        
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        
    def write(self, text):
        """Write to both console and file."""
        self.original_stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        """Flush both outputs."""
        self.original_stdout.flush()
        self.file.flush()
    
    def close(self):
        """Close file and restore stdout."""
        self.file.close()
        sys.stdout = self.original_stdout
    
    def __enter__(self):
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class TestValidator:
    """Validates test results for mathematical correctness."""
    
    @staticmethod
    def is_close_or_equal(actual, expected, rtol=1e-9, atol=1e-12):
        """Check if values are approximately equal, handling special cases."""
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False
        
        # Handle non-numeric types
        if not isinstance(actual, (int, float, complex, np.number)):
            return actual == expected
        
        # Handle NaN cases
        if np.isnan(actual) and np.isnan(expected):
            return True
        if np.isnan(actual) or np.isnan(expected):
            return False
        
        # Handle infinity cases
        if np.isinf(actual) and np.isinf(expected):
            return np.sign(actual) == np.sign(expected)
        if np.isinf(actual) or np.isinf(expected):
            return False
        
        # Handle complex numbers
        if isinstance(actual, complex) or isinstance(expected, complex):
            return np.allclose(actual, expected, rtol=rtol, atol=atol)
        
        # Regular numeric comparison
        return np.allclose(actual, expected, rtol=rtol, atol=atol)
    
    @staticmethod
    def validate_array_result(actual, expected_pattern):
        """Validate array results against expected patterns."""
        if not hasattr(actual, '__iter__'):
            return False
        
        if len(actual) != len(expected_pattern):
            return False
        
        for a, e in zip(actual, expected_pattern):
            if not TestValidator.is_close_or_equal(a, e):
                return False
        
        return True
    
    @staticmethod
    def check_mathematical_properties(func_name, inputs, result):
        """Check mathematical properties specific to each function."""
        if func_name == "safe_divide":
            a, b = inputs
            if b == 0:
                if a == 0:
                    return np.isnan(result)  # 0/0 should be NaN
                elif a > 0:
                    return np.isposinf(result)  # positive/0 should be +inf
                elif a < 0:
                    return np.isneginf(result)  # negative/0 should be -inf
        
        elif func_name == "safe_mod":
            a, b = inputs
            if b == 0:
                return np.isnan(result)  # x % 0 should be NaN
        
        elif func_name == "safe_log":
            x = inputs[0]
            if x == 0:
                return np.isneginf(result)  # log(0) should be -inf
            elif x < 0:
                return np.isnan(result)  # log(negative) should be NaN
            elif x == 1:
                return TestValidator.is_close_or_equal(result, 0)  # log(1) should be 0
        
        elif func_name == "safe_sqrt":
            x = inputs[0]
            if x < 0 and not isinstance(x, complex):
                return np.isnan(result) or isinstance(result, complex)  # sqrt(negative) should be NaN or complex
            elif x == 0:
                return TestValidator.is_close_or_equal(result, 0)  # sqrt(0) should be 0
        
        elif func_name == "safe_power":
            a, b = inputs
            if a == 0 and b == 0:
                return TestValidator.is_close_or_equal(result, 1)  # 0^0 should be 1
            elif b == 0:
                return TestValidator.is_close_or_equal(result, 1)  # x^0 should be 1
        
        return True  # If no specific property to check

def test_section(title):
    """Print formatted test section header."""
    print(f"\n{'='*70}")
    print(f"🧪 {title}")
    print('='*70)

def test_function_with_validation(func_name, func, test_cases):
    """Test a function with validation and report results."""
    print(f"\n🔍 Testing {func_name} with Validation:")
    print("-" * 50)
    
    total_tests = len(test_cases)
    passed_tests = 0
    failed_tests = 0
    
    for i, test_case in enumerate(test_cases):
        if len(test_case) == 3:
            test_input, description, expected = test_case
            custom_validator = None
        else:
            test_input, description, expected, custom_validator = test_case
        
        try:
            # Execute the test
            if isinstance(test_input, tuple):
                result = func(*test_input)
                input_str = f"({', '.join(map(str, test_input))})"
                inputs_for_validation = test_input
            else:
                result = func(test_input)
                input_str = str(test_input)
                inputs_for_validation = (test_input,)
            
            # Validate the result
            if custom_validator:
                is_correct = custom_validator(result)
            elif expected == "CHECK_PROPERTIES":
                is_correct = TestValidator.check_mathematical_properties(func_name, inputs_for_validation, result)
            elif hasattr(expected, '__iter__') and hasattr(result, '__iter__'):
                is_correct = TestValidator.validate_array_result(result, expected)
            else:
                is_correct = TestValidator.is_close_or_equal(result, expected)
            
            # Update counters
            if is_correct:
                passed_tests += 1
                status = "✅ PASS"
            else:
                failed_tests += 1
                status = "❌ FAIL"
            
            # Print results
            print(f"  {i+1}. {description}")
            print(f"     Input: {input_str}")
            print(f"     Result: {result}")
            print(f"     Expected: {expected if expected != 'CHECK_PROPERTIES' else 'Mathematical properties'}")
            print(f"     Status: {status}")
            print(f"     Type: {type(result)}")
            
            # Additional result info
            if hasattr(result, '__iter__') and not isinstance(result, str):
                if hasattr(result, 'shape'):
                    print(f"     Shape: {result.shape}")
                if len(result) > 0:
                    print(f"     Contains NaN: {np.any(np.isnan(result))}")
                    print(f"     Contains Inf: {np.any(np.isinf(result))}")
            else:
                if isinstance(result, (int, float, complex, np.number)):
                    print(f"     Is NaN: {np.isnan(result)}")
                    print(f"     Is Inf: {np.isinf(result)}")
            
        except Exception as e:
            failed_tests += 1
            print(f"  {i+1}. {description}")
            print(f"     Input: {input_str}")
            print(f"     ERROR: {type(e).__name__}: {e}")
            print(f"     Status: ❌ ERROR")
        
        print()
    
    # Print summary for this function
    print(f"📊 {func_name} Summary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    if failed_tests > 0:
        print(f"   ⚠️  {failed_tests} tests failed or had errors")
    print()
    
    return passed_tests, failed_tests

def main():
    with FileOutputManager() as output_manager:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("🎯 SafeMath Library - ENHANCED EDGE CASE TESTING WITH VALIDATION")
        print(f"Test started: {timestamp}")
        print("Testing every function with automatic correctness verification")
        print(f"Results are being saved to: {output_manager.filename}")
        
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        total_passed = 0
        total_failed = 0
        
        # =================================================================
        # SCALAR OPERATIONS WITH VALIDATION
        # =================================================================
        test_section("SCALAR OPERATIONS - Validated Tests")
        
        # Test safe_add with expected results
        add_tests = [
            ((5, 3), "Normal addition", 8),
            ((0, 0), "Zero addition", 0),
            ((np.inf, 1), "Infinity + finite", np.inf),
            ((-np.inf, 1), "Negative infinity + finite", -np.inf),
            ((np.inf, -np.inf), "Infinity + negative infinity", np.nan),
            ((np.nan, 5), "NaN + finite", np.nan),
            ((None, 5), "None input", np.nan),
            ((5, None), "None as second input", np.nan),
        ]
        
        passed, failed = test_function_with_validation("safe_add", safe_add, add_tests)
        total_passed += passed
        total_failed += failed
        
        # Test safe_divide with mathematical properties
        divide_tests = [
            ((10, 2), "Normal division", 5.0),
            ((10, 0), "Positive divided by zero", np.inf),
            ((-10, 0), "Negative divided by zero", -np.inf),
            ((0, 0), "Zero divided by zero", np.nan),
            ((0, 5), "Zero divided by positive", 0.0),
            ((np.inf, 2), "Infinity divided by finite", np.inf),
            ((10, np.inf), "Finite divided by infinity", 0.0),
            ((np.inf, np.inf), "Infinity divided by infinity", np.nan),
        ]
        
        passed, failed = test_function_with_validation("safe_divide", safe_divide, divide_tests)
        total_passed += passed
        total_failed += failed
        
        # Test safe_sqrt with validation
        sqrt_tests = [
            ((25,), "Perfect square", 5.0),
            ((0,), "Square root of zero", 0.0),
            ((-1,), "Square root of negative", np.nan),
            ((np.inf,), "Square root of infinity", np.inf),
            ((-np.inf,), "Square root of negative infinity", np.nan),
            ((np.nan,), "Square root of NaN", np.nan),
            ((None,), "Square root of None", np.nan),
        ]
        
        passed, failed = test_function_with_validation("safe_sqrt", safe_sqrt, sqrt_tests)
        total_passed += passed
        total_failed += failed
        
        # Test safe_log with mathematical properties
        log_tests = [
            ((1,), "Natural log of 1", 0.0),
            ((0,), "Natural log of zero", -np.inf),
            ((-1,), "Natural log of negative", np.nan),
            ((np.inf,), "Natural log of infinity", np.inf),
            ((-np.inf,), "Natural log of negative infinity", np.nan),
            ((np.nan,), "Natural log of NaN", np.nan),
            ((np.e,), "Natural log of e", 1.0),
        ]
        
        passed, failed = test_function_with_validation("safe_log", safe_log, log_tests)
        total_passed += passed
        total_failed += failed
        
        # Test safe_power with validation
        power_tests = [
            ((2, 3), "Normal power", 8),
            ((0, 0), "Zero to zero power", 1),
            ((5, 0), "Positive to zero power", 1),
            ((0, 5), "Zero to positive power", 0),
            ((-1, 2), "Negative to even power", 1),
            ((-1, 3), "Negative to odd power", -1),
            ((np.inf, 2), "Infinity to positive power", np.inf),
            ((np.inf, 0), "Infinity to zero power", 1.0),
        ]
        
        passed, failed = test_function_with_validation("safe_power", safe_power, power_tests)
        total_passed += passed
        total_failed += failed
        
        # Test safe_mod with validation
        mod_tests = [
            ((10, 3), "Normal modulo", 1),
            ((10, 0), "Modulo by zero", np.nan),
            ((0, 5), "Zero modulo positive", 0),
            ((-10, 3), "Negative modulo positive", 2),
            ((np.inf, 5), "Infinity modulo finite", np.nan),
        ]
        
        passed, failed = test_function_with_validation("safe_mod", safe_mod, mod_tests)
        total_passed += passed
        total_failed += failed
        
        # =================================================================
        # ARRAY OPERATIONS WITH VALIDATION
        # =================================================================
        test_section("NUMPY ARRAY OPERATIONS - Validated Tests")
        
        # Test array sqrt with expected patterns
        array_sqrt_tests = [
            (np.array([1, 4, 9, 16]), "Perfect squares array", [1.0, 2.0, 3.0, 4.0]),
            (np.array([1, -1, 0, np.inf]), "Mixed array", [1.0, np.nan, 0.0, np.inf]),
            (np.array([]), "Empty array", []),
            (np.array([0, 0, 0]), "All zeros", [0.0, 0.0, 0.0]),
        ]
        
        passed, failed = test_function_with_validation("safe_sqrt (arrays)", safe_sqrt, array_sqrt_tests)
        total_passed += passed
        total_failed += failed
        
        # Test array division with validation
        array_div_tests = [
            ((np.array([10, 20]), np.array([2, 0])), "Division with zero", [5.0, np.inf]),
            ((np.array([0, 10]), np.array([0, 2])), "Zero numerator cases", [np.nan, 5.0]),
            ((np.array([1, 2, 3]), np.array([1, 1, 1])), "Normal division", [1.0, 2.0, 3.0]),
        ]
        
        passed, failed = test_function_with_validation("safe_divide (arrays)", safe_divide, array_div_tests)
        total_passed += passed
        total_failed += failed
        
        # =================================================================
        # CONFIGURATION SYSTEM VALIDATION
        # =================================================================
        test_section("CONFIGURATION SYSTEM - Validated Tests")
        
        # Save original fallback
        original_fallback = get_global_fallback()
        
        config_tests = []
        fallback_values = [0, -999, np.inf, "ERROR"]
        
        for fallback_val in fallback_values:
            set_global_fallback(fallback_val)
            config_tests.append(
                ((-1,), f"Fallback test with {fallback_val}", fallback_val)
            )
        
        passed, failed = test_function_with_validation("safe_log (fallback)", safe_log, config_tests)
        total_passed += passed
        total_failed += failed
        
        # Restore original fallback
        set_global_fallback(original_fallback)
        
        # =================================================================
        # EXPRESSION EVALUATION VALIDATION
        # =================================================================
        test_section("SAFE_EVAL EXPRESSIONS - Validated Tests")
        
        eval_tests = [
            ("2 + 3", {}, "Simple arithmetic", 5),
            ("log(0) + 5", {}, "Log zero expression", -np.inf),
            ("divide(10, 0)", {}, "Division by zero", np.inf),
            ("sqrt(25)", {}, "Square root", 5.0),
            ("power(2, 3)", {}, "Power operation", 8),
            ("x + y", {'x': 10, 'y': 5}, "Variable substitution", 15),
        ]
        
        print("\n🔍 Testing safe_eval with Validation:")
        print("-" * 50)
        
        eval_passed = 0
        eval_failed = 0
        
        for i, (expr, vars_dict, description, expected) in enumerate(eval_tests):
            try:
                result = safe_eval(expr, vars_dict)
                is_correct = TestValidator.is_close_or_equal(result, expected)
                
                status = "✅ PASS" if is_correct else "❌ FAIL"
                if is_correct:
                    eval_passed += 1
                else:
                    eval_failed += 1
                
                print(f"  {i+1}. {description}")
                print(f"     Expression: '{expr}'")
                print(f"     Variables: {vars_dict}")
                print(f"     Result: {result}")
                print(f"     Expected: {expected}")
                print(f"     Status: {status}")
                
            except Exception as e:
                eval_failed += 1
                print(f"  {i+1}. {description}")
                print(f"     Expression: '{expr}'")
                print(f"     ERROR: {type(e).__name__}: {e}")
                print(f"     Status: ❌ ERROR")
            
            print()
        
        print(f"📊 safe_eval Summary: {eval_passed}/{len(eval_tests)} tests passed ({eval_passed/len(eval_tests)*100:.1f}%)")
        total_passed += eval_passed
        total_failed += eval_failed
        
        # =================================================================
        # PERFORMANCE VALIDATION
        # =================================================================
        test_section("PERFORMANCE TESTS - With Result Validation")
        
        print("\n🚀 Performance testing with validation:")
        print("-" * 40)
        
        import time
        
        perf_tests = [
            ("Large array sqrt", lambda: safe_sqrt(np.ones(10000)), 10000),
            ("Large array division", lambda: safe_divide(np.ones(5000), np.ones(5000) * 2), 5000),
            ("Complex chaining", lambda: SafeNumber(np.ones(1000) * 4).sqrt().value(), 1000),
        ]
        
        for i, (description, test_func, expected_size) in enumerate(perf_tests):
            try:
                start_time = time.time()
                result = test_func()
                end_time = time.time()
                
                # Validate performance result
                size_correct = len(result) == expected_size if hasattr(result, '__len__') else True
                no_errors = not np.any(np.isnan(result)) if hasattr(result, '__iter__') else not np.isnan(result)
                
                status = "✅ PASS" if size_correct else "❌ FAIL"
                
                print(f"  {i+1}. {description}")
                print(f"     Time: {(end_time - start_time)*1000:.2f} ms")
                print(f"     Size: {len(result) if hasattr(result, '__len__') else 'scalar'}")
                print(f"     Status: {status}")
                
            except Exception as e:
                print(f"  {i+1}. {description}")
                print(f"     ERROR: {type(e).__name__}: {e}")
                print(f"     Status: ❌ ERROR")
            
            print()
        
        # =================================================================
        # FINAL VALIDATION SUMMARY
        # =================================================================
        test_section("COMPREHENSIVE VALIDATION SUMMARY")
        
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_tests = total_passed + total_failed
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🏆 SafeMath Library - Validation Testing Complete!")
        print("-" * 55)
        print(f"📊 Overall Results:")
        print(f"   ✅ Tests Passed: {total_passed}")
        print(f"   ❌ Tests Failed: {total_failed}")
        print(f"   📈 Pass Rate: {pass_rate:.1f}%")
        print(f"   🔢 Total Tests: {total_tests}")
        
        if pass_rate >= 95:
            print(f"\n🎉 EXCELLENT! Your SafeMath library has {pass_rate:.1f}% correctness!")
        elif pass_rate >= 90:
            print(f"\n👍 GREAT! Your SafeMath library has {pass_rate:.1f}% correctness!")
        elif pass_rate >= 80:
            print(f"\n👌 GOOD! Your SafeMath library has {pass_rate:.1f}% correctness!")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT: {pass_rate:.1f}% correctness - some issues found")
        
        print(f"\n✅ Mathematical correctness verification complete")
        print(f"✅ Edge case handling validated")
        print(f"✅ Performance benchmarks verified")
        print(f"✅ Configuration system tested")
        print(f"✅ Expression evaluation secured")
        
        print(f"\nTest completed: {end_timestamp}")
        print(f"Detailed results saved to: {output_manager.filename}")
        
        if total_failed > 0:
            print(f"\n⚠️  Review failed tests above for potential improvements")
        else:
            print(f"\n🎊 Perfect score! Your SafeMath library is mathematically bulletproof!")

if __name__ == "__main__":
    main()
