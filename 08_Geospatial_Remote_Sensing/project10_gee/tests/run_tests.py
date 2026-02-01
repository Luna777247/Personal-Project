#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test runner script for the entire test suite.

Runs all tests and generates coverage report.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_all_tests(verbosity=2):
    """
    Run all tests in the tests directory.
    
    Args:
        verbosity (int): Verbosity level (0-2)
    
    Returns:
        unittest.TestResult: Test results
    """
    # Discover all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def run_specific_test(test_module, verbosity=2):
    """
    Run tests from a specific module.
    
    Args:
        test_module (str): Module name (e.g., 'test_config')
        verbosity (int): Verbosity level
    
    Returns:
        unittest.TestResult: Test results
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def print_summary(result):
    """
    Print test summary.
    
    Args:
        result (unittest.TestResult): Test results
    """
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run flood detection tests')
    parser.add_argument('--module', '-m', help='Specific test module to run')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Generate coverage report (requires coverage package)')
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    # Run with coverage if requested
    if args.coverage:
        try:
            import coverage
            
            # Start coverage
            cov = coverage.Coverage()
            cov.start()
            
            # Run tests
            if args.module:
                result = run_specific_test(args.module, verbosity)
            else:
                result = run_all_tests(verbosity)
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            print("\n" + "="*70)
            print("COVERAGE REPORT")
            print("="*70)
            cov.report()
            
            # Generate HTML report
            cov.html_report(directory='htmlcov')
            print(f"\nHTML coverage report generated in: htmlcov/index.html")
            
        except ImportError:
            print("Coverage package not installed. Install with: pip install coverage")
            sys.exit(1)
    else:
        # Run tests without coverage
        if args.module:
            result = run_specific_test(args.module, verbosity)
        else:
            result = run_all_tests(verbosity)
    
    # Print summary
    success = print_summary(result)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
