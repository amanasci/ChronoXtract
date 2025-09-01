#!/usr/bin/env python3
"""
CARMA Test Runner

Runs all CARMA-related tests in the tests directory.
"""

import subprocess
import sys
import os

def run_test(test_file):
    """Run a single test file and return the result."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)

    try:
        result = subprocess.run([sys.executable, test_file],
                              capture_output=True, text=True, cwd=os.path.dirname(test_file))
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False

def main():
    """Run all CARMA tests."""
    print("🚀 CARMA Test Suite Runner")
    print("=" * 60)

    # Get the tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    # List of CARMA test files to run
    carma_tests = [
        'test_carma_basic.py',
        'test_carma_comprehensive.py',
        'test_carma_validation.py',
        'test_carma_benchmark.py',
        'test_carma_performance.py',
        'test_carma_mcmc.py'
    ]

    results = []
    for test_file in carma_tests:
        test_path = os.path.join(tests_dir, test_file)
        if os.path.exists(test_path):
            success = run_test(test_path)
            results.append((test_file, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((test_file, False))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)

    passed = 0
    total = len(results)

    for test_file, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file}: {status}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All CARMA tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
