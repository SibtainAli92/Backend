"""
Master test runner - Executes all tests and generates consolidated report.

This script runs all backend tests and logs results to backend_test_report.log
"""

import sys
import os
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_logger import test_logger


def run_all_tests():
    """Run all backend tests and generate report."""

    test_logger.logger.info("Starting comprehensive backend testing...")
    test_logger.logger.info("")

    # Test files in execution order
    test_files = [
        "tests/test_connection.py",
        "tests/test_models.py",
        "tests/test_rag_tool.py",
        "tests/test_agent.py",
        "tests/test_api.py"
    ]

    # Run pytest with custom options
    pytest_args = [
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "--disable-warnings",  # Cleaner output
        *test_files
    ]

    # Execute tests
    test_logger.logger.info("Executing pytest with all test files...")
    exit_code = pytest.main(pytest_args)

    test_logger.logger.info("")
    test_logger.logger.info("=" * 80)
    test_logger.logger.info("TEST EXECUTION COMPLETE")
    test_logger.logger.info("=" * 80)

    # Generate summary
    summary = test_logger.generate_summary()

    # Print final status
    if summary['failed'] == 0:
        test_logger.logger.info("")
        test_logger.logger.info("✓ ALL TESTS PASSED!")
        test_logger.logger.info("")
    else:
        test_logger.logger.info("")
        test_logger.logger.info(f"✗ {summary['failed']} TESTS FAILED")
        test_logger.logger.info("")

    return exit_code


if __name__ == "__main__":
    sys.exit(run_all_tests())
