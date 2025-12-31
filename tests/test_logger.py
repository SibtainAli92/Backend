"""
Centralized Test Logger for Backend Testing.

All test results are logged to backend_test_report.log with timestamps,
function names, test results, and error messages.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class TestLogger:
    """Centralized logger for all backend tests."""

    def __init__(self, log_file: str = "backend_test_report.log"):
        """
        Initialize test logger.

        Args:
            log_file: Path to log file
        """
        self.log_file = Path(__file__).parent.parent / log_file
        self.test_results = []
        self.start_time = datetime.now()

        # Configure logger
        self.logger = logging.getLogger("BackendTestLogger")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Log test session start
        self.log_header()

    def log_header(self):
        """Log test session header."""
        self.logger.info("=" * 80)
        self.logger.info("BACKEND TEST REPORT - COMPREHENSIVE FUNCTION VERIFICATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Test Session Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log File: {self.log_file.absolute()}")
        self.logger.info("=" * 80)
        self.logger.info("")

    def log_test_start(self, module: str, function: str, test_name: str):
        """
        Log test start.

        Args:
            module: Module name (e.g., 'connection.py')
            function: Function name being tested
            test_name: Test case name
        """
        self.logger.info(f"▶ Testing: {module} → {function} → {test_name}")

    def log_test_pass(self, module: str, function: str, test_name: str, message: str = ""):
        """
        Log test pass.

        Args:
            module: Module name
            function: Function name
            test_name: Test case name
            message: Optional success message
        """
        result = {
            'timestamp': datetime.now(),
            'module': module,
            'function': function,
            'test': test_name,
            'result': 'PASS',
            'message': message
        }
        self.test_results.append(result)

        msg = f"✓ PASS: {module} → {function} → {test_name}"
        if message:
            msg += f" | {message}"
        self.logger.info(msg)

    def log_test_fail(self, module: str, function: str, test_name: str, error: str):
        """
        Log test failure.

        Args:
            module: Module name
            function: Function name
            test_name: Test case name
            error: Error message
        """
        result = {
            'timestamp': datetime.now(),
            'module': module,
            'function': function,
            'test': test_name,
            'result': 'FAIL',
            'message': error
        }
        self.test_results.append(result)

        self.logger.error(f"✗ FAIL: {module} → {function} → {test_name}")
        self.logger.error(f"  Error: {error}")

    def log_test_skip(self, module: str, function: str, test_name: str, reason: str):
        """
        Log test skip.

        Args:
            module: Module name
            function: Function name
            test_name: Test case name
            reason: Skip reason
        """
        result = {
            'timestamp': datetime.now(),
            'module': module,
            'function': function,
            'test': test_name,
            'result': 'SKIP',
            'message': reason
        }
        self.test_results.append(result)

        self.logger.warning(f"⊘ SKIP: {module} → {function} → {test_name}")
        self.logger.warning(f"  Reason: {reason}")

    def log_section(self, section_name: str):
        """
        Log section separator.

        Args:
            section_name: Section name
        """
        self.logger.info("")
        self.logger.info("-" * 80)
        self.logger.info(f"  {section_name}")
        self.logger.info("-" * 80)

    def generate_summary(self):
        """Generate and log final test summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['result'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['result'] == 'FAIL')
        skipped = sum(1 for r in self.test_results if r['result'] == 'SKIP')

        pass_rate = (passed / total * 100) if total > 0 else 0

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("TEST SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Tests:     {total}")
        self.logger.info(f"Passed:          {passed} ({pass_rate:.1f}%)")
        self.logger.info(f"Failed:          {failed}")
        self.logger.info(f"Skipped:         {skipped}")
        self.logger.info(f"Duration:        {duration.total_seconds():.2f} seconds")
        self.logger.info(f"End Time:        {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

        if failed > 0:
            self.logger.info("")
            self.logger.info("FAILED TESTS:")
            self.logger.info("-" * 80)
            for result in self.test_results:
                if result['result'] == 'FAIL':
                    self.logger.info(f"  • {result['module']} → {result['function']} → {result['test']}")
                    self.logger.info(f"    Error: {result['message']}")
            self.logger.info("=" * 80)

        self.logger.info("")
        self.logger.info(f"Full report saved to: {self.log_file.absolute()}")
        self.logger.info("")

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'pass_rate': pass_rate,
            'duration': duration.total_seconds()
        }


# Global test logger instance
test_logger = TestLogger()
