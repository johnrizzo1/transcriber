#!/usr/bin/env python3
"""
Performance test runner script.

This script runs the complete performance test suite and generates reports.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_performance_tests(
    test_type: str = "all",
    verbose: bool = False,
    generate_report: bool = True,
    baseline_file: str = "performance_baseline.txt"
):
    """Run performance tests with specified configuration."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add performance test markers
    if test_type == "all":
        cmd.extend(["-m", "performance"])
    elif test_type == "stress":
        cmd.extend(["-m", "stress"])
    elif test_type == "regression":
        cmd.extend(["-m", "regression"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    else:
        cmd.extend(["-m", test_type])
    
    # Add test directories
    cmd.extend([
        "tests/performance/",
        "tests/integration/test_voice_pipeline_fixed.py"
    ])
    
    # Add options
    cmd.extend([
        "--run-performance",
        f"--performance-baseline={baseline_file}",
        "--tb=short"
    ])
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if generate_report:
        cmd.extend([
            "--performance-report=performance_report.txt",
            "--junit-xml=performance_results.xml"
        ])
    
    # Add coverage if requested
    cmd.extend([
        "--cov=transcriber",
        "--cov-report=html:performance_coverage",
        "--cov-report=term-missing"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AI Voice Agent performance tests"
    )
    
    parser.add_argument(
        "--type",
        choices=["all", "stress", "regression", "integration", "benchmarks"],
        default="all",
        help="Type of performance tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating performance report"
    )
    
    parser.add_argument(
        "--baseline",
        default="performance_baseline.txt",
        help="Path to performance baseline file"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick performance tests only (skip slow tests)"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("transcriber").exists():
        print("Error: Must run from project root directory")
        return 1
    
    # Run the tests
    exit_code = run_performance_tests(
        test_type=args.type,
        verbose=args.verbose,
        generate_report=not args.no_report,
        baseline_file=args.baseline
    )
    
    if exit_code == 0:
        print("\n✅ Performance tests completed successfully!")
        
        if not args.no_report and Path("performance_report.txt").exists():
            print("\n📊 Performance Report:")
            print("-" * 40)
            with open("performance_report.txt") as f:
                print(f.read())
    else:
        print(f"\n❌ Performance tests failed (exit code: {exit_code})")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())