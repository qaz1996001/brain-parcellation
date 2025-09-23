#!/usr/bin/env python
"""
Code quality check script using UV tools.

This script runs all code quality tools and generates a report.
"""
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def run_command(cmd: List[str], check: bool = False) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")


def print_result(tool: str, success: bool, message: str = "") -> None:
    """Print the result of a tool run."""
    status = f"{GREEN}✓ PASSED{RESET}" if success else f"{RED}✗ FAILED{RESET}"
    print(f"{tool}: {status}")
    if message:
        print(f"  {message}")


async def check_ruff() -> bool:
    """Run Ruff linter."""
    print_section("Running Ruff Linter")
    
    # Check code
    code, stdout, stderr = run_command(["uv", "run", "ruff", "check", "code_ai", "backend"])
    
    if code == 0:
        print_result("Ruff Check", True)
    else:
        print_result("Ruff Check", False)
        print(stdout)
        
    # Check formatting
    code, stdout, stderr = run_command(["uv", "run", "ruff", "format", "--check", "code_ai", "backend"])
    
    if code == 0:
        print_result("Ruff Format", True)
        return True
    else:
        print_result("Ruff Format", False, "Run 'uv run ruff format code_ai backend' to fix")
        print(stdout)
        return False


async def check_black() -> bool:
    """Run Black formatter."""
    print_section("Running Black Formatter")
    
    code, stdout, stderr = run_command(["uv", "run", "black", "--check", "code_ai", "backend"])
    
    if code == 0:
        print_result("Black", True)
        return True
    else:
        print_result("Black", False, "Run 'uv run black code_ai backend' to fix")
        print(stdout)
        return False


async def check_mypy() -> bool:
    """Run MyPy type checker."""
    print_section("Running MyPy Type Checker")
    
    code, stdout, stderr = run_command(["uv", "run", "mypy", "code_ai", "backend"])
    
    if code == 0:
        print_result("MyPy", True)
        return True
    else:
        print_result("MyPy", False)
        print(stdout)
        return False


async def check_bandit() -> bool:
    """Run Bandit security checker."""
    print_section("Running Bandit Security Check")
    
    code, stdout, stderr = run_command(["uv", "run", "bandit", "-r", "code_ai", "backend", "-ll"])
    
    if code == 0:
        print_result("Bandit", True)
        return True
    else:
        print_result("Bandit", False)
        print(stdout)
        return False


async def run_tests() -> bool:
    """Run pytest tests."""
    print_section("Running Tests")
    
    code, stdout, stderr = run_command(["uv", "run", "pytest", "-v", "--tb=short"])
    
    if code == 0:
        print_result("Tests", True)
        # Extract coverage info if available
        for line in stdout.split("\n"):
            if "TOTAL" in line and "%" in line:
                print(f"  Coverage: {line.strip()}")
        return True
    else:
        print_result("Tests", False)
        print(stdout)
        return False


async def check_code_complexity() -> bool:
    """Check code complexity metrics."""
    print_section("Checking Code Complexity")
    
    # Check for deeply nested code (Linus style)
    issues = []
    
    for path in ["code_ai", "backend"]:
        for py_file in Path(path).rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            max_indent = 0
            for i, line in enumerate(lines):
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    indent_level = indent // 4
                    if indent_level > max_indent:
                        max_indent = indent_level
                    
                    if indent_level > 3:
                        issues.append(f"{py_file}:{i+1} - {indent_level} levels of nesting")
    
    if not issues:
        print_result("Code Complexity", True, "No deep nesting found")
        return True
    else:
        print_result("Code Complexity", False, f"Found {len(issues)} files with deep nesting")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  {YELLOW}{issue}{RESET}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")
        return False


async def main() -> int:
    """Run all code quality checks."""
    print(f"{BLUE}Code Quality Check Report{RESET}")
    print(f"{BLUE}========================={RESET}")
    
    # Run all checks
    results = await asyncio.gather(
        check_ruff(),
        check_black(),
        check_mypy(),
        check_bandit(),
        check_code_complexity(),
        run_tests(),
    )
    
    # Summary
    print_section("Summary")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"{GREEN}All checks passed! ({passed}/{total}){RESET}")
        return 0
    else:
        print(f"{RED}Some checks failed: {passed}/{total} passed{RESET}")
        print(f"\n{YELLOW}Fix the issues and run this script again.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
