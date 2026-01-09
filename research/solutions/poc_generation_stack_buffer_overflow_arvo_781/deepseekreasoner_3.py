import os
import tarfile
import tempfile
import subprocess
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the source directory (assuming it's the only top-level dir)
            entries = os.listdir(tmpdir)
            if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
                src_dir = os.path.join(tmpdir, entries[0])
            else:
                src_dir = tmpdir
            
            # Look for a build script or makefile
            build_script = None
            for fname in os.listdir(src_dir):
                if fname in ['build.sh', 'Makefile', 'makefile']:
                    build_script = os.path.join(src_dir, fname)
                    break
            
            # Build the vulnerable program
            build_dir = src_dir
            if build_script:
                if build_script.endswith('.sh'):
                    subprocess.run(['/bin/bash', build_script], cwd=build_dir, capture_output=True)
                else:  # Makefile
                    subprocess.run(['make'], cwd=build_dir, capture_output=True)
            
            # Find the vulnerable executable
            exe_path = None
            for root, dirs, files in os.walk(build_dir):
                for f in files:
                    if os.access(os.path.join(root, f), os.X_OK) and not f.endswith('.so'):
                        # Quick test to see if it's the right program
                        test_path = os.path.join(root, f)
                        try:
                            result = subprocess.run([test_path], input=b'test', 
                                                  capture_output=True, timeout=1)
                            # If it runs without immediate crash, assume it's the target
                            if result.returncode == 0 or result.returncode > 1:
                                exe_path = test_path
                                break
                        except:
                            continue
                if exe_path:
                    break
            
            if not exe_path:
                # Last resort: look for any executable
                for root, dirs, files in os.walk(build_dir):
                    for f in files:
                        if os.access(os.path.join(root, f), os.X_OK):
                            exe_path = os.path.join(root, f)
                            break
                    if exe_path:
                        break
            
            # The vulnerability description mentions:
            # "fewer capturing parentheses than the available space in the external ovector"
            # This suggests a regular expression with capturing groups where the ovector
            # (output vector) has more slots than actual captures.
            # We need to create input that causes reading from invalid memory.
            
            # Based on the ground-truth length of 8 bytes, we'll try different
            # 8-byte patterns that could trigger this condition
            
            # Try a simple pattern with mismatched capturing groups
            # Common regex engines use parentheses for capturing
            # An 8-byte input like "(()))())" could cause issues
            candidates = [
                b"(()))())",  # Unbalanced parentheses
                b"(a)(b)(c)",  # 3 capturing groups but might overflow small buffer
                b"\x00" * 8,   # Null bytes might cause issues
                b"(((((((((",  # Many opening parentheses
                b"))))))))",   # Many closing parentheses
                b"(.)*" + b"a"*4,  # Pattern with repetition
            ]
            
            # Test each candidate
            for poc in candidates:
                try:
                    # Run with timeout
                    result = subprocess.run([exe_path], input=poc,
                                          capture_output=True, timeout=2)
                    # Check for crash (non-zero exit) and sanitizer error
                    if result.returncode != 0:
                        if (b'sanitizer' in result.stderr or 
                            b'overflow' in result.stderr or
                            b'invalid' in result.stderr or
                            b'ERROR' in result.stderr):
                            # This looks like a sanitizer-detected error
                            return poc
                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    continue
            
            # If none worked, return the first candidate as fallback
            return candidates[0]