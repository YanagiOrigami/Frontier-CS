class Solution:
    def solve(self, src_path: str) -> bytes:
        return b'\x00' + b'\xff' * 132
