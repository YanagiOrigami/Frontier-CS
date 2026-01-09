import os
import re
import struct
import tarfile
import ast
from typing import Dict, Optional, Iterable, Tuple, List


_MAX_READ = 2_000_000


def _strip_c_comments(s: str) -> str:
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = re.sub(r'//.*', '', s)
    return s


def _parse_defines(text: str, macros: Dict[str, str]) -> None:
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        m = re.match(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.*)$', line)
        if not m:
            continue
        name = m.group(1)
        expr = m.group(2).strip()
        expr = _strip_c_comments(expr).strip()
        if not expr:
            continue
        if len(name) < 3:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_]+', name):
            continue
        # Skip function-like macros
        if "(" in name:
            continue
        # Keep mostly relevant constants, but allow wide net
        if any(name.startswith(p) for p in ("LINKTYPE_", "DLT_", "ETHERTYPE_", "ETH_P_", "GRE_", "WTAP_", "PPP_", "LLC_")) or name in (
            "LINKTYPE_GRE",
            "DLT_GRE",
            "ETH_P_80211_RAW",
            "ETHERTYPE_IEEE802_11",
            "ETHERTYPE_IEEE802_11_RAW",
            "ETHERTYPE_IEEE802_11_RADIOTAP",
        ):
            macros.setdefault(name, expr)


def _safe_eval_int(expr: str) -> Optional[int]:
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None

    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.FloorDiv,
        ast.Div,
        ast.Mod,
        ast.BitOr,
        ast.BitAnd,
        ast.BitXor,
        ast.LShift,
        ast.RShift,
        ast.Invert,
        ast.UAdd,
        ast.USub,
        ast.Constant,
        ast.Num,
        ast.ParenExpr if hasattr(ast, "ParenExpr") else ast.AST,
    )

    def check(n: ast.AST) -> bool:
        if isinstance(n, ast.Expression):
            return check(n.body)
        if isinstance(n, (ast.Num, ast.Constant)):
            v = getattr(n, "n", None)
            if v is None:
                v = getattr(n, "value", None)
            return isinstance(v, int)
        if isinstance(n, ast.UnaryOp):
            return isinstance(n.op, (ast.Invert, ast.UAdd, ast.USub)) and check(n.operand)
        if isinstance(n, ast.BinOp):
            if not isinstance(
                n.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.FloorDiv,
                    ast.Div,
                    ast.Mod,
                    ast.BitOr,
                    ast.BitAnd,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                ),
            ):
                return False
            return check(n.left) and check(n.right)
        return False

    if not check(node):
        return None
    try:
        val = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})
        if isinstance(val, int):
            return val
    except Exception:
        return None
    return None


_CAST_RE = re.compile(
    r'\(\s*(?:'
    r'(?:g?u?int(?:8|16|32|64)_t)|'
    r'(?:guint(?:8|16|32|64))|'
    r'(?:gint(?:8|16|32|64))|'
    r'(?:uint(?:8|16|32|64)_t)|'
    r'(?:unsigned\s+long\s+long)|'
    r'(?:long\s+long)|'
    r'(?:unsigned\s+long)|'
    r'(?:unsigned\s+int)|'
    r'(?:unsigned\s+short)|'
    r'(?:unsigned)|'
    r'(?:long)|'
    r'(?:int)|'
    r'(?:short)|'
    r'(?:size_t)|'
    r'(?:uintptr_t)|'
    r'(?:intptr_t)'
    r')\s*\)'
)


def _normalize_c_int_expr(expr: str) -> str:
    expr = _strip_c_comments(expr).strip()
    expr = _CAST_RE.sub("", expr)
    expr = expr.replace("UL", "").replace("LU", "").replace("ULL", "").replace("LLU", "").replace("LL", "").replace("U", "").replace("L", "")
    expr = expr.strip()
    return expr


def _eval_macro(name: str, macros: Dict[str, str], cache: Dict[str, int], visiting: Optional[set] = None) -> Optional[int]:
    if name in cache:
        return cache[name]
    if name not in macros:
        return None
    if visiting is None:
        visiting = set()
    if name in visiting:
        return None
    visiting.add(name)
    expr = _normalize_c_int_expr(macros[name])

    # Direct literal
    if re.fullmatch(r'0[xX][0-9a-fA-F]+', expr):
        v = int(expr, 16)
        cache[name] = v
        visiting.remove(name)
        return v
    if re.fullmatch(r'[0-9]+', expr):
        v = int(expr, 10)
        cache[name] = v
        visiting.remove(name)
        return v

    def repl(m: re.Match) -> str:
        ident = m.group(0)
        if ident == name:
            return "0"
        if ident in ("sizeof",):
            return "0"
        if ident in macros:
            v = _eval_macro(ident, macros, cache, visiting)
            if v is not None:
                return str(v)
        return "0"

    expr2 = re.sub(r'\b[A-Za-z_]\w*\b', repl, expr)
    expr2 = re.sub(r'(?<=\d)[uUlL]+', '', expr2)
    expr2 = expr2.strip()
    v = _safe_eval_int(expr2)
    if v is not None:
        cache[name] = v
    visiting.remove(name)
    return v


def _parse_int_expr(expr: str, macros: Dict[str, str], cache: Dict[str, int]) -> Optional[int]:
    expr = _normalize_c_int_expr(expr)
    if re.fullmatch(r'0[xX][0-9a-fA-F]+', expr):
        return int(expr, 16)
    if re.fullmatch(r'[0-9]+', expr):
        return int(expr, 10)
    if re.fullmatch(r'[A-Za-z_]\w*', expr):
        return _eval_macro(expr, macros, cache)
    # Try evaluate with macro substitution
    def repl(m: re.Match) -> str:
        ident = m.group(0)
        if ident in ("sizeof",):
            return "0"
        if ident in macros:
            v = _eval_macro(ident, macros, cache)
            if v is not None:
                return str(v)
        return "0"
    expr2 = re.sub(r'\b[A-Za-z_]\w*\b', repl, expr)
    expr2 = re.sub(r'(?<=\d)[uUlL]+', '', expr2)
    return _safe_eval_int(expr2)


def _ipv4_checksum(hdr: bytes) -> int:
    if len(hdr) % 2:
        hdr += b"\x00"
    s = 0
    for i in range(0, len(hdr), 2):
        s += (hdr[i] << 8) | hdr[i + 1]
    s = (s & 0xFFFF) + (s >> 16)
    s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _build_pcap(linktype: int, packet: bytes) -> bytes:
    gh = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 0xFFFF, linktype & 0xFFFFFFFF)
    ph = struct.pack("<IIII", 0, 0, len(packet), len(packet))
    return gh + ph + packet


def _build_ether_ipv4_gre_packet(gre_proto: int, gre_payload: bytes) -> bytes:
    eth_dst = b"\x00\x11\x22\x33\x44\x55"
    eth_src = b"\x66\x77\x88\x99\xaa\xbb"
    ethertype_ipv4 = 0x0800
    eth = eth_dst + eth_src + struct.pack("!H", ethertype_ipv4)

    version_ihl = 0x45
    tos = 0
    total_length = 20 + 4 + len(gre_payload)
    ident = 0
    flags_frag = 0
    ttl = 64
    proto_gre = 47
    csum = 0
    src_ip = 0x01010101
    dst_ip = 0x02020202

    ip_wo_csum = struct.pack("!BBHHHBBHII", version_ihl, tos, total_length, ident, flags_frag, ttl, proto_gre, csum, src_ip, dst_ip)
    csum = _ipv4_checksum(ip_wo_csum)
    ip = struct.pack("!BBHHHBBHII", version_ihl, tos, total_length, ident, flags_frag, ttl, proto_gre, csum, src_ip, dst_ip)

    gre_flags_ver = 0x0000
    gre = struct.pack("!HH", gre_flags_ver, gre_proto & 0xFFFF) + gre_payload
    return eth + ip + gre


def _iter_candidate_texts_from_tar(tar_path: str) -> Iterable[str]:
    keywords = (
        "packet-gre",
        "packet_ieee80211",
        "packet-ieee80211",
        "80211",
        "wlan",
        "gre",
        "pcap",
        "libpcap",
        "etypes",
        "linktype",
        "wiretap",
        "wtap",
    )
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            if m.size <= 0 or m.size > _MAX_READ:
                continue
            name_low = m.name.lower()
            if not (name_low.endswith(".c") or name_low.endswith(".h")):
                continue
            if not any(k in name_low for k in keywords):
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read(_MAX_READ + 1)
            except Exception:
                continue
            if len(data) > _MAX_READ:
                continue
            try:
                yield data.decode("utf-8", errors="ignore")
            except Exception:
                continue


def _iter_candidate_texts_from_dir(dir_path: str) -> Iterable[str]:
    keywords = (
        "packet-gre",
        "packet_ieee80211",
        "packet-ieee80211",
        "80211",
        "wlan",
        "gre",
        "pcap",
        "libpcap",
        "etypes",
        "linktype",
        "wiretap",
        "wtap",
    )
    for root, _, files in os.walk(dir_path):
        for fn in files:
            fn_low = fn.lower()
            if not (fn_low.endswith(".c") or fn_low.endswith(".h")):
                continue
            full = os.path.join(root, fn)
            full_low = full.lower()
            if not any(k in full_low for k in keywords):
                continue
            try:
                with open(full, "rb") as f:
                    data = f.read(_MAX_READ + 1)
            except Exception:
                continue
            if len(data) > _MAX_READ:
                continue
            try:
                yield data.decode("utf-8", errors="ignore")
            except Exception:
                continue


def _extract_gre_proto_for_80211(texts: List[str], macros: Dict[str, str], cache: Dict[str, int]) -> Optional[int]:
    patterns = [
        re.compile(r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([^,]+)\s*,\s*([^)]+)\)\s*;', re.S),
        re.compile(r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([^,]+)\s*,\s*([^)]+)\)\s*,', re.S),
    ]
    for t in texts:
        if "gre.proto" not in t:
            continue
        for pat in patterns:
            for m in pat.finditer(t):
                key_expr = m.group(1).strip()
                handle_expr = m.group(2)
                if re.search(r'(?:ieee\s*802\.?11|802\.?11|80211|wlan)', handle_expr, flags=re.I):
                    v = _parse_int_expr(key_expr, macros, cache)
                    if v is not None:
                        return v & 0xFFFF

    # Try to resolve via known macro names
    for macro_name in ("ETH_P_80211_RAW", "ETHERTYPE_IEEE802_11", "ETHERTYPE_IEEE802_11_RAW", "ETHERTYPE_IEEE_802_11", "ETHERTYPE_IEEE_802_11_RAW"):
        v = _eval_macro(macro_name, macros, cache)
        if v is not None:
            return v & 0xFFFF

    return None


def _extract_linktype_gre(macros: Dict[str, str], cache: Dict[str, int]) -> Optional[int]:
    for nm in ("LINKTYPE_GRE", "DLT_GRE"):
        v = _eval_macro(nm, macros, cache)
        if v is not None and 0 <= v <= 0xFFFFFFFF:
            return int(v)
    return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        macros: Dict[str, str] = {}
        cache: Dict[str, int] = {}
        texts: List[str] = []

        try:
            if os.path.isdir(src_path):
                for t in _iter_candidate_texts_from_dir(src_path):
                    texts.append(t)
            else:
                for t in _iter_candidate_texts_from_tar(src_path):
                    texts.append(t)
        except Exception:
            texts = []

        for t in texts:
            _parse_defines(t, macros)

        gre_proto_80211 = _extract_gre_proto_for_80211(texts, macros, cache)
        if gre_proto_80211 is None:
            gre_proto_80211 = 0x0019  # ETH_P_80211_RAW

        gre_payload = b"\x00"
        gre_packet = struct.pack("!HH", 0x0000, gre_proto_80211 & 0xFFFF) + gre_payload

        linktype_gre = _extract_linktype_gre(macros, cache)
        if linktype_gre is not None:
            return _build_pcap(linktype_gre, gre_packet)

        # Fallback to Ethernet/IPv4/GRE which is widely supported
        eth_ip_gre_packet = _build_ether_ipv4_gre_packet(gre_proto_80211, gre_payload)
        return _build_pcap(1, eth_ip_gre_packet)