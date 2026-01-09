import os
import re
import tarfile
import struct
import ast
from typing import Dict, Optional, Iterable, Tuple, Union


def _strip_c_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//.*", "", s)
    return s


_CAST_RE = re.compile(
    r"\(\s*(?:const\s+)?(?:volatile\s+)?(?:unsigned\s+)?"
    r"(?:long\s+long|long|short|int|char|size_t|ssize_t|ptrdiff_t|"
    r"guint(?:8|16|32|64)?|gint(?:8|16|32|64)?|"
    r"uint(?:8|16|32|64)?_t|int(?:8|16|32|64)?_t|"
    r"u_int(?:8|16|32|64)?_t|gsize|gssize|gboolean|guchar|gushort|guint|gint)\s*\)"
)


def _sanitize_c_expr(expr: str) -> str:
    expr = expr.strip()
    expr = _strip_c_comments(expr).strip()
    if not expr:
        return expr

    # Remove line continuations
    expr = expr.replace("\\\n", " ").replace("\\\r\n", " ")

    # Remove common cast patterns
    for _ in range(8):
        new_expr = _CAST_RE.sub("", expr)
        if new_expr == expr:
            break
        expr = new_expr.strip()

    # Replace known constant wrappers
    wrappers = [
        "G_GUINT64_CONSTANT",
        "G_GINT64_CONSTANT",
        "G_GUINT32_CONSTANT",
        "G_GINT32_CONSTANT",
        "G_GUINT16_CONSTANT",
        "G_GINT16_CONSTANT",
        "G_GUINT8_CONSTANT",
        "G_GINT8_CONSTANT",
    ]
    for w in wrappers:
        expr = re.sub(r"\b" + re.escape(w) + r"\s*\(", "(", expr)

    # Replace common literals
    expr = re.sub(r"\bNULL\b", "0", expr)
    expr = re.sub(r"\bTRUE\b", "1", expr)
    expr = re.sub(r"\bFALSE\b", "0", expr)

    # Remove integer suffixes
    expr = re.sub(r"(?i)\b(0x[0-9a-f]+|\d+)\s*([u]|ul|lu|ull|llu|l)\b", r"\1", expr)

    # Convert C logical ops if present (rare in macros for constants)
    expr = expr.replace("&&", " and ").replace("||", " or ")

    # Handle unary ! (very rare); convert to 'not'
    expr = re.sub(r"!\s*", " not ", expr)

    return expr.strip()


class _ConstResolver:
    def __init__(self) -> None:
        self.exprs: Dict[str, str] = {}
        self.values: Dict[str, int] = {}
        self._resolving: Dict[str, bool] = {}

    def add_defines_and_enums_from_text(self, text: str) -> None:
        if not text:
            return
        raw = text

        # Capture simple #define NAME EXPR (without parameters)
        for m in re.finditer(r"(?m)^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", raw):
            name = m.group(1)
            rest = m.group(2)
            if "(" in name:
                continue
            if re.match(r"^[A-Za-z_]\w*\s*\(", rest):
                # likely macro with args used as expression; skip
                continue
            rest = _sanitize_c_expr(rest)
            if rest:
                self.exprs.setdefault(name, rest)

        # Capture enums
        no_comments = _strip_c_comments(raw)
        for m in re.finditer(r"(?s)\benum\b[^;{]*\{(.*?)\}", no_comments):
            body = m.group(1)
            parts = body.split(",")
            cur_val = -1
            for part in parts:
                item = part.strip()
                if not item:
                    continue
                mm = re.match(r"^([A-Za-z_]\w*)\s*(?:=\s*(.+))?$", item, flags=re.S)
                if not mm:
                    continue
                name = mm.group(1)
                rhs = mm.group(2)
                if rhs is None:
                    cur_val += 1
                    self.values.setdefault(name, cur_val)
                else:
                    rhs = _sanitize_c_expr(rhs)
                    v = self.eval_expr(rhs)
                    if v is None:
                        continue
                    cur_val = v
                    self.values.setdefault(name, v)

    def resolve_name(self, name: str) -> Optional[int]:
        if name in self.values:
            return self.values[name]
        if name not in self.exprs:
            return None
        if self._resolving.get(name, False):
            return None
        self._resolving[name] = True
        v = self.eval_expr(self.exprs[name])
        if v is not None:
            self.values[name] = v
        self._resolving[name] = False
        return v

    def eval_expr(self, expr: str) -> Optional[int]:
        expr = _sanitize_c_expr(expr)
        if not expr:
            return None

        # Fast path for plain integers
        try:
            if re.fullmatch(r"[+-]?\d+", expr) or re.fullmatch(r"[+-]?0x[0-9a-fA-F]+", expr):
                return int(expr, 0)
        except Exception:
            pass

        # Strip outer parentheses
        for _ in range(4):
            if expr.startswith("(") and expr.endswith(")"):
                inner = expr[1:-1].strip()
                if inner.count("(") == inner.count(")"):
                    expr = inner
                else:
                    break
            else:
                break

        try:
            node = ast.parse(expr, mode="eval")
        except Exception:
            return None

        def _eval(n: ast.AST) -> int:
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            if isinstance(n, ast.Constant):
                if isinstance(n.value, bool):
                    return 1 if n.value else 0
                if isinstance(n.value, int):
                    return int(n.value)
                if isinstance(n.value, str):
                    if len(n.value) == 1:
                        return ord(n.value)
                    raise ValueError("string constant not supported")
                raise ValueError("unsupported constant")
            if isinstance(n, ast.Name):
                v = self.resolve_name(n.id)
                if v is None:
                    raise ValueError(f"unknown name {n.id}")
                return int(v)
            if isinstance(n, ast.UnaryOp):
                v = _eval(n.operand)
                if isinstance(n.op, ast.UAdd):
                    return +v
                if isinstance(n.op, ast.USub):
                    return -v
                if isinstance(n.op, ast.Invert):
                    return ~v
                if isinstance(n.op, ast.Not):
                    return 0 if v else 1
                raise ValueError("unsupported unary op")
            if isinstance(n, ast.BinOp):
                a = _eval(n.left)
                b = _eval(n.right)
                op = n.op
                if isinstance(op, ast.Add):
                    return a + b
                if isinstance(op, ast.Sub):
                    return a - b
                if isinstance(op, ast.Mult):
                    return a * b
                if isinstance(op, ast.FloorDiv) or isinstance(op, ast.Div):
                    if b == 0:
                        return 0
                    return int(a // b)
                if isinstance(op, ast.Mod):
                    if b == 0:
                        return 0
                    return a % b
                if isinstance(op, ast.LShift):
                    return a << b
                if isinstance(op, ast.RShift):
                    return a >> b
                if isinstance(op, ast.BitOr):
                    return a | b
                if isinstance(op, ast.BitAnd):
                    return a & b
                if isinstance(op, ast.BitXor):
                    return a ^ b
                raise ValueError("unsupported bin op")
            if isinstance(n, ast.BoolOp):
                if isinstance(n.op, ast.And):
                    cur = 1
                    for v in n.values:
                        cur = 1 if (_eval(v) != 0 and cur != 0) else 0
                    return cur
                if isinstance(n.op, ast.Or):
                    cur = 0
                    for v in n.values:
                        cur = 1 if (_eval(v) != 0 or cur != 0) else 0
                    return cur
                raise ValueError("unsupported bool op")
            if isinstance(n, ast.Compare):
                left = _eval(n.left)
                result = True
                for op, comp in zip(n.ops, n.comparators):
                    right = _eval(comp)
                    if isinstance(op, ast.Eq):
                        ok = left == right
                    elif isinstance(op, ast.NotEq):
                        ok = left != right
                    elif isinstance(op, ast.Lt):
                        ok = left < right
                    elif isinstance(op, ast.LtE):
                        ok = left <= right
                    elif isinstance(op, ast.Gt):
                        ok = left > right
                    elif isinstance(op, ast.GtE):
                        ok = left >= right
                    else:
                        raise ValueError("unsupported compare")
                    result = result and ok
                    left = right
                return 1 if result else 0
            raise ValueError("unsupported AST node")

        try:
            return int(_eval(node))
        except Exception:
            return None


class _SourceIndex:
    def __init__(self, src_path: str) -> None:
        self.src_path = src_path
        self.is_dir = os.path.isdir(src_path)
        self.tar: Optional[tarfile.TarFile] = None
        self.tar_members: Optional[Dict[str, tarfile.TarInfo]] = None

        if not self.is_dir:
            try:
                self.tar = tarfile.open(src_path, "r:*")
                self.tar_members = {m.name: m for m in self.tar.getmembers() if m.isfile()}
            except Exception:
                self.tar = None
                self.tar_members = None

    def close(self) -> None:
        if self.tar is not None:
            try:
                self.tar.close()
            except Exception:
                pass
            self.tar = None
            self.tar_members = None

    def _iter_paths(self) -> Iterable[str]:
        if self.is_dir:
            for root, _, files in os.walk(self.src_path):
                for fn in files:
                    yield os.path.join(root, fn)
        else:
            if not self.tar_members:
                return
            for name in self.tar_members.keys():
                yield name

    def _basename(self, path: str) -> str:
        return os.path.basename(path)

    def _read_file_bytes(self, path: str, max_size: int = 8 * 1024 * 1024) -> Optional[bytes]:
        try:
            if self.is_dir:
                if os.path.getsize(path) > max_size:
                    return None
                with open(path, "rb") as f:
                    return f.read()
            else:
                if not self.tar or not self.tar_members:
                    return None
                ti = self.tar_members.get(path)
                if ti is None or ti.size > max_size:
                    return None
                f = self.tar.extractfile(ti)
                if f is None:
                    return None
                return f.read()
        except Exception:
            return None

    def _read_text_by_basename(self, basenames: Iterable[str], max_size: int = 8 * 1024 * 1024) -> Dict[str, str]:
        want = set(basenames)
        found: Dict[str, str] = {}
        for p in self._iter_paths():
            b = self._basename(p)
            if b not in want:
                continue
            data = self._read_file_bytes(p, max_size=max_size)
            if not data:
                continue
            try:
                txt = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            found[p] = txt
            # don't early-break; multiple variants could exist
        return found

    def _scan_text_files_for_substring(
        self,
        sub: bytes,
        path_hint_substrings: Tuple[str, ...] = (),
        exts: Tuple[str, ...] = (".c", ".h"),
        max_files: int = 200,
        max_size: int = 4 * 1024 * 1024,
    ) -> Dict[str, str]:
        found: Dict[str, str] = {}
        scanned = 0
        for p in self._iter_paths():
            if scanned >= max_files:
                break
            if self.is_dir:
                rel = os.path.relpath(p, self.src_path)
            else:
                rel = p
            low = rel.lower()
            if exts and not any(low.endswith(e) for e in exts):
                continue
            if path_hint_substrings and not any(h in low for h in path_hint_substrings):
                continue
            data = self._read_file_bytes(p, max_size=max_size)
            if not data:
                continue
            scanned += 1
            if sub not in data:
                continue
            try:
                txt = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            found[rel] = txt
        return found

    def _build_resolver_from_texts(self, texts: Iterable[str]) -> _ConstResolver:
        r = _ConstResolver()
        for t in texts:
            r.add_defines_and_enums_from_text(t)
        return r

    def _extract_gre_wlan_keyexpr_from_text(self, txt: str) -> Optional[str]:
        if not txt or "gre.proto" not in txt:
            return None

        patterns = [
            r"\bdissector_add_uint_with_preference\s*\(\s*\"gre\.proto\"\s*,\s*([^,]+)\s*,\s*([^)]+)\)",
            r"\bdissector_add_uint\s*\(\s*\"gre\.proto\"\s*,\s*([^,]+)\s*,\s*([^)]+)\)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, txt):
                keyexpr = m.group(1).strip()
                third = m.group(2)
                ctx = txt[max(0, m.start() - 120): m.end() + 120]
                if re.search(r"(802\.?11|802_11|ieee80211|wlan)", third, flags=re.I) or re.search(
                    r"(802\.?11|802_11|ieee80211|wlan)", ctx, flags=re.I
                ):
                    keyexpr = keyexpr.strip()
                    keyexpr = re.sub(r"^\((?:[^()]+)\)\s*", "", keyexpr).strip()
                    return keyexpr
        return None

    def get_gre_wlan_ptype(self) -> Optional[int]:
        # First: read known likely dissector files
        basenames = [
            "packet-gre.c",
            "packet-gre.h",
            "packet-ieee80211.c",
            "packet-ieee802_11.c",
            "packet-wlan.c",
            "packet-wlan.h",
            "packet-ieee80211.h",
            "packet-ieee802_11.h",
        ]
        texts_map = self._read_text_by_basename(basenames)
        texts = list(texts_map.values())

        # If not found, scan dissectors for gre.proto occurrences
        keyexpr = None
        for t in texts:
            keyexpr = self._extract_gre_wlan_keyexpr_from_text(t)
            if keyexpr:
                break

        if not keyexpr:
            more = self._scan_text_files_for_substring(
                b"gre.proto",
                path_hint_substrings=("epan", "dissectors", "packet-"),
                exts=(".c", ".h"),
                max_files=800,
            )
            for t in more.values():
                texts.append(t)
                keyexpr = self._extract_gre_wlan_keyexpr_from_text(t)
                if keyexpr:
                    break

        # Fallback: find a number near "802.11" in a GRE context
        if not keyexpr:
            for t in texts:
                for m in re.finditer(r"(0x[0-9a-fA-F]{1,6}|\d{1,6}).{0,80}(?:802\.?11|wlan|ieee\s*802\.?11)", t, flags=re.I | re.S):
                    keyexpr = m.group(1)
                    break
                if keyexpr:
                    break

        if not keyexpr:
            return None

        resolver = self._build_resolver_from_texts(texts)

        v = resolver.eval_expr(keyexpr)
        if v is not None:
            return v & 0xFFFF

        # If it's an identifier, try to locate its definition by scanning for #define/enum
        name_match = re.fullmatch(r"[A-Za-z_]\w*", keyexpr)
        if name_match:
            name = keyexpr
            # scan more headers for this identifier
            if self.is_dir:
                for p in self._iter_paths():
                    low = p.lower()
                    if not (low.endswith(".h") or low.endswith(".c")):
                        continue
                    data = self._read_file_bytes(p, max_size=2 * 1024 * 1024)
                    if not data:
                        continue
                    if name.encode("utf-8") not in data:
                        continue
                    try:
                        txt = data.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if f"#define {name}" in txt or re.search(r"\benum\b", txt):
                        resolver.add_defines_and_enums_from_text(txt)
                        vv = resolver.resolve_name(name)
                        if vv is not None:
                            return vv & 0xFFFF
            else:
                if self.tar_members:
                    needle = name.encode("utf-8")
                    for mp, ti in self.tar_members.items():
                        low = mp.lower()
                        if not (low.endswith(".h") or low.endswith(".c")):
                            continue
                        if ti.size > 2 * 1024 * 1024:
                            continue
                        data = self._read_file_bytes(mp, max_size=2 * 1024 * 1024)
                        if not data or needle not in data:
                            continue
                        try:
                            txt = data.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                        if f"#define {name}" in txt or re.search(r"\benum\b", txt):
                            resolver.add_defines_and_enums_from_text(txt)
                            vv = resolver.resolve_name(name)
                            if vv is not None:
                                return vv & 0xFFFF

        return None

    def get_linktype_gre(self) -> Optional[int]:
        basenames = [
            "pcap-common.h",
            "pcap-common.c",
            "libpcap.c",
            "libpcap.h",
            "pcap.h",
            "pcapng.c",
            "wtap.h",
            "pcapio.c",
        ]
        texts_map = self._read_text_by_basename(basenames)
        texts = list(texts_map.values())

        def _extract_from_text(t: str) -> Optional[int]:
            if not t:
                return None
            m = re.search(r"(?m)^\s*#\s*define\s+LINKTYPE_GRE\s+(.+?)\s*$", t)
            if m:
                return _ConstResolver().eval_expr(m.group(1))
            m = re.search(r"(?m)^\s*#\s*define\s+DLT_GRE\s+(.+?)\s*$", t)
            if m:
                return _ConstResolver().eval_expr(m.group(1))
            m = re.search(r"\bLINKTYPE_GRE\b\s*=\s*(\d+)", t)
            if m:
                return int(m.group(1))
            m = re.search(r"\bDLT_GRE\b\s*=\s*(\d+)", t)
            if m:
                return int(m.group(1))
            # Mapping tables e.g. { 778, WTAP_ENCAP_GRE, ... }
            m = re.search(r"\{\s*(\d+)\s*,\s*WTAP_ENCAP_GRE\b", t)
            if m:
                return int(m.group(1))
            m = re.search(r"\{\s*(0x[0-9a-fA-F]+)\s*,\s*WTAP_ENCAP_GRE\b", t)
            if m:
                return int(m.group(1), 0)
            return None

        for t in texts:
            v = _extract_from_text(t)
            if v is not None:
                return int(v)

        # Scan wiretap for LINKTYPE_GRE if not found
        more = self._scan_text_files_for_substring(
            b"LINKTYPE_GRE",
            path_hint_substrings=("wiretap", "pcap"),
            exts=(".c", ".h"),
            max_files=500,
        )
        for t in more.values():
            v = _extract_from_text(t)
            if v is not None:
                return int(v)

        # Fallback: scan for WTAP_ENCAP_GRE mapping
        more2 = self._scan_text_files_for_substring(
            b"WTAP_ENCAP_GRE",
            path_hint_substrings=("wiretap", "pcap"),
            exts=(".c", ".h"),
            max_files=800,
        )
        for t in more2.values():
            v = _extract_from_text(t)
            if v is not None:
                return int(v)

        return None


def _build_pcap(linktype: int, pkt: bytes) -> bytes:
    ghdr = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, int(linktype) & 0xFFFFFFFF)
    phdr = struct.pack("<IIII", 0, 0, len(pkt), len(pkt))
    return ghdr + phdr + pkt


class Solution:
    def solve(self, src_path: str) -> bytes:
        idx = _SourceIndex(src_path)
        try:
            ptype = idx.get_gre_wlan_ptype()
            linktype = idx.get_linktype_gre()
        finally:
            idx.close()

        if linktype is None:
            linktype = 778  # common LINKTYPE_GRE value in libpcap

        if ptype is None:
            # Best-effort fallback if parsing failed; unlikely to work, but keep valid GRE structure
            ptype = 0x0000

        gre_flags_version = 0x0000
        gre_pkt = struct.pack("!HHB", gre_flags_version & 0xFFFF, ptype & 0xFFFF, 0x00)
        return _build_pcap(linktype, gre_pkt)