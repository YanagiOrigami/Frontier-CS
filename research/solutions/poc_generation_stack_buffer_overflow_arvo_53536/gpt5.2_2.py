import os
import re
import tarfile
import tempfile
import subprocess
import shutil
from typing import List, Tuple, Optional, Dict, Set


class Solution:
    _SRC_EXTS = {
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx",
        ".inc", ".inl", ".y", ".l", ".lex", ".yy"
    }

    def solve(self, src_path: str) -> bytes:
        try:
            with tempfile.TemporaryDirectory(prefix="pocgen_") as td:
                root = self._prepare_source_tree(src_path, td)
                files = self._read_sources(root)
                analysis = self._analyze(files)
                candidates = self._generate_payloads(analysis)
                best = self._try_dynamic(root, candidates)
                if best is not None:
                    return best
                return candidates[0] if candidates else (b"<A>" + b"A" * 2000 + b"\n")
        except Exception:
            return b"<A>" + b"A" * 2000 + b"\n"

    def _prepare_source_tree(self, src_path: str, td: str) -> str:
        if os.path.isdir(src_path):
            return src_path
        lower = src_path.lower()
        if lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    self._safe_extract(tf, td)
                return td
            except Exception:
                return td
        return td

    def _safe_extract(self, tf: tarfile.TarFile, path: str) -> None:
        base = os.path.realpath(path) + os.sep
        for m in tf.getmembers():
            name = m.name
            if not name or name.startswith(("/", "\\")):
                continue
            dest = os.path.realpath(os.path.join(path, name))
            if not dest.startswith(base):
                continue
            try:
                tf.extract(m, path)
            except Exception:
                continue

    def _read_sources(self, root: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in self._SRC_EXTS and fn not in ("Makefile", "CMakeLists.txt"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "rb") as f:
                        data = f.read(2_000_000)
                    text = data.decode("utf-8", errors="replace")
                    out.append((p, text))
                except Exception:
                    continue
        return out

    def _analyze(self, files: List[Tuple[str, str]]) -> Dict[str, object]:
        buf_candidates: List[Tuple[int, int, str]] = []
        str_lits: Dict[bytes, int] = {}
        tag_lits: Dict[bytes, int] = {}
        pair_tags: List[Tuple[bytes, bytes, int]] = []
        char_tokens: Dict[bytes, int] = {}
        magic: Dict[bytes, int] = {}

        buf_re = re.compile(r"\bchar\s+([A-Za-z_]\w*)\s*\[\s*(\d{2,6})\s*\]")
        str_re = re.compile(r'"((?:[^"\\]|\\.)*)"')
        tag_line_re = re.compile(r"\btag\b", re.IGNORECASE)
        pair_re = re.compile(r"\{\s*\"((?:[^\"\\]|\\.)*)\"\s*,\s*\"((?:[^\"\\]|\\.)*)*)\"\s*\}")
        # fix pair_re fallback due to nested quoting; use a safer multi-match below
        pair_re2 = re.compile(r"\{\s*\"((?:[^\"\\]|\\.)*)\"\s*,\s*\"((?:[^\"\\]|\\.)*)\"\s*\}")
        char_re = re.compile(r"==\s*'([^'\\]|\\.)'")
        memcmp_re = re.compile(r"\b(memcmp|strncmp)\s*\(\s*[^,]+,\s*\"((?:[^\"\\]|\\.)*)\"\s*,\s*(\d+)\s*\)")

        for path, text in files:
            for m in buf_re.finditer(text):
                name = m.group(1)
                try:
                    sz = int(m.group(2))
                except Exception:
                    continue
                if not (64 <= sz <= 65536):
                    continue
                lname = name.lower()
                weight = 0
                for kw in ("out", "output", "dst", "dest", "result", "res", "buf", "tmp"):
                    if kw in lname:
                        weight += 2
                buf_candidates.append((weight, sz, name))

            for sm in str_re.finditer(text):
                s_raw = sm.group(1)
                b = self._c_unescape_to_bytes(s_raw)
                if not b:
                    continue
                if b"\x00" in b:
                    continue
                if len(b) > 256:
                    continue
                if self._is_mostly_ascii_printable(b):
                    str_lits[b] = str_lits.get(b, 0) + 1

            for line in text.splitlines():
                if tag_line_re.search(line):
                    for sm in str_re.finditer(line):
                        b = self._c_unescape_to_bytes(sm.group(1))
                        if not b or b"\x00" in b or len(b) > 256:
                            continue
                        if self._is_mostly_ascii_printable(b):
                            tag_lits[b] = tag_lits.get(b, 0) + 2
                    for cm in char_re.finditer(line):
                        cb = self._c_unescape_to_bytes(cm.group(1))
                        if cb and cb != b"\x00":
                            char_tokens[cb] = char_tokens.get(cb, 0) + 1

            for pm in pair_re2.finditer(text):
                a = self._c_unescape_to_bytes(pm.group(1))
                b = self._c_unescape_to_bytes(pm.group(2))
                if not a or not b:
                    continue
                if b"\x00" in a or b"\x00" in b:
                    continue
                if len(a) > 128 or len(b) > 512:
                    continue
                if self._looks_like_tag(a):
                    rep_score = max(0, len(b) - len(a))
                    pair_tags.append((a, b, rep_score))

            for mm in memcmp_re.finditer(text):
                s = self._c_unescape_to_bytes(mm.group(2))
                try:
                    n = int(mm.group(3))
                except Exception:
                    n = 0
                if not s or b"\x00" in s:
                    continue
                if 2 <= len(s) <= 16 and n == len(s) and self._is_mostly_ascii_printable(s):
                    if re.fullmatch(rb"[A-Za-z0-9]{2,16}", s) and (s.isupper() or sum(c >= 65 and c <= 90 for c in s) >= len(s) * 0.8):
                        magic[s] = magic.get(s, 0) + 1

        bufsize = self._choose_bufsize(buf_candidates)

        all_tag_candidates: Dict[bytes, int] = {}
        for b, c in str_lits.items():
            if self._looks_like_tag(b):
                all_tag_candidates[b] = all_tag_candidates.get(b, 0) + c
        for b, c in tag_lits.items():
            all_tag_candidates[b] = all_tag_candidates.get(b, 0) + c
        for b, c in char_tokens.items():
            if len(b) == 1 and b in (b"<", b"{", b"[", b"$", b"%", b"@", b"#"):
                all_tag_candidates[b] = all_tag_candidates.get(b, 0) + c

        best_tag, best_rep_len = self._choose_best_tag(all_tag_candidates, pair_tags)
        best_tag = self._normalize_tag(best_tag)

        magic_best = None
        if magic:
            magic_best = max(magic.items(), key=lambda kv: kv[1])[0]
            if magic_best and len(magic_best) <= 16 and b"\x00" not in magic_best:
                magic_best = magic_best + b"\n"
            else:
                magic_best = None

        return {
            "bufsize": bufsize,
            "best_tag": best_tag,
            "best_rep_len": best_rep_len,
            "pair_tags": pair_tags,
            "tag_candidates": all_tag_candidates,
            "magic": magic_best,
        }

    def _choose_bufsize(self, buf_candidates: List[Tuple[int, int, str]]) -> int:
        if not buf_candidates:
            return 1024
        best = None
        for w, sz, name in buf_candidates:
            if sz < 128:
                continue
            score = (w * 100000) - sz
            if best is None or score > best[0]:
                best = (score, sz)
        if best is None:
            szs = sorted(sz for _, sz, _ in buf_candidates if 64 <= sz <= 8192)
            if szs:
                return szs[0]
            return 1024
        return best[1]

    def _choose_best_tag(self, candidates: Dict[bytes, int], pair_tags: List[Tuple[bytes, bytes, int]]) -> Tuple[bytes, int]:
        best_tag = b"<A>"
        best_rep = 0
        best_score = -10**18

        pair_best = None
        for a, b, rep_score in pair_tags:
            if not self._looks_like_tag(a):
                continue
            if not self._is_mostly_ascii_printable(a):
                continue
            if b"\n" in a or b"\r" in a or b"\t" in a:
                continue
            if len(a) < 1 or len(a) > 64:
                continue
            if rep_score <= 0:
                continue
            score = rep_score * 1000 + (64 - min(64, len(a))) * 2
            if pair_best is None or score > pair_best[0]:
                pair_best = (score, a, len(b))
        if pair_best is not None:
            _, t, rep_len = pair_best
            return t, rep_len

        for t, cnt in candidates.items():
            if not t or b"\x00" in t:
                continue
            if len(t) > 128:
                continue
            if not self._is_mostly_ascii_printable(t):
                continue
            if b"\n" in t or b"\r" in t or b"\t" in t:
                continue
            s = self._tag_score(t)
            score = s * 1000 + cnt * 50 - len(t) * 3
            if score > best_score:
                best_score = score
                best_tag = t
                best_rep = 0
        return best_tag, best_rep

    def _generate_payloads(self, analysis: Dict[str, object]) -> List[bytes]:
        bufsize = int(analysis.get("bufsize", 1024))
        best_tag = analysis.get("best_tag", b"<A>")
        if not isinstance(best_tag, (bytes, bytearray)):
            best_tag = b"<A>"
        best_tag = bytes(best_tag)
        magic = analysis.get("magic", None)
        if not isinstance(magic, (bytes, bytearray)) or not magic:
            magic = b""

        target_len = max(1600, bufsize + 600)
        target_len = min(9000, target_len)

        tags: List[bytes] = []
        tags.append(best_tag)

        pair_tags = analysis.get("pair_tags", [])
        if isinstance(pair_tags, list):
            tops: List[Tuple[int, bytes]] = []
            for a, b, rep_score in pair_tags[:200]:
                if not isinstance(a, (bytes, bytearray)):
                    continue
                aa = self._normalize_tag(bytes(a))
                if aa and aa not in tags and self._looks_like_tag(aa):
                    tops.append((int(rep_score), aa))
            tops.sort(key=lambda x: (-x[0], len(x[1])))
            for _, aa in tops[:4]:
                tags.append(aa)

        cand_map = analysis.get("tag_candidates", {})
        if isinstance(cand_map, dict):
            extra = sorted(
                ((cnt, self._normalize_tag(t)) for t, cnt in cand_map.items() if isinstance(t, (bytes, bytearray))),
                key=lambda x: (-x[0], len(x[1]) if x[1] else 9999),
            )
            for cnt, t in extra[:10]:
                if t and t not in tags and self._looks_like_tag(t):
                    tags.append(t)
                if len(tags) >= 7:
                    break

        normalized_tags = [t for t in tags if t]
        if not normalized_tags:
            normalized_tags = [b"<A>"]

        payloads: List[bytes] = []
        for t in normalized_tags[:6]:
            payloads.append(self._make_payload(magic, t, target_len))

        poly = self._make_polyglot_payload(magic, normalized_tags[:6], target_len)
        payloads.append(poly)

        uniq: List[bytes] = []
        seen: Set[bytes] = set()
        for p in payloads:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        uniq.sort(key=len)
        return uniq

    def _make_payload(self, magic: bytes, tag: bytes, target_len: int) -> bytes:
        if not tag:
            tag = b"<A>"
        tag = self._normalize_tag(tag)
        # Put tags first to ensure "tag found" path is hit early
        tag_budget = min(800, max(100, target_len // 3))
        rep = max(8, tag_budget // max(1, len(tag)))
        rep = min(rep, 2000)
        prefix = (tag * rep)[:tag_budget]
        remain = max(0, target_len - len(magic) - len(prefix) - 1)
        filler = b"A" * remain
        return magic + prefix + filler + b"\n"

    def _make_polyglot_payload(self, magic: bytes, tags: List[bytes], target_len: int) -> bytes:
        pieces: List[bytes] = []
        for t in tags:
            tt = self._normalize_tag(t)
            if not tt:
                continue
            pieces.append(tt)
        if not pieces:
            pieces = [b"<A>", b"{{A}}", b"${A}", b"$(A)"]

        head = b""
        for p in pieces:
            head += p * 10
            if len(head) >= min(900, target_len // 2):
                break
        head = head[:min(900, target_len // 2)]
        remain = max(0, target_len - len(magic) - len(head) - 1)
        return magic + head + (b"A" * remain) + b"\n"

    def _try_dynamic(self, root: str, candidates: List[bytes]) -> Optional[bytes]:
        try:
            if not candidates:
                return None
            workdir = root
            if not os.path.isdir(workdir):
                return None

            built = self._attempt_build(workdir)
            if not built:
                return None

            exes = self._find_executables(workdir)
            if not exes:
                return None

            test_payloads = candidates[:8]
            best: Optional[bytes] = None
            best_len = 10**18

            with tempfile.TemporaryDirectory(prefix="pocrun_") as td:
                for payload in test_payloads:
                    if len(payload) >= best_len:
                        continue
                    fpath = os.path.join(td, "poc_input.bin")
                    try:
                        with open(fpath, "wb") as f:
                            f.write(payload)
                    except Exception:
                        continue

                    crashed = False
                    for exe in exes[:10]:
                        if self._run_and_detect_crash(exe, workdir, fpath, payload):
                            crashed = True
                            break
                    if crashed:
                        best = payload
                        best_len = len(payload)

            return best
        except Exception:
            return None

    def _attempt_build(self, workdir: str) -> bool:
        makefile = None
        cmakelists = None
        for dirpath, _, filenames in os.walk(workdir):
            if "Makefile" in filenames and makefile is None:
                makefile = dirpath
            if "CMakeLists.txt" in filenames and cmakelists is None:
                cmakelists = dirpath
        env = os.environ.copy()
        cflags = env.get("CFLAGS", "")
        ldflags = env.get("LDFLAGS", "")
        extra = " -O0 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address,undefined"
        if "sanitize" not in cflags:
            env["CFLAGS"] = (cflags + extra).strip()
        if "sanitize" not in ldflags:
            env["LDFLAGS"] = (ldflags + " -fsanitize=address,undefined").strip()

        def run(cmd, cwd, timeout=240):
            return subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

        if makefile is not None:
            try:
                r = run(["make", "-j8"], makefile, timeout=240)
                if r.returncode == 0:
                    return True
            except Exception:
                pass

        if cmakelists is not None:
            try:
                build_dir = os.path.join(cmakelists, "build_pocgen")
                os.makedirs(build_dir, exist_ok=True)
                r1 = run(["cmake", ".."], build_dir, timeout=240)
                if r1.returncode != 0:
                    return False
                r2 = run(["cmake", "--build", ".", "-j", "8"], build_dir, timeout=240)
                if r2.returncode == 0:
                    return True
            except Exception:
                pass

        return False

    def _find_executables(self, root: str) -> List[str]:
        exes: List[Tuple[int, str]] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if fn.endswith((".o", ".a", ".so", ".dylib", ".dll")):
                    continue
                try:
                    st = os.stat(p)
                    if not (st.st_mode & 0o111):
                        continue
                    if st.st_size < 8 * 1024 or st.st_size > 200 * 1024 * 1024:
                        continue
                    with open(p, "rb") as f:
                        hdr = f.read(4)
                    if hdr != b"\x7fELF":
                        continue
                    exes.append((st.st_size, p))
                except Exception:
                    continue
        exes.sort(key=lambda x: x[0])
        return [p for _, p in exes]

    def _run_and_detect_crash(self, exe: str, cwd: str, file_arg: str, stdin_data: bytes) -> bool:
        def looks_like_asan(s: bytes) -> bool:
            s2 = s.lower()
            return (b"addresssanitizer" in s2) or (b"stack-buffer-overflow" in s2) or (b"undefinedbehavior" in s2) or (b"runtime error" in s2)

        # try as file argument
        for args, use_stdin in (
            ([exe, file_arg], False),
            ([exe], True),
        ):
            try:
                r = subprocess.run(
                    args,
                    cwd=cwd,
                    input=(stdin_data if use_stdin else None),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=3.0,
                )
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue

            if looks_like_asan(r.stderr) or looks_like_asan(r.stdout):
                return True
            if r.returncode < 0:
                return True
            if r.returncode != 0 and (b"segmentation fault" in (r.stderr + r.stdout).lower() or b"stack smashing" in (r.stderr + r.stdout).lower()):
                return True
        return False

    def _c_unescape_to_bytes(self, s: str) -> bytes:
        out = bytearray()
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c != "\\":
                out.append(ord(c) & 0xFF)
                i += 1
                continue
            i += 1
            if i >= n:
                out.append(ord("\\"))
                break
            e = s[i]
            i += 1
            if e == "n":
                out.append(10)
            elif e == "r":
                out.append(13)
            elif e == "t":
                out.append(9)
            elif e == "v":
                out.append(11)
            elif e == "f":
                out.append(12)
            elif e == "b":
                out.append(8)
            elif e == "a":
                out.append(7)
            elif e == "\\":
                out.append(92)
            elif e == "'":
                out.append(39)
            elif e == '"':
                out.append(34)
            elif e == "0":
                out.append(0)
            elif e in "1234567":
                val = ord(e) - 48
                cnt = 1
                while cnt < 3 and i < n and s[i] in "01234567":
                    val = (val << 3) + (ord(s[i]) - 48)
                    i += 1
                    cnt += 1
                out.append(val & 0xFF)
            elif e == "x":
                val = 0
                cnt = 0
                while cnt < 2 and i < n and s[i].lower() in "0123456789abcdef":
                    ch = s[i].lower()
                    if "0" <= ch <= "9":
                        v = ord(ch) - 48
                    else:
                        v = ord(ch) - 87
                    val = (val << 4) | v
                    i += 1
                    cnt += 1
                out.append(val & 0xFF)
            else:
                out.append(ord(e) & 0xFF)
        return bytes(out)

    def _is_mostly_ascii_printable(self, b: bytes) -> bool:
        if not b:
            return False
        good = 0
        for x in b:
            if x in (9, 10, 13):
                good += 1
            elif 32 <= x <= 126:
                good += 1
        return good >= max(1, int(len(b) * 0.9))

    def _looks_like_tag(self, b: bytes) -> bool:
        if not b:
            return False
        if len(b) == 1 and b in (b"<", b"{", b"[", b"$", b"%", b"@", b"#"):
            return True
        specials = b"<>[]{}$%@#"
        if any(ch in specials for ch in b):
            return True
        # Also consider short delimiter words used as tags (e.g., "TAG", "ID3", etc.) but de-prioritize
        if 2 <= len(b) <= 8 and re.fullmatch(rb"[A-Za-z]{2,8}", b):
            return True
        return False

    def _tag_score(self, t: bytes) -> int:
        s = 0
        if b"<" in t and b">" in t:
            s += 50
        if t.startswith(b"${") and t.endswith(b"}"):
            s += 40
        if t.startswith(b"$(") and t.endswith(b")"):
            s += 40
        if t.startswith(b"{{") and t.endswith(b"}}"):
            s += 40
        if t.startswith(b"[") and t.endswith(b"]"):
            s += 25
        if b"%" in t:
            s += 15
        if any(c in t for c in (b"<", b">", b"{", b"}", b"[", b"]", b"$", b"@", b"#")):
            s += 10
        if 3 <= len(t) <= 24:
            s += 10
        if len(t) <= 64:
            s += 5
        if any(x in t for x in (b"\n", b"\r", b"\t", b" ")):
            s -= 30
        if b"\x00" in t:
            s -= 100
        return s

    def _normalize_tag(self, t: bytes) -> bytes:
        if not t:
            return b"<A>"
        if b"\x00" in t:
            t = t.replace(b"\x00", b"")
        if len(t) == 1:
            if t == b"<":
                return b"<A>"
            if t == b"{":
                return b"{A}"
            if t == b"[":
                return b"[A]"
            if t == b"$":
                return b"${A}"
            if t == b"%":
                return b"%A%"
            if t == b"@":
                return b"@A@"
            if t == b"#":
                return b"#A#"
        if t in (b"${",):
            return b"${A}"
        if t in (b"$(",):
            return b"$(A)"
        if t in (b"{{",):
            return b"{{A}}"
        if t in (b"[[",):
            return b"[[A]]"
        # If it looks like an opening-only bracket token, close it
        if t.startswith(b"<") and b">" not in t and len(t) <= 32:
            return t + b"A>"
        if t.startswith(b"${") and not t.endswith(b"}") and len(t) <= 64:
            return t + b"A}"
        if t.startswith(b"$(") and not t.endswith(b")") and len(t) <= 64:
            return t + b"A)"
        if t.startswith(b"{{") and not t.endswith(b"}}") and len(t) <= 64:
            return t + b"A}}"
        if t.startswith(b"[") and not t.endswith(b"]") and len(t) <= 64:
            return t + b"A]"
        # Avoid extremely long tags; shorten deterministically
        if len(t) > 80:
            t = t[:80]
        return t