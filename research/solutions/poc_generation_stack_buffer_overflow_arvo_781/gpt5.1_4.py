import os
import tarfile
import tempfile
import subprocess
import time
import random
import shutil
import stat


class Solution:
    def solve(self, src_path: str) -> bytes:
        base_dir = self._extract_tarball(src_path)
        vuln_dir, fix_dir = self._detect_variant_dirs(base_dir)

        vuln_bin = self._try_compile_project(vuln_dir, "poc_vuln_bin")
        fix_bin = None
        if vuln_bin and fix_dir and os.path.isdir(fix_dir):
            fix_bin = self._try_compile_project(fix_dir, "poc_fix_bin")

        if not vuln_bin or not os.path.isfile(vuln_bin):
            # Fallback: return a generic small payload
            return b"()\nAAAA"

        poc = self._find_poc(vuln_bin, fix_bin)
        if poc is None:
            # As last resort, return something simple
            return b"()\nAAAA"
        return poc

    # ------------------ Tarball & directory handling ------------------ #

    def _extract_tarball(self, src_path: str) -> str:
        tmpdir = tempfile.mkdtemp(prefix="pocgen_")
        with tarfile.open(src_path, "r:*") as tf:
            tf.extractall(tmpdir)
        # Determine base directory: if exactly one non-hidden entry and it's a dir, use it
        entries = [e for e in os.listdir(tmpdir) if not e.startswith(".")]
        if len(entries) == 1:
            candidate = os.path.join(tmpdir, entries[0])
            if os.path.isdir(candidate):
                return candidate
        return tmpdir

    def _detect_variant_dirs(self, base_dir: str):
        vuln_dir = None
        fix_dir = None
        try:
            entries = os.listdir(base_dir)
        except OSError:
            entries = []
        for name in entries:
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            lname = name.lower()
            if lname in ("vuln", "vulnerable", "bug", "orig", "old"):
                vuln_dir = full
            if lname in ("fix", "fixed", "patched", "patch", "new"):
                fix_dir = full
        if vuln_dir is None:
            vuln_dir = base_dir
        return vuln_dir, fix_dir

    # ------------------ Compilation ------------------ #

    def _gather_sources(self, project_dir: str):
        c_exts = (".c",)
        cpp_exts = (".cc", ".cpp", ".cxx", ".C")
        sources = []
        has_cpp = False
        for root, _dirs, files in os.walk(project_dir):
            for f in files:
                path = os.path.join(root, f)
                ext = os.path.splitext(f)[1]
                if ext in c_exts or ext in cpp_exts:
                    sources.append(path)
                    if ext in cpp_exts:
                        has_cpp = True
        return sources, has_cpp

    def _choose_compiler(self, is_cpp: bool):
        if is_cpp:
            for c in ("clang++", "g++"):
                if shutil.which(c):
                    return c
        else:
            for c in ("clang", "gcc"):
                if shutil.which(c):
                    return c
        return None

    def _try_compile_project(self, project_dir: str, out_name: str) -> str or None:
        sources, has_cpp = self._gather_sources(project_dir)
        if not sources:
            return None

        compiler = self._choose_compiler(has_cpp)
        if compiler is None:
            return None

        rel_sources = [os.path.relpath(s, project_dir) for s in sources]
        out_path = os.path.join(project_dir, out_name)

        base_cmd = [compiler, "-g", "-O1", "-fno-omit-frame-pointer"]
        if has_cpp:
            base_cmd += ["-std=c++11"]

        # First try with sanitizers
        cmd_san = base_cmd + [
            "-fsanitize=address",
            "-fsanitize=undefined",
        ] + rel_sources + ["-lm", "-lpthread", "-o", out_name]

        if self._run_compile(cmd_san, project_dir) and self._is_elf(out_path):
            return out_path

        # Fallback: without sanitizers
        cmd_nosan = base_cmd + rel_sources + ["-lm", "-lpthread", "-o", out_name]
        if self._run_compile(cmd_nosan, project_dir) and self._is_elf(out_path):
            return out_path

        # As a last resort, try any existing ELF binary in project_dir
        elf = self._find_any_elf(project_dir)
        return elf

    def _run_compile(self, cmd, cwd: str) -> bool:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def _is_elf(self, path: str) -> bool:
        try:
            if not os.path.isfile(path):
                return False
            with open(path, "rb") as f:
                magic = f.read(4)
            if magic != b"\x7fELF":
                return False
            st = os.stat(path)
            if not (st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                # Not marked executable, but still an ELF; that's fine.
                return True
            return True
        except OSError:
            return False

    def _find_any_elf(self, root_dir: str) -> str or None:
        best_path = None
        best_size = -1
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for f in filenames:
                path = os.path.join(dirpath, f)
                ext = os.path.splitext(f)[1]
                if ext in (".o", ".lo", ".a", ".so", ".so.0", ".so.1", ".so.2"):
                    continue
                try:
                    with open(path, "rb") as fh:
                        magic = fh.read(4)
                    if magic != b"\x7fELF":
                        continue
                    size = os.path.getsize(path)
                    if size > best_size:
                        best_size = size
                        best_path = path
                except OSError:
                    continue
        return best_path

    # ------------------ Fuzzing & PoC search ------------------ #

    def _run_binary(self, bin_path: str, data: bytes, timeout: float = 0.2):
        env = os.environ.copy()
        env.setdefault("ASAN_OPTIONS", "abort_on_error=1:detect_leaks=0:symbolize=0")
        env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1:halt_on_error=1")
        try:
            proc = subprocess.run(
                [bin_path],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, False
        except Exception:
            return False, False

        rc = proc.returncode
        stderr_text = proc.stderr.decode("latin1", errors="ignore")
        is_sanitizer = ("Sanitizer" in stderr_text) or ("runtime error:" in stderr_text)
        did_crash = (rc != 0) and is_sanitizer
        return did_crash, is_sanitizer

    def _triggers_bug(self, vuln_bin: str, fix_bin: str or None, data: bytes) -> bool:
        crashed_vuln, _ = self._run_binary(vuln_bin, data)
        if not crashed_vuln:
            return False
        if fix_bin and os.path.isfile(fix_bin):
            crashed_fix, _ = self._run_binary(fix_bin, data)
            if crashed_fix:
                return False
        return True

    def _find_poc(self, vuln_bin: str, fix_bin: str or None) -> bytes or None:
        rnd = random.Random(0xC0FFEE)

        seeds = [
            b"()\n",
            b"()\na",
            b"(a)\na",
            b"(a)(b)\nab",
            b"(a)(b)(c)\nabc",
            b"(a*)\naaa",
            b"(a|b)\na",
            b"(a*)(b*)\nabab",
            b"([A-Z]+)\nHELLO",
            b"(.*)\nAAAAAAAAAA",
            b"()",
            b"(a*)",
            b"(ab)",
        ]

        # Try seeds directly
        for s in seeds:
            if self._triggers_bug(vuln_bin, fix_bin, s):
                return s

        start = time.time()
        max_time = 15.0

        while time.time() - start < max_time:
            if rnd.random() < 0.5:
                base = rnd.choice(seeds)
                candidate = self._mutate(base, rnd)
            else:
                candidate = self._generate_candidate(rnd)

            if self._triggers_bug(vuln_bin, fix_bin, candidate):
                return candidate

        return None

    # ------------------ Input generation ------------------ #

    def __init__(self):
        self._alphabet = (
            b"abcdefghijklmnopqrstuvwxyz"
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            b"0123456789"
            b".^$*+?()[]{}|\\\n\t "
        )

    def _mutate(self, data: bytes, rnd: random.Random) -> bytes:
        if not data:
            data = b"A"
        b = bytearray(data)
        num_ops = rnd.randint(1, 4)
        for _ in range(num_ops):
            op = rnd.randint(0, 2)
            if op == 0 and len(b) > 0:
                idx = rnd.randrange(len(b))
                b[idx] = rnd.choice(self._alphabet)
            elif op == 1 and len(b) < 128:
                idx = rnd.randrange(len(b) + 1)
                b.insert(idx, rnd.choice(self._alphabet))
            elif op == 2 and len(b) > 1:
                idx = rnd.randrange(len(b))
                del b[idx]
        if len(b) > 256:
            del b[256:]
        return bytes(b)

    def _generate_candidate(self, rnd: random.Random) -> bytes:
        use_newline = rnd.random() < 0.7
        pattern = self._random_pattern(rnd)
        if use_newline:
            subject = self._random_subject(rnd)
            s = pattern + "\n" + subject
        else:
            s = pattern
        b = s.encode("ascii", errors="ignore")
        if len(b) > 256:
            b = b[:256]
        return b

    def _random_pattern(self, rnd: random.Random, max_depth: int = 2) -> str:
        return self._rand_expr(rnd, 0, max_depth)

    def _rand_expr(self, rnd: random.Random, depth: int, max_depth: int) -> str:
        s = self._rand_seq(rnd, depth, max_depth)
        if depth < max_depth and rnd.random() < 0.4:
            s = s + "|" + self._rand_expr(rnd, depth + 1, max_depth)
        return s

    def _rand_seq(self, rnd: random.Random, depth: int, max_depth: int) -> str:
        k = rnd.randint(1, 4)
        parts = [self._rand_piece(rnd, depth, max_depth) for _ in range(k)]
        return "".join(parts)

    def _rand_piece(self, rnd: random.Random, depth: int, max_depth: int) -> str:
        atom = self._rand_atom(rnd, depth, max_depth)
        if rnd.random() < 0.5:
            atom += rnd.choice(["*", "+", "?"])
        return atom

    def _rand_atom(self, rnd: random.Random, depth: int, max_depth: int) -> str:
        r = rnd.random()
        if depth < max_depth and r < 0.3:
            inner = self._rand_expr(rnd, depth + 1, max_depth)
            return "(" + inner + ")"
        elif r < 0.6:
            return rnd.choice("abcd")
        elif r < 0.8:
            return "\\" + rnd.choice(["d", "w", "s"])
        else:
            return "."

    def _random_subject(self, rnd: random.Random) -> str:
        length = rnd.randint(0, 16)
        if length == 0:
            return ""
        alphabet = "abcd"
        return "".join(rnd.choice(alphabet) for _ in range(length))