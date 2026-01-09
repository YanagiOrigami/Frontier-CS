import os
import tarfile
import tempfile
import subprocess
import random
import time
import shutil
import stat
import itertools


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="poc_gen_")
        try:
            self._extract_tarball(src_path, tmpdir)
            bin_paths = self._build_and_find_binaries(tmpdir)

            # Set a global time budget for fuzzing (seconds)
            global_fuzz_budget = 80.0
            fuzz_deadline = time.time() + global_fuzz_budget

            env = os.environ.copy()
            # Encourage sanitizers to abort on error and be quiet about leaks
            asan_opts = env.get("ASAN_OPTIONS", "")
            ubsan_opts = env.get("UBSAN_OPTIONS", "")
            if "abort_on_error" not in asan_opts:
                asan_opts = (asan_opts + ":abort_on_error=1:detect_leaks=0:handle_segv=1:handle_abrt=1").lstrip(":")
            if "halt_on_error" not in ubsan_opts:
                ubsan_opts = (ubsan_opts + ":halt_on_error=1:print_stacktrace=1").lstrip(":")
            env["ASAN_OPTIONS"] = asan_opts
            env["UBSAN_OPTIONS"] = ubsan_opts

            # Try each candidate binary until one yields a crash
            for bin_path in bin_paths:
                time_left = fuzz_deadline - time.time()
                if time_left <= 0:
                    break
                poc = self._fuzz_single_binary(bin_path, env, time_left)
                if poc is not None:
                    return poc

            # If everything fails, return a simple 8-byte guess
            return b"()()()()"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _extract_tarball(self, src_path: str, dst_dir: str) -> None:
        with tarfile.open(src_path, "r:*") as tar:
            def is_within_directory(directory: str, target: str) -> bool:
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

            for member in tar.getmembers():
                member_path = os.path.join(dst_dir, member.name)
                if not is_within_directory(dst_dir, member_path):
                    continue
            tar.extractall(dst_dir)

    def _try_run(self, cmd, cwd, timeout=300.0) -> bool:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def _build_and_find_binaries(self, project_dir: str):
        # First, try common build scripts at project root
        root_scripts = ["build_asan.sh", "build_sanitized.sh", "build.sh", "compile.sh"]
        for script in root_scripts:
            script_path = os.path.join(project_dir, script)
            if os.path.isfile(script_path) and os.access(script_path, os.X_OK):
                self._try_run(["bash", script_path], cwd=project_dir, timeout=300.0)
                break

        # If no root build script, search subdirectories for build scripts
        for root, dirs, files in os.walk(project_dir):
            for script in root_scripts:
                if script in files:
                    script_path = os.path.join(root, script)
                    if os.path.isfile(script_path) and os.access(script_path, os.X_OK):
                        self._try_run(["bash", script_path], cwd=root, timeout=300.0)
                        break

        # As a fallback, try make in project root if Makefile exists
        try:
            if "Makefile" in os.listdir(project_dir):
                self._try_run(["make", "-j", "8"], cwd=project_dir, timeout=300.0)
        except Exception:
            pass

        # Find ELF executables
        elf_candidates = []
        for root, dirs, files in os.walk(project_dir):
            for fname in files:
                path = os.path.join(root, fname)
                if not os.path.isfile(path):
                    continue
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                # Must be executable by user
                if not (st.st_mode & stat.S_IXUSR):
                    continue
                # Skip common library extensions
                if fname.endswith((".so", ".a", ".o", ".lo")):
                    continue
                # Check ELF magic
                try:
                    with open(path, "rb") as f:
                        magic = f.read(4)
                    if magic != b"\x7fELF":
                        continue
                except OSError:
                    continue
                elf_candidates.append(path)

        # Score and sort ELF candidates to prioritize likely harness binaries
        def score(path: str) -> float:
            name = os.path.basename(path).lower()
            s = 0.0
            if any(k in name for k in ["test", "fuzz", "harness", "poc", "demo", "sample", "driver", "regex"]):
                s -= 100.0
            lower_path = path.lower()
            if "/bin/" in lower_path or "/build/" in lower_path:
                s -= 10.0
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            s += size / 1_000_000.0
            return s

        elf_candidates.sort(key=score)
        return elf_candidates

    def _test_input(self, bin_path: str, env, data: bytes, timeout: float) -> bool:
        try:
            proc = subprocess.run(
                [bin_path],
                input=data,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

        rc = proc.returncode
        if rc < 0:
            # Terminated by signal (e.g., SIGSEGV, SIGABRT)
            return True

        err = proc.stderr or b""
        if b"AddressSanitizer" in err or b"UndefinedBehaviorSanitizer" in err or b"Sanitizer" in err:
            return True
        if b"stack-buffer-overflow" in err or b"heap-buffer-overflow" in err:
            return True
        if b"segmentation fault" in err.lower():
            return True

        return False

    def _generate_patterns_deterministic(self, charset: bytes, max_len: int):
        # Yield patterns of lengths 1..max_len using given charset
        for length in range(1, max_len + 1):
            for tup in itertools.product(charset, repeat=length):
                yield bytes(tup)

    def _fuzz_single_binary(self, bin_path: str, env, time_budget: float):
        start_time = time.time()
        deadline = start_time + time_budget

        # Phase 1: Deterministic enumeration of simple regex-like patterns
        # focusing on capturing parentheses and literal characters.
        simple_charset = b"()a"
        for pat in self._generate_patterns_deterministic(simple_charset, max_len=5):
            if time.time() > deadline:
                return None
            # Three structural variants: raw, one-line, two-line (pattern + subject)
            variants = [
                pat,
                pat + b"\n",
                pat + b"\n" + b"A\n",
            ]
            for data in variants:
                if time.time() > deadline:
                    return None
                if self._test_input(bin_path, env, data, timeout=0.2):
                    return data

        # Phase 2: Random fuzzing with richer character set, if time permits
        random.seed(0)
        rich_chars = (
            b"abcdefghijklmnopqrstuvwxyz"
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            b"0123456789"
            b"()[]{}.*+?|^$\\ "
        )
        while time.time() < deadline:
            # Randomly decide structure: 0 -> single line, 1 -> two lines
            structure = random.randint(0, 1)
            if structure == 0:
                length = random.randint(1, 32)
                pat = bytes(random.choice(rich_chars) for _ in range(length))
                data = pat + b"\n"
            else:
                length1 = random.randint(1, 24)
                length2 = random.randint(0, 24)
                pat = bytes(random.choice(rich_chars) for _ in range(length1))
                subj = bytes(random.choice(rich_chars) for _ in range(length2)) if length2 > 0 else b""
                if subj:
                    data = pat + b"\n" + subj + b"\n"
                else:
                    data = pat + b"\n"
            if self._test_input(bin_path, env, data, timeout=0.2):
                return data

        return None