import os
import re
import tarfile
import tempfile
import subprocess
import shutil
import stat
from typing import List, Tuple, Optional, Dict, Callable


def _is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            continue
        tar.extract(member, path=path)


def _find_project_root(extract_dir: str) -> str:
    entries = [os.path.join(extract_dir, x) for x in os.listdir(extract_dir)]
    dirs = [p for p in entries if os.path.isdir(p)]
    if len(dirs) == 1:
        return dirs[0]
    return extract_dir


def _read_text_file(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes)
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _iter_source_files(root: str) -> List[str]:
    ex_dirs = {".git", ".svn", ".hg", "build", "dist", "out", "bin", "obj", "cmake-build-debug", "cmake-build-release", "node_modules"}
    src_exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ex_dirs and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in src_exts:
                out.append(os.path.join(dirpath, fn))
    return out


def _extract_candidates(root: str) -> Dict[str, object]:
    files = _iter_source_files(root)
    main_files = []
    tag_names = []
    magic_strings = []
    buf_sizes = []

    cmp_magic_re = re.compile(r'\b(?:memcmp|strncmp)\s*\(\s*[^,]+,\s*"([^"\n\r]{2,64})"')
    strcmp_tag_re = re.compile(r'\bstrcmp\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"([A-Za-z0-9_\-]{1,32})"\s*\)')
    strncmp_tag_re = re.compile(r'\bstrncmp\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"([A-Za-z0-9_\-]{1,32})"\s*,')
    main_re = re.compile(r'\b(?:int|void)\s+main\s*\(')
    buf_re = re.compile(r'\bchar\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d{2,6})\s*\]')
    tag_hint_re = re.compile(r'\btag\b', re.IGNORECASE)

    for p in files:
        if not os.path.isfile(p):
            continue
        txt = _read_text_file(p)
        if not txt:
            continue

        if main_re.search(txt):
            main_files.append(p)

        for m in buf_re.finditer(txt):
            sz = int(m.group(2))
            if 64 <= sz <= 32768:
                name = m.group(1).lower()
                if "out" in name or "buf" in name or "output" in name or tag_hint_re.search(txt):
                    buf_sizes.append(sz)

        for m in cmp_magic_re.finditer(txt):
            s = m.group(1)
            if any(ch in s for ch in "<>{}[]$@#&") or s.isalpha() or s.startswith("<?") or s.startswith("<!"):
                if 2 <= len(s) <= 24 and all(32 <= ord(c) <= 126 for c in s):
                    magic_strings.append(s)

        for m in strcmp_tag_re.finditer(txt):
            var = m.group(1).lower()
            name = m.group(2)
            if ("tag" in var or "name" in var or "type" in var) and 1 <= len(name) <= 16:
                tag_names.append(name)

        for m in strncmp_tag_re.finditer(txt):
            var = m.group(1).lower()
            name = m.group(2)
            if ("tag" in var or "name" in var or "type" in var) and 1 <= len(name) <= 16:
                tag_names.append(name)

    def _top_k(items: List[str], k: int) -> List[str]:
        freq: Dict[str, int] = {}
        for it in items:
            freq[it] = freq.get(it, 0) + 1
        return [x for x, _ in sorted(freq.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[:k]]

    tag_names = _top_k(tag_names, 10)
    magic_strings = _top_k(magic_strings, 8)

    buf_guess = None
    if buf_sizes:
        freq: Dict[int, int] = {}
        for sz in buf_sizes:
            freq[sz] = freq.get(sz, 0) + 1
        buf_guess = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    return {
        "main_files": main_files,
        "tag_names": tag_names,
        "magic_strings": magic_strings,
        "buf_guess": buf_guess,
    }


def _which(prog: str) -> Optional[str]:
    return shutil.which(prog)


def _compile_single_binary(root: str, analysis: Dict[str, object], out_path: str) -> Optional[str]:
    gcc = _which("gcc")
    gpp = _which("g++")
    if not gcc and not gpp:
        return None

    sources = []
    main_candidates: List[str] = list(analysis.get("main_files") or [])
    if not main_candidates:
        all_sources = []
        for p in _iter_source_files(root):
            if os.path.splitext(p)[1].lower() in {".c", ".cc", ".cpp", ".cxx"}:
                all_sources.append(p)
        main_re = re.compile(r'\b(?:int|void)\s+main\s*\(')
        for p in all_sources:
            txt = _read_text_file(p, max_bytes=400_000)
            if main_re.search(txt):
                main_candidates.append(p)

    def _score_main(path: str) -> Tuple[int, int]:
        txt = _read_text_file(path, max_bytes=600_000)
        score = 0
        if re.search(r'\bfopen\s*\(\s*argv\s*\[\s*1\s*\]', txt):
            score += 5
        if re.search(r'\bgetopt\b', txt):
            score += 2
        if re.search(r'\bstdin\b', txt):
            score += 1
        if re.search(r'\btag\b', txt, re.IGNORECASE):
            score += 2
        depth = path.count(os.sep)
        return (-score, depth)

    if not main_candidates:
        return None

    main_candidates = sorted(set(main_candidates), key=_score_main)
    chosen_main = main_candidates[0]

    ignore_dirs = {"test", "tests", "example", "examples", "doc", "docs", "benchmark", "benchmarks"}
    for dirpath, dirnames, filenames in os.walk(root):
        dbase = os.path.basename(dirpath).lower()
        if dbase in ignore_dirs:
            dirnames[:] = []
            continue
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in {".c", ".cc", ".cpp", ".cxx"}:
                sources.append(os.path.join(dirpath, fn))

    main_re = re.compile(r'\b(?:int|void)\s+main\s*\(')
    mains = []
    for p in sources:
        txt = _read_text_file(p, max_bytes=250_000)
        if main_re.search(txt):
            mains.append(p)

    if chosen_main not in sources:
        sources.append(chosen_main)

    sources_to_build = []
    for p in sources:
        if p != chosen_main and p in mains:
            continue
        sources_to_build.append(p)

    use_cpp = any(os.path.splitext(p)[1].lower() in {".cc", ".cpp", ".cxx"} for p in sources_to_build)
    cc = gpp if (use_cpp and gpp) else (gcc if gcc else gpp)
    if not cc:
        return None

    include_dirs = {root}
    for p in _iter_source_files(root):
        if os.path.splitext(p)[1].lower() in {".h", ".hpp"}:
            include_dirs.add(os.path.dirname(p))
    inc_flags = []
    for d in sorted(include_dirs, key=lambda x: (x.count(os.sep), x)):
        inc_flags.extend(["-I", d])

    cflags = ["-O0", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", "-w"]
    if not use_cpp:
        cflags.extend(["-std=gnu11"])

    cmd = [cc] + cflags + inc_flags + ["-o", out_path] + sources_to_build
    env = dict(os.environ)
    env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1:symbolize=0"
    try:
        p = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if p.returncode == 0 and os.path.isfile(out_path):
            st = os.stat(out_path)
            os.chmod(out_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return out_path
    except Exception:
        return None

    # Retry with -lm if it looks like math is missing
    try:
        err = p.stderr.decode("utf-8", errors="ignore") if "p" in locals() else ""
        if "undefined reference" in err and ("sin" in err or "cos" in err or "pow" in err or "sqrt" in err):
            cmd2 = cmd + ["-lm"]
            p2 = subprocess.run(cmd2, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if p2.returncode == 0 and os.path.isfile(out_path):
                st = os.stat(out_path)
                os.chmod(out_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                return out_path
    except Exception:
        pass

    return None


def _try_build_system(root: str) -> List[str]:
    env = dict(os.environ)
    env["CFLAGS"] = (env.get("CFLAGS", "") + " -O0 -g -fsanitize=address -fno-omit-frame-pointer -w").strip()
    env["CPPFLAGS"] = (env.get("CPPFLAGS", "") + " -w").strip()
    env["LDFLAGS"] = (env.get("LDFLAGS", "") + " -fsanitize=address").strip()
    env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1:symbolize=0"

    def run(cmd: List[str], cwd: str, timeout: int) -> bool:
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            return p.returncode == 0
        except Exception:
            return False

    has_makefile = any(os.path.isfile(os.path.join(root, x)) for x in ("Makefile", "makefile", "GNUmakefile"))
    if os.path.isfile(os.path.join(root, "configure")) and not has_makefile:
        run(["sh", "./configure"], cwd=root, timeout=120)

    if has_makefile:
        run(["make", "-j8"], cwd=root, timeout=180)

    if os.path.isfile(os.path.join(root, "CMakeLists.txt")):
        bdir = os.path.join(root, "_poc_build")
        os.makedirs(bdir, exist_ok=True)
        cmake = _which("cmake")
        if cmake:
            run([cmake, "-S", root, "-B", bdir,
                 "-DCMAKE_BUILD_TYPE=Debug",
                 "-DCMAKE_C_FLAGS=" + env["CFLAGS"],
                 "-DCMAKE_CXX_FLAGS=" + env["CFLAGS"],
                 "-DCMAKE_EXE_LINKER_FLAGS=" + env["LDFLAGS"]], cwd=root, timeout=120)
            run([cmake, "--build", bdir, "-j8"], cwd=root, timeout=180)

    execs = _find_executables(root)
    return execs


def _find_executables(root: str) -> List[str]:
    ex_dirs = {".git", ".svn", ".hg", "node_modules"}
    bad_exts = {".o", ".a", ".so", ".lo", ".la", ".dll", ".dylib", ".py", ".sh", ".txt", ".md", ".html", ".json", ".xml"}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ex_dirs and not d.startswith(".")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in bad_exts:
                continue
            try:
                st = os.stat(p)
            except Exception:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if st.st_size < 8 * 1024 or st.st_size > 200 * 1024 * 1024:
                continue
            if os.access(p, os.X_OK):
                out.append(p)

    def key(p: str) -> Tuple[int, int, str]:
        try:
            st = os.stat(p)
            mtime = int(st.st_mtime)
            size = st.st_size
        except Exception:
            mtime, size = 0, 0
        return (p.count(os.sep), -mtime, size)

    out = sorted(set(out), key=key)
    return out


def _asan_stack_overflow(stderr: bytes) -> bool:
    if b"AddressSanitizer" not in stderr and b"ASan" not in stderr:
        return False
    if b"stack-buffer-overflow" in stderr:
        return True
    # Some builds report just "stack-overflow" or "stack-buffer"
    if b"stack-overflow" in stderr or b"stack-buffer" in stderr:
        return True
    return False


def _run_target(bin_path: str, cwd: str, input_bytes: bytes, mode: Tuple[str, ...], tmpfile_path: str, timeout: float = 1.5) -> Tuple[int, bytes, bytes]:
    env = dict(os.environ)
    env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1:symbolize=0"
    env["UBSAN_OPTIONS"] = "halt_on_error=1:abort_on_error=1:print_stacktrace=0"
    env["MSAN_OPTIONS"] = "abort_on_error=1"

    lib_paths = [cwd, os.path.join(cwd, ".libs"), os.path.join(cwd, "lib"), os.path.join(cwd, "_poc_build")]
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([p for p in lib_paths if os.path.isdir(p)] + ([old_ld] if old_ld else []))

    args = [bin_path]
    stdin_data = None
    if mode == ("stdin",):
        stdin_data = input_bytes
    else:
        try:
            with open(tmpfile_path, "wb") as f:
                f.write(input_bytes)
        except Exception:
            pass

        if mode == ("file",):
            args.append(tmpfile_path)
        elif mode == ("dash_stdin",):
            args.append("-")
            stdin_data = input_bytes
        elif mode == ("-f",):
            args.extend(["-f", tmpfile_path])
        elif mode == ("-i",):
            args.extend(["-i", tmpfile_path])
        elif mode == ("--input",):
            args.extend(["--input", tmpfile_path])
        elif mode == ("--file",):
            args.extend(["--file", tmpfile_path])
        elif mode == ("-in",):
            args.extend(["-in", tmpfile_path])
        else:
            args.append(tmpfile_path)

    try:
        p = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            input=stdin_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, b"", (e.stderr or b"") + b"\nTIMEOUT\n"
    except Exception as e:
        return 125, b"", (str(e).encode("utf-8", errors="ignore") + b"\n")


def _build_patterns(tag_names: List[str]) -> List[Callable[[int], bytes]]:
    base = [
        lambda n: (b"A" * n) + b"\n",
        lambda n: b"<tag>" + (b"A" * n) + b"</tag>\n",
        lambda n: b"<a>" + (b"A" * n) + b"</a>\n",
        lambda n: b"<" + (b"A" * n) + b">\n",
        lambda n: b"<!--" + (b"A" * n) + b"-->\n",
        lambda n: b"{{" + (b"A" * n) + b"}}\n",
        lambda n: b"${" + (b"A" * n) + b"}\n",
        lambda n: b"[[" + (b"A" * n) + b"]]\n",
        lambda n: b"%" + (b"A" * n) + b"%\n",
        lambda n: b"@" + (b"A" * n) + b"@\n",
        lambda n: b"&" + (b"A" * n) + b";\n",
        lambda n: b"<script>" + (b"A" * n) + b"</script>\n",
    ]

    for name in tag_names[:10]:
        try:
            nb = name.encode("ascii", errors="ignore")
            if not nb:
                continue
            base.append(lambda n, nb=nb: b"<" + nb + b">" + (b"A" * n) + b"</" + nb + b">\n")
            base.append(lambda n, nb=nb: b"{{" + nb + b"}}" + (b"A" * n) + b"\n")
            base.append(lambda n, nb=nb: b"${" + nb + b"}" + (b"A" * n) + b"\n")
        except Exception:
            continue
    return base


def _build_prefixes(magic_strings: List[str]) -> List[bytes]:
    prefixes = [b""]
    for s in magic_strings[:8]:
        try:
            b = s.encode("utf-8", errors="ignore")
            if b and b not in prefixes:
                prefixes.append(b + (b"\n" if not b.endswith(b"\n") else b""))
        except Exception:
            continue
    # Common structured text prefixes
    for b in [b"<?xml version=\"1.0\"?>\n", b"<!DOCTYPE html>\n", b"{\n", b"[\n"]:
        if b not in prefixes:
            prefixes.append(b)
    return prefixes[:10]


def _find_crashing_input_for_binary(root: str, bin_path: str, analysis: Dict[str, object]) -> Optional[bytes]:
    buf_guess = analysis.get("buf_guess") or 1024
    if not isinstance(buf_guess, int) or buf_guess <= 0:
        buf_guess = 1024

    tag_names = list(analysis.get("tag_names") or [])
    magic_strings = list(analysis.get("magic_strings") or [])

    patterns = _build_patterns(tag_names)
    prefixes = _build_prefixes(magic_strings)

    modes = [
        ("file",),
        ("stdin",),
        ("dash_stdin",),
        ("-f",),
        ("-i",),
        ("--input",),
        ("--file",),
        ("-in",),
    ]

    tmpfile_path = os.path.join(root, "_poc_input.bin")

    best: Optional[bytes] = None

    def try_case(inp: bytes, mode: Tuple[str, ...]) -> bool:
        nonlocal best
        rc, out, err = _run_target(bin_path, root, inp, mode, tmpfile_path, timeout=1.5)
        if _asan_stack_overflow(err):
            if best is None or len(inp) < len(best):
                best = inp
            return True
        return False

    # quick check for any crash with moderate sizes
    initial_ns = []
    for n in [max(64, buf_guess - 256), max(128, buf_guess - 64), buf_guess, buf_guess + 128, buf_guess + 512]:
        if n not in initial_ns:
            initial_ns.append(n)

    for mode in modes:
        for pref in prefixes:
            for pat in patterns:
                for n in initial_ns:
                    inp = pref + pat(n)
                    if best is not None and len(inp) >= len(best):
                        continue
                    if try_case(inp, mode):
                        return best

    # search for minimal n for each combination
    for mode in modes:
        for pref in prefixes:
            for pat in patterns:
                # exponential search for a crash
                lo = 0
                hi = max(64, buf_guess - 128)
                found = False
                for _ in range(14):
                    inp = pref + pat(hi)
                    if best is not None and len(inp) >= len(best):
                        break
                    if try_case(inp, mode):
                        found = True
                        break
                    lo = hi
                    hi *= 2
                    if hi > 20000:
                        break
                if not found:
                    continue

                # binary search for minimal hi that still crashes
                left = lo + 1
                right = hi
                while left < right:
                    mid = (left + right) // 2
                    inp = pref + pat(mid)
                    if best is not None and len(inp) >= len(best):
                        right = mid
                        continue
                    rc, out, err = _run_target(bin_path, root, inp, mode, tmpfile_path, timeout=1.5)
                    if _asan_stack_overflow(err):
                        best = inp if best is None or len(inp) < len(best) else best
                        right = mid
                    else:
                        left = mid + 1

                # keep searching other patterns; but if very small, can stop
                if best is not None and len(best) <= 256:
                    return best

    return best


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            try:
                with tarfile.open(src_path, "r:*") as tar:
                    _safe_extract_tar(tar, td)
            except Exception:
                return b"<tag>" + (b"A" * 1500) + b"</tag>\n"

            root = _find_project_root(td)
            analysis = _extract_candidates(root)

            poc_bin = os.path.join(root, "_pocbin")
            bin_path = _compile_single_binary(root, analysis, poc_bin)

            execs: List[str] = []
            if bin_path and os.path.isfile(bin_path):
                execs = [bin_path]
            else:
                execs = _try_build_system(root)

            # If no executable found, fall back to heuristic payload
            if not execs:
                n = 1500
                if isinstance(analysis.get("buf_guess"), int):
                    n = max(600, int(analysis["buf_guess"]) + 64)
                tag = b"tag"
                tnames = analysis.get("tag_names") or []
                if tnames:
                    try:
                        tag = str(tnames[0]).encode("ascii", errors="ignore") or b"tag"
                    except Exception:
                        tag = b"tag"
                pref = b""
                m = analysis.get("magic_strings") or []
                if m:
                    try:
                        pref = str(m[0]).encode("utf-8", errors="ignore")
                        if pref and not pref.endswith(b"\n"):
                            pref += b"\n"
                    except Exception:
                        pref = b""
                return pref + b"<" + tag + b">" + (b"A" * n) + b"</" + tag + b">\n"

            best: Optional[bytes] = None
            for bp in execs[:12]:
                if not os.path.isfile(bp) or not os.access(bp, os.X_OK):
                    continue
                cand = _find_crashing_input_for_binary(root, bp, analysis)
                if cand is not None:
                    if best is None or len(cand) < len(best):
                        best = cand
                        if len(best) <= 256:
                            break

            if best is not None:
                return best

            # fallback
            n = 1500
            bg = analysis.get("buf_guess")
            if isinstance(bg, int) and 64 <= bg <= 32768:
                n = max(600, bg + 128)
            return b"<tag>" + (b"A" * n) + b"</tag>\n"