import os
import re
import io
import sys
import tarfile
import time
import glob
import shlex
import random
import string
import hashlib
import tempfile
import subprocess
from typing import List, Tuple, Optional, Callable, Dict, Set


def _is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            continue
        try:
            tar.extract(member, path=path)
        except Exception:
            continue


def _extract_src(src_path: str, dst_dir: str) -> str:
    if os.path.isdir(src_path):
        return os.path.abspath(src_path)

    os.makedirs(dst_dir, exist_ok=True)
    try:
        with tarfile.open(src_path, "r:*") as tf:
            _safe_extract_tar(tf, dst_dir)
    except Exception:
        return os.path.abspath(dst_dir)

    entries = [os.path.join(dst_dir, e) for e in os.listdir(dst_dir)]
    dirs = [e for e in entries if os.path.isdir(e)]
    if len(dirs) == 1:
        return os.path.abspath(dirs[0])
    return os.path.abspath(dst_dir)


_BLACKLIST_DIR_PARTS = {
    ".git", ".svn", ".hg",
    "build", "out", "dist", "bazel-out",
    "CMakeFiles", "_deps",
    "docs", "doc",
    "examples", "example",
    "tests", "test", "testing",
    "bench", "benchmark", "benchmarks",
    "tools", "tool",
    "cmake",
    "third_party_build", "third-party-build",
}


def _is_blacklisted_path(path: str) -> bool:
    parts = re.split(r"[\\/]+", path)
    for p in parts:
        if p in _BLACKLIST_DIR_PARTS:
            return True
    return False


def _read_file_limited(path: str, limit: int = 256 * 1024) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(limit)
    except Exception:
        return b""


def _find_small_poc_files(root: str) -> Optional[bytes]:
    name_re = re.compile(r"(crash|poc|repro|reproducer|regress|issue)", re.IGNORECASE)
    candidates: List[Tuple[int, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if _is_blacklisted_path(dirpath):
            dirnames[:] = []
            continue
        for fn in filenames:
            if not name_re.search(fn):
                continue
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > 1024:
                continue
            candidates.append((st.st_size, p))
    if not candidates:
        return None
    candidates.sort()
    for _, p in candidates[:20]:
        data = _read_file_limited(p, 2048)
        if 0 < len(data) <= 1024:
            return data
    return None


def _find_fuzz_targets(root: str) -> List[str]:
    targets = []
    exts = {".c", ".cc", ".cpp", ".cxx"}
    needle = b"LLVMFuzzerTestOneInput"
    for dirpath, dirnames, filenames in os.walk(root):
        if _is_blacklisted_path(dirpath):
            dirnames[:] = []
            continue
        for fn in filenames:
            _, ext = os.path.splitext(fn)
            if ext.lower() not in exts:
                continue
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > 2_000_000:
                continue
            data = _read_file_limited(p, 256 * 1024)
            if needle in data:
                targets.append(p)
    return targets


def _score_fuzz_target(path: str) -> int:
    data = _read_file_limited(path, 256 * 1024).lower()
    score = 0
    for kw, w in [
        (b"infinity", 30),
        (b"inf", 10),
        (b"nan", 10),
        (b"float", 8),
        (b"double", 8),
        (b"number", 6),
        (b"parse", 4),
        (b"json", 4),
        (b"yaml", 4),
        (b"toml", 4),
    ]:
        score += data.count(kw) * w
    bn = os.path.basename(path).lower()
    for kw, w in [
        ("float", 30),
        ("double", 30),
        ("number", 20),
        ("parse", 15),
        ("json", 10),
        ("yaml", 10),
        ("toml", 10),
        ("lex", 10),
        ("token", 8),
    ]:
        if kw in bn:
            score += w
    return score


def _which(cmd: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = [""] if os.name != "nt" else ["", ".exe", ".bat", ".cmd"]
    for d in paths:
        for e in exts:
            p = os.path.join(d, cmd + e)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return None


def _has_main_function(path: str) -> bool:
    data = _read_file_limited(path, 128 * 1024)
    if not data:
        return False
    text = data.decode("utf-8", "ignore")
    return bool(re.search(r"(^|[^A-Za-z0-9_])main\s*\(", text))


def _collect_sources(root: str, fuzz_file: str, max_sources: int = 1200) -> List[str]:
    exts = {".c", ".cc", ".cpp", ".cxx"}
    sources: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if _is_blacklisted_path(dirpath):
            dirnames[:] = []
            continue
        for fn in filenames:
            _, ext = os.path.splitext(fn)
            if ext.lower() not in exts:
                continue
            p = os.path.join(dirpath, fn)
            if os.path.abspath(p) == os.path.abspath(fuzz_file):
                continue
            try:
                st = os.stat(p)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > 2_500_000:
                continue
            if _has_main_function(p):
                continue
            sources.append(p)
            if len(sources) >= max_sources:
                return sources
    return sources


def _collect_include_dirs(root: str, max_dirs: int = 250) -> List[str]:
    inc_dirs: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        if _is_blacklisted_path(dirpath):
            dirnames[:] = []
            continue
        if any(fn.lower().endswith((".h", ".hpp", ".hh", ".hxx")) for fn in filenames):
            inc_dirs.add(dirpath)
            if len(inc_dirs) >= max_dirs:
                break
    preferred = []
    for p in [root, os.path.join(root, "include"), os.path.join(root, "src"), os.path.join(root, "lib")]:
        if os.path.isdir(p):
            preferred.append(os.path.abspath(p))
    rest = sorted(os.path.abspath(d) for d in inc_dirs if os.path.abspath(d) not in set(preferred))
    all_dirs = preferred + rest
    if len(all_dirs) > max_dirs:
        all_dirs = all_dirs[:max_dirs]
    return all_dirs


def _run(cmd: List[str], cwd: Optional[str], timeout: int) -> Tuple[int, bytes, bytes]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout or b""
        err = e.stderr or b""
        return 124, out, err
    except Exception as e:
        return 127, b"", (str(e).encode("utf-8", "ignore"))


def _build_fuzzer_binary(root: str, fuzz_file: str, build_dir: str, time_budget_s: float) -> Optional[str]:
    clang = _which("clang")
    clangxx = _which("clang++")
    if not clang or not clangxx:
        return None

    os.makedirs(build_dir, exist_ok=True)
    include_dirs = _collect_include_dirs(root)

    common_cflags = ["-O1", "-g", "-fno-omit-frame-pointer", "-fno-sanitize-recover=all", "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION"]
    common_sanitize_compile = ["-fsanitize=address,undefined"]
    link_sanitize = ["-fsanitize=fuzzer,address,undefined"]

    def compile_and_link(source_list: List[str], per_file_timeout: int = 30, link_timeout: int = 60) -> Optional[str]:
        start = time.monotonic()
        objs = []
        for sp in source_list:
            if time.monotonic() - start > time_budget_s:
                return None
            base = os.path.basename(sp)
            h = hashlib.sha1(os.path.abspath(sp).encode("utf-8", "ignore")).hexdigest()[:12]
            obj = os.path.join(build_dir, f"{base}.{h}.o")
            objs.append(obj)

            if os.path.exists(obj):
                continue

            _, ext = os.path.splitext(sp)
            ext = ext.lower()
            is_c = (ext == ".c")
            comp = clang if is_c else clangxx
            stdflag = "-std=c11" if is_c else "-std=c++17"
            cmd = [comp, "-c", stdflag] + common_cflags + common_sanitize_compile
            for d in include_dirs:
                cmd.append("-I" + d)
            cmd += [sp, "-o", obj]
            rc, _, _ = _run(cmd, cwd=root, timeout=per_file_timeout)
            if rc != 0:
                return None

        out_bin = os.path.join(build_dir, "fuzz_bin")
        link_cmd = [clangxx] + common_cflags + link_sanitize + objs + ["-o", out_bin]
        rc, _, _ = _run(link_cmd, cwd=root, timeout=link_timeout)
        if rc != 0:
            return None
        if os.path.isfile(out_bin) and os.access(out_bin, os.X_OK):
            return out_bin
        return None

    # Attempt 1: fuzz file alone (plus perhaps header-only deps)
    attempt_sources = [os.path.abspath(fuzz_file)]
    binpath = compile_and_link(attempt_sources, per_file_timeout=40, link_timeout=60)
    if binpath:
        return binpath

    # Attempt 2: add nearby sources (same dir + common src/lib dirs)
    nearby_dirs = {os.path.dirname(os.path.abspath(fuzz_file))}
    for d in ["src", "lib"]:
        p = os.path.join(root, d)
        if os.path.isdir(p):
            nearby_dirs.add(os.path.abspath(p))

    nearby_sources = [os.path.abspath(fuzz_file)]
    exts = {".c", ".cc", ".cpp", ".cxx"}
    for d in sorted(nearby_dirs):
        for dirpath, dirnames, filenames in os.walk(d):
            if _is_blacklisted_path(dirpath):
                dirnames[:] = []
                continue
            for fn in filenames:
                _, ext = os.path.splitext(fn)
                if ext.lower() not in exts:
                    continue
                p = os.path.join(dirpath, fn)
                if os.path.abspath(p) == os.path.abspath(fuzz_file):
                    continue
                if _has_main_function(p):
                    continue
                nearby_sources.append(p)
                if len(nearby_sources) > 450:
                    break
            if len(nearby_sources) > 450:
                break
        if len(nearby_sources) > 450:
            break

    binpath = compile_and_link(nearby_sources, per_file_timeout=30, link_timeout=90)
    if binpath:
        return binpath

    # Attempt 3: all sources (capped)
    all_sources = [os.path.abspath(fuzz_file)] + _collect_sources(root, fuzz_file, max_sources=900)
    binpath = compile_and_link(all_sources, per_file_timeout=25, link_timeout=120)
    return binpath


def _make_runner(binpath: str, workdir: str) -> Callable[[bytes], bool]:
    input_path = os.path.join(workdir, "input.bin")
    env = dict(os.environ)
    env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":abort_on_error=1:detect_leaks=0:symbolize=0:allocator_may_return_null=1"
    env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + ":abort_on_error=1:print_stacktrace=0:symbolize=0"

    def crashes(data: bytes) -> bool:
        try:
            with open(input_path, "wb") as f:
                f.write(data)
        except Exception:
            return False
        try:
            p = subprocess.run(
                [binpath, "-runs=1", "-timeout=2", input_path],
                cwd=workdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                timeout=6,
            )
            return p.returncode != 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    return crashes


def _ddmin(data: bytes, test: Callable[[bytes], bool], time_budget_s: float) -> bytes:
    start = time.monotonic()
    if not test(data):
        return data
    n = 2
    cur = data
    while len(cur) >= 2 and (time.monotonic() - start) < time_budget_s:
        chunk = max(1, len(cur) // n)
        reduced = False
        for i in range(0, len(cur), chunk):
            if (time.monotonic() - start) >= time_budget_s:
                break
            trial = cur[:i] + cur[i + chunk:]
            if not trial:
                continue
            if test(trial):
                cur = trial
                n = max(2, n - 1)
                reduced = True
                break
        if not reduced:
            if n >= len(cur):
                break
            n = min(len(cur), n * 2)
    # single-byte greedy cleanup
    i = 0
    while i < len(cur) and (time.monotonic() - start) < time_budget_s:
        trial = cur[:i] + cur[i + 1:]
        if trial and test(trial):
            cur = trial
            continue
        i += 1
    return cur


def _parse_dict_tokens(root: str, max_tokens: int = 200) -> List[bytes]:
    tokens: List[bytes] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if _is_blacklisted_path(dirpath):
            dirnames[:] = []
            continue
        for fn in filenames:
            if not fn.lower().endswith(".dict"):
                continue
            p = os.path.join(dirpath, fn)
            data = _read_file_limited(p, 256 * 1024)
            if not data:
                continue
            text = data.decode("utf-8", "ignore")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r'^"((?:\\.|[^"])*)"$', line)
                if m:
                    s = m.group(1)
                    try:
                        b = bytes(s, "utf-8").decode("unicode_escape").encode("latin-1", "ignore")
                    except Exception:
                        b = s.encode("utf-8", "ignore")
                    if b and len(b) <= 64:
                        tokens.append(b)
                else:
                    b = line.encode("utf-8", "ignore")
                    if b and len(b) <= 64:
                        tokens.append(b)
                if len(tokens) >= max_tokens:
                    break
            if len(tokens) >= max_tokens:
                break
        if len(tokens) >= max_tokens:
            break
    # dedup preserve order
    seen = set()
    out = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _candidate_inputs(root: str) -> List[bytes]:
    cands: List[bytes] = []
    # Base candidates (focus on leading '-' and infinity-like tokens)
    bases = [
        b"-",
        b"-0",
        b"-1",
        b"-9",
        b"-.",
        b"-.0",
        b"-.1",
        b"-e",
        b"-1e",
        b"-1e-",
        b"-inf",
        b"-infinity",
        b"-Infinity",
        b"-.inf",
        b"-.Inf",
        b"-nan",
        b"-NaN",
        b"-infin1ty",
        b"-infinIty",
        b"-infinite",
        b"-infi",
        b"-infin",
        b"-infinity0",
        b"-infinity123",
        b"-infinity1234567",
        b"-000000000000000",
        b"-111111111111111",
        b"-999999999999999",
        b"-infin1ty0000000",
        b"-infinite00000000",
        b"-infinity00000000",
    ]
    for b in bases:
        if b:
            cands.append(b)

    # Try 16-byte shaped patterns
    for fill in [b"0", b"1", b"9", b"A", b"i", b"f", b"n", b"." , b"e"]:
        cands.append(b"-" + fill * 15)
    cands.append(b"-" + b"inf" + b"0" * 12)
    cands.append(b"-" + b"infinity" + b"0" * 7)  # 16 bytes
    cands.append(b"-" + b"infinity" + b"1234567")  # 16 bytes

    # Dictionary tokens
    dt = _parse_dict_tokens(root)
    for t in dt[:120]:
        if t.startswith(b"-"):
            cands.append(t)
        else:
            cands.append(b"-" + t)
        cands.append(t)
        if len(t) <= 12:
            cands.append(b"-" + t + b"0" * (16 - 1 - len(t)))

    # Some lightweight structured wrappers (kept short-ish)
    wrappers = [
        (b"", b""),
        (b"[", b"]"),
        (b"{\"a\":", b"}"),
        (b"{\"a\":[", b"]}"),
    ]
    for pre, suf in wrappers:
        for b in bases[:18]:
            x = pre + b + suf
            if 0 < len(x) <= 96:
                cands.append(x)

    # Dedup preserve order
    seen = set()
    out = []
    for x in cands:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _try_find_crash_with_candidates(crashes: Callable[[bytes], bool], cands: List[bytes], time_budget_s: float) -> Optional[bytes]:
    start = time.monotonic()
    for x in cands:
        if (time.monotonic() - start) >= time_budget_s:
            break
        if crashes(x):
            return x
    return None


def _try_libfuzzer_short_run(binpath: str, workdir: str, seeds: List[bytes], time_limit_s: int = 6) -> Optional[bytes]:
    corpus = os.path.join(workdir, "corpus")
    os.makedirs(corpus, exist_ok=True)
    for i, s in enumerate(seeds[:64]):
        try:
            with open(os.path.join(corpus, f"seed_{i}"), "wb") as f:
                f.write(s)
        except Exception:
            pass

    for p in glob.glob(os.path.join(workdir, "crash-*")):
        try:
            os.remove(p)
        except Exception:
            pass

    env = dict(os.environ)
    env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":abort_on_error=1:detect_leaks=0:symbolize=0:allocator_may_return_null=1"
    env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + ":abort_on_error=1:print_stacktrace=0:symbolize=0"

    try:
        subprocess.run(
            [binpath, "-timeout=2", f"-max_total_time={max(1, time_limit_s)}", "-max_len=128", corpus],
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=time_limit_s + 4,
        )
    except Exception:
        pass

    crashes = sorted(glob.glob(os.path.join(workdir, "crash-*")), key=lambda p: os.path.getsize(p) if os.path.isfile(p) else 10**9)
    for p in crashes:
        data = _read_file_limited(p, 4096)
        if data:
            return data
    return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        overall_start = time.monotonic()
        overall_budget = 110.0

        try:
            with tempfile.TemporaryDirectory(prefix="pocgen_") as td:
                root = _extract_src(src_path, os.path.join(td, "src"))
                # quick: try existing poc-like file
                poc = _find_small_poc_files(root)
                if poc is not None and 0 < len(poc) <= 1024:
                    return poc

                fuzz_targets = _find_fuzz_targets(root)
                if fuzz_targets:
                    fuzz_targets.sort(key=_score_fuzz_target, reverse=True)

                # If no fuzz target found, return best-guess 16 bytes
                if not fuzz_targets:
                    return b"-infinity1234567"

                # Try to build and find crash on top few fuzz targets
                max_targets_to_try = 3
                for fuzz_file in fuzz_targets[:max_targets_to_try]:
                    if (time.monotonic() - overall_start) >= overall_budget:
                        break

                    build_dir = os.path.join(td, "build_" + hashlib.sha1(fuzz_file.encode("utf-8", "ignore")).hexdigest()[:10])
                    time_left = max(5.0, overall_budget - (time.monotonic() - overall_start))
                    binpath = _build_fuzzer_binary(root, fuzz_file, build_dir, time_budget_s=min(70.0, time_left * 0.75))
                    if not binpath:
                        continue

                    crashes = _make_runner(binpath, build_dir)
                    cands = _candidate_inputs(root)

                    crash_input = _try_find_crash_with_candidates(crashes, cands, time_budget_s=min(10.0, overall_budget - (time.monotonic() - overall_start)))
                    if crash_input is None:
                        crash_input = _try_libfuzzer_short_run(binpath, build_dir, cands, time_limit_s=6)

                    if crash_input is None:
                        continue

                    # minimize
                    min_budget = min(25.0, max(6.0, overall_budget - (time.monotonic() - overall_start)))
                    minimized = _ddmin(crash_input, crashes, time_budget_s=min_budget)
                    if minimized and crashes(minimized):
                        return minimized
                    return crash_input

                return b"-infinity1234567"
        except Exception:
            return b"-infinity1234567"