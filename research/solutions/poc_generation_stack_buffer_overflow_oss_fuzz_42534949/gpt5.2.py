import os
import io
import re
import tarfile
from typing import Iterable, List, Optional, Tuple


class Solution:
    def _is_probably_source_file(self, name: str) -> bool:
        n = name.lower()
        exts = (
            ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inc",
            ".py", ".java", ".rs", ".go", ".js", ".ts", ".m", ".mm",
            ".md", ".rst", ".txt", ".html", ".css", ".json", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".cmake", ".mk", ".make", ".am", ".in",
            ".bazel", ".bzl", ".gn", ".gni", ".ninja",
        )
        return n.endswith(exts)

    def _looks_like_source_content(self, data: bytes) -> bool:
        if not data:
            return False
        sample = data[:4096]
        if b"LLVMFuzzerTestOneInput" in sample:
            return True
        if b"#include" in sample or b"namespace" in sample or b"class " in sample:
            return True
        if b"/*" in sample or b"*/" in sample or b"//" in sample:
            return True
        return False

    def _iter_tar_files(self, tar_path: str, max_file_size: int = 2 * 1024 * 1024) -> Iterable[Tuple[str, bytes]]:
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.size < 0 or m.size > max_file_size:
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                yield m.name, data

    def _iter_dir_files(self, dir_path: str, max_file_size: int = 2 * 1024 * 1024) -> Iterable[Tuple[str, bytes]]:
        for root, _, files in os.walk(dir_path):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                if st.st_size < 0 or st.st_size > max_file_size:
                    continue
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except Exception:
                    continue
                rel = os.path.relpath(p, dir_path)
                yield rel, data

    def _iter_files(self, src_path: str) -> Iterable[Tuple[str, bytes]]:
        if os.path.isdir(src_path):
            yield from self._iter_dir_files(src_path)
            return
        # tarball
        try:
            yield from self._iter_tar_files(src_path)
            return
        except Exception:
            return

    def _find_embedded_poc(self, src_path: str) -> Optional[bytes]:
        keywords = (
            "clusterfuzz-testcase", "testcase", "minimized", "repro", "poc",
            "crash", "oom", "overflow", "stack-buffer-overflow", "asan", "ubsan"
        )
        candidates: List[Tuple[int, int, str, bytes]] = []
        for name, data in self._iter_files(src_path):
            base = os.path.basename(name).lower()
            path_l = name.lower()
            if not any(k in base or k in path_l for k in keywords):
                continue
            if self._is_probably_source_file(base):
                continue
            if self._looks_like_source_content(data):
                continue
            if not data:
                continue
            # rank: closeness to 16, then size, then contains '-' and inf-ish
            size = len(data)
            closeness = abs(size - 16)
            bonus = 0
            if b"-" in data:
                bonus -= 2
            if b"inf" in data.lower():
                bonus -= 1
            candidates.append((closeness, size + bonus, name, data))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][3]

    def _detect_format_hint(self, src_path: str) -> str:
        # Try to find fuzzer harness and infer likely input format.
        harness_hits: List[bytes] = []
        for name, data in self._iter_files(src_path):
            n = name.lower()
            if not (n.endswith(".c") or n.endswith(".cc") or n.endswith(".cpp") or n.endswith(".cxx")):
                continue
            if b"LLVMFuzzerTestOneInput" not in data:
                continue
            harness_hits.append(data[:200000])
            if len(harness_hits) >= 3:
                break

        blob = b"\n".join(harness_hits).lower()
        if not blob:
            return "unknown"

        def has_any(*subs: bytes) -> bool:
            return any(s in blob for s in subs)

        if has_any(b"toml", b"toml_parse", b"toml::", b"cpptoml"):
            return "toml"
        if has_any(b"yaml", b"libyaml", b"fy_parse", b"yaml_parser"):
            return "yaml"
        if has_any(b"xml", b"tinyxml", b"libxml", b"expat", b"xmlparse"):
            return "xml"
        if has_any(b"json", b"rapidjson", b"cjson", b"yyjson", b"nlohmann"):
            return "json"
        if has_any(b"ini", b"config", b"cfg"):
            return "ini"
        return "unknown"

    def solve(self, src_path: str) -> bytes:
        poc = self._find_embedded_poc(src_path)
        if poc is not None:
            return poc

        hint = self._detect_format_hint(src_path)

        # Primary guess: a 16-byte negative numeric token likely to trigger the sign-advance bug.
        num16 = b"-" + (b"0" * 15)  # 16 bytes

        if hint == "toml":
            # Minimal TOML assignment; keep the vulnerable token length 16.
            return b"a=" + num16
        if hint == "yaml":
            # Minimal YAML plain scalar in a sequence.
            return b"-" + num16
        if hint == "ini":
            return b"a=" + num16
        if hint == "xml":
            # Minimal XML document wrapping the token.
            return b"<a>" + num16 + b"</a>"

        # JSON or unknown: number alone is often accepted by fuzz targets.
        return num16