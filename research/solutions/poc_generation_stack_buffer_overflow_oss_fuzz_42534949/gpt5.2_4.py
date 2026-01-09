import os
import re
import tarfile
from typing import Iterable, Optional


class Solution:
    def _iter_source_files_from_dir(self, root: str, max_files: int = 1500) -> Iterable[tuple[str, Optional[bytes]]]:
        cnt = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", ".svn", ".hg", "build", "out", "dist")]
            for fn in filenames:
                if cnt >= max_files:
                    return
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, root)
                lower = fn.lower()
                if not any(lower.endswith(ext) for ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inc", ".inl", ".l", ".y", ".rl", ".toml", ".txt", ".md")):
                    yield (rel, None)
                    cnt += 1
                    continue
                data = None
                try:
                    st = os.stat(path)
                    if st.st_size <= 512 * 1024:
                        with open(path, "rb") as f:
                            data = f.read(256 * 1024)
                except Exception:
                    data = None
                yield (rel, data)
                cnt += 1

    def _iter_source_files_from_tar(self, tar_path: str, max_files: int = 1500) -> Iterable[tuple[str, Optional[bytes]]]:
        cnt = 0
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf.getmembers():
                    if cnt >= max_files:
                        return
                    if not m.isfile():
                        continue
                    name = m.name
                    base = os.path.basename(name)
                    lower = base.lower()
                    if not any(lower.endswith(ext) for ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inc", ".inl", ".l", ".y", ".rl", ".toml", ".txt", ".md")):
                        yield (name, None)
                        cnt += 1
                        continue
                    data = None
                    try:
                        if m.size <= 512 * 1024:
                            f = tf.extractfile(m)
                            if f is not None:
                                data = f.read(256 * 1024)
                    except Exception:
                        data = None
                    yield (name, data)
                    cnt += 1
        except Exception:
            return

    def _detect_toml_project(self, src_path: str) -> bool:
        it = self._iter_source_files_from_dir(src_path) if os.path.isdir(src_path) else self._iter_source_files_from_tar(src_path)

        name_hits = 0
        content_hits = 0

        toml_name_re = re.compile(r"(?:^|/|\\)(toml\+\+|toml11|toml)(?:$|/|\\)", re.IGNORECASE)

        for name, data in it:
            nlow = name.lower()
            if nlow.endswith(".toml"):
                name_hits += 3
            if toml_name_re.search(name.replace("\\", "/")):
                name_hits += 2
            if "toml" in os.path.basename(nlow):
                name_hits += 1

            if data:
                low = data.lower()
                if b"#include" in low and (b"toml" in low or b"toml++" in low):
                    content_hits += 2
                if b"namespace toml" in low or b"toml::" in low or b"tomlplusplus" in low or b"toml11" in low:
                    content_hits += 2
                if b"toml" in low:
                    content_hits += 1

            if name_hits + content_hits >= 4:
                return True

        return False

    def solve(self, src_path: str) -> bytes:
        # Try to tailor to TOML-like parsers that need a key/value to reach value parsing.
        is_toml = self._detect_toml_project(src_path)

        total_len = 16
        if is_toml:
            prefix = b"a=-i"
        else:
            prefix = b"-i"

        if len(prefix) >= total_len:
            return prefix[:total_len]

        return prefix + (b"A" * (total_len - len(prefix)))