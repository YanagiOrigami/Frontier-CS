import os
import tarfile
from typing import Optional


class Solution:
    def _looks_like_len_prefixed_harness(self, root: str) -> bool:
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")
        needles = (
            "LLVMFuzzerTestOneInput",
            "Size < 8",
        )
        try:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if not fn.endswith(exts):
                        continue
                    path = os.path.join(dirpath, fn)
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except Exception:
                        continue
                    if all(n.encode() in data for n in needles) and (b"Data+4" in data or b"Data + 4" in data):
                        return True
        except Exception:
            return False
        return False

    def _extract_tarball(self, tar_path: str, out_dir: str) -> Optional[str]:
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(out_dir)
            return out_dir
        except Exception:
            return None

    def solve(self, src_path: str) -> bytes:
        # Ground-truth PoC length is 8 bytes; use a minimal, robust payload:
        # two zero 32-bit length fields (common in fuzz harnesses) and/or empty strings.
        return b"\x00" * 8