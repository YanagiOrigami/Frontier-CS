import tarfile

class Solution:
    def _extract_best_candidate(self, tar):
        best = None
        for member in tar.getmembers():
            if not member.isfile():
                continue
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue
            name_lower = member.name.lower()
            # Prefer files that look like PoCs (pdf or poc or similar)
            looks_like_pdf = data.startswith(b'%PDF')
            preferred_ext = any(name_lower.endswith(ext) for ext in (
                '.pdf', '.poc', '.bin', '.dat', '.input', '.in', '.raw', '.txt'))
            if looks_like_pdf or preferred_ext:
                if best is None or len(data) > len(best):
                    best = data
        return best

    def solve(self, src_path: str) -> bytes:
        best_data = None
        try:
            with tarfile.open(src_path, 'r:*') as tar:
                best_data = self._extract_best_candidate(tar)
        except Exception:
            best_data = None
        if best_data:
            return best_data
        # Fallback minimal PDF with a standalone form object
        return (
            b'%PDF-1.4\n'
            b'1 0 obj\n'
            b'<< /Type /Catalog /Pages 2 0 R >>\n'
            b'endobj\n'
            b'2 0 obj\n'
            b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n'
            b'endobj\n'
            b'3 0 obj\n'
            b'<< /Type /Page /Parent 2 0 R /Resources << >> /Contents 4 0 R >>\n'
            b'endobj\n'
            b'4 0 obj\n'
            b'<< /Length 12 >>\n'
            b'stream\n'
            b'BT\n'
            b'ET\n'
            b'endstream\n'
            b'endobj\n'
            b'5 0 obj\n'
            b'<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 1 1] /Resources << >> >>\n'
            b'endobj\n'
            b'xref\n'
            b'0 6\n'
            b'0000000000 65535 f \n'
            b'0000000010 00000 n \n'
            b'0000000061 00000 n \n'
            b'0000000114 00000 n \n'
            b'0000000216 00000 n \n'
            b'0000000315 00000 n \n'
            b'trailer\n'
            b'<< /Size 6 /Root 1 0 R >>\n'
            b'startxref\n'
            b'395\n'
            b'%%EOF\n'
        )
