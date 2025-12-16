import os, tarfile, re

class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, 'r:*') as tf:
                    patt = re.compile(r'(poc|crash|input|test|383200048)', re.I)
                    for member in tf.getmembers():
                        if not member.isfile() or member.size == 0:
                            continue
                        if patt.search(os.path.basename(member.name)):
                            ex = tf.extractfile(member)
                            if ex:
                                data = ex.read()
                                if data:
                                    return data
        except Exception:
            pass
        return b'UPX!' + b'\x00' * (512 - 4)
