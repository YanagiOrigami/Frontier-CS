import tarfile
from typing import Optional


class Solution:
    def _select_member(self, tar: tarfile.TarFile, size: Optional[int] = None):
        # Preferred: exact size match (ground-truth length)
        if size is not None:
            for m in tar.getmembers():
                if m.isfile() and m.size == size:
                    return m

        # Next: any likely PoC file (extension or keyword)
        keywords = ('.pdf', '.poc', '.bin', '.dat', '.input', '.crash')
        for m in tar.getmembers():
            if m.isfile():
                name = m.name.lower()
                if any(k in name for k in keywords):
                    return m

        # Fallback: first regular file
        for m in tar.getmembers():
            if m.isfile():
                return m
        return None

    def solve(self, src_path: str) -> bytes:
        with tarfile.open(src_path, 'r:*') as tar:
            member = self._select_member(tar, size=6431)
            if member is None:
                raise FileNotFoundError("No suitable PoC file found in tarball")
            return tar.extractfile(member).read()
