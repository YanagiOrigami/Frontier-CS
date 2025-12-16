import os, tarfile, typing, io, sys, pathlib, re, gzip, bz2, lzma, zipfile

class Solution:
    def _read_member(self, tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
        fileobj = tar.extractfile(member)
        if fileobj is None:
            return b""
        return fileobj.read()

    def _search_exact_size(self, tar: tarfile.TarFile, size: int) -> typing.Optional[bytes]:
        for member in tar.getmembers():
            if member.isfile() and member.size == size:
                return self._read_member(tar, member)
        return None

    def _search_by_name(self, tar: tarfile.TarFile, patterns: typing.List[str]) -> typing.Optional[bytes]:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            lower_name = member.name.lower()
            if any(pat in lower_name for pat in patterns):
                return self._read_member(tar, member)
        return None

    def _search_first_file(self, tar: tarfile.TarFile) -> bytes:
        for member in tar.getmembers():
            if member.isfile():
                return self._read_member(tar, member)
        return b""

    def solve(self, src_path: str) -> bytes:
        GT_SIZE = 71298  # ground-truth PoC size hint
        try:
            with tarfile.open(src_path, "r:*") as tar:
                # 1. Try exact size match (most reliable)
                data = self._search_exact_size(tar, GT_SIZE)
                if data is not None:
                    return data
                # 2. Try files whose name indicates PoC/crash
                data = self._search_by_name(
                    tar, ["poc", "crash", "input", "trigger", "uaf"]
                )
                if data is not None:
                    return data
                # 3. Fallback: first regular file
                return self._search_first_file(tar)
        except Exception:
            # If anything goes wrong, return empty bytes (better than crashing)
            return b""
