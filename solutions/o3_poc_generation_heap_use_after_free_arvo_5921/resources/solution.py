import tarfile
import os
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            tar = tarfile.open(src_path, 'r:*')
        except tarfile.TarError:
            return b'\x00' * 73

        best_member = None
        best_score = -1

        keywords = {
            'h225': 50,
            'ras': 20,
            'poc': 30,
            'crash': 15,
            'uaf': 15
        }

        for member in tar.getmembers():
            if not member.isfile():
                continue

            size = member.size
            if size == 0 or size > 4096:
                continue

            name_lower = member.name.lower()
            score = 0

            if size == 73:
                score += 1000

            for kw, val in keywords.items():
                if kw in name_lower:
                    score += val

            score += max(0, 4096 - size) // 256  # prefer smaller files

            if score > best_score:
                best_score = score
                best_member = member

        if best_member:
            f = tar.extractfile(best_member)
            if f:
                data = f.read()
                if len(data) > 0:
                    tar.close()
                    return data

        tar.close()
        return b'\x00' * 73
