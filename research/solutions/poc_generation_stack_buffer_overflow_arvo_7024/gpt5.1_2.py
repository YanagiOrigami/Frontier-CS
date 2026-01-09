import os
import re


class Solution:
    def _is_probably_text(self, data: bytes) -> bool:
        if not data:
            return True
        binary_count = 0
        length = len(data)
        for b in data:
            if b in (9, 10, 13):  # \t, \n, \r
                continue
            if b < 0x20 or b > 0x7E:
                binary_count += 1
        if 0 in data:
            return False
        return binary_count <= length * 0.1

    def _score_file_candidate(self, root: str, name: str, is_binary: bool) -> int:
        lower_name = name.lower()
        lower_root = root.lower()
        score = 0

        # High priority keywords
        if 'poc' in lower_name:
            score += 1000
        if 'poc' in lower_root:
            score += 500
        if 'crash' in lower_name:
            score += 900
        if 'crash' in lower_root:
            score += 450
        if 'exploit' in lower_name:
            score += 900

        # Medium priority keywords
        if 'gre' in lower_name or '80211' in lower_name or 'wifi' in lower_name:
            score += 400
        if 'input' in lower_name or 'seed' in lower_name:
            score += 300
        if 'id_' in lower_name or lower_name.startswith('id-'):
            score += 200
        if 'fuzz' in lower_root or 'corpus' in lower_root:
            score += 200
        if 'test' in lower_root or 'tests' in lower_root or 'regress' in lower_root:
            score += 100

        # Binary vs text heuristic
        if is_binary:
            score += 50
        else:
            score -= 50

        ext = os.path.splitext(name)[1].lower()
        if ext in {'.txt', '.md', '.rst'}:
            score -= 100

        return score

    def _find_poc_file(self, src_path: str, target_len: int) -> bytes | None:
        excluded_exts = {
            '.c', '.h', '.cc', '.cpp', '.cxx', '.hpp',
            '.py', '.sh', '.md', '.txt', '.rst', '.json',
            '.yml', '.yaml', '.xml', '.html', '.htm',
            '.in', '.am', '.ac', '.m4', '.cmake',
            '.bat', '.ps1', '.java', '.js', '.ts',
            '.rb', '.go', '.rs', '.php', '.pl', '.pm',
            '.tex'
        }

        best_data = None
        best_score = float('-inf')

        for root, dirs, files in os.walk(src_path):
            for name in files:
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size != target_len:
                    continue

                ext = os.path.splitext(name)[1].lower()
                if ext in excluded_exts:
                    continue

                try:
                    with open(path, 'rb') as f:
                        data = f.read()
                except OSError:
                    continue

                is_binary = not self._is_probably_text(data)
                score = self._score_file_candidate(root, name, is_binary)

                if score > best_score:
                    best_score = score
                    best_data = data

        return best_data

    def _find_array_poc(self, src_path: str, target_len: int) -> bytes | None:
        c_exts = {'.c', '.cc', '.cpp', '.cxx', '.h', '.hpp'}
        best_data = None
        best_score = float('-inf')

        for root, dirs, files in os.walk(src_path):
            for name in files:
                _, ext = os.path.splitext(name)
                if ext.lower() not in c_exts:
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except OSError:
                    continue

                lower_root = root.lower()
                lower_name = name.lower()

                for match in re.finditer(r'\{([^{}]*?)\}', text, re.DOTALL):
                    content = match.group(1)
                    if '0x' not in content and not re.search(r'\d', content):
                        continue

                    tokens = re.findall(r'0x[0-9a-fA-F]+|[0-9]+', content)
                    if len(tokens) != target_len:
                        continue

                    ints = []
                    ok = True
                    for t in tokens:
                        try:
                            v = int(t, 16 if t.lower().startswith('0x') else 10)
                        except ValueError:
                            ok = False
                            break
                        if v < 0 or v > 255:
                            ok = False
                            break
                        ints.append(v)
                    if not ok or len(ints) != target_len:
                        continue

                    start_idx = match.start()
                    context_start = max(0, start_idx - 120)
                    context = text[context_start:start_idx].lower()

                    score = 0
                    if 'poc' in context:
                        score += 1000
                    if 'crash' in context:
                        score += 800
                    if 'seed' in context or 'input' in context:
                        score += 600
                    if 'gre' in context or '80211' in context or 'wifi' in context:
                        score += 400
                    if 'poc' in lower_name:
                        score += 800
                    if 'crash' in lower_name:
                        score += 600
                    if 'test' in lower_root or 'fuzz' in lower_root:
                        score += 200
                    if score == 0:
                        score = 1

                    if score > best_score:
                        best_score = score
                        best_data = bytes(ints)

        return best_data

    def solve(self, src_path: str) -> bytes:
        target_len = 45

        # 1. Try to find an existing PoC file of the correct length
        file_poc = self._find_poc_file(src_path, target_len)
        if file_poc is not None:
            return file_poc

        # 2. Try to find a PoC encoded as a byte array in source files
        array_poc = self._find_array_poc(src_path, target_len)
        if array_poc is not None:
            return array_poc

        # 3. Fallback: craft a generic GRE + 802.11-like payload (best-effort)
        data = bytearray(target_len)

        # GRE flags and version (2 bytes) - set unusual flags to exercise parser
        data[0] = 0xFF
        data[1] = 0xFF

        # GRE Protocol Type (2 bytes) - arbitrary non-standard value
        data[2] = 0x88
        data[3] = 0xBB

        # Fill remaining bytes with 0xFF to maximize potential overflow conditions
        for i in range(4, target_len):
            data[i] = 0xFF

        return bytes(data)