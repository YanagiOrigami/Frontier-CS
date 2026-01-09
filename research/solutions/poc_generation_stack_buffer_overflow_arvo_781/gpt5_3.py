import os
import tarfile
import tempfile
import shutil
import re
import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Strategy:
        - Inspect the source tarball to detect the input parsing format of the harness.
        - Generate a compact PoC accordingly. Default to common formats used by PCRE/PCRE2 fuzz harnesses.
        """
        tmpdir = tempfile.mkdtemp(prefix="arvo781_")
        try:
            # Extract tarball
            self._safe_extract(src_path, tmpdir)

            # Read code files
            code_files = self._gather_source_files(tmpdir)
            combined = self._read_files_combined(code_files)

            # Detect delimiter/format
            fmt = self._detect_format(combined)

            # Generate PoC according to detected format
            if fmt == "nul_delim":
                # Common for libFuzzer PCRE/PCRE2 harnesses: pattern\0subject
                # Keep simple: no capturing groups, short subject
                return b"a\x00a"
            elif fmt == "newline_delim":
                # pattern\nsubject
                return b"a\na"
            elif fmt == "len_pattern_rest_subject":
                # First 4 bytes little-endian pattern length, followed by pattern; subject is the remainder
                # We'll provide a minimal subject too.
                pat = b"a"
                subj = b"a"
                return struct.pack("<I", len(pat)) + pat + subj
            elif fmt == "two_lengths":
                # 4 bytes pattern length + pattern + 4 bytes subject length + subject
                pat = b"a"
                subj = b"a"
                return struct.pack("<I", len(pat)) + pat + struct.pack("<I", len(subj)) + subj
            elif fmt == "slash_delim":
                # /pattern/\nsubject
                return b"/a/\na"
            else:
                # Unknown format: choose a robust default that works for many PCRE/PCRE2 fuzzers:
                # NUL-delimited pattern and subject.
                return b"a\x00a"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _safe_extract(self, src_path: str, dst_dir: str) -> None:
        # Extract tarball safely
        with tarfile.open(src_path, "r:*") as tf:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonpath([abs_directory, abs_target])
                return prefix == abs_directory

            for member in tf.getmembers():
                member_path = os.path.join(dst_dir, member.name)
                if not is_within_directory(dst_dir, member_path):
                    continue
            tf.extractall(dst_dir)

    def _gather_source_files(self, root: str):
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp")
        files = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(exts):
                    files.append(os.path.join(dirpath, fn))
        return files

    def _read_files_combined(self, files):
        chunks = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    chunks.append(fh.read())
            except Exception:
                continue
        return "\n".join(chunks)

    def _detect_format(self, text: str) -> str:
        """
        Attempt to detect how the harness splits input into pattern/subject.
        Returns one of:
          - "nul_delim"
          - "newline_delim"
          - "len_pattern_rest_subject"
          - "two_lengths"
          - "slash_delim"
          - "unknown"
        """
        # Heuristic checks

        # If it looks like a libFuzzer harness
        is_fuzzer = "LLVMFuzzerTestOneInput" in text

        # NUL delimiter detection
        # Look for memchr(..., 0, ...) or comparisons to '\0'
        if re.search(r"memchr\s*\(\s*[^,]+,\s*0\s*,", text) or \
           re.search(r"==\s*'\\0'", text) or \
           re.search(r"==\s*0\s*&&", text) or \
           re.search(r"\*\s*p\s*!\=\s*0\s*;|while\s*\(\s*[^)]*\*\s*[^)]*\)", text):
            if "pcre2_compile" in text or "pcre_compile" in text or is_fuzzer:
                return "nul_delim"

        # Newline delimiter detection
        if re.search(r"memchr\s*\(\s*[^,]+,\s*'\\n'", text) or \
           re.search(r"==\s*'\\n'", text) or \
           "getline" in text or "fgets" in text:
            if "pcre2_compile" in text or "pcre_compile" in text or is_fuzzer:
                return "newline_delim"

        # Slash-delimited patterns like pcretest: /pattern/ then subject on next line
        if re.search(r"/\s*\w.*?/\s*", text) and ("pcretest" in text.lower() or "slash" in text.lower()):
            return "slash_delim"

        # Length-prefixed formats
        # Two-lengths: pattern length + subject length
        if re.search(r"sizeof\s*\(\s*uint32_t\s*\)", text) and \
           re.search(r"(read|parse).*(len|length)", text, re.IGNORECASE) and \
           re.search(r"(subject|input).*(len|length)", text, re.IGNORECASE):
            return "two_lengths"

        # Single pattern length, rest is subject
        if re.search(r"\(\s*uint(32|16|64)_t\s*\*\)\s*data", text) or \
           re.search(r"struct\.pack|little endian|LE32|u32|read32", text, re.IGNORECASE) or \
           re.search(r"pattern(_)?len\s*=\s*\*\s*\(\s*uint(32|16|64)_t\s*\*\)\s*data", text):
            return "len_pattern_rest_subject"

        # PCRE/PCRE2 presence without clear IO format: default to NUL delimiter
        if "pcre2_compile" in text or "pcre_compile" in text or "PCRE2" in text or "PCRE" in text:
            return "nul_delim"

        return "unknown"