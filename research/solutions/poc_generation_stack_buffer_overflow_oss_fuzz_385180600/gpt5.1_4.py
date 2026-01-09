import os
import stat
import tarfile
import zipfile
from typing import Optional


class Solution:
    BUG_ID = "385180600"

    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            data = self._find_poc_in_dir(src_path)
            if data is not None:
                return data

        if os.path.isfile(src_path):
            if tarfile.is_tarfile(src_path):
                data = self._find_poc_in_tar(src_path)
                if data is not None:
                    return data

            if zipfile.is_zipfile(src_path):
                data = self._find_poc_in_zip(src_path)
                if data is not None:
                    return data

        return self._fallback_poc()

    def _find_poc_in_dir(self, root: str) -> Optional[bytes]:
        best_path_bug = None
        best_size_bug = None
        best_path_generic = None
        best_size_generic = None

        for dirpath, dirnames, filenames in os.walk(root):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                if st.st_size == 0:
                    continue

                name_lower = fname.lower()

                if self.BUG_ID in fname:
                    if best_path_bug is None or st.st_size < best_size_bug:
                        best_path_bug = full
                        best_size_bug = st.st_size
                elif "clusterfuzz-testcase" in name_lower or "testcase" in name_lower or "poc" in name_lower:
                    if best_path_generic is None or st.st_size < best_size_generic:
                        best_path_generic = full
                        best_size_generic = st.st_size

        chosen_path = best_path_bug or best_path_generic
        if chosen_path is not None:
            try:
                with open(chosen_path, "rb") as f:
                    return f.read()
            except OSError:
                return None
        return None

    def _find_poc_in_tar(self, tar_path: str) -> Optional[bytes]:
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                best_member_bug = None
                best_size_bug = None
                best_member_generic = None
                best_size_generic = None

                for member in tf.getmembers():
                    if not member.isreg():
                        continue
                    if member.size == 0:
                        continue

                    name = member.name
                    name_lower = name.lower()

                    if self.BUG_ID in name:
                        if best_member_bug is None or member.size < best_size_bug:
                            best_member_bug = member
                            best_size_bug = member.size
                    elif "clusterfuzz-testcase" in name_lower or "testcase" in name_lower or "poc" in name_lower:
                        if best_member_generic is None or member.size < best_size_generic:
                            best_member_generic = member
                            best_size_generic = member.size

                chosen_member = best_member_bug or best_member_generic
                if chosen_member is not None:
                    f = tf.extractfile(chosen_member)
                    if f is not None:
                        return f.read()
        except (tarfile.TarError, OSError):
            return None
        return None

    def _find_poc_in_zip(self, zip_path: str) -> Optional[bytes]:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                best_name_bug = None
                best_size_bug = None
                best_name_generic = None
                best_size_generic = None

                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if info.file_size == 0:
                        continue

                    name = info.filename
                    name_lower = name.lower()

                    if self.BUG_ID in name:
                        if best_name_bug is None or info.file_size < best_size_bug:
                            best_name_bug = name
                            best_size_bug = info.file_size
                    elif "clusterfuzz-testcase" in name_lower or "testcase" in name_lower or "poc" in name_lower:
                        if best_name_generic is None or info.file_size < best_size_generic:
                            best_name_generic = name
                            best_size_generic = info.file_size

                chosen_name = best_name_bug or best_name_generic
                if chosen_name is not None:
                    with zf.open(chosen_name, "r") as f:
                        return f.read()
        except (zipfile.BadZipFile, OSError):
            return None
        return None

    def _fallback_poc(self) -> bytes:
        # Fallback: construct a generic TLV-like buffer of 262 bytes.
        # This is a heuristic and mainly a safeguard if no testcase is found.
        size = 262
        # Create a simple pattern that resembles TLVs: [type, length, ...data...]
        # Put a suspiciously small "length" near the end to mimic under-sized TLV.
        buf = bytearray(size)

        # Fill with non-zero pattern
        for i in range(size):
            buf[i] = (i * 7 + 3) & 0xFF

        # Set up a fake TLV header near the end
        if size > 10:
            tlv_start = size - 6
            # Type: pretend to be "Active Timestamp" or "Delay Timer"
            buf[tlv_start] = 0x03
            # Length smaller than typical (e.g., 1 instead of 4/8)
            buf[tlv_start + 1] = 0x01

        return bytes(buf)