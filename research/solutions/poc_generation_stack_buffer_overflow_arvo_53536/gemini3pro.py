import os
import sys
import tarfile
import tempfile
import subprocess
import glob

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        Strategy:
        1. Extract the source code.
        2. Attempt to compile the project with AddressSanitizer and LibFuzzer.
           - Try CMake first.
           - Fallback to manual compilation of all sources if CMake fails.
        3. Identify the fuzzer binary (linked with libFuzzer).
        4. Run the fuzzer for a limited time to generate a crash.
        5. Return the crash input.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Extract source
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=temp_dir)
            except Exception:
                return b""

            # Setup environment for compilation
            env = os.environ.copy()
            env['CC'] = 'clang'
            env['CXX'] = 'clang++'
            env['CFLAGS'] = '-fsanitize=address,fuzzer -g -O1'
            env['CXXFLAGS'] = '-fsanitize=address,fuzzer -g -O1'
            
            fuzzer_bin = None

            # 2. Build Strategy A: CMake
            # Look for CMakeLists.txt
            cmake_lists = glob.glob(os.path.join(temp_dir, '**', 'CMakeLists.txt'), recursive=True)
            if cmake_lists:
                # Use the top-most CMakeLists.txt
                cmake_lists.sort(key=lambda x: len(x.split(os.sep)))
                root_cmake = os.path.dirname(cmake_lists[0])
                build_dir = os.path.join(root_cmake, 'build_poc')
                os.makedirs(build_dir, exist_ok=True)
                
                try:
                    subprocess.run(['cmake', '..'], cwd=build_dir, env=env, 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['make', '-j8'], cwd=build_dir, env=env, 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Search for the fuzzer binary
                    for root, _, files in os.walk(build_dir):
                        for f in files:
                            path = os.path.join(root, f)
                            if os.access(path, os.X_OK):
                                try:
                                    res = subprocess.run([path, '-help=1'], capture_output=True, text=True)
                                    if "libFuzzer" in res.stderr:
                                        fuzzer_bin = path
                                        break
                                except:
                                    continue
                        if fuzzer_bin: break
                except Exception:
                    pass

            # 2. Build Strategy B: Manual Compilation (Fallback)
            if not fuzzer_bin:
                fuzzer_src = None
                sources = []
                include_dirs = set()
                
                # Scan for sources and include dirs
                for root, _, files in os.walk(temp_dir):
                    include_dirs.add(root)
                    for f in files:
                        path = os.path.join(root, f)
                        if f.endswith(('.c', '.cc', '.cpp')):
                            try:
                                with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
                                    content = fp.read()
                                    if "LLVMFuzzerTestOneInput" in content:
                                        fuzzer_src = path
                                    # Skip files with main() to avoid multiple entry points
                                    elif " main(" not in content and "\nmain(" not in content:
                                        sources.append(path)
                            except:
                                pass
                
                if fuzzer_src:
                    compiler = 'clang++' if fuzzer_src.endswith(('.cc', '.cpp')) else 'clang'
                    out_bin = os.path.join(temp_dir, 'manual_fuzzer')
                    
                    # Compile command
                    cmd = [compiler] + env['CFLAGS'].split() + \
                          [f"-I{d}" for d in include_dirs] + \
                          [fuzzer_src] + sources + ['-o', out_bin]
                    
                    try:
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        fuzzer_bin = out_bin
                    except Exception:
                        pass

            # 3. Fuzzing
            if fuzzer_bin:
                # Identify seed corpus if available
                corpus_dirs = []
                for root, _, files in os.walk(temp_dir):
                    if any(k in os.path.basename(root).lower() for k in ['corpus', 'seed', 'test']):
                        if files:
                            corpus_dirs.append(root)
                
                # Run fuzzer
                # -max_total_time=60: stop after 60s
                # -artifact_prefix: save crashes to temp_dir
                cmd = [fuzzer_bin, '-max_total_time=60', f'-artifact_prefix={temp_dir}/'] + corpus_dirs
                
                try:
                    subprocess.run(cmd, cwd=temp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
                
                # 4. Return Crash
                crashes = glob.glob(os.path.join(temp_dir, 'crash-*'))
                if crashes:
                    try:
                        with open(crashes[0], 'rb') as f:
                            return f.read()
                    except Exception:
                        pass

        # Return empty bytes if failed
        return b""