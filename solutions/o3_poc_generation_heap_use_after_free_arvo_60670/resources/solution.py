class Solution:
    def solve(self, src_path: str) -> bytes:
        return b"""(class foo (bar))

(macro m ((cp classpermission))
    (classpermissionset mycps
        $cp
    )
)

(m (classpermission
        (foo (bar))
))
"""
