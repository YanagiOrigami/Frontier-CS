import os, tarfile, tempfile, glob, re, json, shutil, pathlib, io, sys, itertools, math, random, string, typing, functools, collections

class Solution:
    def _detect_format(self, directory: str) -> str:
        patterns = list(glob.glob(os.path.join(directory, '**', '*.[cC][cCpP]*'), recursive=True))
        for path in patterns:
            try:
                with open(path, 'r', errors='ignore') as f:
                    data = f.read()
            except Exception:
                continue
            if 'polygonToCellsExperimental' not in data:
                continue
            # Look for signs of GeoJSON usage
            if re.search(r'\b(json|rapidjson|nlohmann)::', data):
                return 'geojson'
            # Look for WKT parsing
            if re.search(r'(read_wkt|POLYGON\()', data):
                return 'wkt'
        return 'geojson'  # reasonable default

    def _build_geojson(self) -> bytes:
        # Extremely large rectangle crossing the antimeridian to trigger underestimation
        coords = [
            [179.999999, -89.999999],
            [-179.999999, -89.999999],
            [-179.999999, 89.999999],
            [179.999999, 89.999999],
            [179.999999, -89.999999]
        ]
        obj = {
            "type": "Polygon",
            "coordinates": [coords]
        }
        return json.dumps(obj, separators=(',', ':')).encode()

    def _build_wkt(self) -> bytes:
        # Same rectangle expressed in WKT
        pts = [
            "179.999999 -89.999999",
            "-179.999999 -89.999999",
            "-179.999999 89.999999",
            "179.999999 89.999999",
            "179.999999 -89.999999"
        ]
        wkt = "POLYGON((" + ", ".join(pts) + "))"
        return wkt.encode()

    def solve(self, src_path: str) -> bytes:
        # Extract the tarball to inspect harness and pick format
        tmpdir = tempfile.mkdtemp()
        try:
            with tarfile.open(src_path, 'r:*') as t:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonpath([abs_directory]) + os.sep
                    return os.path.commonpath([abs_directory, abs_target]) == abs_directory
                for m in t.getmembers():
                    if m.isreg():
                        target_path = os.path.join(tmpdir, m.name)
                        if not is_within_directory(tmpdir, target_path):
                            continue
                        t.extract(m, path=tmpdir)
        except Exception:
            pass  # If extraction fails, fallback to default format
        fmt = self._detect_format(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)
        if fmt == 'wkt':
            return self._build_wkt()
        return self._build_geojson()
