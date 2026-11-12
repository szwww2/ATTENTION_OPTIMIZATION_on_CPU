"""Verify the exported evidence without requiring the original machine."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
listed = set()
for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    digest, relative = line.split("  ", 1)
    path = root / relative
    if path.resolve().is_relative_to(root) is False or not path.is_file():
        raise SystemExit("Missing or invalid path: " + relative)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit("SHA256 mismatch: " + relative)
    listed.add(relative)
actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
if actual != listed | {"SHA256SUMS.txt"}:
    raise SystemExit("Unlisted or missing files: " + str(sorted(actual ^ (listed | {"SHA256SUMS.txt"}))))
print("Verified", len(listed), "files; SHA256SUMS.txt is the only self-exclusion.")
