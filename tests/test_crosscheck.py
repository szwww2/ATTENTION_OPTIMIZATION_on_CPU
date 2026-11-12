"""The independent NumPy checker must reject corrupt C++ dumps (requires NumPy)."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "build" / ("attn_dump.exe" if os.name == "nt" else "attn_dump")


class CrosscheckTests(unittest.TestCase):
    def test_real_dump_and_corrupt_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as folder:
            subprocess.run([str(DUMP), "7", "3", "5", "42", "case"], cwd=folder, check=True)
            prefix = Path(folder) / "case"
            output = Path(str(prefix) + "_o32.txt")
            original = output.read_text(encoding="utf-8")
            command = [sys.executable, str(ROOT / "tools/crosscheck_numpy.py"),
                       str(prefix), "7", "3", "5", "42"]
            valid = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
            for bad in ("123", "nan", "inf"):
                with self.subTest(corruption=bad):
                    lines = original.splitlines()
                    lines[0] = bad
                    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    run = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("cpp fp32 outside contract", run.stdout)


if __name__ == "__main__":
    unittest.main()
