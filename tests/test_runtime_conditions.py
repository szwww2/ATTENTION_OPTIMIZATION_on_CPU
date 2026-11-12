"""Check actual child affinity inheritance without running an attention benchmark."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from runtime_conditions import parse_cpu_list, current_cpu_list


class RuntimeConditions(unittest.TestCase):
    def test_invalid_selection(self):
        for value in ('', '0,0', '-1', 'x', '1,,2'):
            with self.assertRaises(ValueError):
                parse_cpu_list(value)
        self.assertEqual(parse_cpu_list('3,1'),[3,1])

    def test_child_inherits_one_cpu(self):
        available=current_cpu_list()
        if not available:self.skipTest('Affinity is unsupported')
        code="from runtime_conditions import *; import subprocess,sys; apply_cpu_affinity(sys.argv[1]); print(subprocess.check_output([sys.executable,'-c','from runtime_conditions import *; import json; print(json.dumps(current_cpu_list()))'],text=True).strip())"
        result=subprocess.check_output([sys.executable,'-c',code,str(available[0])],cwd=Path(__file__).resolve().parents[1]/'tools',text=True)
        self.assertEqual(json.loads(result),[available[0]])
        self.assertEqual(current_cpu_list(),available)


if __name__=='__main__':unittest.main()
