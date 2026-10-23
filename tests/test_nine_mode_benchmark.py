"""Exercise the combined binary, its saved schema and failed team-size gate."""
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'build' / ('attn_bench_all.exe' if os.name == 'nt' else 'attn_bench_all')

class NineModeTests(unittest.TestCase):
    def test_odd_shape_and_clipped_tiles(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            prefix = str(Path(folder)/'nine')
            run = subprocess.run([str(BINARY),'--streaming','--mode','all','--threads','2',
                '--L','31','--d','7','--dv','9','--query-block','64','--key-block','256',
                '--warmup','1','--reps','3','--out',os.path.relpath(prefix,ROOT)],
                cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            with open(prefix+'_summary.csv',newline='') as f: rows=list(csv.DictReader(f))
            with open(prefix+'_raw.csv',newline='') as f: raw=list(csv.DictReader(f))
            self.assertEqual(len(rows),9)
            self.assertEqual(len(raw),27)
            self.assertEqual({(r['route'],r['mode']) for r in rows},
                {(r,m) for r in ('full_matrix','rowwise','blocked_streaming') for m in ('scalar','simd','openmp')})
            for r in rows:
                self.assertEqual(r['contract_pass'],'1')
                self.assertEqual(r['actual_threads'],r['threads'])
                self.assertEqual(r['requested_query_block'],'64')
                self.assertEqual(r['requested_key_block'],'256')
                if r['route']=='blocked_streaming':
                    self.assertEqual(r['key_block'],'31')
                    self.assertEqual(int(r['workspace_bytes']),4*int(r['threads'])*(31*31+31*9+2*31))
            for sample in range(3):
                self.assertEqual(sorted(int(r['order']) for r in raw if int(r['sample'])==sample),list(range(9)))

    def test_constrained_team_cannot_publish(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            prefix = str(Path(folder)/'failed')
            env=dict(os.environ,OMP_THREAD_LIMIT='1')
            run=subprocess.run([str(BINARY),'--streaming','--mode','openmp','--threads','2',
                '--L','3','--out',os.path.relpath(prefix,ROOT)],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertFalse(Path(prefix+'_raw.csv').exists())
            self.assertFalse(Path(prefix+'_summary.csv').exists())

if __name__=='__main__': unittest.main()
