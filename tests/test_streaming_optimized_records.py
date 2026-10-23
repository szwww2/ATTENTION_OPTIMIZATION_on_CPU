"""Regression checks for the recorded SIMD/OpenMP experiment and rejection paths."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_streaming_optimized import summarize


class Records(unittest.TestCase):
    """在内存破坏记录，核验不完整/不合法证据不能被误汇总。"""
    def setUp(self):
        """读取本轮原始CSV作为fixture，不运行耗时内核。"""
        self.text=(ROOT/"examples/streaming_optimized_20261022/raw.csv").read_text(encoding="utf-8")

    def test_complete(self):
        """全部配置重算，检查真实多线程样本和正加速比字段。"""
        rows=summarize(self.text)
        self.assertEqual(len(rows),24)
        self.assertEqual(max(r["actual_threads"] for r in rows),8)
        self.assertTrue(all(r["speedup_vs_scalar"]>0 for r in rows))

    def test_missing_duplicate(self):
        """丢失或重复样本不允许生成摘要。"""
        lines=self.text.splitlines()
        for damaged in (lines[:-1],lines+[lines[1]]):
            with self.assertRaises(ValueError): summarize("\n".join(damaged))

    def test_corrupt(self):
        """耗时/契约/team/顺序/scratch被篡改时拒绝，而非静默丢弃。"""
        lines=self.text.splitlines(); fields=lines[0].split(",")
        for field,value in (("ms","nan"),("contract_pass","0"),("actual_threads","99"),
                            ("workspace_bytes","1"),("order","99")):
            with self.subTest(field=field):
                changed=lines.copy(); row=changed[1].split(",")
                row[fields.index(field)]=value; changed[1]=",".join(row)
                with self.assertRaises(ValueError): summarize("\n".join(changed))

    def test_variable_team(self):
        """同配置实际team变化时不混合统计；即使每个team单独合法也拒绝。"""
        lines=self.text.splitlines(); fields=lines[0].split(",")
        for i in range(1,len(lines)):
            row=lines[i].split(",")
            if row[fields.index("requested_threads")]=="8":
                row[fields.index("actual_threads")]="4"
                lines[i]=",".join(row); break
        with self.assertRaises(ValueError): summarize("\n".join(lines))


if __name__=="__main__":
    unittest.main()
