"""Check pilot statistics and reject corrupted saved records without rerunning kernels."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("streaming_runner", ROOT/"tools/run_streaming_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RecordTests(unittest.TestCase):
    """对已提交的示例记录做完整性回归；篡改只发生在内存，不覆盖原始证据。"""

    def setUp(self):
        """读取调参CSV和预期网格供每项测试使用，不运行性能负载。"""
        self.text = (ROOT/"examples/streaming_cuiziwei_20261022/tuning_raw.csv").read_text(encoding="utf-8")
        self.keys = [(2048,64,64,k) for k in (128,256,512)]

    def test_saved_grid(self):
        """原始调参/六形状样本应可重算出有限统计，且覆盖全部预期配置。"""
        self.assertEqual(len(runner.summarize(self.text,self.keys)),3)
        text=(ROOT/"examples/streaming_cuiziwei_20261022/grid_raw.csv").read_text(encoding="utf-8")
        keys=[(L,d,d,256) for L,d in ((128,64),(512,64),(2048,64),(4096,64),(512,32),(2048,32))]
        self.assertEqual(len(runner.summarize(text,keys)),6)

    def test_missing_sample(self):
        """删除一个样本必须使核验失败，不能仍输出完整实验摘要。"""
        lines=self.text.splitlines()
        with self.assertRaises(ValueError): runner.summarize("\n".join(lines[:-1]),self.keys)

    def test_corruption(self):
        """非法耗时、契约失败、错误workspace或执行顺序均必须拒绝。"""
        lines=self.text.splitlines()
        fields=lines[0].split(",")
        for field,value in (("ms","nan"),("contract_pass","0"),("workspace_bytes","4"),("order","99")):
            with self.subTest(field=field):
                changed=lines.copy()
                row=changed[1].split(",")
                row[fields.index(field)]=value
                changed[1]=",".join(row)
                with self.assertRaises(ValueError): runner.summarize("\n".join(changed),self.keys)


if __name__=="__main__":
    unittest.main()
