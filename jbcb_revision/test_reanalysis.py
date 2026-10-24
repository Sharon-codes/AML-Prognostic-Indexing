"""Mathematical unit tests; toy arrays are not patient predictions."""
import unittest
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("reanalysis",Path(__file__).with_name("reanalysis.py"))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Tests(unittest.TestCase):
    def test_uniform_identical_stability(self):
        self.assertEqual(mod.nogueira({"30":30},30,1000),1)
    def test_recorded_histogram(self):
        h={"1":333,"2":147,"3":44,"4":16,"5":4,"6":3,"7":1,"8":1,"9":1,"15":1}
        self.assertAlmostEqual(mod.nogueira(h,30,1000),0.02042106094718954)
    def test_wilson_all_success(self):
        a=mod.wilson(51,51);self.assertLess(a[0],1);self.assertAlmostEqual(a[1],1)
    def test_wilson_balanced(self):
        a=mod.wilson(30,50);self.assertLess(a[0],.6);self.assertGreater(a[1],.6)
    def test_invalid_counts(self):
        with self.assertRaises(ValueError):mod.wilson(2,1)
if __name__=="__main__":unittest.main()
