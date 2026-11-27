"""End-to-end run on the synthetic dataset (in a temporary folder).   Run:  python -m unittest tests.test_pipeline"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class TestPipeline(unittest.TestCase):
    def test_synthetic_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / 'data', Path(tmp) / 'output'
            env = dict(os.environ, AHE_DATA_DIR=str(data), AHE_OUTPUT_DIR=str(out), PYTHONIOENCODING='utf-8')
            code = (f'import sys; sys.path.insert(0, r"{ROOT / "src"}"); from pathlib import Path; '
                    f'from ahe import synthetic; synthetic.write(out=Path(r"{data}"))')
            subprocess.run([sys.executable, '-c', code], check=True, env=env)
            r = subprocess.run([sys.executable, str(ROOT / 'run_pipeline.py')], env=env, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            split = json.loads((out / 'split.json').read_text(encoding='utf-8'))
            self.assertFalse(set(split['train']) & set(split['test']), 'patients must not be shared between sets')
            w = pd.read_pickle(out / 'intermediate' / 'windows.pkl')
            self.assertTrue((w.groupby('window').size() == 14).all())
            self.assertGreater(w.groupby('window').ahe.last().sum(), 0, 'the synthetic data must contain events')
            for f in ('performance', 'calibration', 'logistic_regression', 'subgroups', 'sensitivity_raw_map',
                      'windows_preceding_ahe', 'false_positive_follow_up', 'shap_importance'):
                self.assertTrue((out / 'results' / f'{f}.csv').exists(), f)


if __name__ == '__main__':
    unittest.main()
