"""Run the pipeline.

    python run_pipeline.py --synthetic      create a small synthetic dataset in data/synthetic and run everything on it
    python run_pipeline.py                  run everything on the tables in AHE_DATA_DIR
    python run_pipeline.py --steps preprocess train evaluate explain

See README.md for the input format and the environment variables.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

STEPS = ['preprocess', 'train', 'evaluate', 'explain']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', nargs='+', choices=STEPS, default=STEPS)
    ap.add_argument('--synthetic', action='store_true', help='generate and use the synthetic example dataset')
    a = ap.parse_args()
    if a.synthetic:
        from ahe import synthetic
        synthetic.write()
    from ahe import config as C
    print(f'Input:  {C.DATA_DIR}\nOutput: {C.OUTPUT_DIR}')
    t0 = time.time()
    if 'preprocess' in a.steps:
        from ahe import preprocess
        preprocess.run()
    if 'train' in a.steps:
        from ahe import model
        model.run()
    if 'evaluate' in a.steps:
        from ahe import evaluation
        evaluation.run()
    if 'explain' in a.steps:
        from ahe import explainability
        explainability.run()
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
