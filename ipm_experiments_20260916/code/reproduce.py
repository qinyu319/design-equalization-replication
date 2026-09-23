"""Reproduce into a NEW directory without modifying any delivered outputs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--destination',required=True)
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    dest=Path(args.destination).resolve()
    if dest.exists():
        raise FileExistsError('Choose a new destination: '+str(dest))
    if dest.is_relative_to(source):
        raise ValueError('Choose a sibling directory, outside the delivered package.')
    dest.mkdir(parents=True)
    for directory in ['code','frozen','inputs']:
        shutil.copytree(source/directory,dest/directory,ignore=shutil.ignore_patterns('__pycache__'))
    for directory in ['results','validation','figures']:
        (dest/directory).mkdir()
    for pattern in ['*.md','requirements*.txt']:
        for path in source.glob(pattern):shutil.copy2(path,dest/path.name)
    commands=[['code/experiments.py','--verify-source','inputs/archive_reference']]
    commands += [['code/experiments.py','--suite',s,'--workers',str(args.workers)] for s in ['E1','E2','E3']]
    commands += [['code/validate_estimator.py'],['code/analyze_results.py'],
                 ['code/correct_archive.py','--source','inputs/archive_reference','--workers',str(args.workers)],
                 ['code/analyze_results.py','--correction-only'],['code/make_figures.py'],
                 ['-m','pytest','code/test_experiments.py','-q']]
    for command in commands:
        subprocess.run([sys.executable,*command],cwd=dest,check=True)
    print('Reproduction completed: '+str(dest))


if __name__=='__main__':main()
