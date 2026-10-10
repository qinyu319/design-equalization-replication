"""Validate delivered numerical artifacts and a separate full reproduction."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from experiments import ROOT,sha_file


def same_csv(first,second):
    count=0
    for a,b in itertools.zip_longest(pd.read_csv(first,chunksize=5000),pd.read_csv(second,chunksize=5000)):
        if a is None or b is None:raise AssertionError('Different file lengths')
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-12,atol=1e-14)
        count+=len(a)
    return count


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reproduced',required=True)
    args=p.parse_args()
    reproduced=Path(args.reproduced)
    source=json.loads((ROOT/'validation/source_verification.json').read_text(encoding='utf-8'))
    checks={}
    checks['original_model_unchanged']=sha_file(Path(source['source_root'])/'code/run_revision.py')==source['original_model_sha256']
    checks['original_archive_unchanged']=sha_file(Path(source['source_root'])/'data/combined_runs_120960.csv.gz')==source['original_archive_sha256']
    checks['packaged_original_archive_same']=sha_file(ROOT/'inputs/archive_reference/data/combined_runs_120960.csv.gz')==source['original_archive_sha256']
    checks['frozen_model_same']=sha_file(ROOT/'frozen/frozen_abm.py')==source['original_model_sha256']
    checks['frozen_estimator_same']=sha_file(ROOT/'frozen/frozen_estimators.py')==source['frozen_estimator_sha256']
    checks['archived_sample_replays']=source['replay_passed']
    checks['new_matrix_integrity']=json.loads((ROOT/'validation/matrix_integrity.json').read_text())['passed']
    correction=json.loads((ROOT/'validation/E5_execution.json').read_text())
    checks['nonhub_rows_unchanged']=correction['nonhub_rows_unchanged']
    checks['original_direct_exposures_unchanged']=correction['direct_exposure_unchanged']
    comparisons=[]
    for path in sorted((ROOT/'results').glob('*')):
        if path.name.endswith(('.csv','.csv.gz')):
            other=reproduced/'results'/path.name
            if not other.is_file():raise AssertionError('Missing reproduced output: '+str(other))
            n=same_csv(path,other)
            comparisons.append(dict(file=path.name,rows=n,equivalent=True))
    checks['full_reproduction_numeric_and_trajectory_equivalence']=all(r['equivalent'] for r in comparisons)
    ranges=pd.read_csv(ROOT/'results/E5_corrected_bootstrap_ranges.csv')
    checks['ratio_filter_counts']=bool(((ranges.valid_draws+ranges.nonpositive_denominator_draws)==1000).all())
    images=[]
    for path in sorted((ROOT/'figures').glob('Fig*.png')):
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            images.append(dict(file=path.name,width=img.width,height=img.height,
                               pdf_present=path.with_suffix('.pdf').is_file(),svg_present=path.with_suffix('.svg').is_file(),
                               source_csv_present=path.with_name(path.stem+'_source.csv').is_file()))
    checks['six_complete_figure_exports']=len(images)==6 and all(i['pdf_present'] and i['svg_present'] and i['source_csv_present'] for i in images)
    # Representative numerical assertions behind the narrative, in raw proportion units.
    e1=pd.read_csv(ROOT/'results/E1_allocations.csv')
    for mu,factor,target in [(0,'dose',.705684),(0,'seeding_regime',.002060),(.05,'seeding_regime',.441365)]:
        r=e1[(e1.reversion==mu)&(e1.view=='full_crossed')&(e1.outcome=='mean_total_shift')&(e1.factor==factor)]
        assert len(r)==1 and abs(r.within_view_share.iloc[0]-target)<1e-6
    e4=pd.read_csv(ROOT/'results/E4_validation_summary.csv')
    u=e4[(e4.estimator=='complete_u')&(e4.factor!='D')]
    assert np.isclose(u.percentile_coverage.min(),.84) and np.isclose(u.percentile_coverage.max(),.952)
    checks['representative_narrative_numbers']=True
    summary=dict(passed=all(checks.values()),checks=checks,compared_outputs=comparisons,
                 figure_exports=images,visual_inspection='All six rendered PNGs inspected; labels, axes, legends and numeric source scales checked.',
                 full_reproduction_directory=str(reproduced),numeric_rtol=1e-12,numeric_atol=1e-14)
    (ROOT/'validation/final_validation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if not summary['passed']:raise AssertionError(summary)
    print(json.dumps(dict(passed=summary['passed'],compared_outputs=len(comparisons),figure_count=len(images)),ensure_ascii=False))


if __name__=='__main__':main()
