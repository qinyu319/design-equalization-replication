"""Independent orthogonal-contrast implementation of complete-U validation."""
from __future__ import annotations
import itertools
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.stats import norm
from experiments import ROOT, sha_file


def contrast_system():
    cells=np.array(list(itertools.product([-1.,1.],repeat=4)))
    masks=list(range(16))
    X=np.column_stack([np.prod(cells[:,[i for i in range(4) if mask>>i&1]],axis=1)
                       if mask else np.ones(16) for mask in masks])
    W=np.array([[((1/mask.bit_count()) if mask>>i&1 else 0.) for mask in masks]
                for i in range(4)])
    return cells,X,W


def estimate(z,W):
    r=z.shape[1]
    totals=z.sum(1)
    squares=(z*z).sum(1)
    u=W@((totals**2-squares)/(r*(r-1)))
    ins=W@(z.mean(1)**2)
    jack=W@(((totals[:,None]-z)**2-(squares[:,None]-z*z))/((r-1)*(r-2)))
    se=np.sqrt((r-1)/r*np.sum((jack-jack.mean(1,keepdims=True))**2,axis=1))
    return u,ins,se


def bootstrap(z,W,counts):
    denom=counts.sum(1)**2-(counts*counts).sum(1)
    if np.any(denom<=0):
        raise AssertionError('Invalid distinct-pair resample.')
    values=(counts@z.T)**2-(counts*counts)@(z*z).T
    return (values/denom[:,None])@W.T


def run():
    start=time.perf_counter()
    cells,X,W=contrast_system()
    a,b,c,d=cells.T
    mu=.8*a+.5*b+.6*a*b+.3*c
    truth=np.array([.82,.43,.09,0.])
    np.testing.assert_allclose(W@((X.T@mu/16)**2),truth,atol=1e-14)
    sig=.6+.6*(a==1)+.4*(b==1)
    v=(a+d)/np.sqrt(2)
    structured=(.4*np.eye(16)+.6*np.outer(v,v))
    structured=structured/np.sqrt(np.outer(np.diag(structured),np.diag(structured)))
    covariances={
        'independent_homoskedastic':np.eye(16),
        'independent_heteroskedastic':np.diag(sig**2),
        'common_shock_heteroskedastic':np.outer(sig,sig)*(.4*np.eye(16)+.6*np.ones((16,16))),
        'structured_shock_heteroskedastic':np.outer(sig,sig)*structured,
    }
    rng=np.random.default_rng(2026091604)
    records=[]
    for noise,cov in covariances.items():
        L=np.linalg.cholesky(cov)
        for r in [10,30,60]:
            exact_bias=W@np.diag(X.T@cov@X/256)/r
            for j in range(1000):
                y=mu[:,None]+L@rng.normal(size=(16,r))
                z=X.T@y/16
                u,ins,se=estimate(z,W)
                coverage={}
                if j<500:
                    counts=rng.multinomial(r,np.ones(r)/r,size=399).astype(float)
                    bt=bootstrap(z,W,counts)
                    lo,hi=np.quantile(bt,[.025,.975],axis=0)
                    coverage=dict(percentile_low=lo,percentile_high=hi,
                                  jackknife_low=u-norm.ppf(.975)*se,
                                  jackknife_high=u+norm.ppf(.975)*se)
                for f,name in enumerate('ABCD'):
                    row=dict(noise=noise,R=r,dataset=j,factor=name,truth=truth[f],
                             complete_u=u[f],in_sample=ins[f],
                             theoretical_in_sample_bias=exact_bias[f])
                    row.update({key:values[f] for key,values in coverage.items()})
                    records.append(row)
            print(f'E4: {noise} R={r}; {time.perf_counter()-start:.1f}s',flush=True)
    raw=pd.DataFrame(records)
    raw.to_csv(ROOT/'results/E4_dataset_estimates.csv.gz',index=False,compression='gzip')
    rows=[]
    for keys,group in raw.groupby(['noise','R','factor'],sort=False):
        for estimator in ['in_sample','complete_u']:
            est=group[estimator].to_numpy()
            truth_value=group.truth.iloc[0]
            errors=est-truth_value
            result=dict(zip(['noise','R','factor'],keys),estimator=estimator,truth=truth_value,
                        datasets=len(est),mean_estimate=est.mean(),bias=errors.mean(),
                        bias_mcse=est.std(ddof=1)/np.sqrt(len(est)),
                        rmse=np.sqrt(np.mean(errors**2)),negative_frequency=np.mean(est<0),
                        theoretical_bias=group.theoretical_in_sample_bias.iloc[0] if estimator=='in_sample' else 0.)
            if estimator=='complete_u':
                for method in ['percentile','jackknife']:
                    valid=group.dropna(subset=[method+'_low'])
                    hit=(valid[method+'_low']<=truth_value)&(truth_value<=valid[method+'_high'])
                    p=hit.mean()
                    result[method+'_coverage']=p
                    result[method+'_coverage_mcse']=np.sqrt(p*(1-p)/len(hit))
                    # Wilson interval remains informative for observed coverage = 0 or 1.
                    den=1+1.96**2/len(hit)
                    center=(p+1.96**2/(2*len(hit)))/den
                    half=1.96*np.sqrt(p*(1-p)/len(hit)+1.96**2/(4*len(hit)**2))/den
                    result[method+'_coverage_wilson_low']=center-half
                    result[method+'_coverage_wilson_high']=center+half
                    result[method+'_mean_width']=(valid[method+'_high']-valid[method+'_low']).mean()
                    result['coverage_datasets']=len(hit)
            rows.append(result)
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/'results/E4_validation_summary.csv',index=False)
    np.savez_compressed(ROOT/'inputs/E4_noise_covariances.npz',**covariances)
    summary=dict(datasets=12000,factor_estimate_rows=len(raw),coverage_datasets=6000,
                 bootstrap_draws_per_dataset=399,noise_settings=list(covariances),
                 elapsed_seconds=time.perf_counter()-start,
                 code_sha256=sha_file(__file__),all_estimates_finite=bool(np.isfinite(raw[['complete_u','in_sample']]).all().all()))
    (ROOT/'validation/E4_execution.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    run()
