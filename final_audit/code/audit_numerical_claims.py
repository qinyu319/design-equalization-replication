"""Independent checks from run-level data; does not import the analysis implementation."""
from pathlib import Path
import itertools,math,json,argparse
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT.parent/'ipm_experiments_20260916/results'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir',type=Path,default=SRC,help='Path to results/ in the computational reproduction package')
args=parser.parse_args();SRC=args.results_dir
(ROOT/'audit').mkdir(parents=True,exist_ok=True)
OUTCOMES=['mean_total_shift','mean_signed_shift','conversion_rate','mean_trajectory_shift']
checks=[]
def eq(name,a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    assert np.allclose(a,b,rtol=2e-10,atol=2e-13),name
    checks.append({'check':name,'max_abs_difference':float(np.max(np.abs(a-b)))})
def allocations(d,factors,outcome):
    values={():0.0};R=d.replication.nunique()
    for k in range(1,len(factors)+1):
        for s in itertools.combinations(factors,k):
            tab=d.groupby(['replication',*s],observed=True)[outcome].mean().unstack(list(s)).to_numpy()
            z=tab-tab.mean(axis=1,keepdims=True)
            values[s]=float(((z.sum(axis=0)**2).sum()-(z*z).sum())/(R*(R-1)*z.shape[1]))
    result={};K=len(factors)
    for f in factors:
        v=0
        for s in values:
            if f in s:continue
            sf=tuple(x for x in factors if x in s or x==f)
            v+=math.factorial(len(s))*math.factorial(K-len(s)-1)/math.factorial(K)*(values[sf]-values[s])
        result[f]=v
    return result
for suite,param in [('E1','reversion'),('E3','stubbornness')]:
    d=pd.read_csv(SRC/f'{suite}_runs.csv.gz');d=d[d.dose>0]
    stored=pd.read_csv(SRC/f'{suite}_allocations.csv')
    cstored=pd.read_csv(SRC/f'{suite}_schedule_contrasts.csv')
    for value,g in d.groupby(param):
        for outcome in OUTCOMES:
            vals=allocations(g,['topology','dose','seeding_regime','strength'],outcome)
            for factor,num in vals.items():
                factor_alias=factor
                row=stored[(stored[param]==value)&(stored.view=='full_crossed')&(stored.outcome==outcome)&(stored.factor==factor_alias)].iloc[0]
                eq(f'{suite}/{value}/{outcome}/{factor}',num,row.B)
            for dose,h in g.groupby('dose'):
                tab=h.groupby(['replication','seeding_regime'])[outcome].mean().unstack()
                contrasts=tab.continuous-tab.single_shot
                mean=contrasts.mean();se=contrasts.std(ddof=1)/np.sqrt(len(contrasts));half=t.ppf(.975,29)*se
                row=cstored[(cstored[param]==value)&(cstored.dose==dose)&(cstored.outcome==outcome)].iloc[0]
                eq(f'{suite}/{value}/{dose}/{outcome}/paired_mean',[mean,se,mean-half,mean+half],row[['mean','mcse','ci_low','ci_high']])
d=pd.read_csv(SRC/'E2_runs.csv.gz');stored=pd.read_csv(SRC/'E2_policy_contrasts.csv')
for _,row in stored.iterrows():
    name=row.comparison
    if name.startswith('corrected_vs_'):continue
    if name.startswith('hub_vs_random_'):
        suffix=name.removeprefix('hub_vs_random_');first='hub_'+suffix;second='random_'+suffix
    else:
        suffix=name.removeprefix('label_retention_');first='label_'+suffix+'_retained';second='label_'+suffix+'_lost'
    sel=d[(d.dose==row.dose)&(d.seeding_regime==row.seeding_regime)]
    tab=sel.groupby(['replication','policy'])[row.outcome].mean().unstack()
    vals=tab[second]-tab[first];m=vals.mean();se=vals.std(ddof=1)/np.sqrt(30);half=t.ppf(.975,29)*se
    eq('E2/'+name+'/'+str(row.dose)+'/'+row.seeding_regime+'/'+row.outcome,[m,se,m-half,m+half],row[['mean','mcse','ci_low','ci_high']])
d=pd.read_csv(SRC/'E4_dataset_estimates.csv.gz');summary=pd.read_csv(SRC/'E4_validation_summary.csv')
for _,r in summary.iterrows():
    g=d[(d.noise==r.noise)&(d.R==r.R)&(d.factor==r.factor)]
    vals=g[r.estimator];errors=vals-g.truth
    eq(f'E4/{r.noise}/{r.R}/{r.factor}/{r.estimator}/bias',errors.mean(),r.bias)
    for meth in ['percentile','jackknife']:
        if r.estimator!='complete_u':continue
        eligible=g[g[meth+'_low'].notna()]
        actual=((eligible[meth+'_low']<=eligible.truth)&(eligible.truth<=eligible[meth+'_high'])).mean()
        eq(f'E4/{r.noise}/{r.R}/{r.factor}/{meth}/coverage',actual,r[meth+'_coverage'])
# Exact expected unique direct reach is a mathematical consequence of uniform independent selections.
reach=[]
for dose in [.05,.5,.95]:
    for schedule,steps in [('single_shot',1),('burst',5),('continuous',50)]:
        k=int(round(dose*1000/steps));reach.append({'dose':dose,'schedule':schedule,'active_steps':steps,'direct_events':1000*dose,'expected_unique_direct_recipients':1000*(1-(1-k/1000)**steps)})
pd.DataFrame(reach).to_csv(ROOT/'audit/expected_direct_reach.csv',index=False)
(ROOT/'audit/numerical_claims.json').write_text(json.dumps({'status':'passed','checks':checks,'independent_algorithm':'Grouped cell means and distinct-block quadratic sums; raw paired block contrasts; dataset-level coverage indicators','scope':'All full-crossed E1/E3 allocations for four outcomes, all E1/E3 schedule contrasts, all 96 primary E2 contrasts, E4 estimator biases and coverages','new_simulation_runs':0},indent=2),encoding='utf8')
print('PASSED',len(checks),'independent numerical comparisons')
