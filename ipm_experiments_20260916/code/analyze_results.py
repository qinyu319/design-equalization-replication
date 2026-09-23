"""Tables and uncertainty summaries for every prespecified extension."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from experiments import ROOT, projection_games, kernels, complete_u, bootstrap_u, make_tasks

OUTCOMES=['mean_total_shift','mean_signed_shift','conversion_rate','mean_trajectory_shift']


def summarize_blocks(frame,groups,values=OUTCOMES):
    blocks=frame.groupby(groups+['replication'],dropna=False,observed=True)[values].mean().reset_index()
    rows=[]
    for keys,g in blocks.groupby(groups,dropna=False,observed=True):
        if not isinstance(keys,tuple): keys=(keys,)
        for outcome in values:
            a=g[outcome].to_numpy()
            n=len(a)
            se=a.std(ddof=1)/np.sqrt(n)
            half=t.ppf(.975,n-1)*se
            rows.append(dict(zip(groups,keys),outcome=outcome,mean=a.mean(),mcse=se,
                             ci_low=a.mean()-half,ci_high=a.mean()+half,replication_blocks=n,
                             interval='95% t interval for Monte Carlo mean; unadjusted'))
    return pd.DataFrame(rows)


def allocation(frame,factors,view,metadata):
    keys=factors+['replication']
    if frame.duplicated(keys).any(): raise AssertionError(f'Duplicate cell: {view}')
    sorted_frame=frame.sort_values(keys).reset_index(drop=True)
    counts=sorted_frame.groupby(factors,observed=True).size()
    if counts.nunique()!=1: raise AssertionError('Unbalanced repetitions.')
    r=int(counts.iloc[0])
    cells=sorted_frame[factors].drop_duplicates().reset_index(drop=True)
    if len(cells)!=np.prod([cells[f].nunique() for f in factors]):
        raise AssertionError('The requested game is not a full Cartesian grid.')
    _,h=projection_games(cells,factors)
    bootcounts=np.random.default_rng(2026091610).multinomial(r,np.ones(r)/r,1000)
    rows=[]
    for outcome in OUTCOMES:
        y=sorted_frame[outcome].to_numpy().reshape(len(cells),r)
        k=kernels(y,h)
        est=complete_u(k)
        draws=bootstrap_u(k,bootcounts)
        lo,hi=np.quantile(draws,[.025,.975],axis=0)
        if not np.isclose(est[:-1].sum(),est[-1],rtol=1e-8,atol=1e-14):
            raise AssertionError('Shapley efficiency failed.')
        for j,factor in enumerate(factors+['TOTAL']):
            rows.append(dict(metadata,view=view,outcome=outcome,factor=factor,B=est[j],
                             B_low=lo[j],B_high=hi[j],B_total=est[-1],cells=len(cells),R=r,
                             within_view_share=est[j]/est[-1] if est[-1]>0 else np.nan,
                             range_label='2.5-97.5 percentile sensitivity range; not calibrated CI'))
    return rows


def analyze_e1():
    data=pd.read_csv(ROOT/'results/E1_runs.csv.gz')
    rows=[]
    for mu,frame in data.groupby('reversion'):
        rows+=allocation(frame,['topology','dose','seeding_regime','strength'],'full_crossed',{'reversion':mu})
        for dose,sub in frame.groupby('dose'):
            rows+=allocation(sub,['topology','seeding_regime','strength'],f'equal_dose_{dose}',{'reversion':mu,'dose':dose})
        for name,mapping in [('bundled_forward',dict(zip(['single_shot','burst','continuous'],[.05,.5,.95]))),
                             ('bundled_reverse',dict(zip(['single_shot','burst','continuous'],[.95,.5,.05])))]:
            sub=frame[np.isclose(frame.dose,frame.seeding_regime.map(mapping))]
            rows+=allocation(sub,['topology','seeding_regime','strength'],name,{'reversion':mu})
    allocations=pd.DataFrame(rows)
    allocations.to_csv(ROOT/'results/E1_allocations.csv',index=False)
    endpoints=summarize_blocks(data,['reversion','dose','seeding_regime'])
    endpoints.to_csv(ROOT/'results/E1_endpoints.csv',index=False)
    # Matched continuous-minus-single contrasts at each dose.
    pairs=[]
    for mu,dose in itertools.product([0.,.05],[.05,.5,.95]):
        sub=data[(data.reversion==mu)&(data.dose==dose)]
        group=sub.groupby(['replication','seeding_regime'])[OUTCOMES].mean()
        difference=group.xs('continuous',level='seeding_regime')-group.xs('single_shot',level='seeding_regime')
        difference=difference.reset_index().assign(reversion=mu,dose=dose)
        pairs.append(difference)
    summarize_blocks(pd.concat(pairs),['reversion','dose']).to_csv(ROOT/'results/E1_schedule_contrasts.csv',index=False)
    return allocations


def analyze_e2():
    data=pd.read_csv(ROOT/'results/E2_runs.csv.gz')
    keys=['topology','dose','seeding_regime','replication']
    baseline=data[data.policy=='none'][keys+OUTCOMES].rename(columns={o:o+'_baseline' for o in OUTCOMES})
    merged=data.merge(baseline,on=keys,validate='many_to_one')
    if len(merged)!=len(data): raise AssertionError('Governance baseline join changed grain.')
    for o in OUTCOMES:
        merged[o+'_raw']=merged[o]
        merged[o]=merged[o+'_baseline']-merged[o]
    merged.to_csv(ROOT/'results/E2_paired_benefits.csv.gz',index=False,compression='gzip')
    summarize_blocks(merged,['policy','selector','dose','seeding_regime']).to_csv(ROOT/'results/E2_policy_benefits.csv',index=False)
    summarize_blocks(merged,['policy','selector','topology','dose','seeding_regime']).to_csv(ROOT/'results/E2_policy_benefits_by_topology.csv',index=False)
    summarize_blocks(data,['policy','selector','dose','seeding_regime']).to_csv(ROOT/'results/E2_policy_endpoints.csv',index=False)
    contrasts=[]
    def contrast(first,second,label):
        # Positive = first achieves a lower endpoint than second.
        a=data[data.policy==first][keys+OUTCOMES].set_index(keys)
        b=data[data.policy==second][keys+OUTCOMES].set_index(keys)
        if not a.index.equals(b.index): b=b.reindex(a.index)
        if a.empty or b.empty or b.isna().any().any(): raise AssertionError(label)
        c=(b-a).reset_index().assign(comparison=label)
        contrasts.append(c)
    for budget,compliance in itertools.product([5,20],[50,100]):
        stem=f'B{budget:02d}_C{compliance:03d}'
        contrast('hub_'+stem,'random_'+stem,'hub_vs_random_'+stem)
    for lam in [30,70]:
        contrast(f'label_L{lam}_retained',f'label_L{lam}_lost',f'label_retention_L{lam}')
    for policy in data[data.selector=='legacy'].policy.unique():
        contrast(policy.removesuffix('_legacy'),policy,'corrected_vs_'+policy)
    paired=pd.concat(contrasts,ignore_index=True)
    paired.to_csv(ROOT/'results/E2_policy_contrast_blocks.csv.gz',index=False,compression='gzip')
    summarize_blocks(paired,['comparison','dose','seeding_regime']).to_csv(ROOT/'results/E2_policy_contrasts.csv',index=False)
    summarize_blocks(paired,['comparison','topology','dose','seeding_regime']).to_csv(ROOT/'results/E2_policy_contrasts_by_topology.csv',index=False)


def analyze_e3():
    data=pd.read_csv(ROOT/'results/E3_runs.csv.gz')
    controls=data[data.dose==0]
    if not np.all(controls[OUTCOMES].to_numpy()==0):
        raise AssertionError('Second-model no-input controls are not zero relative to themselves.')
    active=data[data.dose>0]
    rows=[]
    for stubbornness,frame in active.groupby('stubbornness'):
        rows+=allocation(frame,['topology','dose','seeding_regime','strength'],'full_crossed',{'stubbornness':stubbornness})
        for dose,sub in frame.groupby('dose'):
            rows+=allocation(sub,['topology','seeding_regime','strength'],f'equal_dose_{dose}',{'stubbornness':stubbornness,'dose':dose})
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/'results/E3_allocations.csv',index=False)
    summarize_blocks(active,['stubbornness','dose','seeding_regime']).to_csv(ROOT/'results/E3_endpoints.csv',index=False)
    pairs=[]
    for stubbornness,dose in itertools.product([.2,.8],[.05,.5,.95]):
        sub=active[(active.stubbornness==stubbornness)&(active.dose==dose)]
        group=sub.groupby(['replication','seeding_regime'])[OUTCOMES].mean()
        difference=group.xs('continuous',level='seeding_regime')-group.xs('single_shot',level='seeding_regime')
        pairs.append(difference.reset_index().assign(stubbornness=stubbornness,dose=dose))
    summarize_blocks(pd.concat(pairs),['stubbornness','dose']).to_csv(ROOT/'results/E3_schedule_contrasts.csv',index=False)


def analyze_e5():
    data=pd.read_csv(ROOT/'results/E5_corrected_archive_120960.csv.gz')
    factors=['topology','message_condition','inoculation','seeding_regime']
    outcomes=OUTCOMES[:3]
    arrays={}
    sizes={}
    first=None
    for design,frame in data.groupby('design'):
        frame=frame.sort_values(factors+['replication'])
        cells=frame[factors].drop_duplicates().reset_index(drop=True)
        if first is None:
            first=cells
            _,h=projection_games(cells,factors)
        else:
            assert first.equals(cells)
        r=len(frame)//144
        sizes[design]=r
        arrays[design]={o:frame[o].to_numpy().reshape(144,r) for o in outcomes}
    points={(d,o):complete_u(kernels(y,h)) for d,byo in arrays.items() for o,y in byo.items()}
    rows=[]
    rng=np.random.default_rng(2026091650)
    counts0=rng.multinomial(30,np.ones(30)/30,1000)
    counts1=rng.multinomial(30,np.ones(30)/30,1000)
    ranges=[]
    for outcome in outcomes:
        baseline=points['C-D50',outcome]
        bdraw=bootstrap_u(kernels(arrays['C-D50'][outcome],h),counts0)
        for design in sorted(arrays):
            estimate=points[design,outcome]
            draws=bootstrap_u(kernels(arrays[design][outcome],h),counts0 if sizes[design]==30 else np.c_[counts0,counts1])
            for i,factor in enumerate(factors):
                row=dict(design=design,outcome=outcome,factor=factor,R=sizes[design],
                         corrected_B_total=estimate[-1],corrected_B_factor=estimate[i],
                         corrected_q=estimate[i]/baseline[-1] if baseline[-1]>0 else np.nan,
                         corrected_tau=estimate[-1]/baseline[-1] if baseline[-1]>0 else np.nan,
                         corrected_rho=estimate[i]/baseline[i] if baseline[i]>0 else np.nan)
                rows.append(row)
                def positive_ratio(numerator,denominator):
                    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
                for name,values in [('B_factor',draws[:,i]),('q',positive_ratio(draws[:,i],bdraw[:,-1])),
                                    ('tau',positive_ratio(draws[:,-1],bdraw[:,-1])),('rho',positive_ratio(draws[:,i],bdraw[:,i]))]:
                    valid=np.isfinite(values)
                    lo,hi=np.quantile(values[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
                    ranges.append(dict(design=design,outcome=outcome,factor=factor,metric=name,low=lo,high=hi,
                                       valid_draws=int(valid.sum()),nonpositive_denominator_draws=int((~valid).sum()),
                                       interval='2.5-97.5 percentile sensitivity range; not calibrated CI'))
    corrected=pd.DataFrame(rows)
    original=pd.read_csv(ROOT/'inputs/original_complete_u_point_estimates.csv')
    merged=corrected.merge(original,on=['design','outcome','factor'],validate='one_to_one')
    merged['B_factor_difference']=merged.corrected_B_factor-merged.B_factor_U
    merged['q_difference']=merged.corrected_q-merged.q_proportion
    merged.to_csv(ROOT/'results/E5_original_vs_corrected_allocations.csv',index=False)
    pd.DataFrame(ranges).to_csv(ROOT/'results/E5_corrected_bootstrap_ranges.csv',index=False)
    diff=pd.read_csv(ROOT/'results/E5_run_differences.csv.gz')
    diff['absolute_difference']=diff.difference.abs()
    diff.groupby(['design','topology','outcome']).agg(mean_difference=('difference','mean'),
        mean_absolute_difference=('absolute_difference','mean'),max_absolute_difference=('absolute_difference','max'),
        rows=('difference','size')).reset_index().to_csv(ROOT/'results/E5_correction_impact.csv',index=False)
    return merged


def integrity():
    checks=[]
    for suite in ['E1','E2','E3']:
        frame=pd.read_csv(ROOT/'results'/f'{suite}_runs.csv.gz')
        expected=make_tasks(suite)
        tests=dict(row_count=len(frame)==len(expected),
                   exact_run_ids=set(frame.run_id)==set(t['run_id'] for t in expected),
                   unique_run_ids=not frame.run_id.duplicated().any(),
                   finite_primary_outcomes=bool(np.isfinite(frame[OUTCOMES]).all().all()),
                   direct_dose_conserved=bool(np.all(frame.total_direct_exposures==np.rint(frame.dose*1000))),
                   replication_set=set(frame.replication)==set(range(30)))
        checks.append(dict(suite=suite,rows=len(frame),tests=tests,passed=all(tests.values())))
    result=dict(checks=checks,passed=all(c['passed'] for c in checks),new_matrix_simulations=sum(c['rows'] for c in checks))
    (ROOT/'validation/matrix_integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['passed']: raise AssertionError(result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--correction-only',action='store_true')
    args=p.parse_args()
    if args.correction_only:
        out=analyze_e5()
        print(out[(out.design=='C-D50')&(out.outcome=='mean_total_shift')][['factor','B_factor_U','corrected_B_factor','q_proportion','corrected_q']].to_string(index=False))
    else:
        print(json.dumps(integrity()))
        analyze_e1()
        print('E1 analysis complete',flush=True)
        analyze_e2()
        print('E2 analysis complete',flush=True)
        analyze_e3()
        print('E3 analysis complete',flush=True)


if __name__=='__main__': main()
