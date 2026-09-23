"""Complete cross-replication U-statistic validation on the frozen archive.

No ABM runs are added. Random split averaging is retained as a diagnostic;
the primary point estimator is its exact all-distinct-replication-pairs limit.
"""
from __future__ import annotations
import itertools, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; AN=ROOT/'analysis_outputs'
DATA=ROOT/'data'/'combined_runs_120960.csv.gz'
FACTORS=['topology','message_condition','inoculation','seeding_regime']
OUTCOMES=['mean_total_shift','mean_signed_shift','conversion_rate']
BENCHMARK='C-D50'; EXTENDED=['E-SD','J-S06-D05','J-S18-D05']
SEED=20260805; N_BOOT=1000; SPLIT_GRID=[20,50,100]

def projection_games(cells,factors):
    subsets=[s for k in range(len(factors)+1) for s in itertools.combinations(factors,k)]
    c=len(cells); m=np.eye(c)-np.ones((c,c))/c; hs=[]
    for subset in subsets:
        if not subset: hs.append(np.zeros((c,c))); continue
        code=cells.groupby(list(subset),sort=True,observed=True).ngroup().to_numpy()
        counts=np.bincount(code).astype(float); z=np.zeros((c,len(counts))); z[np.arange(c),code]=1
        p=(z/counts)@z.T; h=m@p@m/c; hs.append((h+h.T)/2)
    hs=np.stack(hs); trans=np.zeros((len(factors),len(subsets))); lookup={s:i for i,s in enumerate(subsets)}
    K=len(factors)
    for fi,factor in enumerate(factors):
        others=[f for f in factors if f!=factor]
        for size in range(K):
            w=math.factorial(size)*math.factorial(K-size-1)/math.factorial(K)
            for sub in itertools.combinations(others,size):
                base=tuple(sorted(sub,key=factors.index)); added=tuple(sorted(sub+(factor,),key=factors.index))
                trans[fi,lookup[added]]+=w; trans[fi,lookup[base]]-=w
    factor_h=np.einsum('fs,sij->fij',trans,hs)
    return factor_h,np.concatenate([factor_h,hs[-1:]],axis=0)

def load_arrays():
    data=pd.read_csv(DATA); keys=FACTORS+['replication']; arrays={}; reps={}; base=None
    for design,frame in data.groupby('design',sort=True):
        frame=frame.sort_values(keys).reset_index(drop=True); r=int(frame.groupby(FACTORS,observed=True).size().iloc[0])
        cells=frame[FACTORS].drop_duplicates().reset_index(drop=True)
        if base is None: base=cells
        elif not cells.equals(base): raise ValueError(f'Cell order differs: {design}')
        arrays[design]={o:frame[o].to_numpy(float).reshape(144,r) for o in OUTCOMES}; reps[design]=r
    if len(data)!=120960 or data.duplicated(['design']+keys).any(): raise ValueError('Archive integrity failure')
    return data,arrays,reps,base

def kernels(y,h): return np.einsum('cr,kcd,ds->krs',y,h,y,optimize=True)
def complete_u(k):
    r=k.shape[-1]; return (k.sum(axis=(-2,-1))-np.trace(k,axis1=-2,axis2=-1))/(r*(r-1))
def in_sample(k):
    r=k.shape[-1]; return k.sum(axis=(-2,-1))/(r*r)
def bootstrap_u(k,counts):
    """Distinct-pair block bootstrap: repeated copies never pair with themselves."""
    quad=np.einsum('br,krs,bs->bk',counts,k,counts,optimize=True)
    diag=np.einsum('br,kr->bk',counts**2,np.diagonal(k,axis1=-2,axis2=-1),optimize=True)
    denom=counts.sum(axis=1)**2-np.sum(counts**2,axis=1)
    return (quad-diag)/denom[:,None]

def point_and_bootstrap(arrays,reps,h):
    rng=np.random.default_rng(SEED+600); c0=rng.multinomial(30,np.full(30,1/30),N_BOOT); c1=rng.multinomial(30,np.full(30,1/30),N_BOOT)
    pnt={}; boot={}; ins={}
    for d in sorted(arrays):
        for o,y in arrays[d].items():
            k=kernels(y,h); pnt[d,o]=complete_u(k); ins[d,o]=in_sample(k)
            boot[d,o]=bootstrap_u(k,c0 if reps[d]==30 else np.concatenate([c0,c1],axis=1))
    prows=[]; rrows=[]
    for o in OUTCOMES:
        bp=pnt[BENCHMARK,o]; bb=boot[BENCHMARK,o]
        for d in sorted(arrays):
            v=pnt[d,o]; bd=boot[d,o]
            for fi,f in enumerate(FACTORS):
                vals={'B_total_U':v[-1],'B_factor_U':v[fi],'q_proportion':v[fi]/bp[-1],
                      'rho_proportion':v[fi]/bp[fi] if bp[fi] else np.nan,'tau_proportion':v[-1]/bp[-1]}
                prows.append({'design':d,'outcome':o,'factor':f,'replications':reps[d],**vals,
                              'in_sample_B_total':ins[d,o][-1],'in_sample_B_factor':ins[d,o][fi]})
                draws={'B_total_U':bd[:,-1],'B_factor_U':bd[:,fi],'q_proportion':bd[:,fi]/bb[:,-1],
                       'rho_proportion':bd[:,fi]/bb[:,fi],'tau_proportion':bd[:,-1]/bb[:,-1]}
                for metric,x in draws.items():
                    lo,med,hi=np.nanquantile(x,[.025,.5,.975])
                    rrows.append({'design':d,'outcome':o,'factor':f,'metric':metric,'point':vals[metric],
                                  'p02_5':lo,'median':med,'p97_5':hi,'range_width':hi-lo,'bootstrap_draws':N_BOOT,
                                  'range_label':'2.5th-97.5th percentile range from distinct-pair replication-block bootstrap',
                                  'replications_design':reps[d],'replications_benchmark':reps[BENCHMARK]})
    points=pd.DataFrame(prows); ranges=pd.DataFrame(rrows)
    points.to_csv(AN/'complete_u_point_estimates.csv',index=False); ranges.to_csv(AN/'complete_u_bootstrap_ranges.csv',index=False)
    return points,ranges

def balanced_parts(r,n,rng):
    out=[]; na=r//2
    for _ in range(n): p=rng.permutation(r); out.append((p[:na],p[na:]))
    return out

def resample_half(idx,b,rng): return idx[rng.integers(0,len(idx),size=(b,len(idx)))]
def cross_draw(k,ia,ib): return k[:,ia[:,:,None],ib[:,None,:]].mean(axis=(-2,-1)).T

def split_count_sensitivity(arrays,h,points):
    rng=np.random.default_rng(SEED+700); bparts=balanced_parts(30,100,rng); eparts=balanced_parts(30,100,rng); B=500
    kb=kernels(arrays[BENCHMARK]['mean_total_shift'],h); kx={d:kernels(arrays[d]['mean_total_shift'],h) for d in EXTENDED}
    sd=np.zeros((B,len(h))); sn={d:np.zeros((B,len(h))) for d in EXTENDED}; pden=np.zeros(len(h)); pn={d:np.zeros(len(h)) for d in EXTENDED}
    rows=[]; start=time.perf_counter()
    for s in range(100):
        a0,b0=bparts[s]; a1,b1=eparts[s]
        ia0,ib0=resample_half(a0,B,rng),resample_half(b0,B,rng); ia1=resample_half(a1,B,rng)+30; ib1=resample_half(b1,B,rng)+30
        sd+=cross_draw(kb,ia0,ib0); pden+=kb[:,a0[:,None],b0[None,:]].mean(axis=(-2,-1))
        ia60=np.concatenate([ia0,ia1],1); ib60=np.concatenate([ib0,ib1],1); pa=np.r_[a0,a1+30]; pb=np.r_[b0,b1+30]
        for d in EXTENDED:
            sn[d]+=cross_draw(kx[d],ia60,ib60); pn[d]+=kx[d][:,pa[:,None],pb[None,:]].mean(axis=(-2,-1))
        S=s+1
        if S not in SPLIT_GRID: continue
        dd=sd/S; dp=pden/S; elapsed=time.perf_counter()-start
        for d in EXTENDED:
            nd=sn[d]/S; npnt=pn[d]/S; q=nd[:,3]/dd[:,-1]; lo,med,hi=np.quantile(q,[.025,.5,.975])
            uq=points[(points.design==d)&(points.outcome=='mean_total_shift')&(points.factor=='seeding_regime')].iloc[0].q_proportion
            rows.append({'design':d,'splits_per_draw':S,'bootstrap_draws':B,'point_split_q':npnt[3]/dp[-1],
                         'point_complete_u_q':uq,'point_absolute_difference':abs(npnt[3]/dp[-1]-uq),
                         'p02_5':lo,'median':med,'p97_5':hi,'range_width':hi-lo,'cumulative_runtime_seconds':elapsed})
    out=pd.DataFrame(rows); out.to_csv(AN/'bootstrap_split_count_sensitivity.csv',index=False); return out

def extension_ratio_sensitivity(arrays,h):
    rng=np.random.default_rng(SEED+800); c0=rng.multinomial(30,np.full(30,1/30),N_BOOT); c1=rng.multinomial(30,np.full(30,1/30),N_BOOT); ci=rng.multinomial(30,np.full(30,1/30),N_BOOT)
    kb=kernels(arrays[BENCHMARK]['mean_total_shift'],h); bp=complete_u(kb)[-1]; bd0=bootstrap_u(kb,c0)[:,-1]; bdi=bootstrap_u(kb,ci)[:,-1]; rows=[]
    for d in EXTENDED:
        y=arrays[d]['mean_total_shift']
        cases=[('first_30_paired',y[:,:30],c0,bd0,'0-29 numerator and benchmark use identical resampled block counts.'),
               ('all_60_stratified',y,np.concatenate([c0,c1],1),bd0,'0-29 numerator shares benchmark counts; 30-59 is resampled independently; q is recomputed per draw.'),
               ('last_30_independent',y[:,30:],c1,bdi,'30-59 numerator and benchmark are independently resampled because no streams are shared.')]
        for sample,yy,counts,bd,rule in cases:
            k=kernels(yy,h); point=complete_u(k)[3]/bp; draw=bootstrap_u(k,counts)[:,3]/bd; lo,med,hi=np.quantile(draw,[.025,.5,.975])
            rows.append({'design':d,'sample':sample,'q_seed_proportion':point,'p02_5':lo,'median':med,'p97_5':hi,
                         'range_width':hi-lo,'bootstrap_draws':N_BOOT,'pairing_rule':rule})
    out=pd.DataFrame(rows); out.to_csv(AN/'extension_ratio_pairing_sensitivity.csv',index=False); return out

def toy_validation():
    from statistics import NormalDist
    rng=np.random.default_rng(SEED+900); factors=['A','B','C','D']; cells=pd.DataFrame(list(itertools.product([-1.,1.],repeat=4)),columns=factors)
    fh,_=projection_games(cells,factors); a,b,c,_=[cells[x].to_numpy() for x in factors]; mu=.8*a+.5*b+.6*a*b+.3*c
    truth=np.einsum('c,kcd,d->k',mu,fh,mu); expected=np.array([.82,.43,.09,0.])
    if not np.allclose(truth,expected,atol=1e-12): raise ValueError(truth)
    outer=500; cov_outer=200; nboot=199; rows=[]; nd=NormalDist()
    for noise in ['homoskedastic','heteroskedastic']:
        sigma=np.ones(16) if noise=='homoskedastic' else .60+.60*(a==1)+.40*(b==1)
        for r in [10,15,30,60]:
            eis=np.empty((outer,4)); eu=np.empty((outer,4)); cover={x:np.zeros((cov_outer,4),bool) for x in ['percentile','basic','bca','jackknife_normal']}
            for m in range(outer):
                y=mu[:,None]+sigma[:,None]*rng.normal(size=(16,r)); k=kernels(y,fh); eis[m]=in_sample(k); eu[m]=complete_u(k)
                if m>=cov_outer: continue
                counts=rng.multinomial(r,np.full(r,1/r),nboot); bt=bootstrap_u(k,counts); plo,phi=np.quantile(bt,[.025,.975],axis=0)
                cover['percentile'][m]=(plo<=truth)&(truth<=phi); cover['basic'][m]=(2*eu[m]-phi<=truth)&(truth<=2*eu[m]-plo)
                total=k.sum(axis=(-2,-1))-np.trace(k,axis1=-2,axis2=-1); rowoff=k.sum(axis=-1)-np.diagonal(k,axis1=-2,axis2=-1)
                jack=(total[:,None]-2*rowoff)/((r-1)*(r-2)); jm=jack.mean(1); se=np.sqrt((r-1)/r*np.sum((jack-jm[:,None])**2,axis=1))
                cover['jackknife_normal'][m]=(eu[m]-1.96*se<=truth)&(truth<=eu[m]+1.96*se)
                for fi in range(4):
                    prop=np.clip(np.mean(bt[:,fi]<eu[m,fi]),1/(2*nboot),1-1/(2*nboot)); z0=nd.inv_cdf(prop); dif=jm[fi]-jack[fi]; den=6*np.sum(dif**2)**1.5; acc=np.sum(dif**3)/den if den>0 else 0
                    probs=[]
                    for alpha in [.025,.975]:
                        za=nd.inv_cdf(alpha); probs.append(np.clip(nd.cdf(z0+(z0+za)/max(1e-12,1-acc*(z0+za))),0,1))
                    lo,hi=np.quantile(bt[:,fi],probs); cover['bca'][m,fi]=lo<=truth[fi]<=hi
            for fi,f in enumerate(factors):
                for estimator,est in [('in-sample',eis[:,fi]),('complete-U',eu[:,fi])]:
                    row={'noise':noise,'R':r,'factor':f,'estimator':estimator,'true_B_allocation':truth[fi],'mean_estimate':est.mean(),
                         'bias':est.mean()-truth[fi],'mean_absolute_error':np.mean(np.abs(est-truth[fi])),'rmse':np.sqrt(np.mean((est-truth[fi])**2)),
                         'negative_probability':np.mean(est<0),'outer_datasets':outer,'coverage_outer_datasets':cov_outer,
                         'heteroskedastic_sigma':'0.60 + 0.60 I(A=1) + 0.40 I(B=1)'}
                    if estimator=='complete-U':
                        for method,hit in cover.items(): row['coverage_'+method]=hit[:,fi].mean()
                    rows.append(row)
    out=pd.DataFrame(rows); out.to_csv(AN/'toy_complete_u_validation.csv',index=False)
    tr=pd.DataFrame({'factor':factors,'true_B_allocation':truth,'derivation':['0.80^2 + 0.60^2/2','0.50^2 + 0.60^2/2','0.30^2','0']}); tr.to_csv(AN/'toy_complete_u_truth.csv',index=False)
    return out,tr

def main():
    start=time.perf_counter(); data,arrays,reps,cells=load_arrays(); _,mh=projection_games(cells,FACTORS)
    points,ranges=point_and_bootstrap(arrays,reps,mh); splits=split_count_sensitivity(arrays,mh,points); ratios=extension_ratio_sensitivity(arrays,mh); toy,truth=toy_validation()
    summary={'archive_rows':len(data),'abm_runs_added':0,'primary_estimator':'complete cross-replication order-2 U-statistic','abm_bootstrap_draws':N_BOOT,
             'split_diagnostic_grid':SPLIT_GRID,'toy_outer_datasets':500,'toy_coverage_outer_datasets':200,'toy_bootstrap_draws':199,
             'elapsed_seconds':time.perf_counter()-start,'all_complete_u_points_finite':bool(np.isfinite(points.B_total_U).all()),
             'all_displayed_points_inside_percentile_ranges':bool(((ranges.point>=ranges.p02_5)&(ranges.point<=ranges.p97_5)).all()),
             'outputs':['complete_u_point_estimates.csv','complete_u_bootstrap_ranges.csv','bootstrap_split_count_sensitivity.csv','extension_ratio_pairing_sensitivity.csv','toy_complete_u_validation.csv','toy_complete_u_truth.csv']}
    (AN/'complete_u_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()



