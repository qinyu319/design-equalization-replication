"""Publication figures; each has an inspectable CSV and PNG/PDF/SVG exports."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from experiments import ROOT

RESULTS=ROOT/'results'
FIGURES=ROOT/'figures'
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titlesize':12,'axes.labelsize':10,
                     'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def save(fig,name,data):
    for ext in ['png','pdf','svg']:
        fig.savefig(FIGURES/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    data.to_csv(FIGURES/f'{name}_source.csv',index=False)
    plt.close(fig)


def figure1():
    d=pd.read_csv(RESULTS/'E1_allocations.csv')
    d=d[(d.view=='full_crossed')&(d.outcome=='mean_total_shift')&(d.factor!='TOTAL')].copy()
    factors=['topology','dose','seeding_regime','strength']
    labels=['Topology','Direct dose','Schedule','Strength']
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.8),layout='constrained')
    for ax,mu in zip(axes,[0,.05]):
        s=d[d.reversion==mu].set_index('factor').loc[factors]
        for i,(_,r) in enumerate(s.iterrows()):
            ax.plot([r.B_low*1e4,r.B_high*1e4],[i,i],color=COLORS[i],lw=2)
            ax.scatter(r.B*1e4,i,color=COLORS[i],s=40,zorder=3)
            ax.annotate(f'{100*r.within_view_share:.2f}%',(r.B*1e4,i),xytext=(7,8),
                        textcoords='offset points',fontsize=9)
        ax.set_yticks(range(4),labels)
        ax.invert_yaxis(); ax.set_ylim(3.5,-.6)
        ax.axvline(0,color='#888888',lw=.8)
        ax.set_xlim(-.02 if mu==.05 else -.12, .32 if mu==.05 else 6.8)
        ax.set_xlabel(r'Complete-U allocation $B_j$ ($\times 10^{-4}$)')
        ax.set_title(f'Reversion = {mu:.2f}')
    fig.suptitle('E1 | Independent crossing separates dose from schedule',fontsize=14)
    save(fig,'Fig1_crossed_attribution',d)


def figure2():
    parts=[]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,(suite,field,values,title) in enumerate([
        ('E1','reversion',[0,.05],'Original dynamics'),
        ('E3','stubbornness',[.2,.8],'Peer-averaging extension')]):
        d=pd.read_csv(RESULTS/f'{suite}_schedule_contrasts.csv')
        parts.append(d.assign(suite=suite))
        for col,outcome in enumerate(['mean_total_shift','mean_trajectory_shift']):
            ax=axes[row,col]
            for j,value in enumerate(values):
                s=d[(d[field]==value)&(d.outcome==outcome)].sort_values('dose')
                ax.errorbar(s.dose,s['mean']*1000,
                    yerr=np.array([s['mean']-s.ci_low,s.ci_high-s['mean']])*1000,
                    color=COLORS[j],marker=['o','s'][j],capsize=3,label=f'{field} = {value:g}')
            ax.axhline(0,color='#888888',lw=.8)
            ax.set_xticks([.05,.5,.95]);ax.set_xlabel('Cumulative direct events per agent')
            ax.set_ylabel(r'Continuous minus single ($\times 10^{-3}$)')
            ax.set_title(title+' | '+('Terminal shift' if col==0 else 'Time-averaged shift'))
            ax.legend(fontsize=9)
    fig.suptitle('E1 & E3 | Timing effects depend on dynamics and the observation window',fontsize=14)
    save(fig,'Fig2_timing_and_outcomes',pd.concat(parts,ignore_index=True))


def figure3():
    d=pd.read_csv(RESULTS/'E2_policy_contrasts.csv')
    d=d[(d.outcome=='mean_total_shift')&~d.comparison.str.startswith('corrected')]
    groups=[(['hub_vs_random_B05_C050','hub_vs_random_B05_C100','hub_vs_random_B20_C050','hub_vs_random_B20_C100'],
             ['5% / 50%','5% / 100%','20% / 50%','20% / 100%'],'Hub advantage over random','Target budget / compliance'),
            (['label_retention_L30','label_retention_L70'],['0.30','0.70'],
             'Retained-label advantage over label loss',r'Label attenuation $\lambda$')]
    fig,axes=plt.subplots(2,2,figsize=(11,7.1),layout='constrained')
    for row,(names,labels,title,xlabel) in enumerate(groups):
        for col,dose in enumerate([.05,.5]):
            ax=axes[row,col]
            for j,(regime,label) in enumerate([('single_shot','Single shot'),('continuous','Continuous')]):
                s=d[(d.dose==dose)&(d.seeding_regime==regime)].set_index('comparison').loc[names]
                xs=np.arange(len(names))+(j-.5)*.12
                ax.errorbar(xs,s['mean']*1000,yerr=np.array([s['mean']-s.ci_low,s.ci_high-s['mean']])*1000,
                            color=COLORS[j],marker=['o','s'][j],ls='none',capsize=3,label=label)
            ax.set_xticks(range(len(names)),labels,fontsize=9)
            ax.set_xlim(-.45,len(names)-.55)
            ax.axhline(0,color='#888888',lw=.8)
            ax.set_title(title+f'\nDirect dose = {dose:g}')
            ax.set_xlabel(xlabel);ax.set_ylabel(r'Extra shift reduction ($\times 10^{-3}$)')
            if row==0 and col==0:ax.legend(fontsize=9)
    fig.suptitle('E2 | Governance comparisons under matched exposure',fontsize=14)
    save(fig,'Fig3_governance_comparisons',d)


def figure4():
    d=pd.read_csv(RESULTS/'E4_validation_summary.csv')
    d=d[d.factor=='D']
    noises=['independent_homoskedastic','independent_heteroskedastic',
            'common_shock_heteroskedastic','structured_shock_heteroskedastic']
    titles=['Independent\nhomoskedastic','Independent\nheteroskedastic',
            'Common shock\nheteroskedastic','Structured shock\nheteroskedastic']
    fig,axes=plt.subplots(1,4,figsize=(12,3.9),sharey=True,layout='constrained')
    for ax,noise,title in zip(axes,noises,titles):
        for j,(est,label) in enumerate([('in_sample','In-sample'),('complete_u','Complete-U')]):
            s=d[(d.noise==noise)&(d.estimator==est)].sort_values('R')
            ax.errorbar(s.R,s.mean_estimate,yerr=1.96*s.bias_mcse,
                        color=COLORS[j],marker=['s','o'][j],capsize=3,label=label)
        ax.axhline(0,color='#444444',lw=.8,ls='--')
        ax.set_xticks([10,30,60]);ax.set_xlabel('Replication blocks R');ax.set_title(title)
    axes[0].set_ylabel('Mean estimated allocation to null factor D')
    axes[0].legend(fontsize=9)
    fig.suptitle('E4 | Point-estimator behavior when the true allocation is zero',fontsize=14)
    save(fig,'Fig4_null_factor_bias',d)


def figure5():
    d=pd.read_csv(RESULTS/'E4_validation_summary.csv')
    d=d[d.estimator=='complete_u']
    noises=['independent_homoskedastic','independent_heteroskedastic',
            'common_shock_heteroskedastic','structured_shock_heteroskedastic']
    titles=['Independent\nhomoskedastic','Independent\nheteroskedastic',
            'Common shock\nheteroskedastic','Structured shock\nheteroskedastic']
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for row,(method,label) in enumerate([('percentile','Distinct-pair percentile'),('jackknife','Jackknife normal')]):
        for col,(noise,title) in enumerate(zip(noises,titles)):
            ax=axes[row,col]
            grid=d[d.noise==noise].pivot(index='factor',columns='R',values=method+'_coverage').loc[list('ABCD'),[10,30,60]]
            im=ax.imshow(grid.to_numpy()*100,cmap='viridis',vmin=80,vmax=100,aspect='auto')
            for i in range(4):
                for j in range(3):
                    val=grid.iloc[i,j]*100
                    ax.text(j,i,f'{val:.1f}',ha='center',va='center',color='white' if val<90 else '#111111',fontsize=10)
            ax.set_xticks(range(3),[10,30,60]);ax.set_yticks(range(4),['A','B','C','D (null)'])
            ax.set_xlabel('Replication blocks R')
            if col==0:ax.set_ylabel(label+'\nFactor')
            if row==0:ax.set_title(title)
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.85,pad=.02)
    cb.set_label('Empirical coverage (%) | nominal 95%')
    fig.suptitle('E4 | Interval calibration varies across signal and noise settings',fontsize=14)
    save(fig,'Fig5_interval_coverage',d)


def figure6():
    audit=pd.read_csv(ROOT/'validation/hub_selector_audit.csv')
    s=audit[audit.seeding_regime=='single_shot'] if 'seeding_regime' in audit else audit[audit.regime=='single_shot']
    s=s.set_index('topology').loc[['BA','ER','WS','SBM']]
    d=pd.read_csv(RESULTS/'E5_original_vs_corrected_allocations.csv')
    d=d[(d.design=='C-D50')&(d.outcome=='mean_total_shift')]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ax=axes[0];x=np.arange(4)
    ax.bar(x-.18,s.selected_mean_degree_legacy,.36,color=COLORS[0],label='Archived selector')
    ax.bar(x+.18,s.selected_mean_degree_corrected,.36,color=COLORS[1],label='Corrected selector')
    ax.set_xticks(x,s.index);ax.set_ylabel('Mean degree among 50 selected nodes')
    ax.set_title('Selector check | one archived seed per topology');ax.legend(fontsize=9)
    ax=axes[1]
    names=['topology','message_condition','inoculation','seeding_regime']
    s2=d.set_index('factor').loc[names]
    for j,(field,label) in enumerate([('q_proportion','Original'),('corrected_q','Corrected')]):
        ax.barh(x+(j-.5)*.32,s2[field]*100,.32,color=COLORS[j],label=label)
    ax.set_yticks(x,['Topology','Message','Inoculation','Schedule']);ax.invert_yaxis()
    ax.set_xlabel('Within-benchmark allocation (%)');ax.set_title('C-D50 | dominant schedule allocation remains')
    ax.legend(fontsize=9)
    save(fig,'Fig6_source_correction',pd.concat([audit.assign(source='selector_audit'),d.assign(source='allocation')],ignore_index=True))


if __name__=='__main__':
    FIGURES.mkdir(exist_ok=True)
    for func in [figure1,figure2,figure3,figure4,figure5,figure6]:
        func()
        print(func.__name__+' exported',flush=True)
