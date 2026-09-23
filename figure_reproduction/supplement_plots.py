"""Reproduce all 12 figures in the expanded IP&M manuscript.

Run: python plot_all_figures.py
Use --help for individual figures, paths, formats and resolution.
Inputs are archived simulation outputs, not human or real-LLM observations.
No simulation is rerun and no network connection is required by this script.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
ROOT=Path(__file__).resolve().parent
import argparse, json, hashlib
from scipy.stats import t
from matplotlib.patches import Rectangle, FancyArrowPatch

RESULTS=ROOT/'data'
FORMATS=['png','pdf','svg']
DPI=300
FIGURES=ROOT/'figures'
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titlesize':12,'axes.labelsize':10,
                     'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def save(fig,name,data):
    for ext in FORMATS:
        fig.savefig(FIGURES/f'{name}.{ext}',dpi=DPI,bbox_inches='tight')
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
    save(fig,'Fig03_crossed_attribution',d)


def figure2():
    parts=[]
    fig,axes=plt.subplots(2,2,figsize=(10.4,7),layout='constrained')
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
    save(fig,'Fig06_timing_and_outcomes',pd.concat(parts,ignore_index=True))


def figure3():
    d=pd.read_csv(RESULTS/'E2_policy_contrasts.csv')
    d=d[(d.outcome=='mean_total_shift')&~d.comparison.str.startswith('corrected')]
    groups=[(['hub_vs_random_B05_C050','hub_vs_random_B05_C100','hub_vs_random_B20_C050','hub_vs_random_B20_C100'],
             ['5% / 50%','5% / 100%','20% / 50%','20% / 100%'],'Hub advantage over random','Target budget / compliance'),
            (['label_retention_L30','label_retention_L70'],['0.30','0.70'],
             'Retained-label advantage over label loss',r'Label attenuation $\lambda$')]
    fig,axes=plt.subplots(2,2,figsize=(10.4,7.1),layout='constrained')
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
    save(fig,'Fig05_governance_comparisons',d)


def figure4():
    d=pd.read_csv(RESULTS/'E4_validation_summary.csv')
    d=d[d.factor=='D']
    noises=['independent_homoskedastic','independent_heteroskedastic',
            'common_shock_heteroskedastic','structured_shock_heteroskedastic']
    titles=['Independent\nhomoskedastic','Independent\nheteroskedastic',
            'Common shock\nheteroskedastic','Structured shock\nheteroskedastic']
    fig,axes=plt.subplots(1,4,figsize=(10.4,3.9),sharey=True,layout='constrained')
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
    save(fig,'FigS03_null_factor_bias',d)


def figure5():
    d=pd.read_csv(RESULTS/'E4_validation_summary.csv')
    d=d[d.estimator=='complete_u']
    noises=['independent_homoskedastic','independent_heteroskedastic',
            'common_shock_heteroskedastic','structured_shock_heteroskedastic']
    titles=['Independent\nhomoskedastic','Independent\nheteroskedastic',
            'Common shock\nheteroskedastic','Structured shock\nheteroskedastic']
    fig,axes=plt.subplots(2,4,figsize=(10.4,6),layout='constrained')
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
    save(fig,'Fig07_interval_coverage',d)


def figure6():
    audit=pd.read_csv(RESULTS/'hub_selector_audit.csv')
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
    save(fig,'FigS05_source_correction',pd.concat([audit.assign(source='selector_audit'),d.assign(source='allocation')],ignore_index=True))


def corrected_original():
    """Preserve the corrected archive figure without importing simulation code."""
    points=pd.read_csv(RESULTS/'E5_original_vs_corrected_allocations.csv')
    endpoints=pd.read_csv(RESULTS/'corrected_original_endpoints.csv')
    d=points[(points.outcome=='mean_total_shift')&(points.factor=='seeding_regime')].set_index('design')
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    names=['CS-0','CS-15','CS-30','CS-45']
    axes[0].plot([0,.15,.3,.45],d.loc[names,'corrected_B_factor']*1000,'o-',color=COLORS[0])
    axes[0].set(xlabel='Assigned dose spread s',ylabel=r'Schedule allocation $B_j$ ($\times 10^{-3}$)',
                title='Fixed mean direct dose = 0.50',ylim=(0,None))
    regimes=['single_shot','burst','continuous']
    for name,color,marker in [('CS-45',COLORS[0],'o'),('CS-45-R',COLORS[1],'s')]:
        e=endpoints[endpoints.design==name].set_index('seeding_regime').loc[regimes]
        axes[1].plot(range(3),e.mean_total_shift,color=color,marker=marker,label=name)
    axes[1].set_xticks(range(3),['Single','Burst','Continuous'])
    axes[1].set(ylabel='Terminal mean absolute shift',title='Reversing the dose assignment',ylim=(0,None))
    axes[1].legend(fontsize=9)
    anchors=d.loc[['C-D1','CS-0','D-A95']]
    axes[2].plot([.05,.5,.95],anchors.corrected_tau*100,'o-',label='Total retained variation',color=COLORS[0])
    axes[2].plot([.05,.5,.95],anchors.corrected_q*100,'s--',label='Schedule contribution',color=COLORS[1])
    axes[2].set(xlabel='Equal direct dose per agent',ylabel='Percentage of C-D50 total variation',
                title='Anchor sensitivity',ylim=(0,None),xticks=[.05,.5,.95])
    axes[2].legend(fontsize=8)
    source=pd.concat([d.reset_index().assign(source='allocation'),endpoints.assign(source='endpoint')],ignore_index=True)
    save(fig,'Fig02_corrected_original',source)


def audit_framework():
    """Conceptual diagram; boxes contain protocol descriptions, not observations."""
    fig,ax=plt.subplots(figsize=(10.4,4.8),layout='constrained')
    ax.set(xlim=(0,10.4),ylim=(0,4.8));ax.axis('off')
    top=[('Specify the model','Record direct dose, schedule,\nstrength and intervention rules'),
         ('Audit the design','Equalize bundled dimensions\nand cross them independently'),
         ('Interpret the evidence','State the reference scale,\noutcome window and model limits')]
    records=[]
    for i,(title,body) in enumerate(top):
        x=.12+i*3.52
        ax.add_patch(Rectangle((x,3.05),3.1,1.15,facecolor='#F4F6F8',edgecolor='#64717A',lw=1))
        ax.text(x+1.55,3.86,title,ha='center',va='center',fontsize=12,weight='bold')
        ax.text(x+1.55,3.40,body,ha='center',va='center',fontsize=10.5,linespacing=1.5)
        if i<2: ax.add_patch(FancyArrowPatch((x+3.13,3.62),(x+3.48,3.62),arrowstyle='-|>',mutation_scale=14,color='#444444'))
        records.append(dict(layer='procedure',order=i+1,title=title,description=body.replace('\n',' ')))
    ax.text(5.2,4.55,'Design equalization and complementary evidence checks',ha='center',va='center',fontsize=14)
    ax.text(5.2,2.65,'Four complementary experiment suites',ha='center',fontsize=12,weight='bold')
    checks=[('E1','Independent crossing','Dose × schedule × strength\nwith and without reversion'),
            ('E2','Governance comparisons','Coverage, compliance\nand label persistence'),
            ('E3','Alternative dynamics','Anchored peer averaging\nand paired no-input controls'),
            ('E4','Statistical calibration','Known truth, noise structure\nand replication count')]
    for i,(suite,title,body) in enumerate(checks):
        x=.12+i*2.62
        ax.add_patch(Rectangle((x,.75),2.3,1.50,facecolor='white',edgecolor='#64717A',lw=1))
        ax.text(x+1.15,1.98,suite,ha='center',fontsize=12,weight='bold',color=COLORS[0])
        ax.text(x+1.15,1.58,title,ha='center',fontsize=10.5,weight='bold')
        ax.text(x+1.15,1.07,body,ha='center',fontsize=9.8,linespacing=1.5)
        records.append(dict(layer='experiment',order=i+1,title=suite+' '+title,description=body.replace('\n',' ')))
    ax.text(5.2,.28,'Implementation correction and reproducibility checks support all stages',ha='center',fontsize=11)
    save(fig,'Fig01_audit_framework',pd.DataFrame(records))


def dose_strength_response():
    """Average four topologies within each replication before computing t intervals."""
    d=pd.read_csv(RESULTS/'E1_terminal_runs.csv')
    keys=['reversion','strength','dose','seeding_regime']
    counts=d.groupby(keys+['replication']).topology.nunique()
    assert counts.eq(4).all(), 'Every block must contain all four topologies.'
    blocks=d.groupby(keys+['replication'],as_index=False).mean_total_shift.mean()
    rows=[]
    for key,g in blocks.groupby(keys):
        assert len(g)==30
        m=g.mean_total_shift.mean();se=g.mean_total_shift.std(ddof=1)/np.sqrt(len(g))
        half=t.ppf(.975,len(g)-1)*se
        rows.append(dict(zip(keys,key),mean=m,mcse=se,ci_low=m-half,ci_high=m+half,
                         replication_blocks=30,topologies_per_block=4))
    source=pd.DataFrame(rows)
    fig,axes=plt.subplots(2,3,figsize=(10.4,6.8),sharex=True,sharey=True,layout='constrained')
    regimes=[('single_shot','Single shot','o','-'),('burst','Burst','s','--'),('continuous','Continuous','^',':')]
    for r,mu in enumerate([0,.05]):
        for c,strength in enumerate([.06,.14,.18]):
            ax=axes[r,c]
            for j,(regime,label,marker,line) in enumerate(regimes):
                g=source[(source.reversion==mu)&(source.strength==strength)&(source.seeding_regime==regime)].sort_values('dose')
                assert len(g)==3
                ax.errorbar(g.dose,g['mean']*1000,yerr=1.0*np.array([g['mean']-g.ci_low,g.ci_high-g['mean']])*1000,
                            color=COLORS[j],marker=marker,ls=line,capsize=3,label=label)
            ax.set_title(f'Strength = {strength:.2f} | Reversion = {mu:.2f}',fontsize=10.5)
            ax.set_xticks([.05,.5,.95]);ax.set_ylim(0,source.ci_high.max()*1000*1.08);ax.grid(axis='y',color='#E5E5E5',lw=.6)
            if r==1:ax.set_xlabel('Cumulative direct events per agent',fontsize=9)
            if c==0:ax.set_ylabel(r'Terminal absolute shift ($\times 10^{-3}$)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=11)
    fig.suptitle('E1 | Dose response at each message strength and reversion setting',fontsize=13)
    save(fig,'Fig04_dose_strength_response',source)


POLICIES=[('none','No intervention'),('label_L30_lost','Label 0.30 / lost'),
 ('label_L30_retained','Label 0.30 / retained'),('label_L70_lost','Label 0.70 / lost'),
 ('label_L70_retained','Label 0.70 / retained'),('random_B05_C050','Random 5% / 50%'),
 ('hub_B05_C050','Hub 5% / 50%'),('random_B05_C100','Random 5% / 100%'),
 ('hub_B05_C100','Hub 5% / 100%'),('random_B20_C050','Random 20% / 50%'),
 ('hub_B20_C050','Hub 20% / 50%'),('random_B20_C100','Random 20% / 100%'),
 ('hub_B20_C100','Hub 20% / 100%'),('combined_random','Combined / random'),('combined_hub','Combined / hub')]


def all_policy_benefits():
    d=pd.read_csv(RESULTS/'E2_policy_benefits.csv')
    d=d[(d.selector=='corrected')&(d.outcome=='mean_total_shift')].copy()
    assert len(d)==60 and d.policy.nunique()==15
    fig,axes=plt.subplots(1,4,figsize=(10.4,6.2),sharey=True,layout='constrained')
    columns=[(.05,'single_shot','Single shot'),(.05,'continuous','Continuous'),(.5,'single_shot','Single shot'),(.5,'continuous','Continuous')]
    for ax,(dose,regime,label) in zip(axes,columns):
        g=d[(d.dose==dose)&(d.seeding_regime==regime)].set_index('policy').loc[[x[0] for x in POLICIES]]
        ax.errorbar(g['mean']*1000,np.arange(15),xerr=np.array([g['mean']-g.ci_low,g.ci_high-g['mean']])*1000,
                    color=COLORS[0],marker='o' if regime=='single_shot' else 's',
                    mfc=COLORS[0] if regime=='single_shot' else 'white',ms=4.5,ls='none',capsize=2)
        group=d[d.dose==dose]
        lo=min(0,float(group.ci_low.min()*1000));hi=float(group.ci_high.max()*1000)
        ax.set_xlim(lo-.03*hi,hi*1.08)
        ax.axvline(0,color='#888888',lw=.8)
        ax.grid(axis='y',color='#E5E5E5',lw=.6)
        ax.set_title(f'Dose = {dose:g}\n{label}',fontsize=11)
        ax.set_yticks(range(15),[x[1] for x in POLICIES],fontsize=10.5)
        ax.set_xlabel(r'Shift reduction ($\times10^{-3}$)',fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle('E2 | All primary policies versus paired no intervention',fontsize=13)
    save(fig,'FigS01_all_policy_benefits',d)


def topology_contrasts():
    d=pd.read_csv(RESULTS/'E2_policy_contrasts_by_topology.csv')
    names=['hub_vs_random_B05_C100','hub_vs_random_B20_C100']
    d=d[(d.outcome=='mean_total_shift')&d.comparison.isin(names)].copy()
    assert len(d)==32
    fig,axes=plt.subplots(2,2,figsize=(10.4,7),sharex=True,sharey='row',layout='constrained')
    for row,dose in enumerate([.05,.5]):
        for col,(name,budget) in enumerate(zip(names,[5,20])):
            ax=axes[row,col]
            for j,(regime,label,marker) in enumerate([('single_shot','Single shot','o'),('continuous','Continuous','s')]):
                g=d[(d.dose==dose)&(d.comparison==name)&(d.seeding_regime==regime)].set_index('topology').loc[['BA','ER','WS','SBM']]
                ax.errorbar(np.arange(4)+(j-.5)*.12,g['mean']*1000,
                    yerr=np.array([g['mean']-g.ci_low,g.ci_high-g['mean']])*1000,
                    color=COLORS[j],marker=marker,ls='none',capsize=3,label=label)
            ax.axhline(0,color='#888888',lw=.8)
            ax.set_xticks(range(4),['BA','ER','WS','SBM']);ax.set_xlim(-.4,3.4)
            ax.set_title(f'Dose = {dose:g} | Target budget = {budget}%',fontsize=12)
            if col==0:ax.set_ylabel(r'Extra hub reduction ($\times10^{-3}$)')
            if row==1:ax.set_xlabel('Network topology')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('E2 | Hub versus random targeting by topology at full compliance',fontsize=13)
    save(fig,'FigS02_topology_contrasts',d)


def rmse_comparison():
    d=pd.read_csv(RESULTS/'E4_validation_summary.csv')
    noises=['independent_homoskedastic','independent_heteroskedastic','common_shock_heteroskedastic','structured_shock_heteroskedastic']
    titles=['Independent\nhomoskedastic','Independent\nheteroskedastic','Common shock\nheteroskedastic','Structured shock\nheteroskedastic']
    fig,axes=plt.subplots(4,4,figsize=(10.4,8.8),sharex=True,sharey='row',layout='constrained')
    for r,factor in enumerate('ABCD'):
        for c,(noise,title) in enumerate(zip(noises,titles)):
            ax=axes[r,c]
            for j,(est,label,marker,line) in enumerate([('in_sample','In-sample','s','--'),('complete_u','Complete-U','o','-')]):
                g=d[(d.factor==factor)&(d.noise==noise)&(d.estimator==est)].sort_values('R')
                assert len(g)==3
                ax.plot(g.R,g.rmse,color=COLORS[j],marker=marker,ls=line,label=label,ms=4)
            ax.set_ylim(0,d.loc[d.factor==factor,'rmse'].max()*1.08);ax.set_xticks([10,30,60]);ax.grid(axis='y',color='#E5E5E5',lw=.6)
            if r==0:ax.set_title(title,fontsize=10.5)
            if c==0:ax.set_ylabel('Factor '+factor+(' (null)' if factor=='D' else '')+'\nRMSE',fontsize=10.5)
            if r==3:ax.set_xlabel('Replication blocks R',fontsize=9.5)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('E4 | Estimation error across all factors and noise structures',fontsize=13)
    save(fig,'FigS04_rmse_comparison',d)


FIGURE_FUNCTIONS={
 '1':audit_framework,'2':corrected_original,'3':figure1,'4':dose_strength_response,
 '5':figure3,'6':figure2,'7':figure5,'S1':all_policy_benefits,'S2':topology_contrasts,
 'S3':figure4,'S4':rmse_comparison,'S5':figure6,
}


def main():
    global RESULTS,FIGURES,DPI,FORMATS
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data',help='Input tables; default is adjacent data/ directory')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'figures',help='Output directory; default is adjacent figures/')
    parser.add_argument('--figures',nargs='+',choices=list(FIGURE_FUNCTIONS),default=list(FIGURE_FUNCTIONS),help='Figure numbers, e.g. 1 4 S2')
    parser.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf','svg'])
    parser.add_argument('--dpi',type=int,default=300)
    args=parser.parse_args()
    if args.dpi<72:parser.error('--dpi must be at least 72')
    RESULTS=args.data_dir.resolve();FIGURES=args.output_dir.resolve();DPI=args.dpi;FORMATS=args.formats
    FIGURES.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data_manifest.json'
    if manifest.exists() and RESULTS==(ROOT/'data').resolve():
        for item in json.loads(manifest.read_text(encoding='utf-8')):
            p=ROOT/item['file']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Input changed from the validated archive: '+item['file'])
    for number in args.figures:
        FIGURE_FUNCTIONS[number]()
        print('Figure '+number+' exported',flush=True)
    print('Outputs: '+str(FIGURES),flush=True)


if __name__=='__main__':
    main()


