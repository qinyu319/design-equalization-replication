"""Reproduce the paper artwork from archived simulation results.

python plot_all_figures.py
python plot_all_figures.py --figures 3 7 --formats pdf eps png --dpi 600
No simulations, fabricated observations, or network access are used.
PDF/EPS are the publication sources; PNG is used in the Word reading copy.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.text import Text
from scipy.stats import t
import supplement_plots as legacy

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'figures'
FORMATS=['pdf','eps','png','svg']; DPI=600
BLUE='#0072B2'; ORANGE='#C45A00'; DARK='#252525'
COLORS=[BLUE,ORANGE,'#557A24','#A84979']
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.titlesize':9,
 'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'lines.linewidth':1.2,'lines.markersize':4,'errorbar.capsize':3,
 'legend.fontsize':8.5,'legend.frameon':False,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
 'text.color':DARK,'axes.labelcolor':DARK})
AUDIT={}

def read(name): return pd.read_csv(DATA/name)
def panel(ax,letter,title=''):
    ax.set_title(title,loc='center',pad=7)
    ax.text(-.02,1.055,f'({letter})',transform=ax.transAxes,fontweight='bold',fontsize=9,ha='right')
def err(g,scale=1): return np.vstack([g['mean']-g.ci_low,g.ci_high-g['mean']])*scale
def quiet(ax):
    ax.grid(axis='y',color='#E4E4E4',linewidth=.5)
    ax.set_axisbelow(True)
def save(fig,name,data):
    fig.canvas.draw()
    texts=[x for x in fig.findobj(Text) if x.get_visible() and x.get_text()]
    AUDIT[name]={'width_inches':float(fig.get_figwidth()),'height_inches':float(fig.get_figheight()),
      'minimum_font_pt':min(x.get_fontsize() for x in texts),'source_rows':len(data)}
    for fmt in FORMATS:
        fig.savefig(OUT/f'{name}.{fmt}',dpi=DPI,facecolor='white')
    data.to_csv(OUT/f'{name}_source.csv',index=False)
    plt.close(fig)

def framework():
    fig,ax=plt.subplots(figsize=(6.5,3.5));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,6.5),ylim=(0,3.5));ax.axis('off')
    records=[]
    steps=[('Specify the model','Record dose, schedule,\nstrength and intervention rules'),
      ('Audit the design','Equalize bundled dimensions\nand cross them independently'),
      ('Interpret the evidence','State the scale, outcome\nwindow and model limits')]
    for i,(title,body) in enumerate(steps):
        x=.12+i*2.15
        ax.add_patch(Rectangle((x,2.42),1.94,.83,facecolor='#F5F6F7',edgecolor='#777',lw=.7))
        ax.text(x+.97,3.03,title,ha='center',va='center',fontsize=9,weight='bold')
        ax.text(x+.97,2.70,body,ha='center',va='center',fontsize=8.3,linespacing=1.4)
        if i<2:ax.add_patch(FancyArrowPatch((x+1.96,2.84),(x+2.13,2.84),arrowstyle='-|>',mutation_scale=9,color='#555',lw=.8))
        records.append({'layer':'procedure','title':title,'description':body.replace('\n',' ')})
    ax.text(3.25,2.12,'Complementary experiments',ha='center',weight='bold',fontsize=9)
    checks=[('E1','Independent crossing','Dose × schedule × strength\nwith and without reversion'),
      ('E2','Governance contrasts','Coverage, compliance\nand label persistence'),
      ('E3','Alternative dynamics','Anchored peer averaging\nand paired no-input controls'),
      ('E4','Statistical calibration','Known truth, covariance\nand replication count')]
    for i,(suite,title,body) in enumerate(checks):
        x=.12+i*1.60
        ax.add_patch(Rectangle((x,.83),1.45,1.02,facecolor='white',edgecolor='#777',lw=.7))
        ax.text(x+.725,1.62,suite,ha='center',weight='bold',fontsize=9,color=BLUE)
        ax.text(x+.725,1.36,title,ha='center',fontsize=8.3,weight='bold')
        ax.text(x+.725,1.01,body,ha='center',fontsize=8,linespacing=1.4)
        records.append({'layer':'experiment','title':suite+' '+title,'description':body.replace('\n',' ')})
    ax.add_patch(Rectangle((.12,.12),6.25,.46,facecolor='#EEF1F3',edgecolor='none'))
    ax.text(3.25,.39,'Verification & reproducibility checks',ha='center',weight='bold',fontsize=8.8)
    ax.text(3.25,.21,'Selector correction · source integrity · paired seeds · reproducible analysis',ha='center',fontsize=8)
    save(fig,'Fig01_audit_framework',pd.DataFrame(records))

def corrected():
    p=read('all_corrected_allocations.csv');p=p[(p.outcome=='mean_total_shift')&(p.factor=='seeding_regime')].set_index('design')
    r=read('all_corrected_ranges.csv');r=r[(r.outcome=='mean_total_shift')&(r.factor=='seeding_regime')]
    e=read('corrected_original_endpoints.csv')
    fig,axes=plt.subplots(1,3,figsize=(6.5,3.45),gridspec_kw={'width_ratios':[.9,1.4,.95]});fig.subplots_adjust(left=.085,right=.985,bottom=.22,top=.86,wspace=.77)
    names=['CS-0','CS-15','CS-30','CS-45']; x=[0,.15,.30,.45]
    g=p.loc[names];bounds=r[r.metric=='B_factor'].set_index('design').loc[names]
    ax=axes[0]
    ax.errorbar(x,g.corrected_B_factor*1000,yerr=np.array([g.corrected_B_factor-bounds.low,bounds.high-g.corrected_B_factor])*1000,
      color=BLUE,marker='o',mfc='white',capsize=4,elinewidth=1.5)
    panel(ax,'a','Mean dose 0.50')
    ax.set(xlabel='Assigned dose spread s',ylabel=r'Schedule allocation $B_j$ (×$10^{-3}$)',xticks=x,ylim=(0,.6));quiet(ax)
    ax=axes[1];regimes=['single_shot','burst','continuous']
    for j,(name,label,color,marker,doses) in enumerate([
      ('CS-45','Forward',BLUE,'o',[.05,.5,.95]),('CS-45-R','Reversed',ORANGE,'s',[.95,.5,.05])]):
        g=e[e.design==name].set_index('seeding_regime').loc[regimes]
        xx=np.arange(3)+(j-.5)*.16
        ax.plot(xx,g.mean_total_shift*1000,ls='none',color=color,marker=marker,label=label,mfc='white' if j else color)
        for k,(_,v) in enumerate(g.iterrows()):
            ax.annotate(f'{doses[k]:.2f}',(xx[k],v.mean_total_shift*1000),xytext=(0,8 if j==0 else -13),textcoords='offset points',ha='center',fontsize=8)
    panel(ax,'b','Dose reassignment')
    ax.set_xticks([0,1,2],['Single-\nshot','Burst','Continuous'],fontsize=8)
    ax.set(ylabel='Terminal absolute\ndisplacement (×$10^{-3}$)',ylim=(-4,80),xlim=(-.45,2.55))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.00),fontsize=8,handlelength=1);quiet(ax)
    ax=axes[2]; names=['C-D1','CS-0','D-A95'];g=p.loc[names];x=np.array([.05,.5,.95])
    for field,metric,label,color,marker,line in [('corrected_tau','tau','Total',BLUE,'o','-'),('corrected_q','q','Schedule',ORANGE,'s','--')]:
        b=r[r.metric==metric].set_index('design').loc[names]
        yy=g[field]*100
        ax.errorbar(x,yy,yerr=np.array([g[field]-b.low,b.high-g[field]])*100,color=color,marker=marker,ls=line,mfc='white',capsize=4,elinewidth=1.5,label=label)
        if metric=='q':
            for xx,val in zip(x,yy):ax.annotate('<0.001' if 0<val<.001 else f'{val:.3f}',(xx,val),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8,color=ORANGE)
    panel(ax,'c','Equal-dose anchors')
    ax.set(xlabel='Equal direct dose',ylabel='C-D50 total variation (%)',xticks=x,ylim=(-.5,19),xlim=(-.08,1.08));quiet(ax)
    ax.legend(loc='upper left',handlelength=1.3,fontsize=8)
    fig.text(.5,.035,'Whiskers in (a, c): resampling sensitivity ranges; dose values label points in (b).',ha='center',fontsize=8)
    save(fig,'Fig02_corrected_original',pd.concat([p.reset_index().assign(source='point'),r.assign(source='sensitivity_range'),e.assign(source='endpoint')],ignore_index=True))

def allocation():
    d=read('E1_allocations.csv');d=d[(d.view=='full_crossed')&(d.outcome=='mean_total_shift')&(d.factor!='TOTAL')]
    names=['topology','dose','seeding_regime','strength'];labels=['Topology','Direct dose','Schedule','Strength']
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.7));fig.subplots_adjust(left=.12,right=.98,top=.84,bottom=.18,wspace=.43)
    def marks(ax,g,shares=True):
        for i,(_,v) in enumerate(g.iterrows()):
            ax.errorbar(v.B*1e4,i,xerr=np.array([[v.B-v.B_low],[v.B_high-v.B]])*1e4,color=BLUE,
              marker='o',mfc='white',ms=3.5,ls='none',elinewidth=1.5,capsize=4,zorder=3)
            if shares:ax.annotate(f'{100*v.within_view_share:.2f}%',(v.B*1e4,i),xytext=(7,6),textcoords='offset points',fontsize=8)
    for j,(ax,mu) in enumerate(zip(axes,[0,.05])):
        g=d[d.reversion==mu].set_index('factor').loc[names];marks(ax,g)
        ax.set_yticks(range(4),labels);ax.set_ylim(6.9,-.7);ax.set_xlim(-.18,6.7)
        ax.set_xticks([0,2,4,6]);ax.axvline(0,color='#888',lw=.7)
        ax.set_xlabel(r'Absolute allocation $B_j$ (×$10^{-4}$)')
        total_label=f'{g.B_total.iloc[0]*1e4:.3f} ×$10^{{-4}}$'
        panel(ax,chr(97+j),f'Reversion = {mu:.2f}\nTotal B = {total_label}')
        if j==1:
            ins=ax.inset_axes([.34,.08,.64,.32]);marks(ins,g,False)
            ins.set_yticks(range(4),['T','D','S','M'],fontsize=8);ins.set_ylim(3.65,-.65)
            ins.set_xlim(-.008,.30);ins.set_xticks([0,.15,.30]);ins.tick_params(labelsize=8,pad=1)
            ins.set_title('Detail (×$10^{-4}$)',fontsize=8,pad=3);ins.axvline(0,color='#999',lw=.6)
            for sp in ins.spines.values():sp.set_visible(True);sp.set_linewidth(.6)
    axes[0].text(.03,.23,'Same absolute scale in both panels.\nLabels: share of each grid total.\nWhiskers: resampling sensitivity.',transform=axes[0].transAxes,fontsize=8,linespacing=1.55)
    fig.text(.5,.035,'Detail labels: T topology; D direct dose; S schedule; M message strength.',ha='center',fontsize=8)
    save(fig,'Fig03_crossed_attribution',d)

def response():
    d=read('E1_terminal_runs.csv');keys=['reversion','strength','dose','seeding_regime']
    assert d.groupby(keys+['replication']).topology.nunique().eq(4).all()
    b=d.groupby(keys+['replication'],as_index=False).mean_total_shift.mean();rows=[]
    for key,g in b.groupby(keys):
        assert len(g)==30
        m=g.mean_total_shift.mean();se=g.mean_total_shift.std(ddof=1)/np.sqrt(30);h=t.ppf(.975,29)*se
        rows.append(dict(zip(keys,key),mean=m,mcse=se,ci_low=m-h,ci_high=m+h,replication_blocks=30,topologies_per_block=4))
    s=pd.DataFrame(rows)
    fig,axes=plt.subplots(2,3,figsize=(6.5,4.55),sharex=True,sharey=True)
    fig.subplots_adjust(left=.16,right=.98,top=.89,bottom=.20,hspace=.35,wspace=.17)
    for r,mu in enumerate([0,.05]):
        for c,st in enumerate([.06,.14,.18]):
            ax=axes[r,c]
            for j,(reg,label,mk,ls) in enumerate([('single_shot','Single-shot','o','-'),('burst','Burst','s','--'),('continuous','Continuous','^',':')]):
                g=s[(s.reversion==mu)&(s.strength==st)&(s.seeding_regime==reg)].sort_values('dose')
                ax.errorbar(g.dose,g['mean']*1000,yerr=err(g,1000),color=COLORS[j],marker=mk,ls=ls,capsize=3,mfc='white',label=label)
            ax.set_ylim(0,s.ci_high.max()*1080);ax.set_xticks([.05,.5,.95]);quiet(ax)
            ax.text(.04,.90,f'({chr(97+r*3+c)})',transform=ax.transAxes,weight='bold')
            if r==0:ax.set_title(f'Strength = {st:.2f}',pad=9)
        fig.text(.105,.71 if r==0 else .395,f'Reversion = {mu:.2f}',rotation=90,ha='center',va='center',fontsize=8.5)
    fig.text(.025,.56,'Terminal absolute displacement (×$10^{-3}$)',rotation=90,ha='center',va='center',fontsize=9)
    fig.text(.56,.11,'Cumulative direct events per agent',ha='center',fontsize=9)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.55,.005),ncol=3)
    save(fig,'Fig04_dose_strength_response',s)

def governance():
    d=read('E2_policy_contrasts.csv');d=d[(d.outcome=='mean_total_shift')&~d.comparison.str.startswith('corrected')]
    groups=[(['hub_vs_random_B05_C050','hub_vs_random_B05_C100','hub_vs_random_B20_C050','hub_vs_random_B20_C100'],
      ['5% /\n50%','5% /\n100%','20% /\n50%','20% /\n100%'],'Hub − random targeting contrast','Target budget / compliance'),
      (['label_retention_L30','label_retention_L70'],['0.30','0.70'],'Retained-label − label-loss contrast','Label attenuation λ')]
    fig,axes=plt.subplots(2,2,figsize=(6.5,5.2),sharey='row')
    fig.subplots_adjust(left=.17,right=.98,bottom=.18,top=.88,wspace=.16,hspace=.92)
    for r,(names,labels,title,xlabel) in enumerate(groups):
        vals=d[d.comparison.isin(names)];lo=min(0,vals.ci_low.min()*1000);hi=vals.ci_high.max()*1000
        for c,dose in enumerate([.05,.5]):
            ax=axes[r,c]
            for j,(reg,label,mk) in enumerate([('single_shot','Single-shot','o'),('continuous','Continuous','s')]):
                g=d[(d.dose==dose)&(d.seeding_regime==reg)].set_index('comparison').loc[names]
                ax.errorbar(np.arange(len(names))+(j-.5)*.14,g['mean']*1000,yerr=err(g,1000),ls='none',marker=mk,
                  color=COLORS[j],mfc='white' if j else COLORS[j],capsize=3,label=label)
            ax.set_xticks(range(len(names)),labels);ax.set_xlim(-.45,len(names)-.55)
            ax.set_ylim(lo-(hi-lo)*.08,hi+(hi-lo)*.12);ax.axhline(0,color='#888',lw=.7);quiet(ax)
            ax.set_xlabel(xlabel);panel(ax,chr(97+r*2+c),f'Direct dose = {dose:.2f}')
        fig.text(.57,.97 if r==0 else .485,title,ha='center',weight='bold',fontsize=9)
    fig.text(.025,.55,'Difference in terminal absolute\ndisplacement reduction (×$10^{-3}$)',rotation=90,ha='center',va='center',fontsize=9)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.57,.035),ncol=2)
    fig.text(.57,.018,'Y-axis scales match across doses within each row; they differ between rows.',ha='center',fontsize=8)
    save(fig,'Fig05_governance_comparisons',d)

def timing():
    parts=[]
    fig,axes=plt.subplots(2,2,figsize=(6.5,4.7));fig.subplots_adjust(left=.13,right=.98,bottom=.16,top=.91,wspace=.36,hspace=.58)
    for r,(suite,field,values) in enumerate([('E1','reversion',[0,.05]),('E3','stubbornness',[.2,.8])]):
        d=read(f'{suite}_schedule_contrasts.csv');parts.append(d.assign(suite=suite))
        for c,outcome in enumerate(['mean_total_shift','mean_trajectory_shift']):
            ax=axes[r,c]
            for j,val in enumerate(values):
                g=d[(d[field]==val)&(d.outcome==outcome)].sort_values('dose')
                ax.errorbar(g.dose,g['mean']*1000,yerr=err(g,1000),color=COLORS[j],marker=['o','s'][j],ls=['-','--'][j],mfc='white',capsize=3,label=f'{field.capitalize()} = {val:g}')
            ax.axhline(0,color='#888',lw=.7);quiet(ax);ax.set_xticks([.05,.5,.95]);ax.legend(loc='best',fontsize=8)
            panel(ax,chr(97+r*2+c),suite+(' terminal' if c==0 else ' time-averaged'))
            if r==1:ax.set_xlabel('Cumulative direct events per agent',fontsize=8.5)
    fig.text(.025,.54,'Continuous − single-shot\nabsolute displacement (×$10^{-3}$)',rotation=90,ha='center',va='center',fontsize=9)
    fig.text(.54,.025,'Y-axis scales differ across panels; lines connect evaluated doses only.',ha='center',fontsize=8)
    save(fig,'Fig06_timing_and_outcomes',pd.concat(parts,ignore_index=True))

def coverage():
    d=read('E4_validation_summary.csv');d=d[d.estimator=='complete_u'].copy()
    noises=['independent_homoskedastic','independent_heteroskedastic','common_shock_heteroskedastic','structured_shock_heteroskedastic']
    titles=['Independent homoskedastic','Independent heteroskedastic','Common-shock heteroskedastic','Structured-shock heteroskedastic']
    cmap=LinearSegmentedColormap.from_list('coverage_deviation',['#2166AC','#F7F7F7','#B35806'])
    norm=TwoSlopeNorm(vmin=-12,vcenter=0,vmax=12)
    fig,axes=plt.subplots(4,2,figsize=(6.5,6.75));fig.subplots_adjust(left=.13,right=.985,bottom=.21,top=.91,wspace=.27,hspace=.49)
    rows=[]
    for r,(noise,title) in enumerate(zip(noises,titles)):
        for c,(method,label) in enumerate([('percentile','Percentile'),('jackknife','Jackknife-normal')]):
            ax=axes[r,c];g=d[d.noise==noise].set_index(['factor','R'])
            vals=np.array([[g.loc[(f,n),method+'_coverage']*100 for n in [10,30,60]] for f in 'ABCD'])
            # pcolormesh, not imshow: each heatmap cell is vector artwork in PDF/EPS.
            im=ax.pcolormesh(np.arange(4)-.5,np.arange(5)-.5,vals-95,cmap=cmap,norm=norm,edgecolors='white',linewidth=.6,rasterized=False)
            for i,f in enumerate('ABCD'):
                for j,n in enumerate([10,30,60]):
                    v=g.loc[(f,n)];low=v[method+'_coverage_wilson_low'];high=v[method+'_coverage_wilson_high'];flag=low>.95 or high<.95
                    ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if vals[i,j]<86 else DARK)
                    if flag:ax.add_patch(Rectangle((j-.475,i-.475),.95,.95,fill=False,edgecolor='black',linewidth=1.1))
                    rows.append(dict(noise=noise,method=method,factor=f,R=n,coverage_percent=vals[i,j],deviation_pp=vals[i,j]-95,wilson_low=low,wilson_high=high,wilson_excludes_95=bool(flag),coverage_datasets=int(v.coverage_datasets)))
            ax.set(xlim=(-.5,2.5),ylim=(3.5,-.5));ax.set_xticks([0,1,2],[10,30,60]);ax.set_yticks(range(4),['A','B','C','D (null)'])
            for spine in ax.spines.values():spine.set_visible(False)
            ax.tick_params(length=0,pad=3);ax.set_title(f'({chr(97+r*2+c)}) {title}',loc='left',fontsize=8.2,pad=6)
            if r==3:ax.set_xlabel('Replication blocks R')
            if r==0:ax.text(.5,1.49,label,transform=ax.transAxes,ha='center',weight='bold',fontsize=10)
    cbax=fig.add_axes([.24,.10,.62,.018]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[-12,-6,0,6,12])
    cb.solids.set_rasterized(False)
    cb.set_label('Coverage − 95% (percentage points)',fontsize=9)
    fig.text(.55,.015,'Black border: pointwise 95% Wilson interval excludes 95% coverage.',ha='center',fontsize=8)
    save(fig,'Fig07_interval_coverage',pd.DataFrame(rows))

def supplementary(number):
    legacy.RESULTS=DATA;legacy.FIGURES=OUT;legacy.COLORS=COLORS
    def export(fig,name,data):
        # Match the 6.5-inch manuscript width and ensure all lettering remains >=8 pt.
        w,h=fig.get_size_inches();ratio=6.5/w
        fig.set_size_inches(6.5,h*ratio)
        for text in fig.findobj(Text):
            text.set_fontfamily('Arial');text.set_fontsize(max(8,text.get_fontsize()*ratio))
            value=text.get_text().replace('Single shot','Single-shot').replace('Extra hub reduction','Hub − random reduction contrast').replace('Shift reduction','Terminal absolute\ndisplacement reduction')
            text.set_text(value)
        if fig._suptitle:fig._suptitle.set_text('')
        if number=='S1':
            for ax in fig.axes:ax.set_xlabel('')
            fig.supxlabel('Terminal absolute displacement reduction (×$10^{-3}$)',fontsize=9)
        if number=='S2':
            for ax in fig.axes:ax.set_ylabel('')
            fig.supylabel('Hub − random contrast in terminal absolute\ndisplacement reduction (×$10^{-3}$)',fontsize=9)
        if number=='S3':
            fig.axes[0].set_ylabel('Mean estimated allocation\nto null factor D',fontsize=9)
        if number=='S5':
            fig.axes[0].set_title('(a) Selector diagnostic',fontsize=9)
            fig.axes[1].set_title('(b) C-D50 allocation',fontsize=9)
        save(fig,name,data)
    legacy.save=export
    legacy.FIGURE_FUNCTIONS[number]()

FUNCTIONS={'1':framework,'2':corrected,'3':allocation,'4':response,'5':governance,'6':timing,'7':coverage}
def main():
    global DATA,OUT,FORMATS,DPI
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=DATA);p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--figures',nargs='+',choices=list(FUNCTIONS)+['S1','S2','S3','S4','S5'],default=list(FUNCTIONS)+['S1','S2','S3','S4','S5'])
    p.add_argument('--formats',nargs='+',choices=['pdf','eps','png','svg'],default=FORMATS);p.add_argument('--dpi',type=int,default=DPI)
    a=p.parse_args();DATA=a.data_dir.resolve();OUT=a.output_dir.resolve();FORMATS=a.formats;DPI=a.dpi;OUT.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data_manifest.json'
    if manifest.exists() and DATA==(ROOT/'data').resolve():
        for item in json.loads(manifest.read_text(encoding='utf8')):
            assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()==item['sha256'],item['file']
    for number in a.figures:
        FUNCTIONS[number]() if number in FUNCTIONS else supplementary(number)
        print('Exported Figure '+number,flush=True)
    (OUT/'artwork_audit.json').write_text(json.dumps(AUDIT,indent=2),encoding='utf8')
if __name__=='__main__':main()
