"""Readable result figures: annual comparison, validation boxes, panel robustness."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/processed/employment_carbon_productivity';ASSETS=ROOT/'paper/assets';ASSETS.mkdir(parents=True,exist_ok=True)
colors=['#286B86','#C17A43'];words={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
a=pd.read_csv(OUT/'annual_split_comparison.csv');cv=pd.read_csv(OUT/'cross_validation.csv').pivot(index='seed',columns='model',values='Q2')
fig,ax=plt.subplots(1,2,figsize=(9,3.65),gridspec_kw={'width_ratios':[1.15,1]})
for col,name,color in [('split_industry_adj_R2','Industry',colors[0]),('energy_adj_R2','Energy',colors[1])]:
 ax[0].plot(a.year,a[col],color=color,lw=2,marker='o',ms=3.6,label=name)
ax[0].set(xlabel='Year',ylabel='Adjusted R²',title='A Fit across years',xticks=[2014,2016,2018,2020,2022,2024],ylim=(.5,.81));ax[0].legend(frameon=False,fontsize=9,loc='upper left');ax[0].grid(axis='y',color='#E4E8EB',lw=.7)
values=[cv.split_industry.to_numpy(),cv.energy.to_numpy()]
bp=ax[1].boxplot(values,positions=[1,2],widths=.5,patch_artist=True,showfliers=True,medianprops={'color':'#202020','linewidth':1.6},whiskerprops={'color':'#666666'},capprops={'color':'#666666'},flierprops={'marker':'o','markersize':3,'markerfacecolor':'#777777','markeredgewidth':0})
for patch,color in zip(bp['boxes'],colors):patch.set(facecolor=color,alpha=.75,edgecolor=color)
for i,v in enumerate(values,1):ax[1].text(i+.29,np.median(v),f'{np.median(v):.3f}',va='center',fontsize=9)
ax[1].set(xticks=[1,2],xticklabels=['Industry','Energy'],ylabel='Q²',title='B Prediction across splits',ylim=(.4,.81),xlim=(.45,2.85));ax[1].grid(axis='y',color='#E4E8EB',lw=.7);ax[1].set_axisbelow(True)
fig.tight_layout(w_pad=2.5)
name='cross_section_result_comparison.png';fig.savefig(ASSETS/name,dpi=270,bbox_inches='tight',facecolor='white');words[name]=len(' '.join(t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_visible()).split());plt.close(fig)
co=pd.read_csv(OUT/'coefficients.csv')
fig,axes=plt.subplots(1,2,figsize=(9,3.55),sharex=True)
settings=[('split_industry',['employment_mining_share','employment_manufacturing_share'],['Mining','Manufacturing'],'A Employment model',colors[0]),('energy',['coal_share','lowcarbon_share','gas_share'],['Coal','Low-carbon','Natural gas'],'B Energy model',colors[1])]
plotted=[]
for ax,(model,variables,labels,title,color) in zip(axes,settings):
 data=co[(co.design=='panel_FE')&co.model.eq(model)].set_index('variable').loc[variables]
 effect=100*np.expm1(.01*data.coef.to_numpy());lo=100*np.expm1(.01*data.lo.to_numpy());hi=100*np.expm1(.01*data.hi.to_numpy());y=np.arange(len(labels))
 ax.errorbar(effect,y,xerr=np.vstack([effect-lo,hi-effect]),fmt='o',color=color,ecolor=color,elinewidth=2,capsize=4,markersize=7)
 for yi,e,l,h,label in zip(y,effect,lo,hi,labels):
  ax.text(e,yi-.16,f'{e:+.2f}%',color=color,ha='center',va='bottom',fontsize=9)
  plotted.append({'model':model,'variable':label,'CP_change_for_1pp_pct':e,'CI95_lower_pct':l,'CI95_upper_pct':h})
 ax.axvline(0,color='#777777',ls='--',lw=1)
 ax.set(yticks=y,yticklabels=labels,title=title,xlim=(-5,12),xticks=[-4,0,4,8,12],ylim=(len(labels)-.4,-.6))
 ax.grid(axis='x',color='#E4E8EB',lw=.7);ax.set_axisbelow(True)
fig.supxlabel('CP change for a 1-percentage-point share increase (%)',fontsize=10)
fig.tight_layout(w_pad=2.8)
name='panel_variable_associations.png';fig.savefig(ASSETS/name,dpi=270,bbox_inches='tight',facecolor='white');words[name]=len(' '.join(t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_visible()).split());plt.close(fig)
pd.DataFrame(plotted).to_csv(OUT/'plotted_panel_associations.csv',index=False)
(OUT/'figure_metadata.json').write_text(json.dumps({'chart_words':words,'notes':'Annual fits use common states within each year. Boxes describe 30 overlapping five-fold splits, not confidence intervals. Panel coefficients are from separate employment and energy models with state/year effects; intervals are pointwise state-clustered 95% intervals. Effects use 100*expm1(0.01*beta).'},indent=2),encoding='utf-8')
print(json.dumps(words,indent=2))
