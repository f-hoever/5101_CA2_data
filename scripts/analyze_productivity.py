"""Employment structure vs energy mix in explaining real carbon productivity.
Main industry proxies: separate mining and manufacturing shares of private
QCEW annual-average covered jobs; combined share retained as sensitivity. Suppressed observations remain missing, never zero.
This independent exploration does not edit submitted reports.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/processed/employment_carbon_productivity';OUT.mkdir(exist_ok=True)
raw=ROOT/'data/raw/qcew_employment'
gaz=pd.read_csv(ROOT/'data/raw/2024_Gaz_state_national.txt',sep='\t',dtype={'GEOID':str});gaz.columns=gaz.columns.str.strip();state_map=dict(zip(gaz.GEOID.str.zfill(2)+'000',gaz.USPS))
existing=pd.read_csv(ROOT/'data/processed/state_panel.csv');state_map={k:v for k,v in state_map.items() if v in set(existing.State)}
records=[]
for year in range(2014,2025):
 for sector in ['10','21','31_33']:
  f=raw/f'{year}_{sector}.csv';d=pd.read_csv(f,dtype={'area_fips':str,'own_code':str,'industry_code':str,'agglvl_code':str,'size_code':str,'disclosure_code':str})
  # Same private ownership scope for numerator and denominator.
  s=d[d.area_fips.isin(state_map)&d.own_code.eq('5')&d.size_code.eq('0')].copy()
  assert not s.area_fips.duplicated().any(),f'Duplicate states in {f}'
  for _,r in s.iterrows():
   disclosed=pd.isna(r.disclosure_code) or str(r.disclosure_code).strip()==''
   records.append({'State':state_map[r.area_fips],'year':year,'sector':sector,'employment':pd.to_numeric(r.annual_avg_emplvl,errors='coerce') if disclosed else np.nan,'suppressed':not disclosed})
audit=pd.DataFrame(records);audit.to_csv(OUT/'employment_missingness.csv',index=False)
employment=audit.pivot(index=['State','year'],columns='sector',values='employment').rename(columns={'10':'private_employment','21':'mining_employment','31_33':'manufacturing_employment'}).reset_index()
panel=existing[existing.year.between(2014,2024)].merge(employment,on=['State','year'],how='left',validate='one_to_one')
panel['employment_industry_share']=(panel.mining_employment+panel.manufacturing_employment)/panel.private_employment
panel['employment_mining_share']=panel.mining_employment/panel.private_employment;panel['employment_manufacturing_share']=panel.manufacturing_employment/panel.private_employment
panel['carbon_productivity']=panel.rgdp/panel.TETCE;panel['ln_cp']=np.log(panel.carbon_productivity)
assert np.allclose(panel.ln_cp,np.log(panel.rgdp)-np.log(panel.TETCE))
groups={'industry':['employment_industry_share'],'energy':['coal_share','lowcarbon_share','gas_share'],'combined':['employment_industry_share','coal_share','lowcarbon_share','gas_share'],'full':['employment_industry_share','coal_share','lowcarbon_share','gas_share','ln_price','ln_density'],'industry_controls':['employment_industry_share','ln_price','ln_density'],'energy_controls':['coal_share','lowcarbon_share','gas_share','ln_price','ln_density'],'split_industry':['employment_mining_share','employment_manufacturing_share'],'split_industry_controls':['employment_mining_share','employment_manufacturing_share','ln_price','ln_density'],'split_full':['employment_mining_share','employment_manufacturing_share','coal_share','lowcarbon_share','gas_share','ln_price','ln_density']}
need=['ln_cp','employment_industry_share','coal_share','lowcarbon_share','gas_share','ln_price','ln_density']
s=panel.dropna(subset=need).copy();assert np.isfinite(s[need].to_numpy()).all()
cs=s[s.year==2024].sort_values('State').reset_index(drop=True)
rows=[];coeff=[];tests=[];cv=[]
def capture(m,design,name,baseline=None):
 row={'design':design,'model':name,'N':int(m.nobs),'R2':m.rsquared,'adj_R2':m.rsquared_adj}
 if baseline is not None:
  row['partial_R2_over_FE_or_year']=1-m.ssr/baseline.ssr
  row['adjusted_partial_R2']=1-(m.ssr/m.df_resid)/(baseline.ssr/baseline.df_resid)
 rows.append(row)
 ci=m.conf_int()
 for v in groups[name]:coeff.append({'design':design,'model':name,'variable':v,'coef':m.params[v],'p':m.pvalues[v],'lo':ci.loc[v,0],'hi':ci.loc[v,1]})
 if name in ['combined','full']:
  tests.append({'design':design,'model':name,'test':'energy_joint','p':float(m.f_test('coal_share = lowcarbon_share = gas_share = 0').pvalue)})
  tests.append({'design':design,'model':name,'test':'industry','p':float(m.pvalues.employment_industry_share)})
# Cross-section: both reduced models exclude controls, as specified.
y=cs.ln_cp.to_numpy();crossmodels={}
for name,x in groups.items():
 m=sm.OLS(y,sm.add_constant(cs[x])).fit(cov_type='HC3',use_t=True);crossmodels[name]=m;capture(m,'2024_cross_section',name)
 X=sm.add_constant(cs[x]).to_numpy();q=[]
 for seed in range(42,72):
  sse=sst=0
  for tr,te in KFold(5,shuffle=True,random_state=seed).split(cs):
   beta=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0];sse+=np.square(y[te]-X[te]@beta).sum();sst+=np.square(y[te]-y[tr].mean()).sum()
  score=1-sse/sst;q.append(score);cv.append({'model':name,'seed':seed,'Q2':score})
 rows[-1]['random_Q2']=float(np.mean(q))
# State/year fixed effects; meaningful comparison excludes FE-explained variation.
counts=s.groupby('State').year.nunique();balanced_states=counts[counts.eq(11)].index
for design,data in [('panel_FE',s),('balanced_panel_FE',s[s.State.isin(balanced_states)]),('panel_FE_no_pandemic',s[~s.year.isin([2020,2021])])]:
 b=smf.ols('ln_cp ~ C(State)+C(year)',data).fit()
 for name in ['industry','energy','combined','full','industry_controls','energy_controls','split_industry','split_industry_controls','split_full']:
  m=smf.ols('ln_cp ~ '+' + '.join(groups[name])+' + C(State)+C(year)',data).fit(cov_type='cluster',cov_kwds={'groups':data.State},use_t=True);capture(m,design,name,b)
# Annual changes are calculated before dropping missing employment observations.
fd=panel.sort_values(['State','year']).copy();cols=list(dict.fromkeys(need+groups['split_industry']))
fd[cols]=fd.groupby('State')[cols].diff();fd=fd.dropna(subset=cols)
b=smf.ols('ln_cp ~ C(year)',fd).fit()
for name in ['industry','energy','combined','full','industry_controls','energy_controls','split_industry','split_industry_controls','split_full']:
 m=smf.ols('ln_cp ~ '+' + '.join(groups[name])+' + C(year)',fd).fit(cov_type='cluster',cov_kwds={'groups':fd.State},use_t=True);capture(m,'annual_changes',name,b)
# Same-year rankings across all eleven years; no selection of best year.
annual=[]
for year,data in s.groupby('year'):
 row={'year':int(year),'N':len(data)}
 for name in ['industry','energy']:
  m=sm.OLS(data.ln_cp,sm.add_constant(data[groups[name]])).fit();row[name+'_adj_R2']=m.rsquared_adj
 annual.append(row)
summary={'employment_source':'BLS QCEW annual average covered private-sector jobs, NAICS 21 and 31-33 divided by private-sector NAICS10; not unique employed persons or all employment','years':[2014,2024],'states_in_raw':50,'cross_section_N':len(cs),'cross_section_excluded':sorted(set(existing.State)-set(cs.State)),'panel_complete_N':len(s),'panel_missing_N':len(panel)-len(s),'balanced_states':len(balanced_states),'balanced_excluded':sorted(set(existing.State)-set(balanced_states)),'carbon_productivity_units':'chained-2017 dollars of state real GDP per metric tonne of energy-related CO2; numerator and denominator million units cancel','warning':'Employment structure has no GDP denominator. It is not value-added structure, covers jobs rather than persons, excludes uncovered/self employment. Definitions and suppressed data must be disclosed. All analyses exploratory.'}
pd.DataFrame(rows).to_csv(OUT/'model_comparison.csv',index=False);pd.DataFrame(coeff).to_csv(OUT/'coefficients.csv',index=False);pd.DataFrame(tests).to_csv(OUT/'group_tests.csv',index=False);pd.DataFrame(cv).to_csv(OUT/'cross_validation.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'annual_comparison.csv',index=False);panel.to_csv(OUT/'analysis_panel.csv',index=False);(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2));print(pd.DataFrame(rows).round(5).to_string(index=False));print('\nFULL COEFFICIENTS\n',pd.DataFrame(coeff).query('model=="full"').round(5).to_string(index=False));print('\nANNUAL RANKING\n',pd.DataFrame(annual).round(4).to_string(index=False))

# All-year and leave-one-state-out checks of the disaggregated employment proxy.
annual_split=[];loo=[]
for year,data in s.groupby('year'):
 row={'year':int(year),'N':len(data)}
 for name in ['industry','split_industry','energy']:
  m=sm.OLS(data.ln_cp,sm.add_constant(data[groups[name]])).fit();row[name+'_adj_R2']=m.rsquared_adj
 annual_split.append(row)
for omitted in cs.State:
 data=cs[cs.State!=omitted];r={'omitted':omitted}
 for name in ['split_industry','energy']:
  m=sm.OLS(data.ln_cp,sm.add_constant(data[groups[name]])).fit();r[name+'_adj_R2']=m.rsquared_adj
 loo.append(r)
pd.DataFrame(annual_split).to_csv(OUT/'annual_split_comparison.csv',index=False);pd.DataFrame(loo).to_csv(OUT/'leave_one_state_out.csv',index=False)
print('ANNUAL SPLIT COMPARISON\n',pd.DataFrame(annual_split).round(4).to_string(index=False))
print('SPLIT WINS AFTER SINGLE STATE OMISSION:',sum(r['split_industry_adj_R2']>r['energy_adj_R2'] for r in loo),'of',len(loo))
equal=[]
for design,data,tail in [('cross_section',cs,''),('panel_FE',s,' + C(State)+C(year)')]:
 for name in ['split_industry','split_full']:
  formula='ln_cp ~ '+' + '.join(groups[name])+tail
  m=smf.ols(formula,data).fit(cov_type='HC3',use_t=True) if design=='cross_section' else smf.ols(formula,data).fit(cov_type='cluster',cov_kwds={'groups':data.State},use_t=True)
  test=m.t_test('employment_mining_share - employment_manufacturing_share = 0')
  equal.append({'design':design,'model':name,'sector_equality_p':float(np.asarray(test.pvalue).item())})
pd.DataFrame(equal).to_csv(OUT/'sector_equality_tests.csv',index=False)

# Controlled head-to-head comparison requested after initial exploration.
controlled=[];annual_controlled=[]
for design,data,tail in [('2024_cross_section',cs,''),('panel_FE',s,' + C(State)+C(year)'),('balanced_panel_FE',s[s.State.isin(balanced_states)],' + C(State)+C(year)'),('panel_FE_no_pandemic',s[~s.year.isin([2020,2021])],' + C(State)+C(year)'),('annual_changes',fd,' + C(year)')]:
 base=smf.ols('ln_cp ~ ln_price + ln_density'+tail,data).fit()
 for name in ['split_industry_controls','energy_controls','split_full']:
  formula='ln_cp ~ '+' + '.join(groups[name])+tail
  m=smf.ols(formula,data).fit(cov_type='HC3',use_t=True) if design=='2024_cross_section' else smf.ols(formula,data).fit(cov_type='cluster',cov_kwds={'groups':data.State},use_t=True)
  controlled.append({'design':design,'model':name,'N':int(m.nobs),'R2':m.rsquared,'adj_R2':m.rsquared_adj,'structure_partial_R2_over_common_controls':1-m.ssr/base.ssr,'adjusted_structure_partial_R2':1-(m.ssr/m.df_resid)/(base.ssr/base.df_resid)})
for year,data in s.groupby('year'):
 row={'year':int(year),'N':len(data)}
 for name in ['split_industry_controls','energy_controls']:
  m=sm.OLS(data.ln_cp,sm.add_constant(data[groups[name]])).fit();row[name+'_adj_R2']=m.rsquared_adj
 annual_controlled.append(row)
pd.DataFrame(controlled).to_csv(OUT/'controlled_comparison.csv',index=False);pd.DataFrame(annual_controlled).to_csv(OUT/'annual_controlled_comparison.csv',index=False)
pivot=pd.DataFrame(cv).pivot(index='seed',columns='model',values='Q2');delta=pivot.split_industry_controls-pivot.energy_controls
cv_summary={'industry_Q2_mean':float(pivot.split_industry_controls.mean()),'energy_Q2_mean':float(pivot.energy_controls.mean()),'industry_better_repetitions':int((delta>0).sum()),'repetitions':len(delta),'difference_range':[float(delta.min()),float(delta.max())],'difference_mean':float(delta.mean()),'warning':'Repeated splits overlap; range is not a confidence interval and no formal difference test is implied.'}
(OUT/'controlled_cv_summary.json').write_text(json.dumps(cv_summary,indent=2),encoding='utf-8')
print('CONTROLLED GROUP CONTRIBUTIONS\n',pd.DataFrame(controlled).round(5).to_string(index=False))
print('CONTROLLED ANNUAL RANKING\n',pd.DataFrame(annual_controlled).round(5).to_string(index=False));print('CONTROLLED CV',cv_summary)

# Joint association tests for the two reduced structural groups.
reduced_tests=[]
for design,data,tail in [('2024',s[s.year==2024],''),('panel',s,' + C(State)+C(year)')]:
 for name,x,h in [('industry',groups['split_industry'],'employment_mining_share = employment_manufacturing_share = 0'),('energy',groups['energy'],'coal_share = lowcarbon_share = gas_share = 0')]:
  formula='ln_cp ~ '+' + '.join(x)+tail
  m=smf.ols(formula,data).fit(cov_type='HC3',use_t=True) if design=='2024' else smf.ols(formula,data).fit(cov_type='cluster',cov_kwds={'groups':data.State},use_t=True)
  f=m.f_test(h);reduced_tests.append({'design':design,'model':name,'F':float(f.fvalue),'p':float(f.pvalue),'df_num':float(f.df_num),'df_denom':float(f.df_denom)})
pd.DataFrame(reduced_tests).to_csv(OUT/'reduced_group_tests.csv',index=False)
