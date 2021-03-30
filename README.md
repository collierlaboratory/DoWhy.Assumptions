# DoWhy.Assumptions
Python code to check propensity score assumptions:  

# Import relevant libraries
import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.append(os.path.abspath("../../../"))import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyreadstat 
from .core import Data, Summary, Propensity, PropensitySelect, Strata
from .estimators import OLS, Blocking, Weighting, Matching, Estimators

# Read in the datafile INDV
df, meta = pyreadstat.read_sav('Users/Desktop/INDV_1819_5.sav' ,encoding='latin1')
print(df.head())

#Restrict the data to relevant variables
restriction_cols = [
    'CHARTER', 'MATH_19_SCORE_RPTD', 'RACE', 'SEX', 'MATH_18_SCORE_RPTD', 'FRLL', 'ELL',
]
missing = nhefs_all[restriction_cols].isnull().any(axis=1)
nhefs = nhefs_all.loc[~missing]

#Treatment variable
df["CHARTER"] = pd.DataFrame([True if i=='YES' else False for i in df["CHARTER"]])
df["CHARTER"]

#Missing Data in Outcome (Y)
df["MATH_19_SCORE_RPTD"].fillna(0, inplace = True)
df["MATH_18_SCORE_RPTD"].fillna(0, inplace = True)
df["MATH_19_SCORE_RPTD"]
df["MATH_18_SCORE_RPTD"]

#Formula for estimating propensity scores
formula = (
    'MATH_19_SCORE_RPTD ~ CHARTER + SEX + RACE + MATH_18_SCORE_RPTD + FRLL + ELL + I(MATH_18_SCORE_RPTD**2) '
    '         + I(FRLL**2) + I(ELL**2)'
)

model = sm.Logit.from_formula(formula, data=nhefs_all) 
res = model.fit(disp=0)

#Collect propensity scores
propensity = res.predict(nhefs_all)
nhefs_all['propensity'] = propensity
ranked = nhefs_all[['seqn', 'propensity']].sort_values('propensity').reset_index(drop=True)
ranked.loc[0]
ranked.loc[ranked.shape[0] - 1]
propensity0 = propensity[nhefs_all.MATH_19_SCORE_RPTD == 0]
propensity1 = propensity[nhefs_all.MATH_19_SCORE_RPTD == 1]

#Histograms
bins = np.arange(0.025, 0.85, 0.05)
top0, _ = np.histogram(propensity0, bins=bins)
top1, _ = np.histogram(propensity1, bins=bins)


fig, ax = plt.subplots(figsize=(8, 6))

ax.set_ylim(-115, 295)

ax.axhline(0, c='gray', linewidth=1)

bars0 = ax.bar(bins[:-1] + 0.025, top0, width=0.04, facecolor='white')
bars1 = ax.bar(bins[:-1] + 0.025, -top1, width=0.04, facecolor='gray')

for bars in (bars0, bars1):
    for bar in bars:
        bar.set_edgecolor("gray")

for x, y in zip(bins, top0):
    ax.text(x + 0.025, y + 10, str(y), ha='center', va='bottom')

for x, y in zip(bins, top1):
    ax.text(x + 0.025, -y - 10, str(y), ha='center', va='top')

ax.text(0.75, 260, "A = 0")
ax.text(0.75, -90, "A = 1")

ax.set_ylabel("No. Subjects", fontsize=14)
ax.set_xlabel("Estimated Propensity Score", fontsize=14);

print('                  mean propensity')
print('    non-quitters: {:>0.3f}'.format(propensity0.mean()))
print('        quitters: {:>0.3f}'.format(propensity1.mean()))


#cut extreme values
nhefs_all[['seqn', 'propensity']].loc[abs(propensity - 0.6563) < 1e-4]
nhefs_all['decile'] = pd.qcut(nhefs_all.propensity, 10, labels=list(range(10)))
nhefs_all.decile.value_counts(sort=False)

# create a model with interaction between CHARTER and L
model = sm.OLS.from_formula(
    'MATH_19_SCORE_RPTD ~ CHARTER *C(decile)', 
    data=nhefs_all
)
res = model.fit()

res.summary().tables[1]

# t-test with contrast DataFrame to get effect estimates
# start with empty DataFrame
contrast = pd.DataFrame(
    np.zeros((2, res.params.shape[0])),
    columns=res.params.index
)

# modify the constant entries
contrast['Intercept'] = 1
contrast['CHARTER'] = [1, 0]

# loop through t-tests, modify the DataFrame for each decile,
# and print out effect estimate and confidence intervals
print('           estimate    95% C.I.\n')
for i in range(10):
    if i != 0:
        # set the decile number
        contrast['C(decile)[T.{}]'.format(i)] = [1, 1]
        contrast['CHARTER:C(decile)[T.{}]'.format(i)] = [1, 0]
    
    ttest = res.t_test(contrast.iloc[0] - contrast.iloc[1])
    est = ttest.effect[0]
    conf_ints = ttest.conf_int(alpha=0.05)
    lo, hi = conf_ints[0, 0], conf_ints[0, 1]

    print('decile {}    {:>5.1f}    ({:>4.1f},{:>4.1f})'.format(i, est, lo, hi))
    
    if i != 0:
        # reset to zero
        contrast['C(decile)[T.{}]'.format(i)] = [0, 0]
        contrast['CHARTER:C(decile)[T.{}]'.format(i)] = [0, 0]
        
 # compare the estimates above to the estimate we get from a model without interaction between CHARTER and L
model = sm.OLS.from_formula(
    'MATH_19_SCORE_RPTD ~ CHARTER + C(decile)', 
    data=nhefs_all
)
res = model.fit()

res.summary().tables[1]

est = res.params.MATH_19_SCORE_RPTD
conf_ints = res.conf_int(alpha=0.05, cols=None)
lo, hi = conf_ints[0]['CHARTER'], conf_ints[1]['CHARTER']

print('         estimate   95% C.I.')
print('effect    {:>5.1f}    ({:>0.1f}, {:>0.1f})'.format(est, lo, hi))

# do "outcome regression E[Y|A, C=0, p(L)] with the estimated propensity score p(L) as a continuous covariate
nhefs['propensity'] = propensity[~nhefs_all.MATH_19_SCORE_RPTD.isnull()]

model = sm.OLS.from_formula('MATH_19_SCORE_RPTD ~ CHARTER + propensity', data=nhefs)
res = model.fit()
res.summary().tables[1]

# use bootstrap to get confidence intervals
def outcome_regress_effect(data):
    model = sm.OLS.from_formula('MATH_19_SCORE_RPTD ~ CHARTER + propensity', data=data)
    res = model.fit()
    
    data_CHARTER_1 = data.copy()
    data_CHARTER_1['CHARTER'] = 1
    
    data_CHARTER_0 = data.copy()
    data_CHARTER_0['CHARTER'] = 0
    
    mean_MATH_19_SCORE_RPTD_1 = res.predict(data_MATH_19_SCORE_RPTD_1).mean()
    mean_MATH_19_SCORE_RPTD_0 = res.predict(data_MATH_19_SCORE_RPTD_0).mean()
    
    return mean_MATH_19_SCORE_RPTD_1 - mean_MATH_19_SCORE_RPTD_0
    
 def nonparametric_bootstrap(data, func, n=1000):
    estimate = func(data)
    
    n_rows = data.shape[0]
    indices = list(range(n_rows))
    
    b_values = []
    for _ in tqdm(range(n)):
        data_b = data.sample(n=n_rows, replace=True)
        b_values.append(func(data_b))
    
    std = np.std(b_values)
    
    return estimate, (estimate - 1.96 * std, estimate + 1.96 * std)
    
  data = nhefs[['MATH_19_SCORE_RPTD', 'CHARTER', 'propensity']]

info = nonparametric_bootstrap(
    data, outcome_regress_effect, n=2000
)

print('         estimate   95% C.I.')
print('effect    {:>5.1f}    ({:>0.1f}, {:>0.1f})'.format(info[0], info[1][0], info[1][1]))


# create a column 'strata' for each element that marks what strata it belongs to
data['strata'] = ((data['df'].rank(ascending=True) / numrows) * numStrata).round(0)

# T_y = outcome iff treated
data['T_y'] = data['CHARTER'] * data['MATH_19_SCORE_RPTD’]

# Tbar = 1 iff untreate
data['Tbar'] = 1 - data['CHARTER’]

# Tbar_y = outcome iff untreated
data['Tbar_y'] = data['Tbar'] * data['MATH_19_SCORE_RPTD']

stratified = data.groupby('strata')

# sum weighted outcomes over all strata  (weight by treated population)
outcomes = stratified.agg({'T':['sum'],'Tbar':['sum'],'T_y':['sum'],'Tbar_y':['sum']}) 

# calculate per-strata effect and weighted sum of effects over all strata
outcomes[‘T_y_mean'] = outcomes[‘T_y_sum'] / outcomes['CHARTER']
outcomes[‘Tbar_y_mean'] = outcomes[‘Tbar_y_sum'] / outcomes['dbar_sum'] 
outcomes['effect'] = outcomes[‘T_y_mean'] - outcomes[‘Tbar_y_mean’]
att = (outcomes['effect'] * outcomes['CHARTER']).sum() / totaltreatmentpopulation




