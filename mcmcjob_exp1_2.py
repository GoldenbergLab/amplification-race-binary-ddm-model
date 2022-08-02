import pickle
import os
import sys
import numpy as np
import pandas as pd
import random
import pystan
import arviz as az
import warnings
warnings.filterwarnings("ignore")

raw = pd.read_csv("final_dataset_for_ddm_dec_21.csv")

# create new dataframe df with only the columns that I need for DDM
df = pd.DataFrame({'ID': raw['Random.ID'], 'choice': raw['emo_binary'],
                   'rt': raw['rt'], 'valence': raw['valence'],
                   'identity': raw['faces'], 'intensity': raw['valence_values'],
                   'ratio': raw['b_person_ratio']})

# convert the nominal variables into integers for Stan; convert identity and intensity strings to lists; convert rt to seconds
df = df.dropna(axis=0).reset_index(drop=True)
df['choice'] = [1 if x == 'Not Emotional' else 2 for x in df['choice']]
df['valence'] = [1 if x == 'Happy' else 2 for x in df['valence']]
df['identity'] = [x.split(', ') for x in df['identity']]
df['intensity'] =  [x.split(', ') for x in df['intensity']]
modinten = []
subt = {'1': 50, '2': 100}
for i, x in enumerate(df['intensity']):
    modinten.append([int(y) - subt[str(df['valence'][i])] for y in df['intensity'][i]])
df['intensity'] = modinten

df['rt'] = [x/1000 for x in df['rt']]

# remove any subjects that lack variability in their emo_binary choices - can't use them for DDM
dellist = []
for x in df['ID'].unique():
    if len(df[df['ID']==x]['choice'].unique()) < 2:
        dellist.append(x)
print('subjects with no variation: %s' % dellist)
df = df[~df['ID'].isin(dellist)]

# remove any observations where rt < 100ms - these are likely false starts
df = df[df['rt'] > 0.1]
df = df.reset_index(drop=True)

df['avg_intensity'] = 0
for i, x in enumerate(df['intensity']):
    df['avg_intensity'][i] = np.mean([int(y) for y in x])

# collapse face identities into either black or white; append NAs to identity lists that are len < 12 so that all vectors are len 12
# likewise, append 0.0 to intensity lists so that all intensity vectors are len 12
identitydict = {'E': 1, 'F': 1, 'B': 2, 'C': 2, 'NA': 0}
for i, x in enumerate(df['identity']):
    while len(x) < 12:
         x.append('NA')
         df['intensity'][i].append(-1)
    df['identity'][i] = [identitydict[e] for e in x]

df['indexer'] = [4*df['ratio'][i] + 3*(df['valence'][i]-1) for i, x in enumerate(df['ID'])]

print(df)
# convert df variables into arrays and vectors for Stan data block
grouped = df.groupby(['ID'], sort=False)
trials_per = grouped.size()
subs = list(trials_per.index)
nsubs = len(subs)
tsubs = list(trials_per)
tmax = max(tsubs)
choice = np.full((nsubs, tmax), -1, dtype=int)
rt = np.full((nsubs, tmax), -1, dtype=float)
valence = np.full((nsubs, tmax), -1, dtype=int)
avg_intensity = np.full((nsubs, tmax), -1, dtype=int)
intensity = np.full((nsubs, tmax, 12), -1, dtype=int)
identity = np.full((nsubs, tmax, 12), -1, dtype=int)
ratio = np.full((nsubs, tmax), -1, dtype=float)
indexer = np.full((nsubs, tmax), -1, dtype=int)
sub_group = iter(grouped)
for s in range(nsubs):
    _, sub_data = next(sub_group)
    t = tsubs[s]
    choice[s][:t] = sub_data['choice']
    rt[s][:t] = sub_data['rt']
    valence[s][:t] = sub_data['valence']
    avg_intensity[s][:t] = sub_data['avg_intensity']
    intensity[s][:t] = np.asarray([np.array(x) for x in sub_data['intensity']])
    identity[s][:t] = np.asarray([np.array(x) for x in sub_data['identity']])
    ratio[s][:t] = sub_data['ratio']
    indexer[s][:t] = sub_data['indexer']
rtmin = np.full(nsubs, -1, dtype=float)
rtbound = 0.1
sub_group = iter(grouped)
for s in range(nsubs):
    _, sub_data = next(sub_group)
    rtmin[s] = min(sub_data['rt'])
data = {
    'N': nsubs,
    'T': tmax,
    'Tsub': tsubs,
    'choice': choice,
    'valence': valence,
    'rt': rt,
    'rtmin': rtmin,
    'rtbound': rtbound,
    'avg_intensity': avg_intensity,
    'intensity': intensity,
    'identity': identity,
    'ratio': ratio,
    'indexer': indexer,
}

def init_f_free1_2():
    res_free1_2 = {'mu_pr': [.5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], #ddmraceall
        'sigma': [1]*19,
        'alpha_pr': [[.5]*nsubs]*6,
        'zed_pr': [[0]*nsubs]*6,
        'delta_pr': [[0]*nsubs]*6,
        'tau_pr': [-1]*nsubs
    }
    return res_free1_2

def init_f_fixed1_2():
    res_fixed1_2 = {'mu_pr': [.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], #ddmraceall
        'sigma': [1]*14,
        'alpha_pr': [.5]*nsubs,
        'zed_pr': [[0]*nsubs]*6,
        'delta_pr': [[0]*nsubs]*6,
        'tau_pr': [-1]*nsubs
    }
    return res_fixed1_2

# import the Stan code from ddm_stan.py; remove and reimport if already imported - this is helpful when making changes to Stan code
if "ddm_stan" in sys.modules:
    sys.modules.pop('ddm_stan')
from ddm_stan import ddmracecode_freealpha_study1_2, ddmracecode_fixedalpha_study1_2

# compile C++ code for model
ddm_sm = pystan.StanModel(model_code=ddmracecode_freealpha_study1_2, model_name='DDM')
ddm_fit = ddm_sm.sampling(data=data, init=init_f_free1_2, iter=10000, warmup=5000, chains=4, seed=101, refresh=1)
azdf = az.from_pystan(posterior=ddm_fit, observed_data=['choice'], log_likelihood={'logy': 'log_lik'},)
az.to_netcdf(azdf, 'ddmrace_freealpha_study1_2.nc')

ddm_sm = pystan.StanModel(model_code=ddmracecode_fixedalpha_study1_2, model_name='DDM')
ddm_fit = ddm_sm.sampling(data=data, init=init_f_fixed1_2, iter=10000, warmup=5000, chains=4, seed=101, refresh=1)
azdf = az.from_pystan(posterior=ddm_fit, observed_data=['choice'], log_likelihood={'logy': 'log_lik'},)
az.to_netcdf(azdf, 'ddmrace_fixedalpha_study1_2.nc')
