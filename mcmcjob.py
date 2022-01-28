# some libraries might not be used for pre-processing and Stan modeling - might be orphans of past notebooks :/
import pickle
import os
import sys
import numpy as np
import pandas as pd
import random
import pystan
import warnings
warnings.filterwarnings("ignore")

old = False

if old == False:
    raw = pd.read_csv("final_dataset_for_ddm_dec_21.csv")
else:
    raw = pd.read_csv("final_dataset_ratings_with_identity_values.csv")

# create new dataframe df with only the columns that I need for DDM
df = pd.DataFrame({'ID': raw['Random.ID'], 'choice': raw['emo_binary'],
                   'rt': raw['rt'], 'valence': raw['valence'],
                   'identity': raw['faces'], 'intensity': raw['valence_values'],
                   'ratio': raw['b_person_ratio']})

# convert the nominal variables into integers for Stan; convert identity and intensity strings to lists; convert rt to seconds
df = df.dropna(axis=0).reset_index(drop=True)
df['choice'] = [1 if x == 'Not Emotional' else 2 for x in df['choice']]
df['valence'] = [1 if x == 'Happy' else 2 for x in df['valence']]
if old == True:
    df['identity'] = [eval(x) for x in df['identity']]
    df['intensity'] = [eval(x) for x in df['intensity']]
else:
    df['identity'] = [x.split(', ') for x in df['identity']]
    df['intensity'] =  [x.split(', ') for x in df['intensity']]
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

# collapse face identities into either black or white; append NAs to identity lists that are len < 12 so that all vectors are len 12
# likewise, append 0.0 to intensity lists so that all intensity vectors are len 12
identitydict = {'E': 1, 'F': 1, 'B': 2, 'C': 2, 'NA': 0}
for i, x in enumerate(df['identity']):
    while len(x) < 12:
        x.append('NA')
        df['intensity'][i].append(0)
    df['identity'][i] = [identitydict[e] for e in x]

df['indexer'] = [4*df['ratio'][i] + 3*(df['valence'][i]-1) for i, x in enumerate(df['ID'])]

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
    'intensity': intensity,
    'identity': identity,
    'ratio': ratio,
    'indexer': indexer,
}

def init_f():
    # res = {'mu_pr': [1, 1, 0, 0, 0, 0, -1], #ddm
    #     'sigma': [1]*7,
    #     'alpha_pr': [[1]*nsubs]*2,
    #     'zed_pr': [[0]*nsubs]*2,
    #     'delta_pr': [[0]*nsubs]*2,
    #     'tau_pr': [-1]*nsubs
    # }
    # res = {'mu_pr': [.5, .5, 0, 0, 0, 0, 0, -1],
    #     'sigma': [1]*8,
    #     'alpha_pr': [[.5]*nsubs]*2,
    #     'zed_pr': [[0]*nsubs]*2,
    #     'delta_pr': [[0]*nsubs]*3,
    #     'tau_pr': [-1]*nsubs
    # }
    res = {'mu_pr': [.5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], #ddmraceall
        'sigma': [1]*19,
        'alpha_pr': [[.5]*nsubs]*6,
        'zed_pr': [[0]*nsubs]*6,
        'delta_pr': [[0]*nsubs]*6,
        'tau_pr': [-1]*nsubs
    }
    # res = {'mu_pr': [.5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    #     'sigma': [1]*15,
    #     'alpha_pr': [[.5]*nsubs]*2,
    #     'zed_pr': [[0]*nsubs]*2,
    #     'lambda_pr': [[0]*nsubs]*2,
    #     'beta_pr':[[0]*nsubs]*4,
    #     'gamma_pr':[[0]*nsubs]*4,
    #     'tau_pr': [-1]*nsubs
    # }
    return res

# import the Stan code from facesddm_stan.py; remove and reimport if already imported - this is helpful when making changes to Stan code
if "facesddm_stan" in sys.modules:
    sys.modules.pop('facesddm_stan')
from facesddm_stan import ddmraceallcode

# compile C++ code for model
ddm_sm = pystan.StanModel(model_code=ddmraceallcode, model_name='DDM')

# fit the model to the data
# more iterations the better, but at cost of compute and memory; shoot for 100,000 iterations, 50,000 of which are warmup
# Use at least 4 chains; thin the samples so that only every other 5 are included in posterior, to reduce autocorrelation
# Set a seed for reproducability
ddm_fit = ddm_sm.sampling(data=data, init=init_f, iter=20000, warmup=10000, chains=4, thin=5, seed=101, refresh=1)

with open("simpleraceallddm.pkl", "wb") as f:
    pickle.dump({'model' : ddm_sm, 'fit' : ddm_fit}, f, protocol=-1)
