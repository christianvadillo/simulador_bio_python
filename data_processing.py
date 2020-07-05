# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:06:17 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from utils.tables import DA_MIN_MAX_DEFINED, MEC_MIN_MAX_DEFINED

da_df = pd.read_csv('OOP_TEST/test1/df_da_3600.csv', index_col=0)
mec_df = pd.read_csv('OOP_TEST/test1/df_mec_3600.csv', index_col=0)
# cmap = sns.cubehelix_palette(light=1)
# sns.pairplot(da_df, height=1, plot_kws={'s':3, 'alpha': 0.5})

def plot_hists(df, log=False, percentils=False):
    n_cols = df.shape[1] // 2
    plt.figure()
    if log:
        for i, var in enumerate(df.columns[:-1]):
            if percentils:
                ulimut = np.percentile(df[var], 99)
                dlimit = np.percentile(df[var], 1)
                data = df.loc[(df[var] > dlimit) & (df[var] < ulimut), var]
            else:
                data = df[var]
            plt.subplot(2,n_cols, i+1)
            sns.distplot(data, hist_kws={'log':True, 'alpha':1}, kde=False)
    else:
        for i, var in enumerate(df.columns[:-1]):
            if percentils:
                ulimut = np.percentile(df[var], 99)
                dlimit = np.percentile(df[var], 1)
                data = df.loc[(df[var] > dlimit) & 
                                       (df[var] < ulimut), var]
            else:
                data = df[var] 
            plt.subplot(2, n_cols, i+1)
            sns.distplot(data, kde=False, hist_kws={'alpha':1})


print(Counter(da_df['labels']))
print(Counter(mec_df['labels']))
# plot_hists(da_df, log=True, percentils=True)
# plot_hists(mec_df, log=True, percentils=False)

# sns.countplot(da_df['labels'])
# sns.countplot(mec_df['labels'])

# sns.pairplot(data=da_df, hue='labels', height=1.5, 
#              palette='dark', markers=['o','x'],
#              plot_kws={'s':20, 'alpha': 0.5})





