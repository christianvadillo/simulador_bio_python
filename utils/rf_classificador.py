# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:06:24 2020

@author: 1052668570
"""

import joblib
import numpy as np
import warnings
# from numba import njit
warnings.filterwarnings("ignore", category=RuntimeWarning)

# da_cls = joblib.load("models/RandomForest_da.sav")
da_cls = joblib.load("models/da_rf_model_7_jul.joblib")
# mec_cls = joblib.load("models/RandomForest_mec.sav")
mec_cls = joblib.load("models/mec_rf_model_7_jul.joblib")
# print(f"classifier loaded")
# da_sc = joblib.load("models/sc_da.sav")
# mec_sc = joblib.load("models/sc_mec.sav")

# clasificar_estado(medicion[da_vars], medicion[mec_vars])

# @njit
def clasificar_estado(*args):
    estados = []

    # for each process measures
    for arg in args:
        x = arg
        length = x.shape[1]
        # print(length)

        # Transform data
        # x = x.reshape(-1, length)
        # print(x.shape)

        if length == 6:
            # print('-'*10)
            # print('da shape', x.shape)
            # print(arg)
            # print('-'*10)
            # print('da cls', length)
            # x = da_sc.transform(x)
            # x = np.nan_to_num(x)
            # print(f"x after sc: {x}")
            estados.append(da_cls.predict(x))
        else:
            # print('-'*10)
            # print('mec shape', x.shape)
            # print(arg)
            # print('-'*10)
            # print('mec cls', length)
            # x = mec_sc.transform(x)
            # x = np.nan_to_num(x)
            # print(f"x after sc: {x}")
            estados.append(mec_cls.predict(x))

    return estados

