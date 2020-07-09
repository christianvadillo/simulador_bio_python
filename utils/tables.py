# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:58:55 2020

@author: 1052668570
"""
import pandas as pd
import numpy as np

# =============================================================================
# Generate graph with min, max and state (Table too)
# =============================================================================
# Tabla para maximos y minimos
MIN_MAX_AT_RUN = pd.DataFrame(
        {"variable": ['da_dil', 'da_agv_in', 'da_dqo_in',
                      'da_biomasa', 'da_dqo_out', 'da_agv_out',
                      'mec_dil', 'mec_agv_in', 'mec_eapp',
                      'mec_ace_out', 'mec_xa', 'mec_xm', 'mec_xh',
                      'mec_mox', 'mec_imec', 'mec_qh2'],
         "min": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         "max": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         })
MIN_MAX_AT_RUN.set_index('variable', inplace=True)

MIN_MAX_DEFINED = pd.DataFrame(
        {"variable": ['da_dil', 'da_agv_in', 'da_dqo_in',
                      'da_biomasa', 'da_dqo_out', 'da_agv_out', 
                      'mec_agv_in', 'mec_dil', 'mec_eapp',
                      'mec_ace_out', 'mec_xa', 'mec_xm', 'mec_xh',
                      'mec_mox', 'mec_imec', 'mec_qh2'],
         "min": [0.31, 50.0, 10.0,
                 4.1365E-131, 2.481, 3002.6,
                 3002.6, 1, 0.1, 
                 226.116877852408, 0, 0, 0, 
                 0, 0, 0],
         "max": [1.0, 150.0, 80.0,
                 415.1761518, 44.99695436, 26061.39625,
                 26061.39625, 3.0, 0.6,
                 23284.91281, 1396.259167, 0.000165502, 14.35906855,
                 0.00889497458104754, 0.088103711, 0.868508462],
        "ErrorMin": np.arange(1, 16*2, 2).tolist(),
        "ErrorMax": np.arange(2, 16*2+1, 2).tolist(),
         })
MIN_MAX_DEFINED.set_index('variable', inplace=True)
