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
# Tabla para máximos y mínimos

MIN_MAX_TABLE = pd.DataFrame(
        {"variable": ['Biomasa(x)', 'DQO_out', 'AGV_out_DA', 'ace',
                      'xa', 'xm', 'xh', 'mox', 'imec', 'QH2'],
         "min": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         "max": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         })
MIN_MAX_TABLE.set_index('variable', inplace=True)
# final_table = MIN_MAX_TABLE.copy()




DA_MIN_MAX_DEFINED = pd.DataFrame(
        {"variable": ['dil', 'agv_in', 'dqo_in',
                      'biomasa', 'dqo_out', 'agv_out'],
         "min": [0.31, 50.0, 10.0,
                 4.1365E-131, 2.481, 3002.6],
         "max": [1.0, 150.0, 80.0,
                 415.1761518, 44.99695436, 26061.39625],
        "ErrorMin": np.arange(1, 6*2, 2).tolist(),
        "ErrorMax": np.arange(2, 6*2+1, 2).tolist(),
         })
DA_MIN_MAX_DEFINED.set_index('variable', inplace=True)


MEC_MIN_MAX_DEFINED = pd.DataFrame(
        {"variable": ['dil', 'agv_in', 'eapp',
                      'ace_out', 'xa', 'xm', 'xh',
                      'mox', 'imec', 'qh2'],
         "min": [1.0, 3002.6, 0.1,
                 226.116877852408, 0.0, 0.0, 0.0, 
                 0.0, 0.0, 0.0],
         "max": [3.0, 26061.39625, 0.6,
                 23284.91281, 1396.259167, 0.000165502, 14.35906855,
                 0.00889497458104754, 0.088103711, 0.868508462],
        "ErrorMin": np.arange(13, 16*2, 2).tolist(),
        "ErrorMax": np.arange(14, 16*2+1, 2).tolist(),
         })
MEC_MIN_MAX_DEFINED.set_index('variable', inplace=True)

# min and max restults from all simulation and filled in "diccionaro_datos.xsls
MIN_MAX_DEFINED = pd.DataFrame(
        {"variable": ['dil', 'agv_in', 'dqo_in',
                      'biomasa', 'dqo_out', 'agv_out', 'agv_in_mec',
                      'Dil2', 'Eapp',
                      'ace_out', 'xa', 'xm', 'xh',
                      'mox', 'imec', 'qh2'],
         "min": [0.31, 50.0, 10.0,
                 4.1365E-131, 2.481, 3002.6, 3002.6,
                 1, 0.1, 
                 226.116877852408, 0, 0,0, 
                 0, 0, 0],
         "max": [1.0, 150.0, 80.0,
                 415.1761518, 44.99695436, 26061.39625,26061.39625,
                 3.0, 0.6,
                 23284.91281, 1396.259167, 0.000165502, 14.35906855,
                 0.00889497458104754, 0.088103711, 0.868508462],
        "ErrorMin": np.arange(1, 16*2, 2).tolist(),
        "ErrorMax": np.arange(2, 16*2+1, 2).tolist(),
         })

MIN_MAX_DEFINED.set_index('variable', inplace=True)


def table_names(number):
    return {
        3: 'dqo_in',
        4: 'biomasa',
        5: 'dqo_out',
        6: 'agv_out',
        7: 'ace_out',
        8:  'xa',
        10: 'xm',
        11: 'xh',
        12: 'mox',
        13: 'imec',
        14: 'qh2',
        }.get(number, 'none')

def plot_labels(number):
    return {
        3: ['DQO, digestor anaerobio: Entrada', '$DQO (gL^{-1})$'],
        4: ['Biomasa, digestor anaerobio: Salida', '$X_{1}(gL^{-1})$'],
        5: ['DQO, digestor anaerobio: Salida', '$DQO (gL^{-1})$'],
        6: ['Acetato, digestor anaerobio: Salida, (entrada MEC)', '$Ace (gL^{-1})$'],
        7: 'ace',
        8:  'xa',
        10: 'xm',
        11: 'xh',
        12: 'mox',
        13: 'imec',
        14: 'QH2',
        }.get(number, 'none')
    
