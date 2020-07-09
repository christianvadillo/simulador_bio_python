# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 08:33:11 2020

@author: 1052668570
"""

class AnomalyDetector:
    def __init__(self, model, ontology):
        self.model = model
        self.ontology = ontology

    def predict(self, x):
        return self.model.predict(x)
    

