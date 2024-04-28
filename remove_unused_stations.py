#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:03:52 2022

@author: tnye
"""

###############################################################################
# Script used to remove unused GNSS and strong motion stations from dataset. 
# New runs do not generate data for these stations, but older runs did. 
###############################################################################

# Imports
import numpy as np
from glob import glob
import os

far_sm = ['PPBI','PSI','CGJI','TSI','CNJI','LASI','MLSI']
noisy_gnss = ['MKMK','LNNG','LAIS','TRTK','MNNA','BTHL']

for stn in far_sm:
    files = glob(f'/Users/tnye/FakeQuakes/FQ_status/new_q_model/new_fault_model/m7.85/new_q_params/**/{stn}*',recursive=True)
    
    for file in files:
        os.remove(file)

for stn in noisy_gnss:
    files = glob(f'/Users/tnye/FakeQuakes/FQ_status/new_q_model/new_fault_model/m7.85/new_q_params/**/{stn}*',recursive=True)
    
    for file in files:
        os.remove(file)