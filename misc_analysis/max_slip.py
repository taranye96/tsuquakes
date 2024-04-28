#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:51:58 2021

@author: tnye
"""

###############################################################################
# Script that calcualtes maximum slip from a set of ruptures.
###############################################################################

# Imports
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

rupt_files = glob('/Users/tnye/FakeQuakes/simulations/final_runs/no_soft_layer/standard/output/ruptures/*.rupt')

max_slip = []

for file in rupt_files:
    
    # Read in rupt file
    rupt = np.genfromtxt(file)
    
    # Get total slip
    ss_slip = rupt[:,8]
    ds_slip = rupt[:,9]
    slip = np.sqrt(ss_slip**2 + ds_slip**2)
    
    max_slip.append(np.max(slip))
    
plt.hist(max_slip, bins=15)
plt.ylabel('Counts')
plt.xlabel('Max Slip')
plt.title('M7.84')
# plt.savefig('/Users/tnye/tsuquakes/plots/misc/rupt_max_slip_oldfault2.png', dpi=300)
# plt.savefig('/Users/tnye/FakeQuakes/parameters/standard/std_m7.84/plots/misc/max_slip.png', dpi=300)
# plt.close()

