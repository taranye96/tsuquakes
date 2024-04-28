#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:58:12 2023

@author: tnye
"""

# Imports
import numpy as np
from mudpy import viewFQ
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import simps

# rupt = '/Users/tnye/FakeQuakes/simulations/ideal_runs/standard/output/ruptures/mentawai.000000.rupt'
# epicenter = [100.14,-3.49]
# t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=0.001,t_total=100,stf_type='dreger',plot=True)

# rupt = '/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/mentawai_fine2.rupt'
# epicenter = [100.14,-3.49]
# t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=0.001,t_total=100,stf_type='dreger',plot=False)

# rupt = '/Users/tnye/FakeQuakes/simulations/test_runs_m7.8/standard/output/ruptures/mentawai.000000.rupt'
# epicenter = [100.14,-3.49]
# t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=0.001,t_total=100,stf_type='dreger',plot=False)

# rupt = '/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/han_yue_depth_adjusted.rupt'
# epicenter = [100.14,-3.49]
# t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=0.001,t_total=100,stf_type='dreger',plot=True)

# rupt = '/Users/tnye/ONC/simulations/cascadia_longer_wfs/output/ruptures/cascadia.000048.rupt'
# epicenter = [236.122328,42.394044]
# t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=0.001,t_total=100,stf_type='dreger',plot=False)

rupt = '/Users/tnye/FakeQuakes/files/vrupt_ruptures/test_runs_m7.8/sf0.4/mentawai.000000.rupt'
epicenter = [100.14,-3.49]
t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=0.001,t_total=500,stf_type='dreger',plot=False)

# exp=np.floor(np.log10(Mr.max()))
exp=18
M1=Mr/(10**exp)

# M0 = np.sum(Mr)
# Mw = (2/3) * (np.log10(M0) - 9.1)
# print(Mw)

M0 = simps(Mr, t)
Mw = (2/3) * (np.log10(M0) - 9.1)
print(Mw)

#%%
fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.fill(t,M1,'b',alpha=0.5)
ax.plot(t,M1,color='k',lw=0.1)
ax.grid()
ax.set_xlabel('Time(s)', fontsize=12)
ax.set_ylabel('Moment Rate ('+r'$\times 10^{'+str(int(exp))+r'}$Nm/s)',fontsize=12)
ax.tick_params(labelsize=12)
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.set_xlim(0,200)
ax.set_ylim(ymin=0)
plt.subplots_adjust(left=0.125, bottom=0.15, right=0.95, top=0.95)

