#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 23:46:37 2021

@author: tnye
"""

###############################################################################
# Script that plots an example synthetic displacement waveform for the NASA
# SET meeting poster. 
###############################################################################

# Imports
from obspy import read
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

obs_disp = read('/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/BSAT.LXE.mseed')[0]
# obs_acc = read('/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr/LWLI.HNE.corr.mseed')[0]
syn_disp = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/risetime/rt_2x/output/waveforms/mentawai.000000/BSAT.LYE.sac')[0]
# syn_acc = read('/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/mentawai.000000/LWLI.bb.HNE.sac')[0]

fig, ax = plt.subplots(1,1, figsize=(8,3.215))
ax.plot(syn_disp.times('matplotlib'), syn_disp.data, color='C1', label='Synthetic')
ax.plot(obs_disp.times('matplotlib'), obs_disp.data, color='black', label='Observed')
# ax.text(14907.61287037, -0.26, 'LYE')
ax.set_xlim(14907.61263888889,14907.615596064816)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax.set_xlabel('UTC Time(hr:min:sec)',fontsize=18)
ax.set_xlabel('Time',fontsize=18)
ax.set_ylabel('Amplitude (m)',fontsize=18)
plt.xticks(rotation=15)
ax.tick_params(width=2,length=5,right=False,top=False,bottom=False,labelleft=True,labelright=False,labelbottom=False,labeltop=False,labelsize=18)
plt.title('BSAT Displacement Waveforms', fontsize=18)
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
plt.legend(fontsize=16)
# plt.savefig('/Users/tnye/tsuquakes/Figures/png/disp_wf.png', dpi=300)
plt.savefig('/Users/tnye/conferences/NASA_SET/figures/disp_wf.png', dpi=300)


# fig, ax = plt.subplots(1,1, figsize=(6,4))
# ax.plot(syn_acc.times('matplotlib'), syn_acc.data, c='C1', label='Synthetic')
# ax.plot(obs_acc.times('matplotlib'), obs_acc.data, c='teal', label='Observed')
# # ax.text(14907.61287037, -0.26, 'LYE')
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax.set_xlabel('UTC Time(hr:min:sec)')
# ax.set_ylabel('Amplitude (m/s/s)')
# plt.title('LWLI HNE Acceleration Waveforms')
# plt.legend()
# plt.savefig('/Users/tnye/tsuquakes/Figures/png/acc_wf.png', dpi=300)