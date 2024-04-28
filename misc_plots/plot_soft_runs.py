#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:51:19 2023

@author: tnye
"""

# Imports
import numpy as np
from obspy import read
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

obs_E = read('/Users/tnye/tsuquakes/data/waveforms/individual/disp/BSAT.LXE.mseed')
obs_N = read('/Users/tnye/tsuquakes/data/waveforms/individual/disp/BSAT.LXN.mseed')
obs_Z = read('/Users/tnye/tsuquakes/data/waveforms/individual/disp/BSAT.LXZ.mseed')

soft1_E = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft1/output/waveforms/mentawai.000000/BSAT.LYE.sac')
soft1_N = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft1/output/waveforms/mentawai.000000/BSAT.LYN.sac')
soft1_Z = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft1/output/waveforms/mentawai.000000/BSAT.LYZ.sac')

soft2_E = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft2/output/waveforms/mentawai.000000/BSAT.LYE.sac')
soft2_N = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft2/output/waveforms/mentawai.000000/BSAT.LYN.sac')
soft2_Z = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft2/output/waveforms/mentawai.000000/BSAT.LYZ.sac')

soft3_E = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft3/output/waveforms/mentawai.000000/BSAT.LYE.sac')
soft3_N = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft3/output/waveforms/mentawai.000000/BSAT.LYN.sac')
soft3_Z = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft3/output/waveforms/mentawai.000000/BSAT.LYZ.sac')

soft4_E = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft4/output/waveforms/mentawai.000000/BSAT.LYE.sac')
soft4_N = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft4/output/waveforms/mentawai.000000/BSAT.LYN.sac')
soft4_Z = read('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft4/output/waveforms/mentawai.000000/BSAT.LYZ.sac')


#%% 
soft1_noise_E = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft1/processed_wfs/disp_noise/mentawai.000000/BSAT.LYE.sac')
soft1_noise_N = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft1/processed_wfs/disp_noise/mentawai.000000/BSAT.LYN.sac')
soft1_noise_Z = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft1/processed_wfs/disp_noise/mentawai.000000/BSAT.LYZ.sac')

soft2_noise_E = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft2/processed_wfs/disp_noise/mentawai.000000/BSAT.LYE.sac')
soft2_noise_N = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft2/processed_wfs/disp_noise/mentawai.000000/BSAT.LYN.sac')
soft2_noise_Z = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft2/processed_wfs/disp_noise/mentawai.000000/BSAT.LYZ.sac')

soft3_noise_E = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft3/processed_wfs/disp_noise/mentawai.000000/BSAT.LYE.sac')
soft3_noise_N = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft3/processed_wfs/disp_noise/mentawai.000000/BSAT.LYN.sac')
soft3_noise_Z = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft3/processed_wfs/disp_noise/mentawai.000000/BSAT.LYZ.sac')

soft5_noise_E = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft5/processed_wfs/disp_noise/mentawai.000000/BSAT.LYE.sac')
soft5_noise_N = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft5/processed_wfs/disp_noise/mentawai.000000/BSAT.LYN.sac')
soft5_noise_Z = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft5/processed_wfs/disp_noise/mentawai.000000/BSAT.LYZ.sac')

soft6_noise_E = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft6/processed_wfs/disp_noise/mentawai.000000/BSAT.LYE.sac')
soft6_noise_N = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft6/processed_wfs/disp_noise/mentawai.000000/BSAT.LYN.sac')
soft6_noise_Z = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft6/processed_wfs/disp_noise/mentawai.000000/BSAT.LYZ.sac')

soft7_noise_E = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft7/processed_wfs/disp_noise/mentawai.000000/BSAT.LYE.sac')
soft7_noise_N = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft7/processed_wfs/disp_noise/mentawai.000000/BSAT.LYN.sac')
soft7_noise_Z = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft7/processed_wfs/disp_noise/mentawai.000000/BSAT.LYZ.sac')

soft8_noise_E = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft8/processed_wfs/disp_noise/mentawai.000000/BSAT.LYE.sac')
soft8_noise_N = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft8/processed_wfs/disp_noise/mentawai.000000/BSAT.LYN.sac')
soft8_noise_Z = read('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/soft_layer_test/sd1.0_soft8/processed_wfs/disp_noise/mentawai.000000/BSAT.LYZ.sac')


#%%
fig, axs = plt.subplots(3,1)
axs[0].set_title('E-component')
axs[0].plot(soft1_E[0].times('matplotlib'),soft1_E[0].data,lw=1,label='soft1')
axs[0].plot(soft1_E[0].times('matplotlib'),soft2_E[0].data,lw=1,label='soft2')
axs[0].plot(soft1_E[0].times('matplotlib'),soft3_E[0].data,lw=1,label='soft3')
axs[0].plot(soft1_E[0].times('matplotlib'),soft4_E[0].data,lw=1,label='soft5')
axs[0].plot(obs_E[0].times('matplotlib'),obs_E[0].data,lw=1.5,c='k',label='observed')

axs[1].set_title('N-component')
axs[1].plot(soft1_N[0].times('matplotlib'),soft1_N[0].data,lw=1,label='soft1')
axs[1].plot(soft1_N[0].times('matplotlib'),soft2_N[0].data,lw=1,label='soft2')
axs[1].plot(soft1_N[0].times('matplotlib'),soft3_N[0].data,lw=1,label='soft3')
axs[1].plot(soft1_N[0].times('matplotlib'),soft4_N[0].data,lw=1,label='soft5')
axs[1].plot(obs_N[0].times('matplotlib'),obs_N[0].data,lw=1.5,c='k',label='observed')
# axs[2].set_title('N-component'axs[2].plot(soft1_Z[0].times('matplotlib'),soft1_Z[0].data,lw=1,label='soft1')
axs[2].plot(soft1_Z[0].times('matplotlib'),soft2_Z[0].data,lw=1,label='soft2')
axs[2].plot(soft1_Z[0].times('matplotlib'),soft3_Z[0].data,lw=1,label='soft3')
axs[2].plot(soft1_Z[0].times('matplotlib'),soft4_Z[0].data,lw=1,label='soft5')
axs[2].plot(obs_Z[0].times('matplotlib'),obs_Z[0].data,lw=1.5,c='k',label='observed')

axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.legend(loc='right')
plt.subplots_adjust(hspace=0.5,top=0.95,left=0.15,right=0.98)


#%%
##############################
fig, axs = plt.subplots(3,1)
axs[0].set_title('E-component')
axs[0].plot(soft1_noise_E[0].times('matplotlib'),soft1_noise_E[0].data,lw=1,label='soft1')
axs[0].plot(soft1_noise_E[0].times('matplotlib'),soft2_noise_E[0].data,lw=1,label='soft2')
axs[0].plot(soft1_noise_E[0].times('matplotlib'),soft3_noise_E[0].data,lw=1,label='soft3')
axs[0].plot(soft1_noise_E[0].times('matplotlib'),soft5_noise_E[0].data,lw=1,label='soft5')
axs[0].plot(soft1_noise_E[0].times('matplotlib'),soft6_noise_E[0].data,lw=1,label='soft6')
axs[0].plot(soft1_noise_E[0].times('matplotlib'),soft7_noise_E[0].data,lw=1,label='soft7')
axs[0].plot(soft1_noise_E[0].times('matplotlib'),soft8_noise_E[0].data,lw=1,label='soft8')
axs[0].plot(obs_E[0].times('matplotlib'),obs_E[0].data,lw=1.5,c='k',label='observed')

axs[1].set_title('N-component')
axs[1].plot(soft1_noise_N[0].times('matplotlib'),soft1_noise_N[0].data,lw=1,label='soft1')
axs[1].plot(soft1_noise_N[0].times('matplotlib'),soft2_noise_N[0].data,lw=1,label='soft2')
axs[1].plot(soft1_noise_N[0].times('matplotlib'),soft3_noise_N[0].data,lw=1,label='soft3')
axs[1].plot(soft1_noise_N[0].times('matplotlib'),soft5_noise_N[0].data,lw=1,label='soft5')
axs[1].plot(soft1_noise_N[0].times('matplotlib'),soft6_noise_N[0].data,lw=1,label='soft6')
axs[1].plot(soft1_noise_N[0].times('matplotlib'),soft7_noise_N[0].data,lw=1,label='soft7')
axs[1].plot(soft1_noise_N[0].times('matplotlib'),soft8_noise_N[0].data,lw=1,label='soft8')
axs[1].plot(obs_N[0].times('matplotlib'),obs_N[0].data,lw=1.5,c='k',label='observed')

axs[2].set_title('Z-component')
axs[2].plot(soft1_noise_Z[0].times('matplotlib'),soft1_noise_Z[0].data,lw=1,label='soft1')
axs[2].plot(soft1_noise_Z[0].times('matplotlib'),soft2_noise_Z[0].data,lw=1,label='soft2')
axs[2].plot(soft1_noise_Z[0].times('matplotlib'),soft3_noise_Z[0].data,lw=1,label='soft3')
axs[2].plot(soft1_noise_Z[0].times('matplotlib'),soft5_noise_Z[0].data,lw=1,label='soft5')
axs[2].plot(soft1_noise_Z[0].times('matplotlib'),soft6_noise_Z[0].data,lw=1,label='soft6')
axs[2].plot(soft1_noise_Z[0].times('matplotlib'),soft7_noise_Z[0].data,lw=1,label='soft7')
axs[2].plot(soft1_noise_Z[0].times('matplotlib'),soft8_noise_Z[0].data,lw=1,label='soft8')
axs[2].plot(obs_Z[0].times('matplotlib'),obs_Z[0].data,lw=1.5,c='k',label='observed')

axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.legend(loc='right')
plt.subplots_adjust(hspace=0.5,top=0.95,left=0.15,right=0.98)



















