#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 01:03:14 2020

@author: tnye
"""

###############################################################################
# Script used to to help determine why the velocity spectra were taking so long
# to compute.  This is my code that I compared to Valerie's in test_velocity_vjs. 
###############################################################################


# Imports
import numpy as np
from mtspec import mtspec
import pandas as pd
import obspy as obs
import matplotlib.pyplot as plt
import time
from scipy.signal import butter,filtfilt,lfilter
from scipy.integrate import cumtrapz
import tsueqs_main_fns as tmf

# datadirectory = '/Users/vjs/tara/velocitytest/'
datadirectory = '/Users/tnye/vel_test/'


station = 'CGJI'
## either: 'CGJI', 'CNJI', or 'KASI'

num_poles = 2
fcorner = 1/15

obs_gain = 100 ## counts / cm/s2

# %%
## REad in files into stream:
obs_file = datadirectory + station +  '.HNE.mseed'
syn_file = datadirectory + station +  '.bb.HNE.sac'

acc_obs_stream = obs.read(obs_file)
acc_syn_stream = obs.read(syn_file)

## gain the observed data, then convert it from cm to m:
acc_obs_stream = tmf.correct_for_gain(acc_obs_stream,obs_gain)

# ## correct for baseline
# nsamples=100
# baseline = tmf.compute_baseline(acc_obs_stream,nsamples)
# acc_obs_stream = tmf.correct_for_baseline(acc_obs_stream,baseline)

## also start a figure:
wf_fig,wf_axes = plt.subplots(nrows=4,ncols=1)
spec_fig,spec_axes = wf_fig = plt.subplots(nrows=4,ncols=1)

# In first panel, plot waveforms:
wf_axes[0].plot(acc_obs_stream[0].times(),acc_obs_stream[0].data,color='blue')
wf_axes[0].plot(acc_syn_stream[0].times(),acc_syn_stream[0].data,color='orange')



## Get spectra...
obs_acc_start = time.time()
[obs_acc_spec,obs_acc_freq] = mtspec(acc_obs_stream[0].data, delta=acc_obs_stream[0].stats.delta,
                                     time_bandwidth=4, number_of_tapers=5,
                                     nfft=acc_obs_stream[0].stats.npts, quadratic=True)
obs_acc_end = time.time()
print('obs acc time %f' % (obs_acc_end - obs_acc_start))

syn_acc_start = time.time() 
[syn_acc_spec,syn_acc_freq] = mtspec(acc_syn_stream[0].data, delta=acc_syn_stream[0].stats.delta,
                                     time_bandwidth=4, number_of_tapers=5,
                                     nfft=acc_syn_stream[0].stats.npts, quadratic=True)
syn_acc_end = time.time()
print('syn acc time %f' % (syn_acc_end - syn_acc_start))

## plot spectra
spec_axes[0].loglog(obs_acc_freq,obs_acc_spec,color='blue')
spec_axes[0].loglog(syn_acc_freq,syn_acc_spec,color='orange')


# %%
## Highpass Filter waveforms. give it the corner frequency divided by nyquist,
##   since the filter goes from 0 to 1 where the nyquist is 1
acc_obs_filt = tmf.highpass(acc_obs_stream,fcorner,acc_obs_stream[0].stats.sampling_rate,num_poles,zerophase=True)

acc_syn_filt = tmf.highpass(acc_syn_stream,fcorner,acc_syn_stream[0].stats.sampling_rate,num_poles,zerophase=True)

## plot the filtered waveforms:
wf_axes[1].plot(acc_obs_stream[0].times(),acc_obs_filt[0].data,color='blue')
wf_axes[1].plot(acc_syn_stream[0].times(),acc_syn_filt[0].data,color='orange')

## Get spectra...
obs_acc_start = time.time()
[obs_acc_filt_spec,obs_acc_filt_freq] = mtspec(acc_obs_filt[0].data, delta=acc_obs_stream[0].stats.delta,
                                               time_bandwidth=4, number_of_tapers=5,
                                               nfft=acc_obs_filt[0].stats.npts, quadratic=True)
obs_acc_end = time.time()
print('obs acc filtered time %f' % (obs_acc_end - obs_acc_start))

syn_acc_start = time.time() 
[syn_acc_filt_spec,syn_acc_filt_freq] = mtspec(acc_syn_filt[0].data, delta=acc_syn_stream[0].stats.delta,
                                               time_bandwidth=4, number_of_tapers=5,
                                               nfft=acc_syn_filt[0].stats.npts, quadratic=True)
syn_acc_end = time.time()
print('syn acc filtered time %f' % (syn_acc_end - syn_acc_start))

## plot spectra
spec_axes[1].loglog(obs_acc_filt_freq,obs_acc_filt_spec,color='blue')
spec_axes[1].loglog(syn_acc_filt_freq,syn_acc_filt_spec,color='orange')

# %%
## INtegrate waveforms
vel_obs = tmf.accel_to_veloc(acc_obs_filt)
vel_obs_amplitude = vel_obs[0].data

vel_syn = tmf.accel_to_veloc(acc_syn_filt)
vel_syn_amplitude = vel_syn[0].data

#plot the velocity waveforms:
wf_axes[2].plot(vel_obs[0].times(),vel_obs_amplitude,color='blue')
wf_axes[2].plot(vel_syn[0].times(),vel_syn_amplitude,color='orange')

## Get spectra...
obs_vel_start = time.time()
[obs_vel_spec,obs_vel_freq] = mtspec(vel_obs_amplitude, delta=acc_obs_stream[0].stats.delta,
                                     time_bandwidth=4, number_of_tapers=5,
                                     nfft=len(vel_obs_amplitude), quadratic=True)
obs_vel_end = time.time()
print('obs vel time %f' % (obs_vel_end - obs_vel_start))

syn_vel_start = time.time() 
[syn_vel_spec,syn_vel_freq] = mtspec(vel_syn_amplitude, delta=acc_syn_stream[0].stats.delta,
                                     time_bandwidth=4, number_of_tapers=5,
                                     nfft=len(vel_syn_amplitude), quadratic=True)
syn_vel_end = time.time()
print('synvel time %f' % (syn_vel_end - syn_vel_start))

## plot spectra
spec_axes[2].loglog(obs_vel_freq,obs_vel_spec,color='blue')
spec_axes[2].loglog(syn_vel_freq,syn_vel_spec,color='orange')


# %%
## Filter again...
## Highpass Filter waveforms. give it the corner frequency divided by nyquist,
##   since the filter goes from 0 to 1 where the nyquist is 1
vel_obs_filt = tmf.highpass(vel_obs_amplitude,fcorner,acc_obs_stream[0].stats.delta,num_poles,zerophase=True)

vel_syn_filt = tmf.highpass(vel_syn_amplitude,fcorner,acc_syn_stream[0].stats.delta,num_poles,zerophase=True)

## plot the filtered waveforms:
wf_axes[3].plot(acc_obs_stream[0].times(),vel_obs_filt,color='blue')
wf_axes[3].plot(acc_syn_stream[0].times(),vel_syn_filt,color='orange')

## Get spectra...
obs_vel__start = time.time()
[obs_vel_filt_spec,obs_vel_filt_freq] = mtspec(vel_obs_filt, delta=acc_obs_stream[0].stats.delta,
                                               time_bandwidth=4, number_of_tapers=5,
                                               nfft=len(vel_obs_filt), quadratic=True)
obs_vel_end = time.time()
print('obs vel filtered time %f' % (obs_acc_end - obs_acc_start))

syn_vel_start = time.time() 
[syn_vel_filt_spec,syn_vel_filt_freq] = mtspec(vel_syn_filt, delta=acc_syn_stream[0].stats.delta,
                                               time_bandwidth=4, number_of_tapers=5,
                                               nfft=len(vel_syn_filt), quadratic=True)
syn_vel_end = time.time()
print('syn vel filtered time %f' % (syn_vel_end - syn_vel_start))

## plot spectra
spec_axes[3].loglog(obs_vel_filt_freq,obs_vel_filt_spec,color='blue')
spec_axes[3].loglog(syn_vel_filt_freq,syn_vel_filt_spec,color='orange')



