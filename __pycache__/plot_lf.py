#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:56:12 2021

@author: tnye
"""

# Imports
import numpy as np
from numpy import r_,diff
from glob import glob
from obspy import read
from mtspec import mtspec
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Read in a waveform 
project = 'standard'
run = 'mentawai.000000'
data_type = 'acc'
region = 'Mentawai'

lf_wfs_list = np.array(sorted(glob(f'/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/{run}/' + '*LYE.sac')))
lf_wf = read(lf_wfs_list[0])


############################### UnivariateSpline ##############################

# Double diff disp waveform to get it into acc using UnivaraiteSpline
disp_spl = UnivariateSpline(lf_wf[0].times(),lf_wf[0].data,s=0,k=4)
disp_spl_2d = disp_spl.derivative(n=2)
acc_data1 = disp_spl_2d(lf_wf[0].times())

# Get spectra using mtspec
amp_squared1, freq1 =  mtspec(acc_data1, delta=lf_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=7, nfft=lf_wf[0].stats.npts, quadratic=True)
amp1 = np.sqrt(amp_squared1)

# Plot low frequency spectra
plt.loglog(freq1,amp1,lw=.8,ls='-',label='low frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s)')
plt.title(f'{region} LF Spectra: UnivariateSpline')
plt.savefig(f'/Users/tnye/tsuquakes/plots/low_frequency/{run}_UniSpline.png', dpi=300)
plt.close()

# Plot low frequency waveform
plt.plot(lf_wf[0].times(),acc_data1)
plt.savefig(f'/Users/tnye/tsuquakes/plots/low_frequency/{run}_US_wf.png', dpi=300)
plt.close()


############################## Finite Differences #############################

# First derivate
dy=np.diff(lf_wf[0].data,1)
dx=np.diff(lf_wf[0].times(),1)
yfirst=dy/dx
xfirst=0.5*(lf_wf[0].data[:-1]+lf_wf[0].data[1:])

# Second derivative 
dyfirst=np.diff(yfirst,1)
dxfirst=np.diff(xfirst,1)
ysecond=dyfirst/dxfirst
acc_data2=0.5*(xfirst[:-1]+xfirst[1:])

# Get spectra using mtspec
amp_squared2, freq2 =  mtspec(acc_data2, delta=lf_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=7, nfft=lf_wf[0].stats.npts, quadratic=True)
amp2 = np.sqrt(amp_squared2)

# Plot low frequency spectra
plt.loglog(freq2,amp2,lw=.8,ls='-',label='low frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s)')
plt.title(f'{region} LF Spectra: Finite Differences')
plt.savefig(f'/Users/tnye/tsuquakes/plots/low_frequency/{run}_FiniteDiff.png', dpi=300)
plt.close()

# Plot low frequency waveform
plt.plot(lf_wf[0].times()[:1022],acc_data2)
plt.savefig(f'/Users/tnye/tsuquakes/plots/low_frequency/{run}_FD_wf.png', dpi=300)
plt.close()


#################################### np.Diff ##################################

# First derivate
dy=np.diff(lf_wf[0].data,1)
dx=np.diff(lf_wf[0].times(),1)
yfirst=dy/dx
xfirst=0.5*(lf_wf[0].data[:-1]+lf_wf[0].data[1:])

# Second derivative 
dyfirst=np.diff(yfirst,1)
dxfirst=np.diff(xfirst,1)
ysecond=dyfirst/dxfirst
acc_data2=0.5*(xfirst[:-1]+xfirst[1:])

# Get spectra using mtspec
amp_squared2, freq2 =  mtspec(acc_data2, delta=lf_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_wf[0].stats.npts, quadratic=True)
amp2 = np.sqrt(amp_squared2)

# Plot low frequency spectra
plt.loglog(freq2,amp2,lw=.8,ls='-',label='low frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s)')
plt.title(f'{region} LF Spectra: Finite Differences')
plt.savefig(f'/Users/tnye/tsuquakes/plots/low_frequency/{run}_FiniteDiff.png', dpi=300)
plt.close()

# Plot low frequency waveform
plt.plot(lf_wf[0].times()[:1022],acc_data2)
plt.savefig(f'/Users/tnye/tsuquakes/plots/low_frequency/{run}_FD_wf.png', dpi=300)
plt.close()


################################ FakeQuakes Diff ##############################

# Get dt
dt=lf_wf[0].stats.delta

# 1st derivative
lf = lf_wf.copy()
lf[0].data=r_[0,diff(lf[0].data)/dt]

# 2nd derivative 
lf[0].data=r_[0,diff(lf[0].data)/dt]
acc_data3 = lf[0].data

# Get spectra using mtspec
amp_squared3, freq3 =  mtspec(acc_data3, delta=lf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf[0].stats.npts, quadratic=True)
amp3 = np.sqrt(amp_squared3)

# Plot low frequency spectra
plt.loglog(freq3,amp3,lw=.8,ls='-',label='low frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s)')
plt.title(f'{region} LF Spectra: FakeQuakes')
plt.savefig(f'/Users/tnye/tsuquakes/plots/low_frequency/{run}_FQs.png', dpi=300)
plt.close()

# Plot low frequency waveform
plt.plot(lf_wf[0].times(),acc_data3)
plt.savefig(f'/Users/tnye/tsuquakes/plots/low_frequency/{run}_FQs_wf.png', dpi=300)
plt.close()