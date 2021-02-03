#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 09:55:34 2021

@author: tnye
"""

###############################################################################
# Script that makes a plot of one of the Cascadia synthetic displacement
# waveforms (but differentiated to acceleration) as a comparison to Mentawai. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
from mtspec import mtspec
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# synthetic run
run = 'cascadia.000000'
station = 'albh'

# Get waveform
lf_wf = read(f'/Users/tnye/Classes/Geol610/project/Cascadia/data/{run}/{station}.LYE.sac')
    
# Low frequency synthetic spectra
    # First, double diff disp waveform to get it into acc using UnivaraiteSpline
disp_spl = UnivariateSpline(lf_wf[0].times(),lf_wf[0].data,s=0,k=4)
disp_spl_2d = disp_spl.derivative(n=2)
acc_data = disp_spl_2d(lf_wf[0].times())

amp_squared, freq =  mtspec(acc_data, delta=lf_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_wf[0].stats.npts, quadratic=True)
amp = np.sqrt(amp_squared)


################################ Make figure ##################################

plt.loglog(freq, amp, lw=0.8, ls='-', label=f'{station}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s)')
plt.title('GNSS Differentiated Acceleration Waveform')
plt.legend()

# Save figure
plt.savefig('/Users/tnye/tsuquakes/plots/misc/lf_Cascadia.png', dpi=300)


