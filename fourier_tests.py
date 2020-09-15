#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:26:02 2019

@author: tnye
"""

###############################################################################
# Script used to make plots of the binned Fourier spectra to determine the best
# number of bins. 
###############################################################################


# Standard Library Imports 
import numpy as np
import pandas as pd
from obspy import read
from glob import glob
from mtspec import mtspec
from scipy.stats import binned_statistic 
import matplotlib.pyplot as plt

# Read in dataframes
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')

# Corrected Miniseed files directories
acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr'
vel_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr'

# GPS files directory 
gps_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'

# Get corrected mseed files
acc_files = np.array(sorted((glob(acc_dir + '/*'))))
filt_vel = np.array(sorted((glob(vel_dir + '/*'))))

# Get GPS data
gps_files = np.array(sorted((glob(gps_dir + '/*'))))


# Using station LASI.HNE for testing


# i = 0
file = acc_files[0]
st = read(file)
tr = st[0]
data = tr.data
delta = tr.stats.delta
station = tr.stats.station
channel = tr.stats.channel
    
spec_amp, freq =  mtspec(data, delta=delta, time_bandwidth = 4, 
                         number_of_tapers=7, quadratic = True)
amp = np.sqrt(spec_amp)

# Remove zero frequencies so that I can take ln
for i, val in enumerate(freq):
    if val == 0:
        freq = np.delete(freq,i)
        amp = np.delete(amp,i)


# 25 bins
bin_sides = [-2.68124124, -2.50603279, -2.33082434, -2.15561589,
                     -1.98040744, -1.80519899, -1.62999054, -1.45478209,
                     -1.27957364, -1.10436519, -0.92915674, -0.75394829,
                     -0.57873984, -0.40353139, -0.22832294, -0.05311449, 
                     0.12209396, 0.29730241, 0.47251086, 0.64771931, 0.82292776, 
                     0.99813621, 1.17334466, 1.3485531 , 1.52376155, 1.69897]

bin_means, bin_edges, binnumber = binned_statistic(np.log10(freq), np.log10(amp), statistic='mean', bins=25)

bins = []
for i in range(len(bin_edges)):
    if bin_edges[i] != bin_edges[-1]:
        bin_pt = np.mean((bin_edges[i], bin_edges[i+1]))
        bins.append(bin_pt)

for i in range(len(bin_edges)):
    bin_edges[i] = 10**bin_edges[i]


plt.loglog(freq,amp)
plt.grid(linestyle='--')
# plt.ylim(ylim)
# plt.xlim(xlim)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s/s)')
plt.title('CGJI Acc Spectra ')

# plt.scatter(bins, bin_means, s=10, color='green')
# # plt.yscale('log')
# # plt.ylim(10**-14, 10**-2)
# # plt.xscale('log')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude (m/s)')
# plt.title('Log Acc--25 bins')

# fig, ax = plt.subplots()
# axis = plt.axes(xscale='log', yscale='log')
# axis.grid()
# ax.axis([10**-3, 20, 10**-12, 10**-2])
# ax.loglog()
# ax.scatter(bins, bin_means, s=10, color='green')

# plt.show()

# figpath = '/Users/tnye/tsuquakes/plots/fourier_spec/bin_tests/logacc_25bins'
# plt.savefig(figpath, dpi=300)
# plt.close()


# # 40 bins
# bin_means, bin_edges, binnumber = binned_statistic(np.log(freq), np.log(amp), statistic='mean', bins=40)

# bins = []
# for i in range(len(bin_edges)):
#     if bin_edges[i] != bin_edges[-1]:
#         bin_pt = np.mean((bin_edges[i], bin_edges[i+1]))
#         bins.append(bin_pt)
        
# plt.scatter(bins, bin_means, s=10, color='green')
# plt.xlabel('ln Frequency (Hz)')
# plt.ylabel('ln Amplitude (m/s)')
# plt.title('Log Acc--40 bins')
# figpath = '/Users/tnye/tsuquakes/plots/fourier_spec/bin_tests/logacc_40bins'
# plt.savefig(figpath, dpi=300)
# plt.close()


# # 10 bins
# bin_means, bin_edges, binnumber = binned_statistic(np.log(freq), np.log(amp), statistic='mean', bins=10)

# bins = []
# for i in range(len(bin_edges)):
#     if bin_edges[i] != bin_edges[-1]:
#         bin_pt = np.mean((bin_edges[i], bin_edges[i+1]))
#         bins.append(bin_pt)
        
# plt.scatter(bins, bin_means, s=10, color='green')
# plt.xlabel('ln Frequency (Hz)')
# plt.ylabel('ln Amplitude (m/s)')
# plt.title('Log Acc--10 bins')
# figpath = '/Users/tnye/tsuquakes/plots/fourier_spec/bin_tests/logacc_10bins'
# plt.savefig(figpath, dpi=300)
# plt.close()

# # 50 bins
# bin_means, bin_edges, binnumber = binned_statistic(np.log(freq), np.log(amp), statistic='mean', bins=50)

# bins = []
# for i in range(len(bin_edges)):
#     if bin_edges[i] != bin_edges[-1]:
#         bin_pt = np.mean((bin_edges[i], bin_edges[i+1]))
#         bins.append(bin_pt)
        
# plt.scatter(bins, bin_means, s=10, color='green')
# plt.xlabel('ln Frequency (Hz)')
# plt.ylabel('ln Amplitude (m/s)')
# plt.title('Log Acc--50 bins')
# figpath = '/Users/tnye/tsuquakes/plots/fourier_spec/bin_tests/logacc_50bins'
# plt.savefig(figpath, dpi=300)
# plt.close()

# # 75 bins
# bin_means, bin_edges, binnumber = binned_statistic(np.log(freq), np.log(amp), statistic='mean', bins=75)

# bins = []
# for i in range(len(bin_edges)):
#     if bin_edges[i] != bin_edges[-1]:
#         bin_pt = np.mean((bin_edges[i], bin_edges[i+1]))
#         bins.append(bin_pt)
        
# plt.scatter(bins, bin_means, s=5, color='green')
# plt.xlabel('ln Frequency (Hz)')
# plt.ylabel('ln Amplitude (m/s)')
# plt.title('Log Acc--75 bins')
# figpath = '/Users/tnye/tsuquakes/plots/fourier_spec/bin_tests/logacc_75bins'
# plt.savefig(figpath, dpi=300)
# plt.close()