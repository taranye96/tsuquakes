#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:52:12 2019

@author: tnye
"""

# Standard Library Imports 
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
from mtspec import mtspec
from scipy.stats import binned_statistic 


# Read in dataframes
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')

# Corrected Miniseed file directories
disp_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'
acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr'
vel_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr'

# Get corrected mseed 
disp_files = np.array(sorted((glob(disp_dir + '/*'))))
acc_files = np.array(sorted((glob(acc_dir + '/*'))))
vel_files = np.array(sorted((glob(vel_dir + '/*'))))

# overall stats
hyplon = np.array(df['hyplon'])
hyplat = np.array(df['hyplat'])
stations = np.array(df['station'])
channel_list = np.array([])
acc_data = np.array([])
vel_data = np.array([])
disp_data = np.array([])
E_data = np.array([])
N_data = np.array([])
Z_data = np.array([])

################################ Acceleration #################################
station_counter = np.array([])
for i, file in enumerate(acc_files):
    st = read(file)
    tr = st[0]
    data = tr.data
    delta = tr.stats.delta
    samprate = tr.stats.sampling_rate
    station = tr.stats.station
    channel = tr.stats.channel 
    
    channel_list = np.append(channel, channel)

    nyquist = 0.5 * samprate

    # Calc spec amplitudes and frequencies
    spec_amp, freq =  mtspec(data, delta=delta, time_bandwidth = 4, 
                             number_of_tapers=7, quadratic = True)
    amp = np.sqrt(spec_amp)

    # Remove zero frequencies so that I can take log and remove data after fc
    indexes = []

    for j, val in enumerate(freq):
        if val == 0:
            indexes.append(j)

        elif val > nyquist: 
            indexes.append(j)

    freq = np.delete(freq,indexes)
    amp = np.delete(amp,indexes) 

    # Create bins
    bin_sides = [-2.68124124, -2.50603279, -2.33082434, -2.15561589,
                 -1.98040744, -1.80519899, -1.62999054, -1.45478209,
                 -1.27957364, -1.10436519, -0.92915674, -0.75394829,
                 -0.57873984, -0.40353139, -0.22832294, -0.05311449, 
                 0.12209396, 0.29730241, 0.47251086, 0.64771931, 0.82292776, 
                 0.99813621, 1.17334466, 1.3485531 , 1.52376155, 1.69897]
    bin_means, bin_edges, binnumber = binned_statistic(np.log10(freq),
                                                       np.log10(amp),
                                                       statistic='mean',
                                                       bins=bin_sides)
    
    # if i == 0:
    #     station_counter = np.append(station_counter, bin_means)
    
    # elif station == read(acc_files[i-1])[0].stats.station:
    #     station_counter = np.append(station_counter, bin_means)

    # else:
    #     if acc_data.size == 0:
    #         acc_data = np.hstack((acc_data, station_counter))
    #     else:
    #         acc_data = np.vstack((acc_data, station_counter))
    #     station_counter = np.array([])
    #     station_counter = np.append(station_counter, bin_means)
    
    # if i == len(acc_files) - 1:
    #     acc_data = np.vstack((acc_data, station_counter))
    
    if 'E' in channel:
        E_data = np.append(E_data, bin_means)
    elif 'N' in channel:
        N_data = np.append(N_data, bin_means)
    elif 'Z' in channel:
        Z_data = np.append(Z_data, bin_means)
        

################################### Velocity ##################################
station_counter = np.array([])
for i, file in enumerate(vel_files):
    st = read(file)
    tr = st[0]
    data = tr.data
    samprate = tr.stats.sampling_rate
    delta = tr.stats.delta
    station = tr.stats.station
    channel = tr.stats.channel
    
    channel_list = np.append(channel, channel)
    
    nyquist = 0.5 * samprate

     # Calc spec amplitudes and frequencies
    spec_amp, freq =  mtspec(data, delta=delta, time_bandwidth = 4, 
                             number_of_tapers=7, quadratic = True)
    amp = np.sqrt(spec_amp)

    # Remove zero frequencies so that I can take log and remove data after fc
    indexes = []

    for j, val in enumerate(freq):
        if val == 0:
            indexes.append(j)

        elif val > nyquist: 
            indexes.append(j)

    freq = np.delete(freq,indexes)
    amp = np.delete(amp,indexes)        

    # Create bins
    bin_sides = [-2.68124124, -2.50603279, -2.33082434, -2.15561589,
                 -1.98040744, -1.80519899, -1.62999054, -1.45478209,
                 -1.27957364, -1.10436519, -0.92915674, -0.75394829,
                 -0.57873984, -0.40353139, -0.22832294, -0.05311449, 
                 0.12209396, 0.29730241, 0.47251086, 0.64771931, 0.82292776, 
                 0.99813621, 1.17334466, 1.3485531 , 1.52376155, 1.69897]
    bin_means, bin_edges, binnumber = binned_statistic(np.log10(freq),
                                                       np.log10(amp),
                                                       statistic='mean',
                                                       bins=bin_sides)

    # if i == 0:
    #     station_counter = np.append(station_counter, bin_means)
    
    # elif station == read(vel_files[i-1])[0].stats.station:
    #     station_counter = np.append(station_counter, bin_means)

    # else:
    #     if vel_data.size == 0:
    #         vel_data = np.hstack((vel_data, station_counter))
    #     else:
    #         vel_data = np.vstack((vel_data, station_counter))
    #     station_counter = np.array([])
    #     station_counter = np.append(station_counter, bin_means)
    
    if i == len(vel_files) - 1:
        vel_data = np.vstack((vel_data, station_counter))
        
    if 'E' in channel:
        E_data = np.append(E_data, bin_means)
    elif 'N' in channel:
        N_data = np.append(N_data, bin_means)
    elif 'Z' in channel:
        Z_data = np.append(Z_data, bin_means)
        
################################# Displacement ################################
disp_stations = []
station_counter = np.array([])
for i, file in enumerate(disp_files):
    st = read(file)
    tr = st[0]
    data = tr.data
    samprate = tr.stats.sampling_rate
    delta = tr.stats.delta
    station = tr.stats.station
    channel = tr.stats.channel
    
    channel_list = np.append(channel, channel)
    
    nyquist = 0.5* samprate

    # Calc spec amplitudes and frequencies
    spec_amp, freq =  mtspec(data, delta=delta, time_bandwidth = 4, 
                             number_of_tapers=7, quadratic = True)
    amp = np.sqrt(spec_amp)

    # Remove zero frequencies so that I can take log
    indexes = []

    for j, val in enumerate(freq):
        if val == 0:
            indexes.append(j)

        elif val > nyquist: 
            indexes.append(j)

    freq = np.delete(freq,indexes)
    amp = np.delete(amp,indexes) 

    # Create bins
    bin_means, bin_edges, binnumber = binned_statistic(np.log10(freq),
                                                       np.log10(amp),
                                                       statistic='mean',
                                                       bins=25)

    
    # if i == 0:
    #     station_counter = np.append(station_counter, bin_means)
    
    # elif station == read(disp_files[i-1])[0].stats.station:
    #     station_counter = np.append(station_counter, bin_means)

    # else:
    #     if disp_data.size == 0:
    #         disp_data = np.hstack((disp_data, station_counter))
    #     else:
    #         disp_data = np.vstack((disp_data, station_counter))
    #     station_counter = np.array([])
    #     station_counter = np.append(station_counter, bin_means)
    
    # if i == len(disp_files) - 1:
    #     disp_data = np.vstack((disp_data, station_counter))
        
    if 'E' in channel:
        E_data = np.append(E_data, bin_means)
    elif 'N' in channel:
        N_data = np.append(N_data, bin_means)
    elif 'Z' in channel:
        Z_data = np.append(Z_data, bin_means)

####################### data frame ############################
# stations
stations = pd.DataFrame(stations, columns=['station'])

# run#
run = np.full_like(stations, 'obs')
run = pd.DataFrame(run, columns=['run'])

# event coordinates
hyplon = pd.DataFrame(hyplon, columns=['hyplon'])
hyplat = pd.DataFrame(hyplat, columns=['hyplat'])


# acc_cols = ['E_acc_bin1', 'E_acc_bin2', 'E_acc_bin3', 'E_acc_bin4',
#             'E_acc_bin5', 'E_acc_bin6', 'E_acc_bin7', 'E_acc_bin8',
#             'E_acc_bin9', 'E_acc_bin10', 'E_acc_bin11', 'E_acc_bin12',
#             'E_acc_bin13', 'E_acc_bin14', 'E_acc_bin15', 'E_acc_bin16',
#             'E_acc_bin17', 'E_acc_bin18', 'E_acc_bin19', 'E_acc_bin20',
#             'E_acc_bin21', 'E_acc_bin22', 'E_acc_bin23', 'E_acc_bin24',
#             'E_acc_bin25', 'N_acc_bin1', 'N_acc_bin2', 'N_acc_bin3',
#             'N_acc_bin4', 'N_acc_bin5', 'N_acc_bin6', 'N_acc_bin7',
#             'N_acc_bin8', 'N_acc_bin9', 'N_acc_bin10', 'N_acc_bin11',
#             'N_acc_bin12', 'N_acc_bin13', 'N_acc_bin14', 'N_acc_bin15',
#             'N_acc_bin16', 'N_acc_bin17', 'N_acc_bin18', 'N_acc_bin19',
#             'N_acc_bin20', 'N_acc_bin21', 'N_acc_bin22', 'N_acc_bin23',
#             'N_acc_bin24', 'N_acc_bin25', 'Z_acc_bin1', 'Z_acc_bin2',
#             'Z_acc_bin3', 'Z_acc_bin4', 'Z_acc_bin5', 'Z_acc_bin6',
#             'Z_acc_bin7', 'Z_acc_bin8', 'Z_acc_bin9', 'Z_acc_bin10',
#             'Z_acc_bin11', 'Z_acc_bin12', 'Z_acc_bin13', 'Z_acc_bin14',
#             'Z_acc_bin15', 'Z_acc_bin16', 'Z_acc_bin17', 'Z_acc_bin18',
#             'Z_acc_bin19', 'Z_acc_bin20', 'Z_acc_bin21', 'Z_acc_bin22',
#             'Z_acc_bin23', 'Z_acc_bin24', 'Z_acc_bin25']

acc_cols = ['acc_bin1', 'acc_bin2', 'acc_bin3', 'acc_bin4', 'acc_bin5',
            'acc_bin6', 'acc_bin7', 'acc_bin8', 'acc_bin9', 'acc_bin10',
            'acc_bin11', 'acc_bin12', 'acc_bin13', 'acc_bin14', 'acc_bin15',
            'acc_bin16', 'acc_bin17', 'acc_bin18', 'acc_bin19', 'acc_bin20',
            'acc_bin21', 'acc_bin22', 'acc_bin23', 'acc_bin24', 'acc_bin25']

acc_df = pd.DataFrame(data=acc_data, columns=acc_cols)

# vel_cols = ['E_vel_bin1', 'E_vel_bin2', 'E_vel_bin3', 'E_vel_bin4',
#             'E_vel_bin5', 'E_vel_bin6', 'E_vel_bin7', 'E_vel_bin8',
#             'E_vel_bin9', 'E_vel_bin10', 'E_vel_bin11', 'E_vel_bin12',
#             'E_vel_bin13', 'E_vel_bin14', 'E_vel_bin15', 'E_vel_bin16',
#             'E_vel_bin17', 'E_vel_bin18', 'E_vel_bin19', 'E_vel_bin20',
#             'E_vel_bin21', 'E_vel_bin22', 'E_vel_bin23', 'E_vel_bin24',
#             'E_vel_bin25', 'N_vel_bin1', 'N_vel_bin2', 'N_vel_bin3',
#             'N_vel_bin4', 'N_vel_bin5', 'N_vel_bin6', 'N_vel_bin7',
#             'N_vel_bin8', 'N_vel_bin9', 'N_vel_bin10', 'N_vel_bin11',
#             'N_vel_bin12', 'N_vel_bin13', 'N_vel_bin14', 'N_vel_bin15',
#             'N_vel_bin16', 'N_vel_bin17', 'N_vel_bin18', 'N_vel_bin19',
#             'N_vel_bin20', 'N_vel_bin21', 'N_vel_bin22', 'N_vel_bin23',
#             'N_vel_bin24', 'N_vel_bin25', 'Z_vel_bin1', 'Z_vel_bin2',
#             'Z_vel_bin3', 'Z_vel_bin4', 'Z_vel_bin5', 'Z_vel_bin6',
#             'Z_vel_bin7', 'Z_vel_bin8', 'Z_vel_bin9', 'Z_vel_bin10',
#             'Z_vel_bin11', 'Z_vel_bin12', 'Z_vel_bin13', 'Z_vel_bin14',
#             'Z_vel_bin15', 'Z_vel_bin16', 'Z_vel_bin17', 'Z_vel_bin18',
#             'Z_vel_bin19', 'Z_vel_bin20', 'Z_vel_bin21', 'Z_vel_bin22',
#             'Z_vel_bin23', 'Z_vel_bin24', 'Z_vel_bin25']

vel_cols = ['vel_bin1', 'vel_bin2', 'vel_bin3', 'vel_bin4', 'vel_bin5',
            'vel_bin6', 'vel_bin7', 'vel_bin8', 'vel_bin9', 'vel_bin10',
            'vel_bin11', 'vel_bin12', 'vel_bin13', 'vel_bin14', 'vel_bin15',
            'vel_bin16', 'vel_bin17', 'vel_bin18', 'vel_bin19', 'vel_bin20',
            'vel_bin21', 'vel_bin22', 'vel_bin23', 'vel_bin24', 'vel_bin25']

vel_df = pd.DataFrame(data=vel_data, columns=vel_cols)

# disp_cols = ['E_disp_bin1', 'E_disp_bin2', 'E_disp_bin3', 'E_disp_bin4',
#              'E_disp_bin5', 'E_disp_bin6', 'E_disp_bin7',
#              'E_disp_bin8', 'E_disp_bin9', 'E_disp_bin10', 'E_disp_bin11',
#              'E_disp_bin12', 'E_disp_bin13', 'E_disp_bin14', 'E_disp_bin15',
#              'E_disp_bin16', 'E_disp_bin17', 'E_disp_bin18', 'E_disp_bin19',
#              'E_disp_bin20', 'E_disp_bin21', 'E_disp_bin22', 'E_disp_bin23', 
#              'E_disp_bin24', 'E_disp_bin25', 'N_disp_bin1', 'N_disp_bin2',
#              'N_disp_bin3', 'N_disp_bin4', 'N_disp_bin5', 'N_disp_bin6',
#              'N_disp_bin7', 'N_disp_bin8', 'N_disp_bin9', 'N_disp_bin10',
#              'N_disp_bin11', 'N_disp_bin12', 'N_disp_bin13', 'N_disp_bin14',
#              'N_disp_bin15', 'N_disp_bin16', 'N_disp_bin17', 'N_disp_bin18',
#              'N_disp_bin19', 'N_disp_bin20', 'N_disp_bin21', 'N_disp_bin22',
#              'N_disp_bin23', 'N_disp_bin24', 'N_disp_bin25', 'Z_disp_bin1',
#              'Z_disp_bin2', 'Z_disp_bin3', 'Z_disp_bin4', 'Z_disp_bin5',
#              'Z_disp_bin6', 'Z_disp_bin7', 'Z_disp_bin8', 'Z_disp_bin9',
#              'Z_disp_bin10', 'Z_disp_bin11', 'Z_disp_bin12', 'Z_disp_bin13',
#              'Z_disp_bin14', 'Z_disp_bin15', 'Z_disp_bin16', 'Z_disp_bin17',
#              'Z_disp_bin18', 'Z_disp_bin19', 'Z_disp_bin20', 'Z_disp_bin21',
#              'Z_disp_bin22', 'Z_disp_bin23', 'Z_disp_bin24', 'Z_disp_bin25']

disp_cols = ['disp_bin1', 'disp_bin2', 'disp_bin3', 'disp_bin4', 'disp_bin5',
            'disp_bin6', 'disp_bin7', 'disp_bin8', 'disp_bin9', 'disp_bin10',
            'disp_bin11', 'disp_bin12', 'disp_bin13', 'disp_bin14', 'disp_bin15',
            'disp_bin16', 'disp_bin17', 'disp_bin18', 'disp_bin19', 'disp_bin20',
            'disp_bin21', 'disp_bin22', 'disp_bin23', 'disp_bin24', 'disp_bin25']

disp_df = pd.DataFrame(data=disp_data, columns=disp_cols)

sm_df = pd.concat([acc_df, vel_df], axis=1)
sm_cols = (acc_cols + vel_cols)



nan_arr = np.empty((len(disp_df), len(sm_df.iloc[0])))
nan_arr[:] = np.nan
sm_spec_df = sm_df.append(pd.DataFrame(nan_arr, columns=sm_cols), ignore_index=True)

nan_arr2 = np.empty((len(sm_df), len(disp_df.iloc[0])))
nan_arr2[:] = np.nan
disp_spec_df = pd.DataFrame(nan_arr2, columns=disp_cols).append(disp_df, ignore_index=True)

fs_df = np.append(sm_spec_df, disp_spec_df, axis=1)
fs_df = pd.DataFrame(fs_df)

df_list = [stations, run, hyplon, hyplat, pga_all, pgv_all, pgd_all, fs_df]
df = pd.concat(df_list, axis=1)

df = np.append(stations, (run, hyplon, hyplat, pga_all, pgv_all, pgd_all, fs_df), axis=1)
df = pd.concat(stations, run, hyplon, hyplat, pga_all, pgv_all, pgd_all, fs_df, join='inner')
