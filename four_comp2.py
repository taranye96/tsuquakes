#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:17:22 2020

@author: tnye
"""

# Imports
from glob import glob
import numpy as np
import pandas as pd
from obspy import read
import tsueqs_main_fns as tmf
import IM_fns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

project = '8min_16cores'
run = 'run.000000'

param_dir = f'/Users/tnye/FakeQuakes/{project}/' 

# data_types = disp, acc vel
data = 'disp'

### Set paths and parameters #### 

# Data directory                          
data_dir = '/Users/tnye/tsuquakes/data'

# Table of earthquake data
eq_table_path = '/Users/tnye/tsuquakes/data/misc/events.csv'   
eq_table = pd.read_csv(eq_table_path)

### Get event data ###
eventname = 'Mentawai2010'
country = eq_table['Country'][11]
hyplon = eq_table['Longitude'][11]
hyplat = eq_table['Latitude'][11]
hypdepth = eq_table['Depth (km)'][11]
    
# Synthetic miniseed dir
disp_syn_dir = param_dir + f'disp/output/waveforms/{run}/'
sm_syn_dir = param_dir + f'sm/output/waveforms/{run}/'
disp_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'
acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr'
vel_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr'

# Filtering
threshold = 0.0
fcorner = 1/15.                          # Frequency at which to high pass filter
order = 2                                # Number of poles for filter  


##################### Data Processing and Calculations ####################

if data == 'disp':
    metadata_file = data_dir + '/' + eventname + '/' + eventname + '_disp.chan'
    syn_files = np.array(sorted(glob(disp_syn_dir + '*.sac')))
    obs_files = np.array(sorted((glob(disp_dir + '/*'))))
    filtering = False
    dtype = 'disp'
elif data == 'acc':
    metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
    syn_files = np.array(sorted(glob(sm_syn_dir + '*.bb*.sac')))
    obs_files = np.array(sorted((glob(acc_dir + '/*'))))
    filtering = True
    dtype = 'sm'
elif data == 'vel':
    metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
    syn_files = np.array(sorted(glob(sm_syn_dir + '*.bb*.sac')))
    obs_files = np.array(sorted((glob(vel_dir + '/*'))))
    dtype = 'sm'

metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                      names=['net', 'sta', 'loc', 'chan', 'lat',
                             'lon', 'elev', 'samplerate', 'gain', 'units'])

# There might be white spaces in the station name, so remove those
metadata.sta = metadata.sta.astype(str)
metadata.sta = metadata.sta.str.replace(' ','')


############################# Synthetic Spectra ###############################
# Create lists to add station names, channels, and miniseed files to 
stn_name_list = []
channel_list = []
syn_mseed_list = []

# Group all files by station
N = 3
syn_stn_files = [syn_files[n:n+N] for n in range(0, len(syn_files), N)]

# Loop over files to get the list of station names, channels, and mseed files 
for station in syn_stn_files:
    components = []
    mseeds = []

    stn_name = station[0].split('/')[-1].split('.')[0]
    stn_name_list.append(stn_name)
    
    for mseed_file in station:
        if data == 'disp':
            channel_code = mseed_file.split('/')[-1].split('.')[1]
        elif data == 'acc':
            channel_code = mseed_file.split('/')[-1].split('.')[2]
        elif data == 'vel':
            channel_code = mseed_file.split('/')[-1].split('.')[2]

        components.append(channel_code)
        mseeds.append(mseed_file)

    channel_list.append(components)
    syn_mseed_list.append(mseeds)

obs_mseed_list = []

# Group all files by station
N = 3
obs_stn_files = [obs_files[n:n+N] for n in range(0, len(obs_files), N)]

# Loop over files to get the list of station names, channels, and mseed files 
for station in obs_stn_files:
    mseeds = []

    for mseed_file in station:
        channel_code = mseed_file.split('/')[-1].split('.')[1]
        components.append(channel_code)
        mseeds.append(mseed_file)

    channel_list.append(components)
    obs_mseed_list.append(mseeds)

# Loop over the stations for this earthquake, and start to run the computations:
for i, station in enumerate(stn_name_list):
    
    # Get the instrument (HN or LX) and component (E,N,Z) for this station
    components = []
    for channel in channel_list[i]:
        components.append(channel[2])

        
        # Get the metadata for this station from the chan file - put it into
    #     a new dataframe and reset the index so it starts at 0
    if country == 'Japan':
        station_metadata = metadata[(metadata.net == station[0:2]) & (metadata.sta == station[2:])].reset_index(drop=True)
        
    else:
        station_metadata = metadata[metadata.sta == station].reset_index(drop=True)       # what is going on here
          
  
    # Pull out the data. Take the first row of the subset dataframe, 
    #    assuming that the gain, etc. is always the same:
    stlon = station_metadata.loc[0].lon
    stlat = station_metadata.loc[0].lat
    stelev = station_metadata.loc[0].elev
    stsamprate = station_metadata.loc[0].samplerate
  

    ##################### Start computations ######################        

    # Compute the hypocentral distance
    hypdist = tmf.compute_rhyp(stlon,stlat,stelev,hyplon,hyplat,hypdepth)

    # Get the components
    components = np.asarray(components)
    
    # Only look at E component 
    # Get index for E component 
    E_index = np.where(components=='E')[0][0]
    
    # Read in observed station data
    E_obs = read(obs_mseed_list[i][E_index])

    # Read in synthetic station data
    E_raw = read(syn_mseed_list[i][E_index])

    # High pass filter strong motion data at fcorner specified above
    if filtering == True:
        E_filt = tmf.highpass(E_raw,fcorner,stsamprate,order,zerophase=True)
        E_syn = E_filt
    else:
        E_syn = E_raw

    # Integrate if using velocity     
    if data == 'vel':
        # Convert acceleration record to velocity 
        E_raw = tmf.accel_to_veloc(E_raw)

        # High pass filter velocity 
        E_syn = tmf.highpass(E_raw,fcorner,stsamprate,order,zerophase=True)


    ################################# Spectra ########################W########

    # Calc obs spectra 
    obs_spec_data, obs_freq, obs_amp = IM_fns.calc_spectra(E_obs, dtype)

    # Calculate syn spectra
    syn_spec_data, syn_freq, syn_amp = IM_fns.calc_spectra(E_syn, dtype)
    
    # Plot spectra
    obs_tr = E_obs[0]
    syn_tr = E_syn[0]
    station = obs_tr.stats.station
    
    # Loop through frequencies and amplitudes 
    # Units
    if data == 'disp':
        units = 'm*s'
        channel = 'LX' 
        ylim = 10**-4, 6*10**-1
        xlim = 2*10**-3, 5*10**-1
    elif data == 'acc':
        units = 'm/s'
        channel = 'HN'
        ylim = 6*10**-15, 6*10**-2
        xlim = .002, 50
    elif data == 'vel':
        units = 'm'
        channel = 'HN'
        ylim = 6*10**-15, 6*10**-2
        xlim = .002, 50
            
    # Define label 
    component = 'E'
    label = channel + component 

    # Plot parameters 
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=52, vmax=952)
    color = cmap(norm(hypdist))
    
    # Plot obs and syn
    fig, ax = plt.subplots()
    ax.loglog(obs_freq, obs_amp, color=color, ls='-', lw=.8)
    ax.loglog(syn_freq, syn_amp, color=color, ls='--', lw=.8)
    # axs[i].grid(linestyle='--')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # fig.colorbar(im, ax=ax)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    
    # im.show()
    
    
### Plot

# Colorbar setup
# s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
# s_map.set_array(colorparams)
# # cbar = plt.figure().colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
# # If color parameters is a linspace, we can set boundaries in this way
# halfdist = (colorparams[1] - colorparams[0])/2.0
# boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

# # Use this to emphasize the discrete color values
# cbar = plt.colorbar(s_map, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g')

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cmap = cm.cool
norm = mcolors.Normalize(vmin=np.min(hypdists), vmax=np.max(hypdists))
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), orientation='horizontal', label='Some Units')

# plt.show()