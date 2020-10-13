#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:31:02 2020

@author: tnye
"""

###############################################################################
# Script that calculates the Fourier spectra for the observed Mentawai
# waveforms and saves them as .csv files.  These files are used in 
# synthetic_calc_mpi.py to make spectra comparison plots. 
###############################################################################

# Imports
from glob import glob
import numpy as np
import pandas as pd
from obspy import read
import IM_fns

data_types = ['disp', 'acc', 'vel']

for data in data_types:
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
    
    # Filtering
    threshold = 0.0
    fcorner = 1/15.                          # Frequency at which to high pass filter
    order = 2                                # Number of poles for filter  
            
    ##################### Data Processing and Calculations ####################
    
    # Obs data directories
    disp_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'
    acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr'
    vel_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr'
            
    if data == 'disp':
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_disp.chan'
        obs_files = np.array(sorted((glob(disp_dir + '/*'))))
        filtering = False
        dtype = 'disp'
    elif data == 'acc':
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
        obs_files = np.array(sorted((glob(acc_dir + '/*'))))
        filtering = True
        dtype = 'sm'
    elif data == 'vel':
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
        obs_files = np.array(sorted((glob(vel_dir + '/*'))))
        filtering = True
        dtype = 'sm'
    
    metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                          names=['net', 'sta', 'loc', 'chan', 'lat',
                                 'lon', 'elev', 'samplerate', 'gain', 'units'])
    
    # There might be white spaces in the station name, so remove those
    metadata.sta = metadata.sta.astype(str)
    metadata.sta = metadata.sta.str.replace(' ','')
    
    syn_freqs = []
    syn_amps = []
    obs_freqs = []
    obs_amps = []
    hypdists = []
    
    
    ################################### Observed ###########3######################
    
    # Create lists to add station names, channels, and miniseed files to 
    stn_name_list = []
    channel_list = []
    mseed_list = []
    
    # Group all files by station
    N = 3
    stn_files = [obs_files[n:n+N] for n in range(0, len(obs_files), N)]
    
    # Loop over files to get the list of station names, channels, and mseed files 
    for station in stn_files:
        components = []
        mseeds = []
    
        stn_name = station[0].split('.')[0].split('/')[-1]
        if stn_name != 'SISI':
            stn_name_list.append(stn_name)
            
            for mseed_file in station:
                channel_code = mseed_file.split('/')[-1].split('.')[1]
                components.append(channel_code)
                mseeds.append(mseed_file)
        
            channel_list.append(components)
            mseed_list.append(mseeds)
    
    # Loop over the stations for this earthquake, and start to run the computations:
    for i, station in enumerate(stn_name_list):
            
        # Get the components for this station (E, N, and Z):
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
        stnetwork = station_metadata.loc[0].net
        stlon = station_metadata.loc[0].lon
        stlat = station_metadata.loc[0].lat
        stelev = station_metadata.loc[0].elev
        stsamprate = station_metadata.loc[0].samplerate
        stgain = station_metadata.loc[0].gain
    
    
        ######################### Start computations ##########################       
        
        # List for all spectra at station
        station_spec = []
    
        # Get the components
        components = np.asarray(components)
            
        # Get index for E component 
        E_index = np.where(components=='E')[0][0]
        # Read file into a stream object
        E_record = read(mseed_list[i][E_index])
    
    
        ####################### IMs ########################
    
        # Calc Spectra
        E_spec_data, freqE, ampE = IM_fns.calc_spectra(E_record, dtype)
        
        # Append spectra to observed list
        obs_freqs.append(freqE.tolist())
        obs_amps.append(ampE.tolist())
        
    freq_df = pd.DataFrame(obs_freqs)
    amp_df = pd.DataFrame(obs_amps)
    flatfile_df = pd.concat([freq_df, amp_df], axis=1)
    
    flatfile_df.to_csv(f'/Users/tnye/tsuquakes/data/obs_spectra/{data}_spec.csv',index=False)
    