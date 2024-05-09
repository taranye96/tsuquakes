#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 20:41:36 2020

@author: tnye
"""

###############################################################################
# Script that stores the trace data for the observed Mentawai waveforms and 
# saves them as .csv files.  These files are used in parallel_calc_IMs.py to
# make waveform comparison plots. 
###############################################################################

# Imports
from glob import glob
import numpy as np
import pandas as pd
from obspy import read
from os import path, makedirs
from obspy import UTCDateTime,Stream
import signal_average_fns as avg
import IM_fns
from rotd50 import compute_rotd50

data_types = ['disp', 'acc', 'vel']
home_dir = '/Users/tnye/tsuquakes/data/waveforms'

for data in data_types:
    
    if not path.exists(f'{home_dir}/average/eucnorm_3comp/{data}'):
        makedirs(f'{home_dir}/average/eucnorm_3comp/{data}')
    if not path.exists(f'{home_dir}/average/eucnorm_2comp/{data}'):
        makedirs(f'{home_dir}/average/eucnorm_2comp/{data}')
    if not path.exists(f'{home_dir}/average/geom_3comp/{data}'):
        makedirs(f'{home_dir}/average/geom_3comp/{data}')
    if not path.exists(f'{home_dir}/average/geom_2comp/{data}'):
        makedirs(f'{home_dir}/average/geom_2comp/{data}')
    if not path.exists(f'{home_dir}/average/rotd50/{data}'):
        makedirs(f'{home_dir}/average/rotd50/{data}')
    
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
    

    ##################### Data Processing and Calculations ####################
    
    disp_dir = '/Users/tnye/tsuquakes/data/waveforms/individual/disp'
    acc_dir = '/Users/tnye/tsuquakes/data/waveforms/individual/acc'
    vel_dir = '/Users/tnye/tsuquakes/data/waveforms/individual/vel'
            
    if data == 'disp':
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_disp.chan'
        obs_files = np.array(sorted((glob(disp_dir + '/*'))))
        dtype = 'gnss'
    elif data == 'acc':
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
        obs_files = np.array(sorted((glob(acc_dir + '/*'))))
        dtype = 'sm'
    elif data == 'vel':
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
        obs_files = np.array(sorted((glob(vel_dir + '/*'))))
        dtype = 'sm'
    
    metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                          names=['net', 'sta', 'loc', 'chan', 'lat',
                                 'lon', 'elev', 'samplerate', 'gain', 'units'])
    
    # There might be white spaces in the station name, so remove those
    metadata.sta = metadata.sta.astype(str)
    metadata.sta = metadata.sta.str.replace(' ','')
    
    # Lists to store stream data in
    obs_times = []
    obs_amps_euc3comp = []
    obs_amps_euc2comp = []
    obs_amps_rotd50 = []
    obs_amps_Ecomp = []
    obs_freqs = []
    obs_spec = []
    obs_freqs_trim = []
    obs_spec_trim = []
    
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
        if data == 'disp':
            if stn_name in pd.read_csv('/Users/tnye/FakeQuakes/files/stn_info/gnss_clean.gflist',delimiter='\t')['#station'].values:
                stn_name_list.append(stn_name)
                
                for mseed_file in station:
                    channel_code = mseed_file.split('/')[-1].split('.')[1]
                    components.append(channel_code)
                    mseeds.append(mseed_file)
            
                channel_list.append(components)
                mseed_list.append(mseeds)
        else:
            if stn_name in pd.read_csv('/Users/tnye/FakeQuakes/files/stn_info/sm_close.gflist',delimiter='\t')['#station'].values:
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
    
        ######################### Start computations ##########################       
    
        # Get the components
        components = np.asarray(components)
            
        # Get index for E component 
        E_index = np.where(components=='E')[0][0]
        # Read file into a stream object
        E_record = read(mseed_list[i][E_index])
        
        # Get index for E component 
        N_index = np.where(components=='N')[0][0]
        # Read file into a stream object
        N_record = read(mseed_list[i][N_index])
        
        # Get index for E component 
        Z_index = np.where(components=='Z')[0][0]
        # Read file into a stream object
        Z_record = read(mseed_list[i][Z_index])
    
    
        ########################## Waveforms ##########################
            
        # Get traces
        tr_E = E_record[0]
        tr_N = N_record[0]
        tr_Z = Z_record[0]
        
        tr_E_short = tr_E.copy().trim(endtime=UTCDateTime("2010-10-25T14:44:47"))
        tr_N_short = tr_N.copy().trim(endtime=UTCDateTime("2010-10-25T14:44:47"))
        tr_Z_short = tr_Z.copy().trim(endtime=UTCDateTime("2010-10-25T14:44:47"))
        
        # Compute averages
        euc_3comp_amps = avg.get_eucl_norm_3comp(tr_E.data,tr_N.data,tr_Z.data)
        euc_2comp_amps = avg.get_eucl_norm_2comp(tr_E.data,tr_N.data)
        rotd50_amps = compute_rotd50(tr_E.data,tr_N.data)
        
        if data == 'disp':

            # Euclidean norm 3 component
            st_euc3comp = Stream(tr_E.copy())
            st_euc3comp[0].data = euc_3comp_amps
            st_euc3comp[0].stats.channel = 'LXNEZ'
            filename = f'{home_dir}/average/eucnorm_3comp/{data}/{station}.{st_euc3comp[0].stats.channel}.mseed' 
            st_euc3comp[0].write(filename, format='MSEED')
            
            # Euclidean norm 2 component 
            st_euc2comp = Stream(tr_E.copy())
            st_euc2comp[0].data = euc_2comp_amps
            st_euc2comp[0].stats.channel = 'LXNE'
            filename = f'{home_dir}/average/eucnorm_2comp/{data}/{station}.{st_euc2comp[0].stats.channel}.mseed' 
            st_euc2comp[0].write(filename, format='MSEED')
            
            # rotd50
            st_rotd50 = Stream(tr_E.copy())
            st_rotd50[0].data = rotd50_amps
            st_rotd50[0].stats.channel = 'LXNE'
            filename = f'{home_dir}/average/rotd50/{data}/{station}.{st_rotd50[0].stats.channel}.mseed' 
            st_rotd50[0].write(filename, format='MSEED')
            
            # # Append trace data and times to lists
            # obs_times.append(tr_E.times('matplotlib').tolist())
            # obs_amps_euc2comp.append(st_euc2comp[0].data.tolist())
            # obs_amps_euc3comp.append(st_euc3comp[0].data.tolist())
            # obs_amps_rotd50.append(st_rotd50[0].data.tolist())
            # obs_amps_Ecomp.append(E_record[0].data.tolist())
            
        else:
            
            # Euclidean norm 3 component
            st_euc3comp = E_record.copy()
            st_euc3comp[0].data = euc_3comp_amps
            st_euc3comp[0].stats.channel = 'HNNEZ'
            filename = f'{home_dir}/average/eucnorm_3comp/{data}/{station}.{st_euc3comp[0].stats.channel}.mseed' 
            st_euc3comp[0].write(filename, format='MSEED')
            
            # Euclidean norm 2 component 
            st_euc2comp = E_record.copy()
            st_euc2comp[0].data = euc_2comp_amps
            st_euc2comp[0].stats.channel = 'HNNE'
            filename = f'{home_dir}/average/eucnorm_2comp/{data}/{station}.{st_euc2comp[0].stats.channel}.mseed' 
            st_euc2comp[0].write(filename, format='MSEED')
            
            # rotd50
            st_rotd50 = E_record.copy()
            st_rotd50[0].data = rotd50_amps
            st_rotd50[0].stats.channel = 'HNNE'
            filename = f'{home_dir}/average/rotd50/{data}/{station}.{st_rotd50[0].stats.channel}.mseed' 
            st_rotd50[0].write(filename, format='MSEED')
            
            # # Append trace data and times to lists
            # obs_times.append(E_record[0].times('matplotlib').tolist())
            # obs_amps_euc2comp.append(st_euc2comp[0].data.tolist())
            # obs_amps_euc3comp.append(st_euc3comp[0].data.tolist())
            # obs_amps_rotd50.append(st_rotd50[0].data.tolist())
            # obs_amps_Ecomp.append(E_record[0].data.tolist())
            
            
        ########################## Spectra ##########################
        
        # Calc Spectra full waveforms
        bins, E_spec_data = IM_fns.calc_spectra(Stream(tr_E), dtype)
        bins, N_spec_data = IM_fns.calc_spectra(Stream(tr_N), dtype)
        
        # Calc Spectra trimmed waveforms
        bins_trim, E_spec_data_trim = IM_fns.calc_spectra(Stream(tr_E_short), dtype)
        bins_trim, N_spec_data_trim = IM_fns.calc_spectra(Stream(tr_N_short), dtype)
        
        # Get avg of horizontals
        NE_data = np.sqrt(E_spec_data**2 + N_spec_data**2)
        NE_data_trim = np.sqrt(E_spec_data_trim**2 + N_spec_data_trim**2)
                
        # Append spectra to observed list
        obs_freqs.append(bins)
        obs_spec.append(NE_data.tolist())
        obs_freqs_trim.append(bins_trim)
        obs_spec_trim.append(NE_data_trim.tolist())
    
    # Set up dataframe
    freq_df = pd.DataFrame(obs_freqs)
    spec_df = pd.DataFrame(obs_spec)
    flatfile_df = pd.concat([freq_df, spec_df], axis=1)
    
    freq_trim_df = pd.DataFrame(obs_freqs_trim)
    spec_trim_df = pd.DataFrame(obs_spec_trim)
    flatfile_trim_df = pd.concat([freq_trim_df, spec_trim_df], axis=1)
    
    flatfile_df.to_csv(f'/Users/tnye/tsuquakes/data/obs_avg_spectra/{data}_binned_spec.csv',index=False)
    flatfile_trim_df.to_csv(f'/Users/tnye/tsuquakes/data/obs_avg_spectra/{data}_binned_spec_trim.csv',index=False)
    
    
    # times_df = pd.DataFrame(obs_times)
    # euc_3comp_amp_df = pd.DataFrame(obs_amps_euc3comp)
    # euc_2comp_amp_df = pd.DataFrame(obs_amps_euc2comp)
    # rotd50_amp_df = pd.DataFrame(obs_amps_rotd50)
    # Ecomp_amp_df = pd.DataFrame(E_record[0].data.tolist())
    # Ncomp_amp_df = pd.DataFrame(N_record[0].data.tolist())
    # Zcomp_amp_df = pd.DataFrame(Z_record[0].data.tolist())
    
    # euc3comp_df = pd.concat([times_df, euc_3comp_amp_df], axis=1)
    # euc2comp_df = pd.concat([times_df, euc_2comp_amp_df], axis=1)
    # rotd50_df = pd.concat([times_df, rotd50_amp_df], axis=1)
    # Ecomp_df = pd.concat([times_df, Ecomp_amp_df], axis=1)
    # Ncomp_df = pd.concat([times_df, Ncomp_amp_df], axis=1)
    # Zcomp_df = pd.concat([times_df, Zcomp_amp_df], axis=1)
    
    # euc3comp_df.to_csv(f'{home_dir}/average/flatfiles/{data}_euc3comp.csv',index=False)
    # euc2comp_df.to_csv(f'{home_dir}/average/flatfiles/{data}_euc2comp.csv',index=False)
    # rotd50_df.to_csv(f'{home_dir}/average/flatfiles/{data}_rotd50comp.csv',index=False)
    # Ecomp_df.to_csv(f'{home_dir}/individual/flatfiles/{data}_Ecomp.csv',index=False)
    # Ncomp_df.to_csv(f'{home_dir}/individual/flatfiles/{data}_Ncomp.csv',index=False)
    # Zcomp_df.to_csv(f'{home_dir}/individual/flatfiles/{data}_Zcomp.csv',index=False)
    
    
    
    
    
    
    
    