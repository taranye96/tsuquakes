#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:04:38 2022

@author: tnye
"""

# Standard library Imports 
import time
import sys
from glob import glob
from os import makedirs, path
import numpy as np
from numpy import genfromtxt
import pandas as pd
from obspy import read

# Local Imports
import tsueqs_main_fns as tmf
import signal_average_fns as avg
import IM_fns
import comparison_fns as comp

###################### Set up parallelization parameters ######################

home = '/Users/tnye/FakeQuakes/FQ_status/new_Q_model'
param_dir = f'{home}/standard_parameters'                      
data_dir = '/Users/tnye/tsuquakes/data'


rupture_list = genfromtxt(f'{param_dir}/data/ruptures.list',dtype='U')

data = 'sm'
avg_comp = '2-comp'
obs_wf_dir = f'/Users/tnye/tsuquakes/data/waveforms/individual'

# Observed spectra
acc_obs_spec_df = pd.read_csv('/Users/tnye/tsuquakes/data/obs_spectra/acc_binned_spec.csv')
acc_obs_freqs = np.array(acc_obs_spec_df.iloc[:,:int(acc_obs_spec_df.shape[1]/2)])
acc_obs_spec = np.array(acc_obs_spec_df.iloc[:,int(acc_obs_spec_df.shape[1]/2):])

vel_obs_spec_df = pd.read_csv('/Users/tnye/tsuquakes/data/obs_spectra/vel_binned_spec.csv')
vel_obs_freqs = np.array(vel_obs_spec_df.iloc[:,:int(vel_obs_spec_df.shape[1]/2)])
vel_obs_spec = np.array(vel_obs_spec_df.iloc[:,int(vel_obs_spec_df.shape[1]/2):])

disp_obs_spec_df = pd.read_csv('/Users/tnye/tsuquakes/data/obs_spectra/disp_binned_spec.csv')
disp_obs_freqs = np.array(disp_obs_spec_df.iloc[:,:int(disp_obs_spec_df.shape[1]/2)])
disp_obs_spec = np.array(disp_obs_spec_df.iloc[:,int(disp_obs_spec_df.shape[1]/2):])

# Observed waveforms
acc_obs_wf_df = pd.read_csv(f'{obs_wf_dir}/acc.csv')
acc_obs_times = np.array(acc_obs_wf_df.iloc[:,:int(acc_obs_wf_df.shape[1]/2)])
acc_obs_amps = np.array(acc_obs_wf_df.iloc[:,int(acc_obs_wf_df.shape[1]/2):])

vel_obs_wf_df = pd.read_csv(f'{obs_wf_dir}/vel.csv')
vel_obs_times = np.array(vel_obs_wf_df.iloc[:,:int(vel_obs_wf_df.shape[1]/2)])
vel_obs_amps = np.array(vel_obs_wf_df.iloc[:,int(vel_obs_wf_df.shape[1]/2):])

disp_obs_wf_df = pd.read_csv(f'{obs_wf_dir}/disp.csv')
disp_obs_times = np.array(disp_obs_wf_df.iloc[:,:int(disp_obs_wf_df.shape[1]/2)])
disp_obs_amps = np.array(disp_obs_wf_df.iloc[:,int(disp_obs_wf_df.shape[1]/2):])


################################ Set up Folders ###############################

# Set up folder for velocity mseed files
if not path.exists(f'{param_dir}/vel'):
    makedirs(f'{param_dir}/vel')

# Set up folder for flatfile
if not path.exists(f'{param_dir}/flatfiles'):
    makedirs(f'{param_dir}/flatfiles')
if not path.exists(f'{param_dir}/flatfiles/IMs'):
    makedirs(f'{param_dir}/flatfiles/IMs')


############################### Do Calculations ###############################

### Set paths and parameters #### 

# Table of earthquake data
eq_table_path = f'{data_dir}/misc/events.csv'   
eq_table = pd.read_csv(eq_table_path)

### Get event data ###
origin = pd.to_datetime('2010-10-25T14:42:22')
eventname = 'Mentawai2010'
country = eq_table['Country'][11]
origintime = eq_table['Origin Time (UTC)*'][11]
hyplon = eq_table['Longitude'][11]
hyplat = eq_table['Latitude'][11]
hypdepth = eq_table['Depth (km)'][11]
mw = eq_table['Mw'][11]
m0 = 10**(mw*(3/2.) + 9.1)
mechanism = eq_table['Mechanism'][11]  

for rupture in rupture_list:
    print(f'Working on run {rupture}')
    run = rupture.rsplit('.', 1)[0]
    # print(f"Processor {rank} beginning {run}")
        
    # Synthetic miniseed dirs
    sm_dir = f'{param_dir}/output/waveforms/{run}/'
    
    # Gather displacement and strong motion files
    sm_files = np.array(sorted(glob(sm_dir + '*.bb*.sac')))
    
    # Path to send flatfile
    gnss_flatfile_path = f'{param_dir}/flatfiles/IMs/{run}_gnss.csv'
    sm_flatfile_path = f'{param_dir}/flatfiles/IMs/{run}_sm.csv'
    
    # Filtering
    threshold = 0.0
    fcorner = 1/15.                          # Frequency at which to high pass filter
    order = 2                                # Number of poles for filter  
    
    
    ##################### Data Processing and Calculations ####################
        
    # Create lists for of the event and station info for the df
    eventnames = np.array([])
    countries = np.array([])
    origintimes = np.array([])
    hyplons = np.array([])
    hyplats = np.array([])
    hypdepths = np.array([])
    mws = np.array([])
    m0s = np.array([])
    mechanisms = np.array([])
    networks = np.array([])
    stations = np.array([])
    stn_type_list = np.array([])
    stlons = np.array([])
    stlats = np.array([])
    stelevs = np.array([])
    hypdist_list = np.array([])
    instrument_codes = np.array([])
    E_Td_list = np.array([])
    N_Td_list = np.array([])
    Z_Td_list = np.array([])
    horiz_Td_list = np.array([])
    comp3_Td_list = np.array([])
    pga_list = np.array([])
    pgv_list = np.array([])
    pgd_list = np.array([])
    tPGD_orig_list = np.array([])
    tPGD_parriv_list = np.array([])
    tPGA_orig_list = np.array([])
    tPGA_parriv_list = np.array([])
    disp_speclist = []
    acc_speclist = []
    vel_speclist = []
        
    metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
    files = sm_files
    IMs = ['pga', 'pgv']
    filtering = True

    metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                          names=['net', 'sta', 'loc', 'chan', 'lat',
                                  'lon', 'elev', 'samplerate', 'gain', 'units'])

    # There might be white spaces in the station name, so remove those
    metadata.sta = metadata.sta.astype(str)
    metadata.sta = metadata.sta.str.replace(' ','')

    # Create lists to add station names, channels, and miniseed files to 
    stn_list = []
    channel_list = []
    mseed_list = []
    
    # Group all files by station
    N = 3
    stn_files = [files[n:n+N] for n in range(0, len(files), N)]
    
    # Lists to make spectra and wf comparison plots
    syn_freqs = []
    syn_spec = []
    syn_times = []
    syn_amps = []
    hypdists = []
    
    # Lists to make spectra and wf compaison plots for velocity if doing sm
    syn_freqs_v = []
    syn_spec_v = []
    syn_times_v = []
    syn_amps_v = []
    
    # Loop over the stations for this earthquake, and start to run the computations:        
    for i, group in enumerate(stn_files):
        
        stn = group[0].split('/')[-1].split('.')[0]
        stn_list.append(stn)
        components = np.array([])
        
        for mseed in group:
            # Get the instrument component (E,N,Z) for this station
            channel = mseed.split('/')[-1].split('.sac')[0][-3:]
            components = np.append(components,channel[2])
        
        # Get the metadata for this station from the chan file - put it into
        #    a new dataframe and reset the index so it starts at 0
        station_metadata = metadata[metadata.sta == stn].reset_index(drop=True)       # what is going on here
  
        # Pull out the data. Take the first row of the subset dataframe, 
        #    assuming that the gain, etc. is always the same:
        stnetwork = station_metadata.loc[0].net
        stlon = station_metadata.loc[0].lon
        stlat = station_metadata.loc[0].lat
        stelev = station_metadata.loc[0].elev
        stsamprate = station_metadata.loc[0].samplerate
        stgain = station_metadata.loc[0].gain
  

        ##################### Start computations ######################        
        
        # Compute the hypocentral distance
        hypdist = tmf.compute_rhyp(stlon,stlat,stelev,hyplon,hyplat,hypdepth)

        # Append the earthquake and station info for this station
        eventnames = np.append(eventnames,eventname)
        countries = np.append(countries,country)
        origintimes = np.append(origintimes,origintime)
        hyplons = np.append(hyplons,hyplon)
        hyplats = np.append(hyplats,hyplat)
        hypdepths = np.append(hypdepths,hypdepth)
        mws = np.append(mws,mw)
        m0s = np.append(m0s,m0)
        mechanisms = np.append(mechanisms,mechanism)
        networks = np.append(networks,stnetwork)
        stations = np.append(stations,stn)
        stlons = np.append(stlons,stlon)
        stlats = np.append(stlats,stlat)
        stelevs = np.append(stelevs,stelev)
        hypdist_list = np.append(hypdist_list,hypdist)
        if data == 'disp':
            stn_type_list = np.append(stn_type_list, 'GNSS')
        elif data == 'sm':
            stn_type_list = np.append(stn_type_list, 'SM')
        
        # List for all spectra at station
        station_spec = []
        
        # Set up directories for filtered acceleration data 
        if not path.exists(f'{param_dir}/processed_wfs/acc/{run}'):
                makedirs(f'{param_dir}/processed_wfs/acc/{run}')
        
        ########## East component ##########

        # Find and read in East component
        E_index = np.where(components=='E')[0][0]
        E_raw = read(group[E_index])

        # High pass filter strong motion data at fcorner specified above
        E_filt = tmf.highpass(E_raw,fcorner,stsamprate,order,zerophase=True)
        E_record = E_filt
        # Save mseed
        E_record[0].stats.channel = 'HNE'
        E_record_filename = f'{param_dir}/processed_wfs/acc/{run}/{stn}.{E_record[0].stats.channel}.mseed' 
        E_record[0].write(E_record_filename, format='MSEED')

        # Get the duration, stream file time of start, and time of stop of shaking
        E_Td, E_start, E_end = tmf.determine_Td(threshold,E_record)      
        E_Td_list = np.append(E_Td_list,E_Td)

        ########## North component ##########

        # Find and read in North component 
        N_index = np.where(components=='N')[0][0]
        N_raw = read(group[N_index])

        # High pass filter strong motion data at fcorner specified above
        N_filt = tmf.highpass(N_raw,fcorner,stsamprate,order,zerophase=True)
        N_record = N_filt
        # Save mseed
        N_record[0].stats.channel = 'HNN'
        N_record_filename = f'{param_dir}/processed_wfs/acc/{run}/{stn}.{N_record[0].stats.channel}.mseed' 
        N_record[0].write(N_record_filename, format='MSEED')
        
        # Get the duration, stream file time of start, and time of stop of shaking
        N_Td, N_start, N_end = tmf.determine_Td(threshold,N_record)      
        N_Td_list = np.append(N_Td_list,N_Td)

        ########## Vertical component ##########

        # Find and read in vertical component
        Z_index = np.where(components=='Z')[0][0]                       
        Z_raw = read(group[Z_index])

        # High pass filter strong motion data at fcorner specified above
        Z_filt = tmf.highpass(Z_raw,fcorner,stsamprate,order,zerophase=True)
        Z_record = Z_filt
        # Save mseed
        Z_record[0].stats.channel = 'HNZ'
        Z_record_filename = f'{param_dir}/processed_wfs/acc/{run}/{stn}.{Z_record[0].stats.channel}.mseed' 
        Z_record[0].write(N_record_filename, format='MSEED')
        
        # Get the duration, stream file time of start, and time of stop of shaking
        Z_Td, Z_start, Z_end = tmf.determine_Td(threshold,Z_record)  
        Z_Td_list = np.append(Z_Td_list,Z_Td)    
        
        ############### Velocity ###############

        # Convert acceleration record to velocity (HP filter)
        E_vel_unfilt = tmf.accel_to_veloc(E_record)
        N_vel_unfilt = tmf.accel_to_veloc(N_record)
        Z_vel_unfilt = tmf.accel_to_veloc(Z_record)
        
        # High pass filter velocity 
        E_vel = tmf.highpass(E_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
        N_vel = tmf.highpass(N_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
        Z_vel = tmf.highpass(Z_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
        
        # Save vel mseed files
        # Set up folder for velocity mseed files
        if not path.exists(f'{param_dir}/processed_wfs/vel/{run}'):
            makedirs(f'{param_dir}/processed_wfs/vel/{run}')

        E_vel[0].stats.channel = 'HNE'
        E_vel_filename = f'{param_dir}/processed_wfs/vel/{run}/{stn}.{E_vel[0].stats.channel}.mseed' 
        E_vel[0].write(E_vel_filename, format='MSEED')
        
        N_vel[0].stats.channel = 'HNN'
        N_vel_filename = f'{param_dir}/processed_wfs/vel/{run}/{stn}.{N_vel[0].stats.channel}.mseed' 
        N_vel[0].write(N_vel_filename, format='MSEED')
        
        Z_vel[0].stats.channel = 'HNZ'
        Z_vel_filename = f'{param_dir}/processed_wfs/vel/{run}/{stn}.{Z_vel[0].stats.channel}.mseed' 
        Z_vel[0].write(Z_vel_filename, format='MSEED')


        ####################### Horizontal calc #######################
            
        # Take the min time of E and N start times to be the start
        EN_start = np.min([E_start,N_start])
        
        # Take the max time of E and N end times to be the end
        EN_end = np.max([E_end,N_end])
        
        # Get the duration to be the time between these
        EN_Td = EN_end - EN_start
        horiz_Td_list = np.append(horiz_Td_list,EN_Td)

            
        ####################### 3 component calc ######################

        # Take the min time of the E,N,Z start times to be the start
        ENZ_start = np.min([E_start,N_start,Z_start])
        
        # Take the max of the E,N,Z end times to be the end
        ENZ_end = np.max([E_end,N_end,Z_end])
        
        # Get the duration to be the time between these
        ENZ_Td = ENZ_end - ENZ_start
        comp3_Td_list = np.append(comp3_Td_list,ENZ_Td)
        
        
        ############################ Waveforms ############################

        ## Append tr data to lists to make wf comparison plots
        
        # Displacement and acceleration waveforms
            # Get trace (just using E component)
        tr = E_record[0]
        
        # Append trace data, times, and hypocentral distance to lists
        if obs_wf_dir.split('/')[-1] == 'average':
            if avg_comp == '3-comp':
                wf_amps = avg.get_eucl_norm_3comp(E_record[0].data,N_record[0].data,Z_record[0].data)
            elif avg_comp == '2-comp':
                wf_amps = avg.get_eucl_norm_2comp(E_record[0].data,N_record[0].data)
        elif obs_wf_dir.split('/')[-1] == 'individual':
            wf_amps = tr.data
            
        syn_times.append(tr.times('matplotlib'))
        syn_amps.append(wf_amps.tolist())
            
        if data == 'sm':
            
            # Velocity waveforms
                # Get trace (just using E component)
            tr_v = E_vel[0]
            
            # Append trace data, times, and hypocentral distance to lists
            syn_times_v.append(tr_v.times('matplotlib').tolist())
            syn_amps_v.append(tr_v.data.tolist())


        ######################## Intensity Measures #######################
            
        ######################### Acceleration ########################
        
        # Get euclidean norm of acceleration components 
        acc_euc_norm = avg.get_eucl_norm_3comp(E_record[0].data,
                                            N_record[0].data,
                                            Z_record[0].data)
       
        # Calculate PGA
        pga = np.max(np.abs(acc_euc_norm))
        pga_list = np.append(pga_list,pga)
        
        # Calcualte tPGA from origin and p-arrival
        tPGA_orig, tPGA_parriv = IM_fns.calc_time_to_peak(pga, E_record[0],
                                                        np.abs(acc_euc_norm),
                                                        origin, hypdist)
        tPGA_orig_list = np.append(tPGA_orig_list,tPGA_orig)
        tPGA_parriv_list = np.append(tPGA_parriv_list,tPGA_parriv)

        # Acc Spectra
        bins, E_spec_data = IM_fns.calc_spectra(E_record, data)                
        bins, N_spec_data = IM_fns.calc_spectra(N_record, data)
        bins, Z_spec_data = IM_fns.calc_spectra(Z_record, data)
        
        # Get avg of horizontals
        NE_data = (E_spec_data - N_spec_data)/(np.log(E_spec_data)-np.log(N_spec_data))
        
        # Append spectra to lists to make spectra comparison plots
        syn_freqs.append(bins)
        syn_spec.append(NE_data.tolist())
        hypdists.append(hypdist)
        
        # Save spectra as .out file
        if not path.exists(f'{param_dir}/spectra/{run}/acc/'):
            makedirs(f'{param_dir}/spectra/{run}/acc/')
            
        outfile = open(f'{param_dir}/spectra/{run}/acc/{stnetwork}_{stn}_HHNE.out', 'w')
        file_data = np.array([bins, E_spec_data, N_spec_data, Z_spec_data],dtype=object)
        file_data = file_data.T
        outfile.write(f'#bins \t \t acc_spec_E_m/s \t acc_spec_N_m/s \t acc_spec_Z_m/s \n')
        np.savetxt(outfile, file_data, fmt=['%E', '%E', '%E', '%E'], delimiter='\t')
        outfile.close()


        ########################### Velocity ##########################
        
        # Get euclidean norm of velocity components 
        vel_euc_norm = avg.get_eucl_norm_3comp(E_vel[0].data,
                                            N_vel[0].data,
                                            Z_vel[0].data)
        
        # Calculate PGV
        pgv = np.max(np.abs(vel_euc_norm))
        pgv_list = np.append(pgv_list,pgv)
        
        ## Vel Spectra
        bins, E_spec_vel = IM_fns.calc_spectra(E_vel, data)
        bins, N_spec_vel = IM_fns.calc_spectra(N_vel, data)
        bins, Z_spec_vel = IM_fns.calc_spectra(Z_vel, data)
        
        # Get avg of horizontals
        NE_data_v = (E_spec_vel - N_spec_vel)/(np.log(E_spec_vel)-np.log(N_spec_vel))
        
        # Append spectra to lists to make spectra comparison plots
        syn_freqs_v.append(bins)
        syn_spec_v.append(NE_data_v.tolist())
        
        # Save spectra as .out file
        if not path.exists(f'{param_dir}/spectra/{run}/vel/'):
            makedirs(f'{param_dir}/spectra/{run}/vel/')
            
        outfile = open(f'{param_dir}/spectra/{run}/vel/{stnetwork}_{stn}_HHNE.out', 'w')
        file_data = np.array([bins, E_spec_vel, N_spec_vel, Z_spec_vel],dtype=object)
        file_data = file_data.T
        outfile.write(f'#bins \t \t vel_spec_E_m \t vel_spec_N_m \t vel_spec_Z_m \n')
        np.savetxt(outfile, file_data, fmt=['%E', '%E', '%E', '%E'], delimiter='\t')
        outfile.close()

    
        ############################## Dataframe ##############################
        
        # Crete dictionary 
        dataset_dict = {'eventname':eventnames,'country':countries, 'origintime':origintimes,
                        'hyplon':hyplons, 'hyplat':hyplats, 'hypdepth (km)':hypdepths,
                        'mw':mws, 'm0':m0s, 'network':networks, 'station':stations,
                        'station_type':stn_type_list, 'stlon':stlons, 'stlat':stlats, 'stelev':stelevs,
                        'mechanism':mechanisms, 'hypdist':hypdist_list, 'duration_e':E_Td_list,
                        'duration_n':N_Td_list, 'duration_z':Z_Td_list, 'duration_horiz':horiz_Td_list,
                        'duration_3comp':comp3_Td_list, 'pga':pga_list, 'pgv':pgv_list,
                        'tPGA_origin':tPGA_orig_list, 'tPGA_parriv':tPGA_parriv_list}
        
        # Create and save dataframe
        main_df = pd.DataFrame(data=dataset_dict)
        main_df.to_csv(sm_flatfile_path,index=False)


