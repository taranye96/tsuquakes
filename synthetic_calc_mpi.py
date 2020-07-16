#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:13:49 2020

@author: tnye
"""

# Imports
import time
from mpi4py import MPI
from stringcolor import *
from glob import glob
from os import makedirs
import numpy as np
from numpy import genfromtxt
import pandas as pd
from math import ceil
from obspy import read
import tsueqs_main_fns as tmf
import signal_average_fns as avg
import IM_fns
import sys


start_time = time.time()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# ncpus = 4

# parameter = 'stress_drop_runs'
# project = 'sd0.3_etal_standard'

ncpus = sys.argv[0]
parameter = sys.argv[1]
project = sys.argv[2]

param_dir = f'/Users/tnye/FakeQuakes/{parameter}/{project}/' 

rupture_list = genfromtxt(f'/Users/tnye/FakeQuakes/{parameter}/{project}/disp/data/ruptures.list',dtype='U')

# runs = ['run.000000', 'run.000001', 'run.000002', 'run.000003', 'run.000004',
#         'run.000005', 'run.000006', 'run.000007', 'run.000008', 'run.000009',
#         'run.000010', 'run.000011', 'run.000012', 'run.000013', 'run.000014',
#         'run.000015']

data_types = ['disp','sm']
# data_types = ['sm']


############################# Start Parallelization ###########################

# Set up full array of data on main process 
if rank == 0:
    fulldata = np.arange(len(runs), dtype=int)
    # print(cs(f"I'm {rank} and fulldata is: {fulldata}", "OrangeRed"))
else:
    fulldata=None

# Number of items on each process
count = len(runs)//ncpus

# Set up empty array for each process to receive data
subdata = np.empty(count, dtype=int)

# Scatter data
comm.Scatter(fulldata,subdata,root=0)
# print(cs(f"After Scatter, I'm {rank} and my data is: {subdata}", "Pink5"))


############################### Do Calculations ###############################

### Set paths and parameters #### 

# Set up folder for flatfile
makedirs(f'/Users/tnye/tsuquakes/flatfiles/{parameter}/{project}')

# Data directory                          
data_dir = '/Users/tnye/tsuquakes/data'

# Table of earthquake data
eq_table_path = '/Users/tnye/tsuquakes/data/misc/events.csv'   
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

for index in subdata:
    rupture = rupture_list[index]
    run = rupture.rsplit('.', 1)[0]
    # print(cs(f"Processor {rank} beginning {run}...", "Gold"))
    print(f"Processor {rank} beginning {run}")
    
    # Set up folders for fourier plots
    makedirs(f'/Users/tnye/tsuquakes/plots/fourier_spec/synthetic/{parameter}/{project}/{run}')
    makedirs(f'/Users/tnye/tsuquakes/plots/fourier_spec/synthetic/{parameter}/{project}/{run}/acc')
    makedirs(f'/Users/tnye/tsuquakes/plots/fourier_spec/synthetic/{parameter}/{project}/{run}/vel')
    makedirs(f'/Users/tnye/tsuquakes/plots/fourier_spec/synthetic/{parameter}/{project}/{run}/disp')
    
    # Synthetic miniseed dir
    disp_dir = param_dir + f'disp/output/waveforms/{run}/'
    sm_dir = param_dir + f'sm/output/waveforms/{run}/'
    
    # Gather displacement and strong motion files
    disp_files = np.array(sorted(glob(disp_dir + '*.sac')))
    sm_files = np.array(sorted(glob(sm_dir + '*.bb*.sac')))
    
    # Path to send flatfile
    flatfile_path = f'/Users/tnye/tsuquakes/flatfiles/{parameter}/{project}/{run}.csv'
    
    # Filtering
    threshold = 0.0
    fcorner = 1/15.                          # Frequency at which to high pass filter
    order = 2                                # Number of poles for filter  
    
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
    hypdists = np.array([])
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
    
    ##################### Data Processing and Calculations ####################
    
    for data in data_types:

        if data == 'disp':
            metadata_file = data_dir + '/' + eventname + '/' + eventname + '_disp.chan'
            files = disp_files
            IMs = ['pgd']
            filtering = False
            
#            print(cs(f'Rank {rank} beginning {run} {data} processing...', 'DeepPink6'))
            
        elif data == 'sm':
            metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
            files = sm_files
            IMs = ['pga', 'pgv']
            filtering = True
            
 #           print(cs(f'Rank {rank} beginning {run} {data} processing...', 'Chartreuse3'))

        metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                              names=['net', 'sta', 'loc', 'chan', 'lat',
                                      'lon', 'elev', 'samplerate', 'gain', 'units'])
    
        # There might be white spaces in the station name, so remove those
        metadata.sta = metadata.sta.astype(str)
        metadata.sta = metadata.sta.str.replace(' ','')

        # Create lists to add station names, channels, and miniseed files to 
        stn_name_list = []
        channel_list = []
        mseed_list = []
        
        # Group all files by station
        N = 3
        stn_files = [files[n:n+N] for n in range(0, len(files), N)]
        
        # Loop over files to get the list of station names, channels, and mseed files 
        for station in stn_files:
            components = []
            mseeds = []
        
            stn_name = station[0].split('/')[-1].split('.')[0]
            stn_name_list.append(stn_name)
            
            for mseed_file in station:
                if data == 'disp':
                    channel_code = mseed_file.split('/')[-1].split('.')[1]
                elif data == 'sm':
                    channel_code = mseed_file.split('/')[-1].split('.')[2]
        
                components.append(channel_code)
                mseeds.append(mseed_file)
        
            channel_list.append(components)
            mseed_list.append(mseeds)
        
        # Loop over the stations for this earthquake, and start to run the computations:
        if data == 'disp':
            color = 'DeepPink6'
        elif data =='sm':
            color = 'Chartreuse3'
#        print(cs(f'Rank {rank} looping through {run} {data} stations...', color))
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
            stnetwork = station_metadata.loc[0].net
            stlon = station_metadata.loc[0].lon
            stlat = station_metadata.loc[0].lat
            stelev = station_metadata.loc[0].elev
            stsamprate = station_metadata.loc[0].samplerate
            stgain = station_metadata.loc[0].gain
  
    
            ##################### Start computations ######################        
#            print(cs(f'Rank {rank} beginning {run} computations...', 'SlateBlue2'))
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
            stations = np.append(stations,station)
            stlons = np.append(stlons,stlon)
            stlats = np.append(stlats,stlat)
            stelevs = np.append(stelevs,stelev)
            hypdists = np.append(hypdists,hypdist)
            if data == 'disp':
                stn_type_list = np.append(stn_type_list, 'GNSS')
            elif data == 'sm':
                stn_type_list = np.append(stn_type_list, 'SM')
            
            # List for all spectra at station
            station_spec = []
    
            # Get the components
            components = np.asarray(components)
            
            ########## East component ##########
    
            # Find and read in East component
            E_index = np.where(components=='E')[0][0]
            E_raw = read(mseed_list[i][E_index])
    
            # High pass filter strong motion data at fcorner specified above
            if filtering == True:
                E_filt = tmf.highpass(E_raw,fcorner,stsamprate,order,zerophase=True)
                E_record = E_filt
            else:
                E_record = E_raw
            
            # Get the duration, stream file time of start, and time of stop of shaking
            E_Td, E_start, E_end = tmf.determine_Td(threshold,E_record)      
            E_Td_list = np.append(E_Td_list,E_Td)
    
            ########## North component ##########
    
            # Find and read in North component 
            N_index = np.where(components=='N')[0][0]
            N_raw = read(mseed_list[i][N_index])
    
            # High pass filter strong motion data at fcorner specified above
            if filtering == True:
                N_filt = tmf.highpass(N_raw,fcorner,stsamprate,order,zerophase=True)
                N_record = N_filt
            else:
                N_record = N_raw
            
            # Get the duration, stream file time of start, and time of stop of shaking
            N_Td, N_start, N_end = tmf.determine_Td(threshold,N_record)      
            N_Td_list = np.append(N_Td_list,N_Td)
    
            ########## Vertical component ##########
    
            # Find and read in vertical component
            Z_index = np.where(components=='Z')[0][0]                       
            Z_raw = read(mseed_list[i][Z_index])
    
            # High pass filter strong motion data at fcorner specified above
            if filtering == True:
                Z_filt = tmf.highpass(Z_raw,fcorner,stsamprate,order,zerophase=True)
                Z_record = Z_filt
            else:
                Z_record = Z_raw
            
            # Get the duration, stream file time of start, and time of stop of shaking
            Z_Td, Z_start, Z_end = tmf.determine_Td(threshold,Z_record)  
            Z_Td_list = np.append(Z_Td_list,Z_Td)    
            
            ### Velocity 
            
            if data == 'sm':
                # # Convert acceleration record to velocity (no HP filter)
                # E_vel_unfilt = tmf.accel_to_veloc(E_raw)
                # N_vel_unfilt = tmf.accel_to_veloc(N_raw)
                # Z_vel_unfilt = tmf.accel_to_veloc(Z_raw)
    
                # Convert acceleration record to velocity (HP filter)
                E_vel_unfilt = tmf.accel_to_veloc(E_record)
                N_vel_unfilt = tmf.accel_to_veloc(N_record)
                Z_vel_unfilt = tmf.accel_to_veloc(Z_record)
                
                # High pass filter velocity 
                E_vel = tmf.highpass(E_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
                N_vel = tmf.highpass(N_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
                Z_vel = tmf.highpass(Z_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
    
    
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
    
    
            ######################### Intensity Measures ######################
            if data == 'disp':
                # print(cs(f'Rank {rank}', rc), cs(f'beginning disp IMs for {run}', 'Pink'), cs(f'({station})', 'Khaki'))
                print(f'....Processor {rank} working on disp IMs for {run} {station}')
                ## PGD
                # Get euclidean norm of displacement components 
                euc_norm = avg.get_eucl_norm_3comp(E_record[0].data,
                                                    N_record[0].data,
                                                    Z_record[0].data)
                # Calculate PGD
                pgd = np.max(np.abs(euc_norm))
                pgd_list = np.append(pgd_list,pgd)
                # Calcualte tPGD from origin and p-arrival
                tPGD_orig, tPGD_parriv = IM_fns.calc_tPGD(pgd, E_record[0],
                                                                np.abs(euc_norm),
                                                                origin, hypdist)
                tPGD_orig_list = np.append(tPGD_orig_list,tPGD_orig)
                tPGD_parriv_list = np.append(tPGD_parriv_list,tPGD_parriv)
    
                ## Disp Spectra
                E_spec_data, freqE, ampE = IM_fns.calc_spectra(E_record, data)
                N_spec_data, freqN, ampN = IM_fns.calc_spectra(N_record, data)
                Z_spec_data, freqZ, ampZ = IM_fns.calc_spectra(Z_record, data)
                # Combine into one array and append to main list
                disp_spec = np.concatenate([E_spec_data,N_spec_data,Z_spec_data])
                disp_speclist.append(disp_spec.tolist())
                # Plot spectra
                freqs = [freqE,freqN,freqZ]
                amps = [ampE,ampN,ampZ]
                IM_fns.plot_spectra(E_record, freqs, amps, 'disp', parameter=parameter, project=project, run=run)
    
            else:
                pgd_list = np.append(pgd_list,np.nan)
                tPGD_orig_list = np.append(tPGD_orig_list,np.nan)
                tPGD_parriv_list = np.append(tPGD_parriv_list,np.nan)
                disp_spec = [np.nan] * 75
                disp_speclist.append(disp_spec)
                
            if data == 'sm':
                # print(cs(f'Rank {rank}', rc), cs(f'beginning acc IMs for {run}', 'SpringGreen3'), cs(f'({station})', 'Khaki'))
                print(f'....Processor {rank} working on acc IMs for {run} {station}')
                ## PGA         
                # Get euclidean norm of acceleration components 
                acc_euc_norm = avg.get_eucl_norm_3comp(E_record[0].data,
                                                   N_record[0].data,
                                                   Z_record[0].data)
                
                # Calculate PGA
                pga = np.max(np.abs(acc_euc_norm))
                pga_list = np.append(pga_list,pga)
                # Calcualte tPGD from origin and p-arrival
                tPGA_orig, tPGA_parriv = IM_fns.calc_tPGD(pga, E_record[0],
                                                                np.abs(acc_euc_norm),
                                                                origin, hypdist)
                tPGA_orig_list = np.append(tPGA_orig_list,tPGA_orig)
                tPGA_parriv_list = np.append(tPGA_parriv_list,tPGA_parriv)
    
                ## Acc Spectra
                E_spec_data, freqE, ampE = IM_fns.calc_spectra(E_record, data)
                N_spec_data, freqN, ampN = IM_fns.calc_spectra(N_record, data)
                Z_spec_data, freqZ, ampZ = IM_fns.calc_spectra(Z_record, data)
                # Combine into one array and append to main list
                acc_spec = np.concatenate([E_spec_data,N_spec_data,Z_spec_data])
                acc_speclist.append(acc_spec.tolist())
                # Plot spectra
                freqs = [freqE,freqN,freqZ]
                amps = [ampE,ampN,ampZ]
                IM_fns.plot_spectra(E_record, freqs, amps, 'acc', parameter=parameter, project=project, run=run)
    
                # print(cs(f'Rank {rank}', rc), cs(f'beginning vel IMs for {run}', 'DodgerBlue'), cs(f'({station})', 'Khaki'))
                print(f'....Processor {rank} working on vel IMs for {run} {station}')
                ## PGV
                # Convert acceleration record to velocity 
                E_vel = tmf.accel_to_veloc(E_record)
                N_vel = tmf.accel_to_veloc(N_record)
                # Calculate PGV on each component
                pgv_E = np.max(np.abs(E_vel))
                pgv_N = np.max(np.abs(N_vel))
                # Get geometric average of the PGVs
                pgv = tmf.get_geom_avg(pgv_E,pgv_N)
                pgv_list = np.append(pgv_list,pgv)
                
                ## Vel Spectra
                E_spec_vel, freqE_v, ampE_v = IM_fns.calc_spectra(E_vel, data)
                N_spec_vel, freqN_v, ampN_v = IM_fns.calc_spectra(N_vel, data)
                Z_spec_vel, freqZ_v, ampZ_v = IM_fns.calc_spectra(Z_vel, data)
                # Combine into one array and append to main list
                vel_spec = np.concatenate([E_spec_data,N_spec_data,Z_spec_data])
                vel_speclist.append(vel_spec.tolist())
                # Plot spectra
                freqs_v = [freqE_v,freqN_v,freqZ_v]
                amps_v = [ampE_v,ampN_v,ampZ_v]
                IM_fns.plot_spectra(E_vel, freqs_v, amps_v, 'vel', parameter=parameter, project=project, run=run)
    
            else:
                pga_list = np.append(pga_list,np.nan)
                pgv_list = np.append(pgv_list,np.nan)
                tPGA_orig_list = np.append(tPGA_orig_list,np.nan)
                tPGA_parriv_list = np.append(tPGA_parriv_list,np.nan)
                
                acc_spec = [np.nan] * 75
                acc_speclist.append(acc_spec)
                vel_spec = [np.nan] * 75
                vel_speclist.append(vel_spec)
                            

    
    ## Now, put all the final arrays together into a pandas dataframe. First mak a dict:
    dataset_dict = {'eventname':eventnames,'country':countries, 'origintime':origintimes,
                    'hyplon':hyplons, 'hyplat':hyplats, 'hypdepth (km)':hypdepths,
                    'mw':mws, 'm0':m0s, 'network':networks, 'station':stations,
                    'stlon':stlons, 'stlat':stlats, 'stelev':stelevs,
                    'mechanism':mechanisms, 'hypdist':hypdists, 'duration_e':E_Td_list,
                    'duration_n':N_Td_list, 'duration_z':Z_Td_list, 'duration_horiz':horiz_Td_list,
                    'duration_3comp':comp3_Td_list, 'pgd':pgd_list, 'pga':pga_list, 'pgv':pgv_list,
                    'tPGD_origin':tPGD_orig_list, 'tPGD_parriv':tPGD_parriv_list,
                    'tPGA_origin':tPGA_orig_list, 'tPGA_parriv':tPGA_parriv_list}
    

    disp_cols = ['E_disp_bin1', 'E_disp_bin2', 'E_disp_bin3', 'E_disp_bin4',
                  'E_disp_bin5', 'E_disp_bin6', 'E_disp_bin7',
                  'E_disp_bin8', 'E_disp_bin9', 'E_disp_bin10', 'E_disp_bin11',
                  'E_disp_bin12', 'E_disp_bin13', 'E_disp_bin14', 'E_disp_bin15',
                  'E_disp_bin16', 'E_disp_bin17', 'E_disp_bin18', 'E_disp_bin19',
                  'E_disp_bin20', 'E_disp_bin21', 'E_disp_bin22', 'E_disp_bin23', 
                  'E_disp_bin24', 'E_disp_bin25', 'N_disp_bin1', 'N_disp_bin2',
                  'N_disp_bin3', 'N_disp_bin4', 'N_disp_bin5', 'N_disp_bin6',
                  'N_disp_bin7', 'N_disp_bin8', 'N_disp_bin9', 'N_disp_bin10',
                  'N_disp_bin11', 'N_disp_bin12', 'N_disp_bin13', 'N_disp_bin14',
                  'N_disp_bin15', 'N_disp_bin16', 'N_disp_bin17', 'N_disp_bin18',
                  'N_disp_bin19', 'N_disp_bin20', 'N_disp_bin21', 'N_disp_bin22',
                  'N_disp_bin23', 'N_disp_bin24', 'N_disp_bin25', 'Z_disp_bin1',
                  'Z_disp_bin2', 'Z_disp_bin3', 'Z_disp_bin4', 'Z_disp_bin5',
                  'Z_disp_bin6', 'Z_disp_bin7', 'Z_disp_bin8', 'Z_disp_bin9',
                  'Z_disp_bin10', 'Z_disp_bin11', 'Z_disp_bin12', 'Z_disp_bin13',
                  'Z_disp_bin14', 'Z_disp_bin15', 'Z_disp_bin16', 'Z_disp_bin17',
                  'Z_disp_bin18', 'Z_disp_bin19', 'Z_disp_bin20', 'Z_disp_bin21',
                  'Z_disp_bin22', 'Z_disp_bin23', 'Z_disp_bin24', 'Z_disp_bin25']
    
    acc_cols = ['E_acc_bin1', 'E_acc_bin2', 'E_acc_bin3', 'E_acc_bin4',
                'E_acc_bin5', 'E_acc_bin6', 'E_acc_bin7', 'E_acc_bin8',
                'E_acc_bin9', 'E_acc_bin10', 'E_acc_bin11', 'E_acc_bin12',
                'E_acc_bin13', 'E_acc_bin14', 'E_acc_bin15', 'E_acc_bin16',
                'E_acc_bin17', 'E_acc_bin18', 'E_acc_bin19', 'E_acc_bin20',
                'E_acc_bin21', 'E_acc_bin22', 'E_acc_bin23', 'E_acc_bin24',
                'E_acc_bin25', 'N_acc_bin1', 'N_acc_bin2', 'N_acc_bin3',
                'N_acc_bin4', 'N_acc_bin5', 'N_acc_bin6', 'N_acc_bin7',
                'N_acc_bin8', 'N_acc_bin9', 'N_acc_bin10', 'N_acc_bin11',
                'N_acc_bin12', 'N_acc_bin13', 'N_acc_bin14', 'N_acc_bin15',
                'N_acc_bin16', 'N_acc_bin17', 'N_acc_bin18', 'N_acc_bin19',
                'N_acc_bin20', 'N_acc_bin21', 'N_acc_bin22', 'N_acc_bin23',
                'N_acc_bin24', 'N_acc_bin25', 'Z_acc_bin1', 'Z_acc_bin2',
                'Z_acc_bin3', 'Z_acc_bin4', 'Z_acc_bin5', 'Z_acc_bin6',
                'Z_acc_bin7', 'Z_acc_bin8', 'Z_acc_bin9', 'Z_acc_bin10',
                'Z_acc_bin11', 'Z_acc_bin12', 'Z_acc_bin13', 'Z_acc_bin14',
                'Z_acc_bin15', 'Z_acc_bin16', 'Z_acc_bin17', 'Z_acc_bin18',
                'Z_acc_bin19', 'Z_acc_bin20', 'Z_acc_bin21', 'Z_acc_bin22',
                'Z_acc_bin23', 'Z_acc_bin24', 'Z_acc_bin25']
    
    vel_cols = ['E_vel_bin1', 'E_vel_bin2', 'E_vel_bin3', 'E_vel_bin4',
                'E_vel_bin5', 'E_vel_bin6', 'E_vel_bin7', 'E_vel_bin8',
                'E_vel_bin9', 'E_vel_bin10', 'E_vel_bin11', 'E_vel_bin12',
                'E_vel_bin13', 'E_vel_bin14', 'E_vel_bin15', 'E_vel_bin16',
                'E_vel_bin17', 'E_vel_bin18', 'E_vel_bin19', 'E_vel_bin20',
                'E_vel_bin21', 'E_vel_bin22', 'E_vel_bin23', 'E_vel_bin24',
                'E_vel_bin25', 'N_vel_bin1', 'N_vel_bin2', 'N_vel_bin3',
                'N_vel_bin4', 'N_vel_bin5', 'N_vel_bin6', 'N_vel_bin7',
                'N_vel_bin8', 'N_vel_bin9', 'N_vel_bin10', 'N_vel_bin11',
                'N_vel_bin12', 'N_vel_bin13', 'N_vel_bin14', 'N_vel_bin15',
                'N_vel_bin16', 'N_vel_bin17', 'N_vel_bin18', 'N_vel_bin19',
                'N_vel_bin20', 'N_vel_bin21', 'N_vel_bin22', 'N_vel_bin23',
                'N_vel_bin24', 'N_vel_bin25', 'Z_vel_bin1', 'Z_vel_bin2',
                'Z_vel_bin3', 'Z_vel_bin4', 'Z_vel_bin5', 'Z_vel_bin6',
                'Z_vel_bin7', 'Z_vel_bin8', 'Z_vel_bin9', 'Z_vel_bin10',
                'Z_vel_bin11', 'Z_vel_bin12', 'Z_vel_bin13', 'Z_vel_bin14',
                'Z_vel_bin15', 'Z_vel_bin16', 'Z_vel_bin17', 'Z_vel_bin18',
                'Z_vel_bin19', 'Z_vel_bin20', 'Z_vel_bin21', 'Z_vel_bin22',
                'Z_vel_bin23', 'Z_vel_bin24', 'Z_vel_bin25']

    
    disp_spec_df = pd.DataFrame(disp_speclist, columns=disp_cols)
    acc_spec_df = pd.DataFrame(acc_speclist, columns=acc_cols)
    vel_spec_df = pd.DataFrame(vel_speclist, columns=vel_cols)
    
    # Make main dataframe
    main_df = pd.DataFrame(data=dataset_dict)
    
    
    # Combine dataframes 
    flatfile_df = pd.concat([main_df, disp_spec_df.reindex(main_df.index),
                            acc_spec_df.reindex(main_df.index),
                            vel_spec_df.reindex(main_df.index)], axis=1)
    
    ## Save to file:
    flatfile_df.to_csv(flatfile_path,index=False)


############################### End Parallelization ###########################

# print(f'Rank: {rank}, sendbuf: {subdata}')

#Set up empty array to gather data on
recvbuf=None
if rank == 0:
    # recvbuf = np.empty(count*size, dtype='d')
    recvbuf = np.empty(count*size, dtype=int)

comm.Gather(subdata, recvbuf, root=0)
# print(cs(f"After Gather, I'm {rank} and my data is: {recvbuf}", "Pink5"))

total_time = (time.time() - start_time)
if total_time < 60:
    print(cs(f"--- Total duration for {len(runs)} runs on {ncpus} CPUS is ~{round(total_time)} seconds ---", 'SandyBrown'))
elif total_time >= 60 and total_time < 3600:
    print(cs(f"--- Total duration for {len(runs)} runs on {ncpus} CPUS is ~{round(total_time/60)} minutes ---", 'SandyBrown'))
elif total_time >= 3600 and total_time < 86400:
    print(cs(f"--- Total duration for {len(runs)} runs on {ncpus} CPUS is ~{round(total_time/3600)} hours ---", 'SandyBrown'))
else:
    print(cs(f"--- Total duration for {len(runs)} runs on {ncpus} CPUS is ~{round(total_time/86400)} days ---", 'SandyBrown'))
