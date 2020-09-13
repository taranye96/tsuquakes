#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:45:11 2020

@author: tnye
"""

###############################################################################
# Script that calculates and plots synthetic disp, acc, and vel waveforms a
# against observed
###############################################################################

# Imports
from mpi4py import MPI
from glob import glob
import numpy as np
from numpy import genfromtxt
import pandas as pd
from obspy import read
import tsueqs_main_fns as tmf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from os import makedirs, path

# MPI parameters
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# ncpus = 4
ncpus = size

# Parameters
parameter = 'rise_time'
project = 'rt1.5x'
data_types = ['disp', 'acc', 'vel']

rupture_list = genfromtxt(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/disp/data/ruptures.list',dtype='U')


############################# Start Parallelization ###########################

# Set up full array of data on main process 
if rank == 0:
    fulldata = np.arange(len(rupture_list), dtype=int)
    # print(cs(f"I'm {rank} and fulldata is: {fulldata}", "OrangeRed"))
else:
    fulldata=None

# Number of items on each process
count = len(rupture_list)//ncpus

# Set up empty array for each process to receive data
subdata = np.empty(count, dtype=int)

# Scatter data
comm.Scatter(fulldata,subdata,root=0)


################################## Start Plots ################################

# Make sure path to save plots to exists
if not path.exists(f'/Users/tnye/tsuquakes/plots/waveform_comp/{parameter}/{project}'):
    makedirs(f'/Users/tnye/tsuquakes/plots/waveform_comp/{parameter}/{project}')
        
# Loop through projects
for index in subdata:
    rupture = rupture_list[index]
    run = rupture.rsplit('.', 1)[0]
        
    # Loop through data types
    for data in data_types:

        ### Set paths and parameters #### 
        
        # Project directory 
        proj_dir = f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/' 
        
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
        disp_syn_dir = proj_dir + f'disp/output/waveforms/{run}/'
        sm_syn_dir = proj_dir + f'sm/output/waveforms/{run}/'
        disp_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'
        acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr'
        vel_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr'
        
        # Filtering
        threshold = 0.0
        fcorner = 1/15.      # Frequency at which to high pass filter
        order = 2            # Number of poles for filter  
        
        ##################### Data Processing and Calculations ####################
        
        # Set parameters based on data type
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
            filtering = True
            dtype = 'sm'
        
        metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                              names=['net', 'sta', 'loc', 'chan', 'lat',
                                     'lon', 'elev', 'samplerate', 'gain', 'units'])
        
        # There might be white spaces in the station name, so remove those
        metadata.sta = metadata.sta.astype(str)
        metadata.sta = metadata.sta.str.replace(' ','')
        
        # Initialize lists to save waveform times and amplitudes to
        syn_times = []
        syn_amps = []
        obs_times = []
        obs_amps = []
        hypdists = []
        
        
        ############################# Synthetics ##########################
        
        # Create lists to add station names, channels, and miniseed files to 
        stn_name_list = []
        channel_list = []
        mseed_list = []
        
        # Group all files by station
        N = 3
        stn_files = [syn_files[n:n+N] for n in range(0, len(syn_files), N)]
        
        # Loop over files to get the list of station names, channels, and mseed files 
        for station in stn_files:
            components = []
            mseeds = []
        
            # Obtain station name from filename and append to list
            stn_name = station[0].split('/')[-1].split('.')[0]
            stn_name_list.append(stn_name)
            
            # Loop stations, append all mseed files to a list, and append 
                # all channel codes to a list
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
            mseed_list.append(mseeds)
        
        # Loop over the stations for this earthquake, and start to run the computations:
        for i, station in enumerate(stn_name_list):
            
            # Get the instrument (HN or LX) and component (E,N,Z) for this station
            components = []
            for channel in channel_list[i]:
                components.append(channel[2])
        
                
            # Get the metadata for this station from the chan file - put it into
                # a new dataframe and reset the index so it starts at 0
            if country == 'Japan':
                station_metadata = metadata[(metadata.net == station[0:2]) &
                                            (metadata.sta == station[2:])].reset_index(drop=True)
                
            else:
                station_metadata = metadata[metadata.sta == station].reset_index(drop=True)       # what is going on here
                  
          
            # Pull out the station data. Take the first row of the subset dataframe, 
                # assuming that the gain, etc. is always the same:
            stlon = station_metadata.loc[0].lon
            stlat = station_metadata.loc[0].lat
            stelev = station_metadata.loc[0].elev
            stsamprate = station_metadata.loc[0].samplerate
          
        
            ##################### Start computations ######################        
        
            # Compute hypocentral distance
            hypdist = tmf.compute_rhyp(stlon,stlat,stelev,hyplon,hyplat,hypdepth)
        
            # Turn list of components into an array
            components = np.asarray(components)
            
            ## East component 
        
            # Find and read in East component
            E_index = np.where(components=='E')[0][0]
            E_raw = read(mseed_list[i][E_index])
        
            # High pass filter strong motion data at fcorner specified above
            if filtering == True:
                E_filt = tmf.highpass(E_raw,fcorner,stsamprate,order,zerophase=True)
                E_record = E_filt
            else:
                E_record = E_raw
        
            
            ### Velocity 
            
            if data == 'vel':
                # Convert acceleration record to velocity 
                E_raw = tmf.accel_to_veloc(E_record)
        
                # High pass filter velocity 
                E_record = tmf.highpass(E_raw,fcorner,stsamprate,order,zerophase=True)
        
        
            ############################ Waveforms ###############W########
            
            # Get trace (just using E component)
            tr = E_record[0]
            station = tr.stats.station 
            
            # Append trace data, times, and hypocentral distance to lists
            syn_times.append(tr.times('matplotlib').tolist())
            syn_amps.append(tr.data.tolist())
            hypdists.append(hypdist)
            
        
        ############################## Observed ###########################
        
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
        
            # Obtain station name from filename and append to list
            stn_name = station[0].split('.')[0].split('/')[-1]
            if stn_name != 'SISI':
                stn_name_list.append(stn_name)
                
                # Loop stations, append all mseed files to a list, and append 
                    # all channel codes to a list
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
        
        
            ####################### Start computations ####################     
        
            # Turn list of components into an array
            components = np.asarray(components)
                
            # Get index for E component 
            E_index = np.where(components=='E')[0][0]
            # Read file into a stream object
            E_record = read(mseed_list[i][E_index])
        
        
            ########################## Waveforms ##########################
        
            # Get trace (just using E component)
            tr = E_record[0]
            station = tr.stats.station 
            
            # Append trace data and times to lists
            obs_times.append(tr.times('matplotlib').tolist())
            obs_amps.append(tr.data.tolist())     
            
        
        ############################ Make Figure ##########################
        
        # Set figure parameters based on data type
        if data == 'disp':
               units = 'm'
               channel = 'LX' 
               dim = 5,3
               figsize = 10,20
        elif data == 'acc':
               units = 'm/s/s'
               channel = 'HN'
               dim = 6,3
               figsize = 10,30
        elif data == 'vel':
               units = 'm/s'
               channel = 'HN'
               dim = 6,3
               figsize = 10,30
        
        # Sort hypdist and get sorted indices
        sort_id = np.argsort(hypdists)
        sort_hypdists = np.sort(hypdists)
        
        # Function to sort list based on list of indices 
        def sort_list(list1, list2): 
            zipped_pairs = zip(list2, list1) 
            z = [x for _, x in sorted(zipped_pairs)] 
            return z 
        
        # Sort times and amps based off hypdist
        sort_syn_times = sort_list(syn_times, sort_id)
        sort_syn_amps = sort_list(syn_amps, sort_id)
        sort_obs_times = sort_list(obs_times, sort_id)
        sort_obs_amps = sort_list(obs_amps, sort_id)
        sort_stn_name = sort_list(stn_name_list, sort_id)
        
        # Make figure and subplots
        if data == 'disp':
            # Set up figure
            fig, axs = plt.subplots(dim[0],dim[1],figsize=figsize)
            k = 0     # subplot index
            # Loop rhough rows
            for i in range(dim[0]):
                # Loop through columns
                for j in range(dim[1]):
                    # Only make enough subplots for length of station list
                    if k+1 <= len(stn_name_list):
                        axs[i][j].plot(sort_syn_times[k],sort_syn_amps[k],
                                         color='C1',lw=0.4,label='synthetic')
                        axs[i][j].plot(sort_obs_times[k],sort_obs_amps[k],
                                         'k-',lw=0.4,label='observed')
                        axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                        axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                        axs[i][j].text(0.625,5E-2,f'Hypdist={int(sort_hypdists[k])}km',
                                        transform=axs[i][j].transAxes,size=7)
                        axs[i][j].text(0.025,5E-2,'LXE',transform=axs[i][j].transAxes,size=7)
                        if i < dim[0]-2:
                            axs[i][j].set_xticklabels([])
                        if i == dim[0]-2 and j == 0:
                            axs[i][j].set_xticklabels([])
                        k += 1
            fig.text(0.5, 0.005, 'UTC Time(hr:min:sec)', ha='center')
            fig.text(0.005, 0.5, f'Amplitude ({units})', va='center', rotation='vertical')
            axs_list = []
            handles, labels = axs[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc=(0.74,0.09), framealpha=None)
            if data == 'disp':
                    fig.delaxes(axs[4][1])
                    fig.delaxes(axs[4][2])
            else:
                    fig.delaxes(axs[5][1])
                    fig.delaxes(axs[5][2])
            fig.suptitle('Waveform Comparison', fontsize=12, y=1)
            fig.text(0.385, 0.135, (r"$\bf{" + 'Project:' + "}$" + '' + project))
            fig.text(0.385, 0.115, (r'$\bf{' + 'Run:' + '}$' + '' + run))
            fig.text(0.385, 0.09, (r'$\bf{' + 'DataType:' '}$' + '' + data))
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.075, top=0.925,
                                wspace=0.3, hspace=0.4)
    
            plt.savefig(f'/Users/tnye/tsuquakes/plots/waveform_comp/{parameter}/{project}/{run}_{data}.png', dpi=300)
            plt.close()
            
        else:
            # Set up figure
            fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,15))
            # fig, axs = plt.subplots(dim[0],dim[1])
            k = 0     # subplot index
            # Loop through rows
            for i in range(dim[0]):
                # Loop through columns 
                for j in range(dim[1]):
                    # Only make enough subplots for length of station list
                    if k+1 <= len(stn_name_list):
                        axs[i][j].plot(sort_syn_times[k],sort_syn_amps[k],
                                         color='C1',lw=0.4,label='synthetic')
                        axs[i][j].plot(sort_obs_times[k],sort_obs_amps[k],
                                         'k-',lw=0.4,label='observed')
                        axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                        axs[i][j].text(0.7,5E-13,f'Hypdist={int(sort_hypdists[k])}km',size=7)
                        axs[i][j].text(0.025,5E-2,'HNE',transform=axs[i][j].transAxes,size=7)
                        axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                        if i < dim[0]-2:
                            axs[i][j].set_xticklabels([])
                        if i == dim[0]-2 and j == 0:
                            axs[i][j].set_xticklabels([])
                        k += 1
            fig.text(0.51, 0.005, 'UTC Time(hr:min:sec))', ha='center')
            fig.text(0.005, 0.5, f'Amplitude ({units})', va='center', rotation='vertical')
            axs_list = []
            handles, labels = axs[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc=(0.74,0.08), framealpha=None)
            if data == 'disp':
                    fig.delaxes(axs[4][1])
                    fig.delaxes(axs[4][2])
            else:
                    fig.delaxes(axs[5][1])
                    fig.delaxes(axs[5][2])
            fig.suptitle('Waveform Comparison', fontsize=12, y=1)
            fig.text(0.435, 0.125, (r"$\bf{" + 'Project:' + "}$" + '' + project))
            fig.text(0.435, 0.105, (r'$\bf{' + 'Run:' + '}$' + '' + run))
            fig.text(0.435, 0.08, (r'$\bf{' + 'DataType:' '}$' + '' + data))
            # fig.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.075, top=0.925,
                                wspace=0.4, hspace=0.4)
    
            plt.savefig(f'/Users/tnye/tsuquakes/plots/waveform_comp/{parameter}/{project}/{run}_{data}.png', dpi=300)
            plt.close()
            