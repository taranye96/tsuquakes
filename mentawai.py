#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:16:28 2019

@author: tnye
"""

###############################################################################
# Script that goes through observed waveforms from the 2010 M7.8 Mentawai event,
# calculates intensity measures, and stores it all in a flatefile.  The IMs
# this script calculates are:
    # PGD
    # PGA
    # PGV
    # Displacement spectra bin averages for 20 bins
    # Acceleration spectra bin averages for 20 bins
    # Velocity spectra bin averages for 20 bins
###############################################################################

# Imports
import numpy as np
import pandas as pd
import obspy
from glob import glob

# Local Imports
import tsueqs_main_fns as tmf
import signal_average_fns as avg
import IM_fns

################################ Parameters ###################################

# Used for directory paths
earthquake_name = 'Mentawai2010'

# Data types to loop through.  I have a folder for displacement ('disp') and a 
    # folder for acceleration ('accel'), so those are my data types. 
data_types = ['disp', 'accel']

# Project directory 
proj_dir = '/Users/tnye/tsuquakes' 

# Table of earthquake data
eq_table_path = '/Users/tnye/tsuquakes/data/misc/events.csv'   
eq_table = pd.read_csv(eq_table_path)

# Data directories- one for displacement and one for strong motion (acc)     
data_dir = proj_dir + '/data' 
disp_dir = data_dir + '/' + earthquake_name + '/disp'
sm_dir = data_dir + '/' + earthquake_name + '/accel'

# Path to send flatfiles of intensity measures
flatfile_path = proj_dir + '/flatfiles/obs_IMs2.csv'     

# Parameters for integration to velocity and filtering 
fcorner = 1/15.                          # Frequency at which to high pass filter
order = 2                                # Number of poles for filter  

# Gather displacement and strong motion files
disp_files = np.array(sorted(glob(disp_dir + '/*.mseed')))
sm_files = np.array(sorted(glob(sm_dir + '/*.mseed')))


################################ Event Data ###################################

origin = pd.to_datetime('2010-10-25T14:42:22')

eventname = earthquake_name
country = eq_table['Country'][11]
origintime = eq_table['Origin Time (UTC)*'][11]
hyplon = eq_table['Longitude'][11]
hyplat = eq_table['Latitude'][11]
hypdepth = eq_table['Depth (km)'][11]
mw = eq_table['Mw'][11]
m0 = 10**(mw*(3/2.) + 9.1)
nostations_i = eq_table['No. Sta'][11]
mechanism = eq_table['Mechanism'][11]


########## Initialize lists for the event and station info for the df #########

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
nostations = np.array([])
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
tPGA_list = np.array([])
disp_speclist = []
acc_speclist = []
vel_speclist = []


###################### Data Processing and Calculations #######################

# Threshold- used to calculate duration 
threshold = 0.0

# Loop through data types
for data in data_types:
    
    ###################### Set parameters for data type #######################
    
    if data == 'disp':
        
        # Get metadata file
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_disp.chan'
        
        # Get mseed files
        files = disp_files
        
        # Types of IMs associated with this data type
        IMs = ['pgd']
        
        # Number of samples to use in computing the pre-event baseline
        nsamples = 10
        
        # Channel code prefix
        code = 'LX'
        
        # Filtering
            # Displacement data don't need to be highpass filtered 
        filtering = False

    
    elif data == 'accel':
        
        # Get metadata file
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
        
        # Get mseed files
        files = sm_files
        
        # Remove SISI sm station (SNR too low)
        for file in files:
            if 'SISI' in file:
                files = np.delete(files, np.argwhere(files == file))
        
        # Types of IMs associated with this data type
        IMs = ['pga', 'pgv']
  
        # Number of samples to use in computing the pre-event baseline
        nsamples = 100
        
        # Channel code prefix
        code = 'HN'
        
        # Filtering
            # Acceleration data need to be highpass fitlered 
        filtering = True


    ############################# Get metadata ################################
    
    # Read in metadata file
    metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                          names=['net', 'sta', 'loc', 'chan', 'lat',
                                 'lon', 'elev', 'samplerate', 'gain', 'units'])
    
    # There might be white spaces in the station name, so remove those
    metadata.sta = metadata.sta.astype(str)
    metadata.sta = metadata.sta.str.replace(' ','')

    # Obtain gain and units
    gain = metadata['gain'][0]
    units = metadata['units'][0]
    
    
    ######################## Get station data and files #######################
    
    # Create lists to add station names, channels, and miniseed files to 
    stn_name_list = []
    channel_list = []
    mseed_list = []
    
    # Group all files by station since there should be 3 components for each 
        # station
    N = 3
    stn_files = [files[n:n+N] for n in range(0, len(files), N)]
    
    # Loop over files to get the list of station names, channels, and mseed files 
    for station in stn_files:
        
        # Initialize lists for components and mseed files for this station
        components = []
        mseeds = []
    
        # Get station name and append to station name list
        stn_name = station[0].split('.')[0].split('/')[-1]
        stn_name_list.append(stn_name)
        
        # Loop through station mseed files
        for mseed_file in station:
            
            # Get channel code and append to components list
            channel_code = mseed_file.split('/')[-1].split('.')[1]
            components.append(channel_code)
            
            # Append mseed file to mseed files list
            mseeds.append(mseed_file)
        
        # Append station's channel code list to channel list for all stations
        channel_list.append(components)
        # Append station's mseed files list to mseed files list for all stations
        mseed_list.append(mseeds)

    
    #################### Begin Processing and Calculations ####################
    
    # Loop over the stations for this earthquake, and start to run the computations:
    for i, station in enumerate(stn_name_list):
        
        # Get the components for this station (E, N, and Z):
        components = []
        for channel in channel_list[i]:
            components.append(channel[2])
            
        # Get the metadata for this station from the chan file - put it into
            # a new dataframe and reset the index so it starts at 0
        if country == 'Japan':
            station_metadata = metadata[(metadata.net == station[0:2]) & (metadata.sta == station[2:])].reset_index(drop=True)
            
        else:
            station_metadata = metadata[metadata.sta == station].reset_index(drop=True)

        # Pull out the data. Take the first row of the subset dataframe, 
            # assuming that the gain, etc. is always the same:
        stnetwork = station_metadata.loc[0].net
        stlon = station_metadata.loc[0].lon
        stlat = station_metadata.loc[0].lat
        stelev = station_metadata.loc[0].elev
        stsamprate = station_metadata.loc[0].samplerate
        stgain = station_metadata.loc[0].gain


        ######################### Start computations ##########################       

        # Compute the hypocentral distance
        hypdist = tmf.compute_rhyp(stlon,stlat,stelev,hyplon,hyplat,hypdepth)

        # Append the earthquake and station info for this station to their lists
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
        elif data == 'accel':
            stn_type_list = np.append(stn_type_list, 'SM')
        
        # Initialize list for all spectra at this station
        station_spec = []

        # Turn the components list into an array 
        components = np.asarray(components)
        
        # Get the values for the E component 
        if 'E' in components:
          
            # Get index for E component 
            E_index = np.where(components=='E')[0][0]
           
            # Read file into a stream object
            E_raw = obspy.read(mseed_list[i][E_index])
            
            # Correct by gain, so everything is in meters
            E_gaincorr = tmf.correct_for_gain(E_raw ,stgain)
            
            # Get the pre-event baseline
            E_baseline = tmf.compute_baseline(E_gaincorr,nsamples)
            
            # Get the baseline corrected stream object
            E_basecorr = tmf.correct_for_baseline(E_gaincorr,E_baseline)

            # High pass filter strong motion data at fcorner specified above
            if filtering == True:
                E_filt = tmf.highpass(E_basecorr,fcorner,stsamprate,order,zerophase=True)
                E_record = E_filt
            else:
                E_record = E_basecorr
            
            # Get the duration, stream file time of start, and time of stop of shaking
            E_Td, E_start, E_end = tmf.determine_Td(threshold,E_record)      
            E_Td_list = np.append(E_Td_list,E_Td)

            # Save corrected mseed file
            tra = E_record[0]
            tra.stats.channel = code + 'E'
            filename = '/Users/tnye/tsuquakes/data/Mentawai2010/' + data + '_corr/' + tra.stats.station + '.' + tra.stats.channel + '.mseed' 
            tra.write(filename, format='MSEED')

            
        # Get the values for the N component
        if 'N' in components:
           
            # Get index for N component 
            N_index = np.where(components=='N')[0][0]
            
            # Read file into a stream object
            N_raw = obspy.read(mseed_list[i][N_index])
            
            # Correct by gain, so everything is in meters
            N_gaincorr = tmf.correct_for_gain(N_raw,stgain)
            
            # Get the pre-event baseline
            N_baseline = tmf.compute_baseline(N_gaincorr,nsamples)
            
            # Get the baseline corrected stream object
            N_basecorr = tmf.correct_for_baseline(N_gaincorr,N_baseline)
            
            # High pass filter strong motion data at fcorner specified above
            if filtering == True:
                N_filt = tmf.highpass(N_basecorr,fcorner,stsamprate,order,zerophase=True)
                N_record = N_filt
            else:
                N_record = N_basecorr
                
            # Get the duration, stream file time of start, and time of stop of shaking
            N_Td, N_start, N_end = tmf.determine_Td(threshold,N_record)  
            N_Td_list = np.append(N_Td_list,N_Td)

            # Save corrected mseed file
            tra = N_record[0]
            tra.stats.channel = code + 'N'
            filename = '/Users/tnye/tsuquakes/data/Mentawai2010/' + data + '_corr/' + tra.stats.station + '.' + tra.stats.channel + '.mseed' 
            tra.write(filename, format='MSEED')


        # Get the values for the Z component
        if 'Z' in components:
         
            # Get index for Z component 
            Z_index = np.where(components=='Z')[0][0]     
           
            # Read file into a stream object                     
            Z_raw = obspy.read(mseed_list[i][Z_index])
            
            # Correct by gain, so everything is in meters
            Z_gaincorr = tmf.correct_for_gain(Z_raw,stgain)
            
            # Get the pre-event baseline
            Z_baseline = tmf.compute_baseline(Z_gaincorr,nsamples)
            
            # Get the baseline corrected stream object
            Z_basecorr = tmf.correct_for_baseline(Z_gaincorr,Z_baseline)
            
            # High pass filter strong motion data at fcorner specified above
            if filtering == True:
                Z_filt = tmf.highpass(Z_basecorr,fcorner,stsamprate,order,zerophase=True)
                Z_record = Z_filt
            else:
                Z_record = Z_basecorr
            
            # Get the duration, stream file time of start, and time of stop of shaking
            Z_Td, Z_start, Z_end = tmf.determine_Td(threshold,Z_record)  
            Z_Td_list = np.append(Z_Td_list,Z_Td)

            # Save corrected mseed file
            tra = Z_record[0]
            tra.stats.channel = code + 'Z'
            filename = '/Users/tnye/tsuquakes/data/Mentawai2010/' + data + '_corr/' + tra.stats.station + '.' + tra.stats.channel + '.mseed' 
            tra.write(filename, format='MSEED')


        # Get the values for the horizontal components 
        if ('E' in components) and ('N' in components):
            
            # Take the min time of E and N start times to be the start
            EN_start = np.min([E_start,N_start])
            
            # Take the max time of E and N end times to be the end
            EN_end = np.max([E_end,N_end])
            
            # Get the duration to be the time between these
            EN_Td = EN_end - EN_start
            horiz_Td_list = np.append(horiz_Td_list,EN_Td)

        else:
            # Append nan to the overall arrays if horizontals don't exist:
            horizon_Td_list = np.append(horiz_Td_list,np.nan)
            
            
        # Get the values for all 3 components
        if ('E' in components) and ('N' in components) and ('Z' in components):
            
            # Take the min time of the E,N,Z start times to be the start
            ENZ_start = np.min([E_start,N_start,Z_start])
            
            # Take the max of the E,N,Z end times to be the end
            ENZ_end = np.max([E_end,N_end,Z_end])
            
            # Get the duration to be the time between these
            ENZ_Td = ENZ_end - ENZ_start
            comp3_Td_list = np.append(comp3_Td_list,ENZ_Td)

        else:
            # Append nan to the overall arrays if all 3 components don't exist:
            comp3_Td_list = np.append(comp3_Td_list,np.nan)


        ############################### Velocity ##############################

        # Integrate acceleration data to velocity 
        if data == 'accel':
        
         ### Integrate filtered acc data to get velocity data
        
            ## East component 
            E_vel_unfilt = tmf.accel_to_veloc(E_record)
            E_vel = tmf.highpass(E_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
            
            # Save filtered velocity mseed file
            trv = E_vel[0]
            trv.stats.channel = 'HNE'
            filename_vel = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr/' + trv.stats.station + '.HNE.mseed' 
            trv.write(filename_vel, format='MSEED')

            ## North component 
            N_vel_unfilt = tmf.accel_to_veloc(N_record)
            N_vel = tmf.highpass(N_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
            
            # Save filtered velocity mseed file
            trv = N_vel[0]
            trv.stats.channel = 'HNN'
            filename_vel = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr/' + trv.stats.station + '.HNN.mseed' 
            trv.write(filename_vel, format='MSEED')
            
            ## Vertical component 
            Z_vel_unfilt = tmf.accel_to_veloc(Z_record)
            Z_vel = tmf.highpass(Z_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
            
            # Save filtered velocity mseed file
            trv = Z_vel[0]
            trv.stats.channel = 'HNZ'
            filename_vel = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr/' + trv.stats.station + '.HNZ.mseed' 
            trv.write(filename_vel, format='MSEED')
            

        ########################### Intensity Measures ########################
        
        # Calculate displacement intensity measures
        if data == 'disp':
            ## PGD
            # Get euclidean norm of displacement components 
            euc_norm = avg.get_eucl_norm_3comp(E_record[0].data,
                                               N_record[0].data,
                                               Z_record[0].data)
            # Calculate PGD
            pgd = np.max(np.abs(euc_norm))
            pgd_list = np.append(pgd_list,pgd)
            # Calcualte tPGD from origin and p-arrival
            tPGD_orig, tPGD_parriv = IM_fns.calc_time_to_peak(pgd, E_record[0],
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
            IM_fns.plot_spectra(E_record, freqs, amps, 'disp', '', synthetic=False)
    
        # If data type is not displacement, append 'nans'
        else:
            pgd_list = np.append(pgd_list,np.nan)
            tPGD_orig_list = np.append(tPGD_orig_list,np.nan)
            tPGD_parriv_list = np.append(tPGD_parriv_list,np.nan)
            disp_spec = [np.nan] * 60
            disp_speclist.append(disp_spec)

        # Calculate acceleration and velocity intensity measures
        if data == 'accel':
            ## PGA         
            # Get euclidean norm of acceleration components 
            acc_euc_norm = avg.get_eucl_norm_3comp(E_record[0].data,
                                               N_record[0].data,
                                               Z_record[0].data)
            
            # Calculate PGA
            pga = np.max(np.abs(acc_euc_norm))
            pga_list = np.append(pga_list,pga)
            # Calcualte tPGD from origin and p-arrival
            tPGA_orig, tPGA_parriv = IM_fns.calc_time_to_peak(pga, E_record[0],
                                                            np.abs(acc_euc_norm),
                                                            origin, hypdist)
            tPGA_orig_list = np.append(tPGA_orig_list,tPGA_orig)
            tPGA_parriv_list = np.append(tPGA_parriv_list,tPGA_parriv)

            ## Acc Spectra
            E_spec_data, freqE, ampE = IM_fns.calc_spectra(E_record, 'sm')
            N_spec_data, freqN, ampN = IM_fns.calc_spectra(N_record, 'sm')
            Z_spec_data, freqZ, ampZ = IM_fns.calc_spectra(Z_record, 'sm')
            # Combine into one array and append to main list
            acc_spec = np.concatenate([E_spec_data,N_spec_data,Z_spec_data])
            acc_speclist.append(acc_spec.tolist())
            # Plot spectra
            freqs = [freqE,freqN,freqZ]
            amps = [ampE,ampN,ampZ]
            IM_fns.plot_spectra(E_record, freqs, amps, 'acc', '', synthetic=False)


            ## PGV      
            # Get euclidean norm of velocity components 
            vel_euc_norm = avg.get_eucl_norm_3comp(E_vel[0].data,
                                                N_vel[0].data,
                                                Z_vel[0].data)
            # Calculate PGV
            pgv = np.max(np.abs(vel_euc_norm))
            pgv_list = np.append(pgv_list,pgv)
            
            ## Vel Spectra
            E_spec_vel, freqE_v, ampE_v = IM_fns.calc_spectra(E_vel, 'sm')
            N_spec_vel, freqN_v, ampN_v = IM_fns.calc_spectra(N_vel, 'sm')
            Z_spec_vel, freqZ_v, ampZ_v = IM_fns.calc_spectra(Z_vel, 'sm')
            # Combine into one array and append to main list
            vel_spec = np.concatenate([E_spec_data,N_spec_data,Z_spec_data])
            vel_speclist.append(vel_spec.tolist())
            # Plot spectra
            freqs_v = [freqE_v,freqN_v,freqZ_v]
            amps_v = [ampE_v,ampN_v,ampZ_v]
            IM_fns.plot_spectra(E_vel, freqs_v, amps_v, 'vel', '', synthetic=False)

        # If data type is not acceleration, append 'nans'
        else:
            pga_list = np.append(pga_list,np.nan)
            pgv_list = np.append(pgv_list,np.nan)
            tPGA_orig_list = np.append(tPGA_orig_list,np.nan)
            tPGA_parriv_list = np.append(tPGA_parriv_list,np.nan)
            
            acc_spec = [np.nan] * 60
            acc_speclist.append(acc_spec)
            vel_spec = [np.nan] * 60
            vel_speclist.append(vel_spec)


########################### Put together dataframe ############################

# First, make a dictionary for main part of dataframe:
dataset_dict = {'eventname':eventnames,'country':countries,'origintime':origintimes,
                    'hyplon':hyplons,'hyplat':hyplats,'hypdepth (km)':hypdepths,
                    'mw':mws,'m0':m0s,'network':networks,'station':stations,
                    'station_type':stn_type_list,'stlon':stlons,'stlat':stlats,
                    'stelev':stelevs,'mechanism':mechanisms,'hypdist':hypdists,
                    'duration_e':E_Td_list,'duration_n':N_Td_list,'duration_z':Z_Td_list,
                    'duration_horiz':horiz_Td_list,'duration_3comp':comp3_Td_list,
                    'pga':pga_list, 'pgv':pgv_list,'pgd':pgd_list,'tPGD_origin':tPGD_orig_list,
                    'tPGD_parriv':tPGD_parriv_list, 'tPGA_origin':tPGA_orig_list,
                    'tPGA_parriv':tPGA_parriv_list}

# Create list of column names for the displacement spectra bins
disp_cols = ['E_disp_bin1', 'E_disp_bin2', 'E_disp_bin3', 'E_disp_bin4',
                 'E_disp_bin5', 'E_disp_bin6', 'E_disp_bin7',
                 'E_disp_bin8', 'E_disp_bin9', 'E_disp_bin10', 'E_disp_bin11',
                 'E_disp_bin12', 'E_disp_bin13', 'E_disp_bin14', 'E_disp_bin15',
                 'E_disp_bin16', 'E_disp_bin17', 'E_disp_bin18', 'E_disp_bin19',
                 'E_disp_bin20', 'N_disp_bin1', 'N_disp_bin2',
                 'N_disp_bin3', 'N_disp_bin4', 'N_disp_bin5', 'N_disp_bin6',
                 'N_disp_bin7', 'N_disp_bin8', 'N_disp_bin9', 'N_disp_bin10',
                 'N_disp_bin11', 'N_disp_bin12', 'N_disp_bin13', 'N_disp_bin14',
                 'N_disp_bin15', 'N_disp_bin16', 'N_disp_bin17', 'N_disp_bin18',
                 'N_disp_bin19', 'N_disp_bin20', 'Z_disp_bin1',
                 'Z_disp_bin2', 'Z_disp_bin3', 'Z_disp_bin4', 'Z_disp_bin5',
                 'Z_disp_bin6', 'Z_disp_bin7', 'Z_disp_bin8', 'Z_disp_bin9',
                 'Z_disp_bin10', 'Z_disp_bin11', 'Z_disp_bin12', 'Z_disp_bin13',
                 'Z_disp_bin14', 'Z_disp_bin15', 'Z_disp_bin16', 'Z_disp_bin17',
                 'Z_disp_bin18', 'Z_disp_bin19', 'Z_disp_bin20']

# Create list of column names for the acceleration spectra bins
acc_cols = ['E_acc_bin1', 'E_acc_bin2', 'E_acc_bin3', 'E_acc_bin4',
            'E_acc_bin5', 'E_acc_bin6', 'E_acc_bin7', 'E_acc_bin8',
            'E_acc_bin9', 'E_acc_bin10', 'E_acc_bin11', 'E_acc_bin12',
            'E_acc_bin13', 'E_acc_bin14', 'E_acc_bin15', 'E_acc_bin16',
            'E_acc_bin17', 'E_acc_bin18', 'E_acc_bin19', 'E_acc_bin20',
            'N_acc_bin1', 'N_acc_bin2', 'N_acc_bin3',
            'N_acc_bin4', 'N_acc_bin5', 'N_acc_bin6', 'N_acc_bin7',
            'N_acc_bin8', 'N_acc_bin9', 'N_acc_bin10', 'N_acc_bin11',
            'N_acc_bin12', 'N_acc_bin13', 'N_acc_bin14', 'N_acc_bin15',
            'N_acc_bin16', 'N_acc_bin17', 'N_acc_bin18', 'N_acc_bin19',
            'N_acc_bin20', 'Z_acc_bin1', 'Z_acc_bin2',
            'Z_acc_bin3', 'Z_acc_bin4', 'Z_acc_bin5', 'Z_acc_bin6',
            'Z_acc_bin7', 'Z_acc_bin8', 'Z_acc_bin9', 'Z_acc_bin10',
            'Z_acc_bin11', 'Z_acc_bin12', 'Z_acc_bin13', 'Z_acc_bin14',
            'Z_acc_bin15', 'Z_acc_bin16', 'Z_acc_bin17', 'Z_acc_bin18',
            'Z_acc_bin19', 'Z_acc_bin20']

# Create list of column names for the velocity spectra bins  
vel_cols = ['E_vel_bin1', 'E_vel_bin2', 'E_vel_bin3', 'E_vel_bin4',
            'E_vel_bin5', 'E_vel_bin6', 'E_vel_bin7', 'E_vel_bin8',
            'E_vel_bin9', 'E_vel_bin10', 'E_vel_bin11', 'E_vel_bin12',
            'E_vel_bin13', 'E_vel_bin14', 'E_vel_bin15', 'E_vel_bin16',
            'E_vel_bin17', 'E_vel_bin18', 'E_vel_bin19', 'E_vel_bin20',
            'N_vel_bin1', 'N_vel_bin2', 'N_vel_bin3',
            'N_vel_bin4', 'N_vel_bin5', 'N_vel_bin6', 'N_vel_bin7',
            'N_vel_bin8', 'N_vel_bin9', 'N_vel_bin10', 'N_vel_bin11',
            'N_vel_bin12', 'N_vel_bin13', 'N_vel_bin14', 'N_vel_bin15',
            'N_vel_bin16', 'N_vel_bin17', 'N_vel_bin18', 'N_vel_bin19',
            'N_vel_bin20', 'Z_vel_bin1', 'Z_vel_bin2',
            'Z_vel_bin3', 'Z_vel_bin4', 'Z_vel_bin5', 'Z_vel_bin6',
            'Z_vel_bin7', 'Z_vel_bin8', 'Z_vel_bin9', 'Z_vel_bin10',
            'Z_vel_bin11', 'Z_vel_bin12', 'Z_vel_bin13', 'Z_vel_bin14',
            'Z_vel_bin15', 'Z_vel_bin16', 'Z_vel_bin17', 'Z_vel_bin18',
            'Z_vel_bin19', 'Z_vel_bin20']

# Create spectra dataframes for each data type
disp_spec_df = pd.DataFrame(disp_speclist, columns=disp_cols)
acc_spec_df = pd.DataFrame(acc_speclist, columns=acc_cols)
vel_spec_df = pd.DataFrame(vel_speclist, columns=vel_cols)

# Make main dataframe
main_df = pd.DataFrame(data=dataset_dict)

# Combine main dataframe with spectra dataframes 
flatfile_df = pd.concat([main_df, disp_spec_df.reindex(main_df.index),
                        acc_spec_df.reindex(main_df.index),
                        vel_spec_df.reindex(main_df.index)], axis=1)

# Save df to file:
flatfile_df.to_csv(flatfile_path,index=False)
