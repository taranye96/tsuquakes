#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:16:28 2019

@author: tnye
"""

###############################################################################
# Script that goes through observed waveforms from the 2010 M7.8 Mentawai event,
# calculates intensity measures, and stores them all in a flatefile.  The IMs
# this script calculates are:
    # PGD
    # PGA
    # Displacement spectra bin averages for 10 bins
    # Acceleration spectra bin averages for 10 bins
###############################################################################

# Imports
import numpy as np
import pandas as pd
import obspy
from glob import glob
from obspy import UTCDateTime, Stream

# Local Imports
import tsuquakes_main_fns as tmf

################################ Parameters ###################################

# Stations not used for analyses because they are too noisy or too far away
unused_stns = ['PPBI','PSI','CGJI','TSI','CNJI','LASI','MLSI','MKMK','LNNG','LAIS','TRTK','MNNA','BTHL']

# Used for directory paths
earthquake_name = 'Mentawai2010'

# Data types to loop through.  I have a folder for displacement ('disp') and a 
    # folder for acceleration ('accel'), so those are my data types. 
data_types = ['disp', 'accel']
# data_types = ['disp']

# Project directory 
proj_dir = '/Users/tnye/tsuquakes' 

# Waveforms home directory
home_dir = '/Users/tnye/tsuquakes/data/processed_waveforms/individual'

# Table of earthquake data  
eq_table = pd.read_csv('/Users/tnye/tsuquakes/data/events.csv')

# Data directories
disp_dir = '/Users/tnye/tsuquakes/data/GNSS_data_processed_Dara_SAC/events/'
data_dir = proj_dir + '/data'
sm_dir = data_dir + '/' + earthquake_name + '/accel'

# Path to save flatfiles of intensity measures
gnss_flatfile_path = proj_dir + '/flatfiles/obs_IMs_gnss.csv'     
sm_flatfile_path = proj_dir + '/flatfiles/obs_IMs_sm.csv'  

# Parameters for integration to velocity and filtering 
fcorner_high = 1/15.                     # Frequency at which to high pass filter
fcorner_low = 0.4
order = 2                                # Number of poles for filter  

# Gather displacement and strong motion files
disp_files = np.array(sorted(glob(disp_dir + '/*.mseed')))
sm_files = np.array(sorted(glob(sm_dir + '/*.mseed')))


################################ Event Data ###################################

# Metadata
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


###################### Data Processing and Calculations #######################

# Threshold- used to calculate duration 
threshold = 0.0

# Loop through data types
for data in data_types:
    
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
    disp_freqs = []
    sm_freqs = []
    disp_speclist = []
    acc_speclist = []
    
    ###################### Set parameters for data type #######################
    
    if data == 'disp':
        
        dtype = 'disp'
        
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
        filtering = 'lowpass'

    
    elif data == 'accel':
        
        dtype = 'acc'
        
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
        filtering = 'highpass'


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
    obs_times = []
    obs_amps = []
    obs_amps_v = []
    
    # Group all files by station since there should be 3 components for each 
        # station
    N = 3
    stn_files = [files[n:n+N] for n in range(0, len(files), N)]
    
    # Loop over files to get the list of station names, channels, and mseed files 
    for station in stn_files:
        
        name = station[0].split('/')[-1].split('.')[0]
        
        if name not in unused_stns:
            
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

        # Turn the components list into an array 
        components = np.asarray(components)
        
        ########## East component ########## 
          
        # Get index for E component 
        E_index = np.where(components=='E')[0][0]
       
        # Read file into a stream object
        E_raw = obspy.read(mseed_list[i][E_index])
        
        if data == 'disp':
            E_gaincorr = E_raw
        else:
            # Correct by gain, so everything is in meters
            E_gaincorr = tmf.correct_for_gain(E_raw,stgain)
        
        # Get the pre-event baseline
        E_baseline = tmf.compute_baseline(E_gaincorr,nsamples)
        
        # Get the baseline corrected stream objectda
        E_basecorr = tmf.correct_for_baseline(E_gaincorr,E_baseline)

        # High pass filter strong motion data at fcorner specified above
        if filtering == 'lowpass':
            E_filt = tmf.lowpass(E_basecorr,fcorner_low,stsamprate,order,zerophase=True)
        elif filtering == 'highpass':
            E_filt = tmf.highpass(E_basecorr,fcorner_high,stsamprate,order,zerophase=True)
        E_record = E_filt
        
        # Get the duration, stream file time of start, and time of stop of shaking
        E_Td, E_start, E_end = tmf.determine_Td(threshold,E_record)      
        E_Td_list = np.append(E_Td_list,E_Td)

        # Save corrected mseed file
        tra = E_record[0]
        tra.stats.channel = code + 'E'
        filename = f'/Users/tnye/tsuquakes/data/processed_waveforms/individual/{dtype}/{tra.stats.station}.{tra.stats.channel}.mseed' 
        tra.write(filename, format='MSEED')

            
        ########## North component ##########
           
        # Get index for N component 
        N_index = np.where(components=='N')[0][0]
        
        # Read file into a stream object
        N_raw = obspy.read(mseed_list[i][N_index])
        
        if data == 'disp':
            N_gaincorr = N_raw
        else:
            # Correct by gain, so everything is in meters
            N_gaincorr = tmf.correct_for_gain(N_raw,stgain)
        
        
        # Get the pre-event baseline
        N_baseline = tmf.compute_baseline(N_gaincorr,nsamples)
        
        # Get the baseline corrected stream object
        N_basecorr = tmf.correct_for_baseline(N_gaincorr,N_baseline)
        
        # High pass filter strong motion data at fcorner specified above
        if filtering == 'lowpass':
            N_filt = tmf.lowpass(N_basecorr,fcorner_low,stsamprate,order,zerophase=True)
        elif filtering == 'highpass':
            N_filt = tmf.highpass(N_basecorr,fcorner_high,stsamprate,order,zerophase=True)
        N_record = N_filt
            
        # Get the duration, stream file time of start, and time of stop of shaking
        N_Td, N_start, N_end = tmf.determine_Td(threshold,N_record)  
        N_Td_list = np.append(N_Td_list,N_Td)

        # Save corrected mseed file
        tra = N_record[0]
        tra.stats.channel = code + 'N'
        filename = f'/Users/tnye/tsuquakes/data/processed_waveforms/individual/{dtype}/{tra.stats.station}.{tra.stats.channel}.mseed' 
        tra.write(filename, format='MSEED')

        ########## Vertical component ##########
         
        # Get index for Z component 
        Z_index = np.where(components=='Z')[0][0]     
       
        # Read file into a stream object                     
        Z_raw = obspy.read(mseed_list[i][Z_index])
        
        if data == 'disp':
            Z_gaincorr = Z_raw
        else:
            # Correct by gain, so everything is in meters
            Z_gaincorr = tmf.correct_for_gain(Z_raw,stgain)
        
        
        # Get the pre-event baseline
        Z_baseline = tmf.compute_baseline(Z_gaincorr,nsamples)
        
        # Get the baseline corrected stream object
        Z_basecorr = tmf.correct_for_baseline(Z_gaincorr,Z_baseline)
        
        # High pass filter strong motion data at fcorner specified above
        if filtering == 'lowpass':
            Z_filt = tmf.lowpass(Z_basecorr,fcorner_low,stsamprate,order,zerophase=True)
        elif filtering == 'highpass':
            Z_filt = tmf.highpass(Z_basecorr,fcorner_high,stsamprate,order,zerophase=True)
        Z_record = Z_filt
        
        # Get the duration, stream file time of start, and time of stop of shaking
        Z_Td, Z_start, Z_end = tmf.determine_Td(threshold,Z_record)  
        Z_Td_list = np.append(Z_Td_list,Z_Td)

        # Save corrected mseed file
        tra = Z_record[0]
        tra.stats.channel = code + 'Z'
        filename = f'/Users/tnye/tsuquakes/data/processed_waveforms/individual/{dtype}/{tra.stats.station}.{tra.stats.channel}.mseed' 
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
            horiz_Td_list = np.append(horiz_Td_list,np.nan)
            
            
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
            

        ########################### Intensity Measures ########################
        
        # Shorten waveforms to avoid calculating PGD on long period noise for some
            # stations and for calculating spectra
        tr_E_short = E_record[0].trim(endtime=UTCDateTime("2010-10-25T14:47:02"))
        tr_N_short = N_record[0].trim(endtime=UTCDateTime("2010-10-25T14:47:02"))
        tr_Z_short = Z_record[0].trim(endtime=UTCDateTime("2010-10-25T14:47:02"))
        
        # Calculate displacement intensity measures
        if data == 'disp':
            
            ## PGD
            #Get euclidean norm of displacement components 
            eucnorm = tmf.get_eucl_norm_3comp(tr_E_short.data,
                                                tr_N_short.data,
                                                tr_Z_short.data)
            
            # Calculate PGD
            pgd = np.max(np.abs(eucnorm))
            pgd_list = np.append(pgd_list,pgd)
            
            # Calculate tPGD from origin and p-arrival
            tPGD_orig = tmf.calc_time_to_peak(pgd, tr_E_short,
                                                            np.abs(eucnorm),
                                                            origin, hypdist)
            tPGD_orig_list = np.append(tPGD_orig_list,tPGD_orig)
            
            ## Disp Spectra
            bins, E_spec_data = tmf.calc_spectra(Stream(tr_E_short), 'gnss')
            bins, N_spec_data = tmf.calc_spectra(Stream(tr_N_short), 'gnss')
            bins, Z_spec_data = tmf.calc_spectra(Stream(tr_Z_short), 'gnss')
            
            NE_data = np.sqrt(E_spec_data**2 + N_spec_data**2)
            
            # Append spectra to lists to make spectra comparison plots
            disp_freqs.append(bins)
            disp_speclist.append(NE_data.tolist())
    
        # Calculate acceleration and velocity intensity measures
        if data == 'accel':
            
            ## PGA         
            #Get rotd50 of the acceleration horizontal components 
            rotd50 = tmf.compute_rotd50(E_record[0].data,N_record[0].data)
            
            # Calculate PGA
            pga = np.max(np.abs(rotd50))
            pga_list = np.append(pga_list,pga)
            
            # Calcualte tPGA from origin and p-arrival
            tPGA_orig = tmf.calc_time_to_peak(pga, E_record[0],
                                                            np.abs(rotd50),
                                                            origin, hypdist)
            tPGA_orig_list = np.append(tPGA_orig_list,tPGA_orig)

            ## Acc Spectra
            bins, E_spec_data = tmf.calc_spectra(Stream(tr_E_short), 'sm')
            bins, N_spec_data = tmf.calc_spectra(Stream(tr_N_short), 'sm')
            bins, Z_spec_data = tmf.calc_spectra(Stream(tr_Z_short), 'sm')
            
            NE_data = np.sqrt(E_spec_data**2 + N_spec_data**2)
            
            # Append spectra to lists to make spectra comparison plots
            sm_freqs.append(bins)
            acc_speclist.append(NE_data.tolist())

        
    ############################## Dataframe ##############################
    
    disp_speclist = np.array(disp_speclist)
    acc_speclist = np.array(acc_speclist)

    if data == 'disp':
    
        # First, make a dictionary for main part of dataframe:
        gnss_dataset_dict = {'eventname':eventnames,'country':countries,'origintime':origintimes,
                            'hyplon':hyplons,'hyplat':hyplats,'hypdepth (km)':hypdepths,
                            'mw':mws,'m0':m0s,'network':networks,'station':stations,
                            'station_type':stn_type_list,'stlon':stlons,'stlat':stlats,
                            'stelev':stelevs,'mechanism':mechanisms,'hypdist':hypdists,
                            'duration_e':E_Td_list,'duration_n':N_Td_list,'duration_z':Z_Td_list,
                            'duration_horiz':horiz_Td_list,'duration_3comp':comp3_Td_list,
                            'pgd':pgd_list,'tPGD_origin':tPGD_orig_list,
                            'Spectra_Disp_Bin1_E':disp_speclist[:,0],'Spectra_Disp_Bin2_E':disp_speclist[:,1],
                            'Spectra_Disp_Bin3_E':disp_speclist[:,2],'Spectra_Disp_Bin4_E':disp_speclist[:,3],
                            'Spectra_Disp_Bin5_E':disp_speclist[:,4],'Spectra_Disp_Bin6_E':disp_speclist[:,5],
                            'Spectra_Disp_Bin7_E':disp_speclist[:,6],'Spectra_Disp_Bin8_E':disp_speclist[:,7],
                            'Spectra_Disp_Bin9_E':disp_speclist[:,8],'Spectra_Disp_Bin10_E':disp_speclist[:,9]}
        
        # Make main dataframe
        gnss_flatfile_df = pd.DataFrame(data=gnss_dataset_dict)
        
        # Save df to file:
        gnss_flatfile_df.to_csv(gnss_flatfile_path,index=False)
        
 
    elif data == 'accel':
        sm_dataset_dict = {'eventname':eventnames,'country':countries,'origintime':origintimes,
                            'hyplon':hyplons,'hyplat':hyplats,'hypdepth (km)':hypdepths,
                            'mw':mws,'m0':m0s,'network':networks,'station':stations,
                            'station_type':stn_type_list,'stlon':stlons,'stlat':stlats,
                            'stelev':stelevs,'mechanism':mechanisms,'hypdist':hypdists,
                            'duration_e':E_Td_list,'duration_n':N_Td_list,'duration_z':Z_Td_list,
                            'duration_horiz':horiz_Td_list,'duration_3comp':comp3_Td_list,
                            'pga':pga_list, 'tPGA_origin':tPGA_orig_list,
                            'Spectra_Acc_Bin1_E':acc_speclist[:,0],'Spectra_Acc_Bin2_E':acc_speclist[:,1],
                            'Spectra_Acc_Bin3_E':acc_speclist[:,2],'Spectra_Acc_Bin4_E':acc_speclist[:,3],
                            'Spectra_Acc_Bin5_E':acc_speclist[:,4],'Spectra_Acc_Bin6_E':acc_speclist[:,5],
                            'Spectra_Acc_Bin7_E':acc_speclist[:,6],'Spectra_Acc_Bin8_E':acc_speclist[:,7],
                            'Spectra_Acc_Bin9_E':acc_speclist[:,8],'Spectra_Acc_Bin10_E':acc_speclist[:,9]}
    
        # Make main dataframe
        sm_flatfile_df = pd.DataFrame(data=sm_dataset_dict)
        
        # Save df to file:
        sm_flatfile_df.to_csv(sm_flatfile_path,index=False)

