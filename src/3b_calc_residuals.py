#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:08:17 2020

@author: tnye
"""

###############################################################################
# This script calculates and plots residuals between synthetic and observed IMs
# and spectra.
###############################################################################

# Imports
import numpy as np
from numpy import genfromtxt
import pandas as pd
from os import path,makedirs,chdir
import tsuquakes_main_fns as tmf


# Name of directory that contains variations of rupture parameters for the same
    # set of simulated rupture files
simulation_set = 'tse_simulations'

# If the residuals for the standard parameter simulations have not been run
# yet, standard should == True
standard = True

# Name of subdirectory in simulation set
    # This only matters if standard == False
parameter = 'stress_drop'


#%%

##### Determine project directories based on parameters from previous cell ####

# Are we computing residuals for simulations generated using standard parameters?
if standard == True:
    
    # Set home directory
    home_dir = '/Users/tnye/tsuquakes/simulations'

    # Set parameter and project names
    parameter = simulation_set
    projects = ['standard' ]  # array of projects (parameter variations) for the parameter
    data_types = ['disp', 'acc']
    stations = ['gnss','sm']
    
else:
    
    if simulation_set == 'test_simulations':
        
        # Set home directory
        home_dir = f'/Users/tnye/tsuquakes/simulations/{simulation_set}'
        
        if parameter == 'stress_drop':
           
            #Set parameters
            parameter = 'stress_drop'      # parameter being varied
            projects = ['sd0.1','sd1.0','sd2.0']  # array of projects for the parameter
            param_vals = ['0.1', '1.0', '2.0']        # array of parameter values associated w/ the projects
        
        elif parameter == 'risetime':
        
            # Set parameters
            parameter = 'risetime'      # parameter being varied
            projects = ['rt2x','rt3x']  # array of projects for the parameter
            param_vals = ['10.8','16.2']        # array of parameter values associated w/ the projects
        
        elif parameter == 'vrupt':
    
            # Set parameters
            parameter = 'vrupt'      # parameter being varied
            projects = ['sf0.3', 'sf0.4']  # array of projects for the parameter
            param_vals = ['1.0', '1.3']        # array of parameter values associated w/ the projects
        
        data_types = ['disp', 'acc']
        stations = ['gnss','sm']
    
    elif simulation_set == 'gpr_simulations':
        
        # Set home directory
        home_dir = f'/Users/tnye/tsuquakes/simulations/{simulation_set}'
        
        if parameter == '2D':
            
            # Set parameters
            projects = np.array(['rt1.0x_sf0.37','rt1.0x_sf0.4','rt1.0x_sf0.43','rt1.0x_sf0.46',
                                  'rt1.0x_sf0.49','rt1.1x_sf0.37','rt1.1x_sf0.4','rt1.1x_sf0.43',
                                  'rt1.1x_sf0.46','rt1.1x_sf0.49','rt1.2x_sf0.37','rt1.2x_sf0.4',
                                  'rt1.2x_sf0.43','rt1.2x_sf0.46','rt1.2x_sf0.49','rt1.3x_sf0.37',
                                  'rt1.3x_sf0.4','rt1.3x_sf0.43','rt1.3x_sf0.46','rt1.3x_sf0.49',
                                  'rt1.4x_sf0.37','rt1.4x_sf0.4','rt1.4x_sf0.43','rt1.4x_sf0.46',
                                  'rt1.4x_sf0.49','rt1.5x_sf0.37','rt1.5x_sf0.4','rt1.5x_sf0.43',
                                  'rt1.5x_sf0.46','rt1.5x_sf0.49','rt1.6x_sf0.37','rt1.6x_sf0.4',
                                  'rt1.6x_sf0.43','rt1.6x_sf0.46','rt1.6x_sf0.49','rt1.7x_sf0.37',
                                  'rt1.7x_sf0.4','rt1.7x_sf0.43','rt1.7x_sf0.46','rt1.7x_sf0.49',
                                  'rt1.8x_sf0.37','rt1.8x_sf0.4','rt1.8x_sf0.43','rt1.8x_sf0.46',
                                  'rt1.8x_sf0.49','rt1.9x_sf0.37','rt1.9x_sf0.4','rt1.9x_sf0.43',
                                  'rt1.9x_sf0.46','rt1.9x_sf0.49','rt2.0x_sf0.37','rt2.0x_sf0.4',
                                  'rt2.0x_sf0.43','rt2.0x_sf0.46','rt2.0x_sf0.49','rt2.1x_sf0.37',
                                  'rt2.1x_sf0.4','rt2.1x_sf0.43','rt2.1x_sf0.46','rt2.1x_sf0.49',
                                  'rt2.2x_sf0.37','rt2.2x_sf0.4','rt2.2x_sf0.43','rt2.2x_sf0.46',
                                  'rt2.2x_sf0.49','rt2.3x_sf0.37','rt2.3x_sf0.4','rt2.3x_sf0.43',
                                  'rt2.3x_sf0.46','rt2.3x_sf0.49'
                                  ])
        
            data_types = ['disp']
            stations = ['gnss']
            
        elif parameter == '1D':
            
            # Set parameters
            projects = np.array(['sd0.1','sd0.5','sd1.0','sd1.5','sd2.0','sd3.0'])
            
            data_types = ['acc']
            stations = ['sm']
        
    elif simulation_set == 'tse_simulations':
        
        # Set home directory
        home_dir = '/Users/tnye/tsuquakes/simulations'
        
        parameter = 'tse_simulations'
        
        # Set parameters
        parameter = 'tse_simulations'
        projects = np.array(['rt1.234x_sf0.41_sd1.196','rt1.954x_sf0.469_sd1.196','rt1.4x_sf0.45_sd1.0',
                              'rt1.4x_sf0.45_sd2.0','rt1.75x_sf0.42_sd1.0','rt1.75x_sf0.42_sd2.0'])
        param_vals = [r'$\mu$-a',r'$\mu$-b',r'$\sigma$-a1',r'$\sigma$-a3',r'$\sigma$-b1',r'$\sigma$-b3']        # array of parameter values associated w/ the projects
        
        data_types = ['disp', 'acc']
        stations = ['gnss','sm']


#%%

############################## Compute residuals ##############################

# Loop through projects
for project in projects:  
    
    # Project directory 
    project_dir = f'{home_dir}/{parameter}/{project}'
    
    # List of synthetic rupture scenarios generated by FakeQuakes to be compared
        # to observed 
    try:
        rupture_list = genfromtxt(f'{home_dir}/{parameter}/{project}/data/ruptures.sublist',dtype=str)
    except:
        rupture_list = genfromtxt(f'{home_dir}/{parameter}/{project}/data/ruptures.list',dtype=str)
    
    # Initialize lists to append residuals and run names from all the project's 
        # runs to    
    gnss_run_list = []
    sm_run_list = []
    pgd_res_list = []
    pga_res_list = []
    tPGD_res_ln_list = []
    tPGD_res_linear_list = []
    tPGA_res_ln_list = []
    tPGA_res_linear_list = []
    disp_spec_res_list = []
    acc_spec_res_list = []
    
    # Loop through rupture scenarios
    for rupture in rupture_list:
        
        # Flatfile with observed data
        sm_obs_file = '/Users/tnye/tsuquakes/files/flatfiles/obs_IMs_sm.csv'
        gnss_obs_file = '/Users/tnye/tsuquakes/files/flatfiles/obs_IMs_gnss.csv'
        
        # Remove .rupt to obtain run name
        run = rupture.rsplit('.', 1)[0]
        
        # Calculate residuals
        if 'gnss' in stations:
            pgd_res, tPGD_res_ln, tPGD_res_linear, disp_spec_res = tmf.calc_res(gnss_obs_file,home_dir,parameter,project,run,'gnss')
            pgd_res_list.append(pgd_res.tolist())
            tPGD_res_ln_list.append(tPGD_res_ln.tolist())
            tPGD_res_linear_list.append(tPGD_res_linear.tolist())
            disp_spec_res_list.append(disp_spec_res.tolist())
            
            for i in range(len(pgd_res)):
                gnss_run_list.append(run)
            
        if 'sm' in stations:
            pga_res, tPGA_res_ln, tPGA_res_linear, acc_spec_res = tmf.calc_res(sm_obs_file,home_dir,parameter,project,run,'sm')
            pga_res_list.append(pga_res.tolist())
            tPGA_res_ln_list.append(tPGA_res_ln.tolist())
            tPGA_res_linear_list.append(tPGA_res_linear.tolist())
            acc_spec_res_list.append(acc_spec_res.tolist())
    
            # Append run to main run list
            for i in range(len(pga_res)):
                sm_run_list.append(run)

    # Function to flatten lists
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    # Flatten residual lists
    pgd_res_list = flatten(pgd_res_list)
    pga_res_list = flatten(pga_res_list)
    tPGD_res_ln_list = flatten(tPGD_res_ln_list)
    tPGD_res_linear_list = flatten(tPGD_res_linear_list)
    tPGA_res_ln_list = flatten(tPGA_res_ln_list)
    tPGA_res_linear_list = flatten(tPGA_res_linear_list)
    disp_spec_res_list = flatten(disp_spec_res_list)
    acc_spec_res_list = flatten(acc_spec_res_list)
    
    # Make sure there is a folder for the residual flatfiles
    if not path.exists(f'{project_dir}/flatfiles/residuals'):
        makedirs(f'{project_dir}/flatfiles/residuals')
    
    # Set up dataframe
    if 'gnss' in stations:    
        gnss_IM_dict = {'run':gnss_run_list,'pgd_res':pgd_res_list,'tPGD_res_ln':tPGD_res_ln_list,
                        'tPGD_res_linear':tPGD_res_linear_list}
        gnss_spec_cols = ['Spectra_Disp_Res_Bin1', 'Spectra_Disp_Res_Bin2', 'Spectra_Disp_Res_Bin3', 'Spectra_Disp_Res_Bin4',
                      'Spectra_Disp_Res_Bin5', 'Spectra_Disp_Res_Bin6', 'Spectra_Disp_Res_Bin7',
                      'Spectra_Disp_Res_Bin8', 'Spectra_Disp_Res_Bin9', 'Spectra_Disp_Res_Bin10']
    
        gnss_obs_df = pd.read_csv(gnss_obs_file)
        
        main_gnss_df = gnss_obs_df.iloc[:,:16]
        main_gnss_df = main_gnss_df.reset_index(drop=True)
        
        gnss_IM_df = pd.DataFrame(data=gnss_IM_dict)
        
        gnss_spec_df = pd.DataFrame(disp_spec_res_list, columns=gnss_spec_cols)
        
        gnss_main_df_full = pd.concat([main_gnss_df]*len(rupture_list), ignore_index=True)
        
        gnss_res_df = pd.concat([gnss_main_df_full,gnss_IM_df,gnss_spec_df], axis=1)
        
        # Save to flatfile
        gnss_res_df.to_csv(f'{project_dir}/flatfiles/residuals/{project}_gnss.csv',index=False)

        
    if 'sm' in stations:    
        sm_IM_dict = {'run':sm_run_list,'pga_res':pga_res_list,
                      'tPGA_res_ln':tPGA_res_ln_list,'tPGA_res_linear':tPGA_res_linear_list}

        acc_spec_cols = ['Spectra_Acc_Res_Bin1', 'Spectra_Acc_Res_Bin2', 'Spectra_Acc_Res_Bin3', 'Spectra_Acc_Res_Bin4',
                      'Spectra_Acc_Res_Bin5', 'Spectra_Acc_Res_Bin6', 'Spectra_Acc_Res_Bin7', 'Spectra_Acc_Res_Bin8',
                      'Spectra_Acc_Res_Bin9', 'Spectra_Acc_Res_Bin10']
        
        sm_obs_df = pd.read_csv(sm_obs_file)
        
        main_sm_df = sm_obs_df.iloc[:,:16]
        main_sm_df = main_sm_df.reset_index(drop=True)
        
        sm_IM_df = pd.DataFrame(data=sm_IM_dict)
        
        acc_spec_df = pd.DataFrame(acc_spec_res_list, columns=acc_spec_cols)
        
        sm_main_df_full = pd.concat([main_sm_df]*len(rupture_list), ignore_index=True)
        
        sm_res_df = pd.concat([sm_main_df_full,sm_IM_df,acc_spec_df], axis=1)
    
        # Save to flatfile
        sm_res_df.to_csv(f'{project_dir}/flatfiles/residuals/{project}_sm.csv',index=False)

