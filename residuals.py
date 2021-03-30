#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:08:17 2020

@author: tnye
"""

###############################################################################
# Script that calculates and plots residuals between synthetic and observed IMs
# and spectra
###############################################################################

# Imports
from numpy import genfromtxt
import pandas as pd
from os import path,makedirs
import residual_fns as res

################################ Group Res Plots ##############################

### Plots residuals for all runs of all projects for one parameter (i.e. stress
 ## drops of 0.3, 1.0, and 2.0 MPa) on one figure

# Set parameters
parameter = 'stress_drop'      # parameter being varied
projects = ['sd0.1', 'sd1.0', 'sd2.0']  # array of projects for the parameter
param_vals = ['0.1', '1.0', '2.0']        # array of parameter values associated w/ the projects

# # Set parameters
# parameter = 'rise_time'      # parameter being varied
# projects = ['rt2x', 'rt3x', 'rt4x']  # array of projects for the parameter
# param_vals = ['2', '3', '4']        # array of parameter values associated w/ the projects

# # Set parameters
# parameter = 'vrupt'      # parameter being varied
# projects = ['sf0.25', 'sf0.5']  # array of projects for the parameter
# param_vals = ['0.25', '0.5']        # array of parameter values associated w/ the projects

# parameter = 'test'      # parameter being varied
# projects = ['kappa_test']  # array of projects for the parameter
# param_vals = ['test']        # array of parameter values associated w/ the projects

# Set to true if you want the natural log of the residuals 
ln=True

# Set to true if you want to include the standard values too
default=True

# Flatfile with observed data
obs_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')

# Loop through projects
for project in projects:  
    
    # Project directory 
    project_dir = f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}'
    
    # List of synthetic rupture scenarios generated by FakeQuakes to be compared
        # to observed 
    rupture_list = genfromtxt(f'/Users/tnye/FakeQuakes/parameters/{parameter}/'
                          f'{project}/disp/data/ruptures.sublist',dtype='U')
    
    # Initialize lists to append residuals and run names from all the project's 
        # runs to    
    run_list = []
    pgd_res_list = []
    pga_res_list = []
    pgv_res_list = []
    tPGD_orig_res_list = []
    tPGD_parriv_res_list = []
    tPGA_orig_res_list = []
    tPGA_parriv_res_list = []
    spectra_res_list = []
    
    # Loop through rupture scenarios
    for rupture in rupture_list:
        
        # Remove .rupt to obtain run name
        run = rupture.rsplit('.', 1)[0]
        
        # Calculate residuals
        pgd_res, pga_res, pgv_res, tPGD_orig_res, tPGD_parriv_res, tPGA_orig_res, \
            tPGA_parriv_res, spectra_res = res.calc_res(parameter,project,run,ln=ln)
        
        # Append individual run's residuals to main residual lists
        pgd_res_list.append(pgd_res.tolist())
        pga_res_list.append(pga_res.tolist())
        pgv_res_list.append(pgv_res.tolist())
        tPGD_orig_res_list.append(tPGD_orig_res.tolist())
        tPGD_parriv_res_list.append(tPGD_parriv_res.tolist())
        tPGA_orig_res_list.append(tPGA_orig_res.tolist())
        tPGA_parriv_res_list.append(tPGA_parriv_res.tolist())
        spectra_res_list.append(spectra_res.tolist())
        
        # Append run to main run list
        for i in range(len(pga_res)):
            run_list.append(run)

    # Function to flatten lists
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    # Flatten residual lists
    pgd_res_list = flatten(pgd_res_list)
    pga_res_list = flatten(pga_res_list)
    pgv_res_list = flatten(pgv_res_list)
    tPGD_orig_res_list = flatten(tPGD_orig_res_list)
    tPGD_parriv_res_list = flatten(tPGD_parriv_res_list)
    tPGA_orig_res_list = flatten(tPGA_orig_res_list)
    tPGA_parriv_res_list = flatten(tPGA_parriv_res_list)
    spectra_res_list = flatten(spectra_res_list)
    
    
    # Set up dataframe    
    IM_dict = {'run':run_list, 'pgd_res':pgd_res_list, 'pga_res':pga_res_list,
                'pgv_res':pgv_res_list, 'tPGD_origin_res':tPGD_orig_res_list,
                'tPGD_parriv_res':tPGD_parriv_res_list, 'tPGA_origin_res':tPGA_orig_res_list,
                'tPGA_parriv_res':tPGA_parriv_res_list}
    
    # Initialize spectra column names 
    spec_cols = ['E_disp_bin1', 'E_disp_bin2', 'E_disp_bin3', 'E_disp_bin4',
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
                  'Z_disp_bin18', 'Z_disp_bin19', 'Z_disp_bin20',
                  'E_acc_bin1', 'E_acc_bin2', 'E_acc_bin3', 'E_acc_bin4',
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
                  'Z_acc_bin19', 'Z_acc_bin20',
                  'E_vel_bin1', 'E_vel_bin2', 'E_vel_bin3', 'E_vel_bin4',
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
    
    
    # Separate out columns from observed dataframe to be used in residual dataframes
    main_df = obs_df.iloc[:,:16]
    main_df = main_df.reset_index(drop=True)
    
    # Make IM dataframe
    IM_df = pd.DataFrame(data=IM_dict)
    
    # Make spectra dataframe
    spec_df = pd.DataFrame(spectra_res_list, columns=spec_cols)
    
    # Extend main dataframe so that info repeates for each new run
    main_df_full = pd.concat([main_df]*len(rupture_list), ignore_index=True)
    
    # Combine all dataframes 
    res_df = pd.concat([main_df_full,IM_df,spec_df], axis=1)
    
    # Make sure there is a folder for the residual flatfiles
    if not path.exists(f'{project_dir}/flatfiles/residuals'):
        makedirs(f'{project_dir}/flatfiles/residuals')
    
    # Save to flatfile
    if ln:
        res_df.to_csv(f'{project_dir}/flatfiles/residuals/{project}_lnres.csv',index=False)
    else:
        res_df.to_csv(f'{project_dir}/flatfiles/residuals/{project}_res.csv',index=False)
    
    # Plot spectra residuals 
    res.plot_spec_res(parameter, project, ln=ln, outliers=True, default=default)
    res.plot_spec_res(parameter, project, ln=ln, outliers=False, default=default)


# Plot IM residuals 
res.plot_IM_res_full(parameter, projects, param_vals, ln=ln, outliers=True, default=default)
res.plot_IM_res_full(parameter, projects, param_vals, ln=ln, outliers=False, default=default)
