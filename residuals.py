#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:08:17 2020

@author: tnye
"""

# Imports
import pandas as pd
import IM_fns
import sys

parameter = 'stress_drop_runs'
project = 'sd2.0_etal_standard'
# project=sys.argv[1]

# runs list

# runs = ['run.000000', 'run.000001', 'run.000002', 'run.000003', 'run.000004',
#         'run.000005', 'run.000006', 'run.000007']
# runs = ['run.000000', 'run.000001', 'run.000002', 'run.000003', 'run.000004',
#         'run.000005', 'run.000006', 'run.000007', 'run.000008', 'run.000009',
#         'run.000010', 'run.000011']
runs = ['run.000000', 'run.000001', 'run.000002', 'run.000003', 'run.000004',
        'run.000005', 'run.000006', 'run.000007', 'run.000008', 'run.000009',
        'run.000010', 'run.000011', 'run.000012', 'run.000013', 'run.000014',
        'run.000015']

obs_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')

# All stations
# main_df = obs_df.iloc[:,:16]

# If just using sm stations
# main_df = obs_df.iloc[13:,:16]
main_df = obs_df.iloc[:,:16]

main_df = main_df.reset_index(drop=True)

for run in runs:
    
    ############################ Individual Res Plots #########################
    
    # Calculate residuals
    pgd_res,pga_res,pgv_res,tPGD_orig_res,tPGD_parriv_res,spectra_res = IM_fns.calc_res(parameter,project,run)
    
    # Set up dataframe    
    IM_dict = {'pgd_res':pgd_res,'pga_res':pga_res,'pgv_res':pgv_res,
               'tPGD_origin_res':tPGD_orig_res,'tPGD_parriv_res':tPGD_parriv_res}
    
    spec_cols = ['E_disp_bin1', 'E_disp_bin2', 'E_disp_bin3', 'E_disp_bin4',
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
                 'Z_disp_bin22', 'Z_disp_bin23', 'Z_disp_bin24', 'Z_disp_bin25',
                 'E_acc_bin1', 'E_acc_bin2', 'E_acc_bin3', 'E_acc_bin4',
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
                 'Z_acc_bin23', 'Z_acc_bin24', 'Z_acc_bin25',
                 'E_vel_bin1', 'E_vel_bin2', 'E_vel_bin3', 'E_vel_bin4',
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
    
    # Make main dataframe
    IM_df = pd.DataFrame(data=IM_dict)
    
    # Make spectra dataframe
    spec_df = pd.DataFrame(spectra_res, columns=spec_cols)
    
    # Combine dataframes 
    res_df = pd.concat([main_df,IM_df.reindex(main_df.index),spec_df.reindex(main_df.index)], axis=1)
    # res_df = pd.concat([main_df,IM_df,spec_df], ignore_index=True, axis=1)
    
    ## Save to file:
    res_df.to_csv('/Users/tnye/tsuquakes/flatfiles/residuals/'+parameter+'/'+project+'/runs/'+run+ '.csv',index=False)
    
    
################################ Group Res Plots ##############################
    
# run_list = []
# pgd_res_list = []
# pga_res_list = []
# pgv_res_list = []
# tPGD_orig_res_list = []
# tPGD_parriv_res_list = []
# spectra_res_list = []
    
# for run in runs:
    
#     # Calculate residuals
#     pgd_res,pga_res,pgv_res,tPGD_orig_res,tPGD_parriv_res,spectra_res = IM_fns.calc_res(parameter,project,run)
    
#     # Append run residuals to main list
#     pgd_res_list.append(pgd_res.tolist())
#     pga_res_list.append(pga_res.tolist())
#     pgv_res_list.append(pgv_res.tolist())
#     tPGD_orig_res_list.append(tPGD_orig_res.tolist())
#     tPGD_parriv_res_list.append(tPGD_parriv_res.tolist())
#     spectra_res_list.append(spectra_res.tolist())
    
#     for i in range(len(pga_res)):
#         run_list.append(run)

# lists = [pgd_res_list, pga_res_list, pgv_res_list, tPGD_orig_res_list,
#          tPGD_parriv_res_list, spectra_res_list]

# flatten = lambda l: [item for sublist in l for item in sublist]
# # for l in lists:         
# #     l = flatten(l)

# pgd_res_list = flatten(pgd_res_list)
# pga_res_list = flatten(pga_res_list)
# pgv_res_list = flatten(pgv_res_list)
# tPGD_orig_res_list = flatten(tPGD_orig_res_list)
# tPGD_parriv_res_list = flatten(tPGD_parriv_res_list)
# spectra_res_list = flatten(spectra_res_list)


# # # Set up dataframe    
# # IM_dict = {'run':run_list, 'pgd_res':pgd_res_list,'pga_res':pga_res_list,
# #            'pgv_res':pgv_res_list,'tPGD_origin_res':tPGD_orig_res_list,
# #            'tPGD_parriv_res':tPGD_parriv_res_list}
# # Set up dataframe    

# IM_dict = {'run':run_list, 'pga_res':pga_res_list,
#             'pgv_res':pgv_res_list}


# spec_cols = ['E_disp_bin1', 'E_disp_bin2', 'E_disp_bin3', 'E_disp_bin4',
#              'E_disp_bin5', 'E_disp_bin6', 'E_disp_bin7',
#              'E_disp_bin8', 'E_disp_bin9', 'E_disp_bin10', 'E_disp_bin11',
#              'E_disp_bin12', 'E_disp_bin13', 'E_disp_bin14', 'E_disp_bin15',
#              'E_disp_bin16', 'E_disp_bin17', 'E_disp_bin18', 'E_disp_bin19',
#              'E_disp_bin20', 'E_disp_bin21', 'E_disp_bin22', 'E_disp_bin23', 
#              'E_disp_bin24', 'E_disp_bin25', 'N_disp_bin1', 'N_disp_bin2',
#              'N_disp_bin3', 'N_disp_bin4', 'N_disp_bin5', 'N_disp_bin6',
#              'N_disp_bin7', 'N_disp_bin8', 'N_disp_bin9', 'N_disp_bin10',
#              'N_disp_bin11', 'N_disp_bin12', 'N_disp_bin13', 'N_disp_bin14',
#              'N_disp_bin15', 'N_disp_bin16', 'N_disp_bin17', 'N_disp_bin18',
#              'N_disp_bin19', 'N_disp_bin20', 'N_disp_bin21', 'N_disp_bin22',
#              'N_disp_bin23', 'N_disp_bin24', 'N_disp_bin25', 'Z_disp_bin1',
#              'Z_disp_bin2', 'Z_disp_bin3', 'Z_disp_bin4', 'Z_disp_bin5',
#              'Z_disp_bin6', 'Z_disp_bin7', 'Z_disp_bin8', 'Z_disp_bin9',
#              'Z_disp_bin10', 'Z_disp_bin11', 'Z_disp_bin12', 'Z_disp_bin13',
#              'Z_disp_bin14', 'Z_disp_bin15', 'Z_disp_bin16', 'Z_disp_bin17',
#              'Z_disp_bin18', 'Z_disp_bin19', 'Z_disp_bin20', 'Z_disp_bin21',
#              'Z_disp_bin22', 'Z_disp_bin23', 'Z_disp_bin24', 'Z_disp_bin25',
#              'E_acc_bin1', 'E_acc_bin2', 'E_acc_bin3', 'E_acc_bin4',
#              'E_acc_bin5', 'E_acc_bin6', 'E_acc_bin7', 'E_acc_bin8',
#              'E_acc_bin9', 'E_acc_bin10', 'E_acc_bin11', 'E_acc_bin12',
#              'E_acc_bin13', 'E_acc_bin14', 'E_acc_bin15', 'E_acc_bin16',
#              'E_acc_bin17', 'E_acc_bin18', 'E_acc_bin19', 'E_acc_bin20',
#              'E_acc_bin21', 'E_acc_bin22', 'E_acc_bin23', 'E_acc_bin24',
#              'E_acc_bin25', 'N_acc_bin1', 'N_acc_bin2', 'N_acc_bin3',
#              'N_acc_bin4', 'N_acc_bin5', 'N_acc_bin6', 'N_acc_bin7',
#              'N_acc_bin8', 'N_acc_bin9', 'N_acc_bin10', 'N_acc_bin11',
#              'N_acc_bin12', 'N_acc_bin13', 'N_acc_bin14', 'N_acc_bin15',
#              'N_acc_bin16', 'N_acc_bin17', 'N_acc_bin18', 'N_acc_bin19',
#              'N_acc_bin20', 'N_acc_bin21', 'N_acc_bin22', 'N_acc_bin23',
#              'N_acc_bin24', 'N_acc_bin25', 'Z_acc_bin1', 'Z_acc_bin2',
#              'Z_acc_bin3', 'Z_acc_bin4', 'Z_acc_bin5', 'Z_acc_bin6',
#              'Z_acc_bin7', 'Z_acc_bin8', 'Z_acc_bin9', 'Z_acc_bin10',
#              'Z_acc_bin11', 'Z_acc_bin12', 'Z_acc_bin13', 'Z_acc_bin14',
#              'Z_acc_bin15', 'Z_acc_bin16', 'Z_acc_bin17', 'Z_acc_bin18',
#              'Z_acc_bin19', 'Z_acc_bin20', 'Z_acc_bin21', 'Z_acc_bin22',
#              'Z_acc_bin23', 'Z_acc_bin24', 'Z_acc_bin25',
#              'E_vel_bin1', 'E_vel_bin2', 'E_vel_bin3', 'E_vel_bin4',
#              'E_vel_bin5', 'E_vel_bin6', 'E_vel_bin7', 'E_vel_bin8',
#              'E_vel_bin9', 'E_vel_bin10', 'E_vel_bin11', 'E_vel_bin12',
#              'E_vel_bin13', 'E_vel_bin14', 'E_vel_bin15', 'E_vel_bin16',
#              'E_vel_bin17', 'E_vel_bin18', 'E_vel_bin19', 'E_vel_bin20',
#              'E_vel_bin21', 'E_vel_bin22', 'E_vel_bin23', 'E_vel_bin24',
#              'E_vel_bin25', 'N_vel_bin1', 'N_vel_bin2', 'N_vel_bin3',
#              'N_vel_bin4', 'N_vel_bin5', 'N_vel_bin6', 'N_vel_bin7',
#              'N_vel_bin8', 'N_vel_bin9', 'N_vel_bin10', 'N_vel_bin11',
#              'N_vel_bin12', 'N_vel_bin13', 'N_vel_bin14', 'N_vel_bin15',
#              'N_vel_bin16', 'N_vel_bin17', 'N_vel_bin18', 'N_vel_bin19',
#              'N_vel_bin20', 'N_vel_bin21', 'N_vel_bin22', 'N_vel_bin23',
#              'N_vel_bin24', 'N_vel_bin25', 'Z_vel_bin1', 'Z_vel_bin2',
#              'Z_vel_bin3', 'Z_vel_bin4', 'Z_vel_bin5', 'Z_vel_bin6',
#              'Z_vel_bin7', 'Z_vel_bin8', 'Z_vel_bin9', 'Z_vel_bin10',
#              'Z_vel_bin11', 'Z_vel_bin12', 'Z_vel_bin13', 'Z_vel_bin14',
#              'Z_vel_bin15', 'Z_vel_bin16', 'Z_vel_bin17', 'Z_vel_bin18',
#              'Z_vel_bin19', 'Z_vel_bin20', 'Z_vel_bin21', 'Z_vel_bin22',
#              'Z_vel_bin23', 'Z_vel_bin24', 'Z_vel_bin25']

# # Make main dataframe
# IM_df = pd.DataFrame(data=IM_dict)

# # Make spectra dataframe
# spec_df = pd.DataFrame(spectra_res_list, columns=spec_cols)

# # Add main_df for each run
# main_df_full = pd.concat([main_df]*len(runs), ignore_index=True)

# # Combine dataframes 
# res_df = pd.concat([main_df_full,IM_df,spec_df], axis=1)

# ## Save to file:
# res_df.to_csv('/Users/tnye/tsuquakes/flatfiles/residuals/'+parameter+'/'+project+'/'+project+'_res.csv',index=False)


# # Plot residuals 
# IM_fns.plot_res(parameter, project, runs)
    
    
    
    
    





            

