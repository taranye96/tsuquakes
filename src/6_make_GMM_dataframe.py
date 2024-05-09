#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:23:28 2022

@author: tnye
"""

###############################################################################
# Script used to find location of code or other files. 
###############################################################################

# Imports
import numpy as np
import scipy as sp
import pandas as pd
from glob import glob
from obspy import read
import single_shakemaps
from rotd50 import compute_rotd50

df = pd.read_csv('/Users/tnye/FakeQuakes/cacadia_attenuation_test/flatfiles/IMs/cascadia.000000.csv')
acc_wfs_E = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.87/standard_parameters/processed_wfs/acc/*/*HNE*'))
acc_wfs_N = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.87/standard_parameters/processed_wfs/acc/*/*HNN*'))
vel_wfs_E = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.87/standard_parameters/processed_wfs/vel/*/*HNE*'))
vel_wfs_N = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.87/standard_parameters/processed_wfs/vel/*/*HNN*'))

rrup_list = np.array([])
vs30 = np.array([])
ngasub_pga = np.array([])
ngasub_pga_sd = np.array([])
ngasub_pgv = np.array([])
ngasub_pgv_sd = np.array([])
BCHydro_pga = np.array([])
BCHydro_pga_sd = np.array([])
fq_pga = np.array([])
fq_pgv = np.array([])
stations = np.array([])
stlons = []
stlats = []
mws = []
hyplons = []
hyplats = []
hypdepths = []
hypdists = []

M = 7

for i in range(len(acc_wfs_E)):
    
    stn = acc_wfs_E[i].split('/')[-1].split('.')[0]
    stations = np.append(stations, stn)
    ind = np.where(df['station']==stn)[0][0]
    
    acc_st_E = read(acc_wfs_E[i])
    acc_st_N = read(acc_wfs_N[i])
    acc = compute_rotd50(acc_st_E[0].data,acc_st_N[0].data)
    
    vel_st_E = read(vel_wfs_E[i])
    vel_st_N = read(vel_wfs_N[i])
    vel = compute_rotd50(vel_st_E[0].data,vel_st_N[0].data)
    
    fq_pga = np.append(fq_pga, np.max(np.abs(acc)))
    fq_pgv = np.append(fq_pgv, np.max(np.abs(vel)))
    mws.append(df['mw'].iloc[ind])
    hyplons.append(df['hyplon'].iloc[ind])
    hyplats.append(df['hyplat'].iloc[ind])
    hypdepths.append(df['hypdepth (km)'].iloc[ind])
    hypdists.append(df['hypdist'].iloc[ind])
    stlons.append(df['stlon'].iloc[ind])
    stlats.append(df['stlat'].iloc[ind])

rupt_files = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.87/standard_parameters/output/ruptures/mentawai*.rupt'))
vs30_csv = '/Users/tnye/tsuquakes/data/vs30/sm_vs30.txt'

for rupt_file in rupt_files:
    single_shakemaps.run_shakemaps(rupt_file, vs30_csv)
    
    batch = rupt_file.split('/')[-4]
    run = rupt_file.split('/')[-1].strip('.rupt')

    # Read in dfs for BCHydro and NGA-Sub and extract IM info
    shake_file = f'/Users/tnye/tsuquakes/shakemaps/shakefiles/{batch}/{run}.shake'
    shake_df = pd.read_csv(shake_file,delimiter=',')
    
    BCHydro_pga = np.append(BCHydro_pga, np.array(shake_df['PGA_bchydro2018_g'])*9.8)
    BCHydro_pga_sd = np.append(BCHydro_pga_sd, np.exp(np.array(shake_df['PGAsd_bchydro2018_lng']))*9.8)
    ngasub_pga = np.append(ngasub_pga, np.array(shake_df['PGA_NGASub_g'])*9.8)
    ngasub_pga_sd = np.append(ngasub_pga_sd, np.exp(np.array(shake_df['PGAsd_NGASub_lng']))*9.8)
    ngasub_pgv = np.append(ngasub_pgv, np.array(shake_df['PGV_NGASub_cm/s'])/100)
    ngasub_pgv_sd = np.append(ngasub_pgv_sd, np.exp(np.array(shake_df['PGVsd_NGASub_lncm/s']))/100)
    
    rrup_list = np.append(rrup_list, np.array(shake_df['Rrupt(km)']))

ngasub_pga_res = np.log10(ngasub_pga) - np.log10(fq_pga)
ngasub_pgv_res = np.log10(ngasub_pgv) - np.log10(fq_pgv)
BCHydro_pga_res = np.log10(BCHydro_pga) - np.log10(fq_pga)

dataset_dict = {'Station':stations,'stlon':stlons,'stlat':stlats,'Mw':mws,
                'hyplon':hyplons,'hyplat':hyplats,'hypdepth(km)':hypdepths,
                'Rhyp(km)':hypdists,'Rrup(km)':rrup_list,
                'syn_PGA_m/s2':fq_pga,'syn_PGV_m/s':fq_pgv,'BCHydro_pga_m/s2':BCHydro_pga,
                'BCHydro_sd_m/s2':BCHydro_pga_sd,'NGASub_pga_m/s2':ngasub_pga,
                'NGA_Sub_sd_m/s2':ngasub_pga_sd,'NGA_Sub_pgv_m/s':ngasub_pgv,
                'NGA_Sub_sd_m/s':ngasub_pgv_sd,'lnPGA_res_BCHydro':BCHydro_pga_res,
                'lnPGA_res_NGASub':ngasub_pga_res,'lnPGV_res_NGASub':ngasub_pgv_res}

# Make main dataframe using dictionary 
df = pd.DataFrame(data=dataset_dict)

# Save to flatfile:
df.to_csv('/Users/tnye/tsuquakes/GMM/new_runs_m7.87_HF_residuals.csv',index=False)

 
 
 
    
    