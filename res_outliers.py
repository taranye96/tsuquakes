#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:12:35 2020

@author: tnye
"""

import numpy as np
import pandas as pd
from glob import glob
import sys

sys.stdout = open('/Users/tnye/tsuquakes/data/res_outlier_stns.txt','wt')
print('Outlier stations for IMs for simulation 2min_16cores')

res_dfs = np.array(sorted(glob('/Users/tnye/tsuquakes/flatfiles/residuals/2min_16cores/' + '*.csv')))

for file in res_dfs:
    run = file.split('/')[-1].split('.')[1]
    res_df = pd.read_csv(file)
    
    pgd_res = np.array(res_df['pgd_res'])[:13]
    pga_res = np.array(res_df['pga_res'])[13:]
    pgv_res = np.array(res_df['pgv_res'])[13:]
    gnss_stations = np.array(res_df['station'])[:13]
    sm_stations = np.array(res_df['station'])[13:]
    
    pgd_out = np.max((np.abs(pgd_res)))
    pgd_out_id = np.where(np.abs(pgd_res)==pgd_out)
    pga_out = np.max((np.abs(pga_res)))
    pga_out_id = np.where(np.abs(pga_res)==pga_out)
    pgv_out = np.max((np.abs(pgv_res)))
    pgv_out_id = np.where(np.abs(pgv_res)==pgv_out)
    
    pgd_out_stn = gnss_stations[pgd_out_id]
    pga_out_stn = sm_stations[pga_out_id]
    pgv_out_stn = sm_stations[pgv_out_id]
    
    print(f'run {run}: pgd={pgd_out_stn} pga={pga_out_stn} pgv={pgv_out_stn}')