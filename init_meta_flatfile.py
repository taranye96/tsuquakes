#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:59:00 2019

@author: tnye
"""

# Imports
import pandas as pd

# Define columns 
columns = ['station', 'run #', 'hyplon', 'hyplat', 'risetime', 'vrup (km/s)',
           'PGA (m/s/s)', 'PGV (m/s)', 'PGD (m)', 'Efs_acc bin1',
           'Efs_acc bin2', 'Efs_acc bin3', 'Efs_acc bin4', 'Efs_acc bin5',
           'Efs_acc bin6', 'Efs_acc bin7', 'Efs_acc bin8', 'Efs_acc bin9',
           'Efs_acc bin10', 'Efs_acc bin11', 'Efs_acc bin12', 'Efs_acc bin13',
           'Efs_acc bin14', 'Efs_acc bin15', 'Efs_acc bin16', 'Efs_acc bin17',
           'Efs_acc bin18', 'Efs_acc bin19', 'Efs_acc bin20', 'Efs_acc bin21',
           'Efs_acc bin22', 'Efs_acc bin23', 'Efs_acc bin24', 'Efs_acc bin25',
           'Nfs_acc bin1',
           'Nfs_acc bin2', 'Nfs_acc bin3', 'Nfs_acc bin4', 'Nfs_acc bin5',
           'Nfs_acc bin6', 'Nfs_acc bin7', 'Nfs_acc bin8', 'Nfs_acc bin9',
           'Nfs_acc bin10', 'Nfs_acc bin11', 'Nfs_acc bin12', 'Nfs_acc bin13',
           'Nfs_acc bin14', 'Nfs_acc bin15', 'Nfs_acc bin16', 'Nfs_acc bin17',
           'Nfs_acc bin18', 'Nfs_acc bin19', 'Nfs_acc bin20', 'Nfs_acc bin21',
           'Nfs_acc bin22', 'Nfs_acc bin23', 'Nfs_acc bin24', 'Nfs_acc bin25',
           'Zfs_acc bin1',
           'Zfs_acc bin2', 'Zfs_acc bin3', 'Zfs_acc bin4', 'Zfs_acc bin5',
           'Zfs_acc bin6', 'Zfs_acc bin7', 'Zfs_acc bin8', 'Zfs_acc bin9',
           'Zfs_acc bin10', 'Zfs_acc bin11', 'Zfs_acc bin12', 'Zfs_acc bin13',
           'Zfs_acc bin14', 'Zfs_acc bin15', 'Zfs_acc bin16', 'Zfs_acc bin17',
           'Zfs_acc bin18', 'Zfs_acc bin19', 'Zfs_acc bin20', 'Zfs_acc bin21',
           'Zfs_acc bin22', 'Zfs_acc bin23', 'Zfs_acc bin24', 'Zfs_acc bin25',
           'Efs_vel bin1', 'Efs_vel bin2', 'Efs_vel bin3', 'Efs_vel bin4',
           'Efs_vel bin5', 'Efs_vel bin6', 'Efs_vel bin7', 'Efs_vel bin8',
           'Efs_vel bin9', 'Efs_vel bin10', 'Efs_vel bin11', 'Efs_vel bin12',
           'Efs_vel bin13', 'Efs_vel bin14', 'Efs_vel bin15', 'Efs_vel bin16',
           'Efs_vel bin17', 'Efs_vel bin18', 'Efs_vel bin19', 'Efs_vel bin20',
           'Efs_vel bin21', 'Efs_vel bin22', 'Efs_vel bin23', 'Efs_vel bin24',
           'Efs_vel bin25', 'Nfs_vel bin1', 'Nfs_vel bin2', 'Nfs_vel bin3', 'Nfs_vel bin4',
           'Nfs_vel bin5', 'Nfs_vel bin6', 'Nfs_vel bin7', 'Nfs_vel bin8',
           'Nfs_vel bin9', 'Nfs_vel bin10', 'Nfs_vel bin11', 'Nfs_vel bin12',
           'Nfs_vel bin13', 'Nfs_vel bin14', 'Nfs_vel bin15', 'Nfs_vel bin16',
           'Nfs_vel bin17', 'Nfs_vel bin18', 'Nfs_vel bin19', 'Nfs_vel bin20',
           'Nfs_vel bin21', 'Nfs_vel bin22', 'Nfs_vel bin23', 'Nfs_vel bin24',
           'Nfs_vel bin25', 'Zfs_vel bin1', 'Zfs_vel bin2', 'Zfs_vel bin3', 'Zfs_vel bin4',
           'Zfs_vel bin5', 'Zfs_vel bin6', 'Zfs_vel bin7', 'Zfs_vel bin8',
           'Zfs_vel bin9', 'Zfs_vel bin10', 'Zfs_vel bin11', 'Zfs_vel bin12',
           'Zfs_vel bin13', 'Zfs_vel bin14', 'Zfs_vel bin15', 'Zfs_vel bin16',
           'Zfs_vel bin17', 'Zfs_vel bin18', 'Zfs_vel bin19', 'Zfs_vel bin20',
           'Zfs_vel bin21', 'Zfs_vel bin22', 'Zfs_vel bin23', 'Zfs_vel bin24',
           'Zfs_vel bin25', 'Efs_disp bin1', 'Efs_disp bin2', 'Efs_disp bin3',
           'Efs_disp bin4', 'Efs_disp bin5', 'Efs_disp bin6', 'Efs_disp bin7',
           'Efs_disp bin8', 'Efs_disp bin9', 'Efs_disp bin10', 'Efs_disp bin11',
           'Efs_disp bin12', 'Efs_disp bin13', 'Efs_disp bin14', 'Efs_disp bin15',
           'Efs_disp bin16', 'Efs_disp bin17', 'Efs_disp bin18', 'Efs_disp bin19',
           'Efs_disp bin20', 'Efs_disp bin21', 'Efs_disp bin22', 'Efs_disp bin23', 
           'Efs_disp bin24', 'Efs_disp bin25', 'Nfs_disp bin1', 'Nfs_disp bin2', 'Nfs_disp bin3',
           'Nfs_disp bin4', 'Nfs_disp bin5', 'Nfs_disp bin6', 'Nfs_disp bin7',
           'Nfs_disp bin8', 'Nfs_disp bin9', 'Nfs_disp bin10', 'Nfs_disp bin11',
           'Nfs_disp bin12', 'Nfs_disp bin13', 'Nfs_disp bin14', 'Nfs_disp bin15',
           'Nfs_disp bin16', 'Nfs_disp bin17', 'Nfs_disp bin18', 'Nfs_disp bin19',
           'Nfs_disp bin20', 'Nfs_disp bin21', 'Nfs_disp bin22', 'Nfs_disp bin23', 
           'Nfs_disp bin24', 'Nfs_disp bin25', 'Zfs_disp bin1', 'Zfs_disp bin2', 'Zfs_disp bin3',
           'Zfs_disp bin4', 'Zfs_disp bin5', 'Zfs_disp bin6', 'Zfs_disp bin7',
           'Zfs_disp bin8', 'Zfs_disp bin9', 'Zfs_disp bin10', 'Zfs_disp bin11',
           'Zfs_disp bin12', 'Zfs_disp bin13', 'Zfs_disp bin14', 'Zfs_disp bin15',
           'Zfs_disp bin16', 'Zfs_disp bin17', 'Zfs_disp bin18', 'Zfs_disp bin19',
           'Zfs_disp bin20', 'Zfs_disp bin21', 'Zfs_disp bin22', 'Zfs_disp bin23', 
           'Zfs_disp bin24', 'Zfs_disp bin25']

# Create and save empty dataframe
df = pd.DataFrame(columns=columns)

flatfile_path = '/Users/tnye/tsuquakes/flatfiles/meta_flatfile.csv'

df.to_csv(flatfile_path,index=False)