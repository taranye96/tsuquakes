#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:58:46 2020

@author: tnye
"""

import numpy as np
import pandas as pd

# Load in .csv for Q map from Eberhart-Phillips 2016)
q_file = pd.read_csv('/Users/tnye/kappa/eberhart_phillips_q.csv')

# Find rows in Q file that are within my dataset range and are either 4 or 6 km deep
lat_range = q_file['Latitude'].between(36, 39)
lon_range = q_file['Longitude'].between(-124, -120.5)
q4_range = q_file['Depth(km_BSL)'] == 4
q6_range = q_file['Depth(km_BSL)'] == 6

# Find indexes of rows that meet all of the requirements
q4_indexes = np.where(lat_range==True,lon_range==True,q4_range==True)
q6_indexes = np.where(lat_range==True,lon_range==True,q6_range==True)

# Make a df of the indexes
q4_df = q_file[q4_indexes]
q6_df = q_file[q6_indexes]

# Q mean for depth of 4 km and 6 km
q4_mean = np.mean(q4_df['Qs'])
q6_mean = np.mean(q6_df['Qs'])

# Overall mean Q
q_mean = np.mean([q4_mean, q6_mean])
