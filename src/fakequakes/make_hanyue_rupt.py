#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:33:55 2022

@author: tnye
"""

import numpy as np
import pandas as pd

han_yue = pd.read_csv('/Users/tnye/tsuquakes/files/han_yue_fault.txt', delimiter=',')

nsub = np.arange(1,len(han_yue)+1,1)
lon = np.array(han_yue['lon'])
lat = np.array(han_yue['lat'])
depth = np.array(han_yue['depth'])
strike = np.array(han_yue['strike'])
dip = np.array(han_yue['dip'])
rise = np.full_like(dip, 0)
dura = np.full_like(dip, 0)
ss_slip = np.full_like(dip, 0)
ds_slip = np.array(han_yue['slip(m)'])
width = np.full_like(dip, 14250)
length = np.full_like(dip, 15000)
rupt_time = np.full_like(dip, 0)
rigidity = np.full_like(dip, 0)


outfile = open(f'/Users/tnye/tsuquakes/data/fault_info/han_yue.rupt', 'w')
file_data = np.array([nsub,lon,lat,depth,strike,dip,rise,dura,ss_slip,ds_slip,width,length,rupt_time,rigidity],dtype=object)
file_data = file_data.T
np.savetxt(outfile, file_data, fmt='%E', delimiter='\t')
outfile.close()
