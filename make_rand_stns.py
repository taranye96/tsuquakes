#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:08:41 2022

@author: tnye
"""

###############################################################################
# Makes some random station locations around Mentawai to compare with GMM. 
###############################################################################

import random
import numpy as np

lon = np.array([])
lat = np.array([])
nstns = 20

for i in range(nstns):
    lon = np.append(lon, round(random.uniform(99.500, 100.500), 3))
    lat = np.append(lat, round(random.uniform(-4, -2.5), 3))

out = np.c_[lon,lat]
np.savetxt('/Users/tnye/tsuquakes/files/random_stns.csv',out,fmt='%.3f,%.3f',header='lon,lat')


