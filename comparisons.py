#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:07:20 2019

@author: tnye
"""

###############################################################################
# Script that makes comparison plots between IMs (PGA, PGV, PGD) and hypocentral
# distance. 
##############################################################################


# Standard Library Imports 
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

# Read in dataframes
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/pga_pgv.csv')
df_pgd = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/pgd_flatfile.csv')

# Corrected Miniseed files directories
acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr'
vel_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_filt'

# GPS files directory 
gps_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp'

# Get corrected mseed files
corr_acc = np.array(sorted((glob(acc_dir + '/*'))))
filt_vel = np.array(sorted((glob(vel_dir + '/*'))))

# Get GPS data
gps_files = np.array(sorted((glob(gps_dir + '/*'))))

# Get arrays of pga, pgv, and hypocentral distance 
pga = np.array(df['pga'])
pgv = np.array(df['pgv'])
pgd = np.array(df_pgd['pgd_meters'])
hypdist = np.array(df['hypdist'])
hypdist_pgd = np.array(df_pgd['hypdist'])

origintime = pd.to_datetime('2010-10-25T14:42:22')

# Comparison vs hypocentral distance 

# PGA
plt.scatter(hypdist, np.log(pga))
plt.xlabel('Hypocentral Dist (m)')
plt.ylabel('lnPGA (m/s/s)')
plt.title('PGA vs Hypdist')

figpath = '/Users/tnye/tsuquakes/plots/comparisons/pga_dist.png'

plt.savefig(figpath, dpi=300)
plt.close()

# PGV 
plt.scatter(hypdist, np.log(pgv))
plt.xlabel('Hypocentral Dist (m)')
plt.ylabel('lnPGV (m/s)')
plt.title('PGV vs Hypdist')

figpath = '/Users/tnye/tsuquakes/plots/comparisons/pgv_dist.png'

plt.savefig(figpath, dpi=300)
plt.close()

# PGD
plt.scatter(hypdist_pgd, pgd)
plt.xlabel('Hypocentral Dist (m)')
plt.ylabel('PGD (m)')
plt.title('PGD vs Hypdist')

figpath = '/Users/tnye/tsuquakes/plots/comparisons/pgd_dist.png'

plt.savefig(figpath, dpi=300)
plt.close()