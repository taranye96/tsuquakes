#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:17:28 2020

@author: tnye
"""

###############################################################################
# Script that makes signal to noise ratio vs hypocentral distance plots for
# PGA, PGV, and PGD at for the GNSS and sm stations. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in gnss and strong motion SNR dataframes
gnss_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/gnss_SNR.csv')
sm_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/sm_SNR.csv')

# Get IM, distance and SNR values from dataframes
pgd = [np.array(gnss_df['pgd(m)'])[index] for index in sorted(np.unique(np.array(gnss_df['pgd(m)']),
                return_index=True)[1])]
pga = [np.array(sm_df['pga'])[index] for index in sorted(np.unique(np.array(sm_df['pga']),
                return_index=True)[1])]
pgv = [np.array(sm_df['pgv'])[index] for index in sorted(np.unique(np.array(sm_df['pgv']),
                return_index=True)[1])]
gnss_dist = [np.array(gnss_df['hypdist'])[index] for index in sorted(np.unique(np.array(gnss_df['hypdist']),
                return_index=True)[1])]
sm_dist = [np.array(sm_df['hypdist'])[index] for index in sorted(np.unique(np.array(sm_df['hypdist']),
                return_index=True)[1])]
gnss_snr = [np.array(gnss_df['SNR'])[index] for index in sorted(np.unique(np.array(gnss_df['SNR']),
                return_index=True)[1])]
sm_snr = [np.array(sm_df['SNR'])[index] for index in sorted(np.unique(np.array(sm_df['SNR']),
                return_index=True)[1])]

##### PGA #####

plt.figure()
plt.scatter(sm_dist, np.log10(pga), c=np.log10(sm_snr), cmap='Purples')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGA) (m/s/s)')
plt.title('PGA vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')
# plt.savefig('/Users/tnye/tsuquakes/plots/SNR/pga_purples.png', dpi=300)

plt.figure()
plt.scatter(sm_dist, np.log10(pga), c=np.log10(sm_snr), cmap='cool')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGA) (m/s/s)')
plt.title('PGA vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')
# plt.savefig('/Users/tnye/tsuquakes/plots/SNR/pga_cool.png', dpi=300)


##### PGV #####

plt.figure()
plt.scatter(sm_dist, np.log10(pgv), c=np.log10(sm_snr), cmap='Purples')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGV) (m/s)')
plt.title('PGV vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')
# plt.savefig('/Users/tnye/tsuquakes/plots/SNR/pgv_purples.png', dpi=300)

plt.figure()
plt.scatter(sm_dist, np.log10(pgv), c=np.log10(sm_snr), cmap='cool')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGV) (m/s)')
plt.title('PGV vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')
# plt.savefig('/Users/tnye/tsuquakes/plots/SNR/pgv_cool.png', dpi=300)


##### PGD #####

plt.figure()
plt.scatter(gnss_dist, np.log10(pgd), c=np.log10(gnss_snr), cmap='Purples')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGD) (m)')
plt.title('PGD vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')
# plt.savefig('/Users/tnye/tsuquakes/plots/SNR/pgd_purples.png', dpi=300)


