#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:17:28 2020

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/gnss_SNR.csv')

pga = [np.array(df['pga'])[index] for index in sorted(np.unique(np.array(df['pga']),
                return_index=True)[1])]
pgv = [np.array(df['pgv'])[index] for index in sorted(np.unique(np.array(df['pgv']),
                return_index=True)[1])]
pgd = [np.array(df['pgd(m)'])[index] for index in sorted(np.unique(np.array(df['pgd(m)']),
                return_index=True)[1])]
dist = [np.array(df['hypdist'])[index] for index in sorted(np.unique(np.array(df['hypdist']),
                return_index=True)[1])]
snr = [np.array(df['SNR'])[index] for index in sorted(np.unique(np.array(df['SNR']),
                return_index=True)[1])]

##### PGA
plt.scatter(dist, np.log10(pga), c=np.log10(snr), cmap='Purples')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGA) (m/s/s)')
plt.title('PGA vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')

figpath = '/Users/tnye/tsuquakes/plots/SNR/pga_purples.png'
plt.savefig(figpath, dpi=300)

plt.close()

plt.scatter(dist, np.log10(pga), c=np.log10(snr), cmap='cool')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGA) (m/s/s)')
plt.title('PGA vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')

figpath = '/Users/tnye/tsuquakes/plots/SNR/pga_cool.png'
plt.savefig(figpath, dpi=300)

plt.close()


##### PGV
plt.scatter(dist, np.log10(pgv), c=np.log10(snr), cmap='Purples')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGV) (m/s)')
plt.title('PGV vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')

figpath = '/Users/tnye/tsuquakes/plots/SNR/pgv_purples.png'
plt.savefig(figpath, dpi=300)

plt.close()

plt.scatter(dist, np.log10(pgv), c=np.log10(snr), cmap='cool')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGV) (m/s)')
plt.title('PGV vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')

figpath = '/Users/tnye/tsuquakes/plots/SNR/pgv_cool.png'
plt.savefig(figpath, dpi=300)

plt.close()

##### PGD
plt.scatter(dist, np.log10(pgd), c=np.log10(snr), cmap='Purples')
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('log$_{10}$(PGD) (m)')
plt.title('PGD vs Hypdist')
plt.colorbar(label='log$_{10}$(SNR)')

figpath = '/Users/tnye/tsuquakes/plots/SNR/pgd_purples.png'
plt.savefig(figpath, dpi=300)

plt.close()

