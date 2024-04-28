#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:23:41 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

dfs_5km = glob('/Users/tnye/FakeQuakes/simulations/fault_coarseness/mentawai_5km/flatfiles/IMs/*_sm.csv')
dfs_3km = glob('/Users/tnye/FakeQuakes/simulations/fault_coarseness/mentawai_3km/flatfiles/IMs/*_sm.csv')
dfs_2km = glob('/Users/tnye/FakeQuakes/simulations/fault_coarseness/mentawai_2km/flatfiles/IMs/*_sm.csv')

pga_5km = np.array([])
pgv_5km = np.array([])
pga_3km = np.array([])
pgv_3km = np.array([])
pga_2km = np.array([])
pgv_2km = np.array([])
hypdist = np.array([])

for file in dfs_5km:
    df = pd.read_csv(file)
    pga_5km = np.append(pga_5km, df['pga'])
    pgv_5km = np.append(pgv_5km, df['pgv'])
    hypdist = np.append(hypdist,df['hypdist'])
    
for file in dfs_3km:
    df = pd.read_csv(file)
    pga_3km = np.append(pga_3km, df['pga'])
    pgv_3km = np.append(pgv_3km, df['pgv'])
    
for file in dfs_2km:
    df = pd.read_csv(file)
    pga_2km = np.append(pga_2km, df['pga'])
    pgv_2km = np.append(pgv_2km, df['pgv'])
    

# Plot
fig, ax = plt.subplots(1,1)
ax.scatter(hypdist, pga_5km, label='len = 5km')
ax.scatter(hypdist, pga_3km, label='len = 3km')
ax.scatter(hypdist, pga_2km, label='len = 2km')
ax.set_xlabel('Hypdist (km)')
ax.set_ylabel('PGA (m/s/s)')
plt.legend()
plt.subplots_adjust(left=0.15, bottom=0.15)


fig, ax = plt.subplots(1,1)
ax.scatter(hypdist, pgv_5km, label='len = 5km')
ax.scatter(hypdist, pgv_3km, label='len = 3km')
ax.scatter(hypdist, pgv_2km, label='len = 2km')
ax.set_xlabel('Hypdist (km)')
ax.set_ylabel('PGV (m/s)')
plt.legend()
plt.subplots_adjust(left=0.15, bottom=0.15)