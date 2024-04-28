#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:52:08 2023

@author: tnye
"""

# Imports
import numpy as np
from obspy import read
import matplotlib.pyplot as plt
import IM_fns

E_file = '/Users/tnye/tsuquakes/data/waveforms/individual/disp/BSAT.LXE.mseed'
N_file = '/Users/tnye/tsuquakes/data/waveforms/individual/disp/BSAT.LXN.mseed'
Z_file = '/Users/tnye/tsuquakes/data/waveforms/individual/disp/BSAT.LXZ.mseed'

st_E = read(E_file)
st_N = read(N_file)
st_Z = read(Z_file)

bins, E_spec_data = IM_fns.calc_spectra(st_E, 'gnss')
bins, N_spec_data = IM_fns.calc_spectra(st_N, 'gnss')
bins, Z_spec_data = IM_fns.calc_spectra(st_Z, 'gnss')

NE_data2 = np.sqrt(E_spec_data**2 + N_spec_data**2)
NE_data3 = np.sqrt(E_spec_data**2 + N_spec_data**2 + Z_spec_data**2)


fig, ax = plt.subplots()
ax.loglog(bins, NE_data2, label='2-component')
ax.loglog(bins, NE_data3, label='3-component')
plt.legend()