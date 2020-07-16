#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:17:48 2020

@author: tnye
"""

import IM_fns
from obspy import read

stream = read('/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/BSAT.LXE.corr')

bin_means, freqs, amps = IM_fns.calc_spectra(stream, 'disp')
