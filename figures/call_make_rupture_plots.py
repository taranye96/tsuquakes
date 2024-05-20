#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 21:18:18 2022

@author: tnye
"""

###############################################################################
# This script calls make_ruputre_plots.gmt which plots the individual rupture
# models.
###############################################################################

# Imports
from glob import glob
from os import chdir, path, makedirs
import subprocess

# chdir('/Users/tnye/tsuquakes/GMT/scripts')

parameter = 'ideal_runs_m7.8'

cpt='magma_white'

slip_files = sorted(glob(f'/Users/tnye/tsuquakes/rupture_models/{parameter}/*.txt'))

figdir = f'/Users/tnye/tsuquakes/manuscript/figures/rupture_models/{parameter}/{cpt}'
if not path.exists(figdir):
    makedirs(figdir)
    
# a = []
for slip_file in slip_files:
    rupt_name = slip_file.split('/')[-1].strip('.txt')
    rupt_file = f'/Users/tnye/tsuquakes/simulations/{parameter}/standard/output/ruptures/{rupt_name}.rupt'
    subprocess.run(['/Users/tnye/tsuquakes/code/figures/call_plot_example_ruptures.gmt',cpt,rupt_name,rupt_file,slip_file,figdir])
