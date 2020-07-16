#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:42:11 2020

@author: tnye
"""

import numpy as np

def calc_MMI(pgm, pgm_type):
    """
    """
    
    if pgm_type == 'pga':
        c1 = 1.78
        c2 = 1.55
        c3 = -1.60
        c4 = 3.70
        
        if np.log10(pgm) <= 1.57:
            MMI = c1 + c2*np.log10(pgm)
        else:
            MMI = c3 + c4*np.log10(pgm)
            
    elif pgm_type == 'pgv':
        c1 = 3.78
        c2 = 1.47
        c3 = 2.89
        c4 = 3.16
        
        if np.log10(pgm) <= 0.53:
            MMI = c1 + c2*np.log10(pgm)
        else:
            MMI = c3 + c4*np.log10(pgm)
    
    return(MMI)
    