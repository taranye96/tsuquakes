#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:24:50 2020

@author: tnye
"""

def get_geom_avg_3comp(E_record, N_record, Z_record):
    """
    Get the geometric average of the three components of a record.

    Inputs:
        E_record(array): Full record of East-West component.
        N_record(array): Full record of North-South component
        Z_record(array): Full record of vertical component 
    
    Return:
        geom_avg(array): Full record of geometric average.
    """

    import numpy as np

    geom_avg = np.cbrt(E_record * N_record * Z_record)

    return geom_avg


def get_geom_avg_2comp(E_record, N_record):
    """
    Get the geometric average of two components of a record (most likely the
    horizontal components).

    Inputs:
        E_record(array): Full record of East-West component.
        N_record(array): Full record of North-South component
    
    Return:
        geom_avg(array): Full record of geometric average.
    """

    import numpy as np

    geom_avg = np.sqrt(E_record * N_record)

    return geom_avg


def get_eucl_norm_3comp(E_record, N_record, Z_record):
    """
    Get the euclidean norm of the three components of a record.  This is
    equivalent to calculating the magnitude of a vector. 

    Inputs:
        E_record(array): Record of East-West component data.
        N_record(array): Record of North-South component data.
        Z_record(array): Record of vertical component data.
    
    Return:
        eucl_norm(array): Full record of euclidian norm.
    """

    import numpy as np

    eucl_norm = np.sqrt(E_record**2 + N_record**2 + Z_record**2)

    return eucl_norm


def get_eucl_norm_2comp(E_record, N_record):
    """
    Get the euclidean norm of two components of a record (most likely the
    horizontal components).  This is equivalent to calculating the magnitude of
    a vector. 

    Inputs:
        E_record(array): Full record of East-West component.
        N_record(array): Full record of North-South component
    
    Return:
        eucl_norm(array): Full record of euclidian norm.
    """

    import numpy as np

    eucl_norm = np.sqrt(E_record**2 + N_record**2)

    return eucl_norm