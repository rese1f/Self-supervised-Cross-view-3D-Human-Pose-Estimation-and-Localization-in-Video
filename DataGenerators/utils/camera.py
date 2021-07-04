# this file is to 

import numpy as np

def get_center():
    '''
    input: first frame
    output: center location
    '''

def get_endpoint():
    '''
    input: distance, height, cross, center location
    output: two endpoint location
    '''

def T():
    '''
    input: two endpoint location, frame
    output: location of camera by the time
    '''
    
def R():
    '''
    input: location of camera by the time, tracking, center location
    output: rotation of camera by the time
    '''

def exmat():
    '''
    input: T and R
    output: exmat of camera by time
    '''

def w2c():
    '''
    input: exmat, data_3d_std
    output: data_c_std
    '''

def c2s():
    '''
    input: w2c
    output: data_2d_std
    '''