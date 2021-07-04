# this file is to 

import numpy as np

def get_center():
    '''
    input: first frame infomation
    output: center location
    '''

def get_endpoint():
    '''
    input: distance, height, cross, center location
    output: two endpoint location
    '''
    get_center()

def T():
    '''
    input: two endpoint location, frame
    output: location of camera by the time
    '''
    get_endpoint()
    
def exmat():
    '''
    input: location of camera by the time, tracking, center location
    output: exmat of camera by the time
    '''
    T()
    get_center()

def w2c():
    '''
    input: exmat, data_3d_std
    output: data_c_std
    '''
    exmat()

def c2s():
    '''
    input: w2c
    output: data_2d_std
    '''
    w2c()