import numpy as np
import matplotlib.pyplot as plt

def make_gaus(size = 128, res = 0.1, center = None, data_spacing = 0.01):

    #Make an x and y axis
    x = np.arange(0, size*data_spacing, data_spacing, float)
    y = x[:,np.newaxis]
    
    #Find center of Guassian
    if center is None:
        x0 = (x[0]+x[-1])/2
        y0 = (y[0]+y[-1])/2
    else:
        x0 = center[0]
        y0 = center[1]
        
    #Generate matrix
    gaus = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / res**2)
      

    return gaus 

def make_file(filename, size = 128, res = 0.1, center = None, data_space = 0.01):

