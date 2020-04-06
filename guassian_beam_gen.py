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
      
    return gaus, x0, y0[0] 

def make_file(filename, size = 128, res = 0.1, center = None, data_spacing = 0.01):
    gaus, x0, y0 = make_gaus(size = size, res = res, center = center, data_spacing = data_spacing)
    with open(str(filename)+'.txt', 'w') as f:
        f.write('Pure Guassian beam\nData spacing is: ' + str(data_spacing) + ' degrees \nData area is: ' + str(size*data_spacing) + ' degrees square \nCenter point is: ' + str(x0) + ', ' + str(y0) + '\nValues are relative intensity\n\n')

        for i in range(len(gaus[0])):
            for j in range(len(gaus[0])):
                f.write(str(gaus[i][j]) + '\t')
            f.write('\n')
         
    return

make_file('test')
