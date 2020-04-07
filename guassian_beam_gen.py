import numpy as np
import matplotlib.pyplot as plt

def gaus2d(x,y, x0, y0, res):
    return np.exp(-4*np.log(2) * (((x - x0)**2 + (y - y0)**2) / (res*2)**2))

def make_gaus(filename, size = 128, res = 0.1, center = None, data_spacing = 0.01):

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
        
    #Compute Matrix
    gaus = gaus2d(x,y,x0,y0, res) #np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / res**2) 
    
    #Make 1D slice
    plt.imshow(np.log(gaus))
    plt.savefig(str(filename)+'.pdf')
    plt.show()

    midpoint = int(len(gaus[1])/2)

    y = gaus[midpoint]
    xmin, xmax = -size*data_spacing/2, size*data_spacing/2
    x = np.linspace(xmin, xmax, len(y))
    
    plt.plot(x,np.log(y))
    plt.xlim(xmin, xmax)
    plt.ylim(-20,0)
    #plt.axvline(x = -res, color = 'r')
    #plt.axvline(x = res, color = 'r')
    plt.savefig(str(filename)+'_central_slice.pdf')
    plt.show()


    return gaus, x0, y0[0] 

def load_beam_txt(filename):
    meta = {}
    mode = 'preamble'
    data = []
    for line in open(fn, 'rb'):
        line = line.decode('latin_1')
        #print(mode)
        if mode == 'preamble':
            if (line.strip() == 'HUYGENS PSF'):
                mode = 'header'

        elif mode == 'header':
            if line.strip() == '':
                mode = 'data'
            for regstr, key, nelem, cast in [
                    ('(.*) .m at (.*), (.*) mm', 'params', 3, float),
                    ('Data spacing is (.*) deg.', 'spacing', 1, float),
                    ]:
                m = re.search(regstr, line)
                if m is not None:
                    vals = [cast(m.group(i+1)) for i in range(nelem)]
                    if len(vals) == 1:
                        vals = vals[0]
                    meta[key] = vals

        elif mode == 'data':
            w = line.split()
            data.append(list(map(float, w)))

    data = np.array(data)
    return data, meta


def make_file(filename, size = 128, res = 0.1, center = None, data_spacing = 0.01):
    gaus, x0, y0 = make_gaus(filename = filename, size = size, res = res, center = center, data_spacing = data_spacing)
    with open(str(filename)+'.txt', 'w') as f:
        f.write('Pure Guassian beam\nData spacing is: ' + str(data_spacing) + ' degrees \nData area is: ' + str(size*data_spacing) + ' degrees square \nCenter point is: ' + str(x0) + ', ' + str(y0) + '\nValues are relative intensity\n\n')

        for i in range(len(gaus[0])):
            for j in range(len(gaus[0])):
                f.write(str(gaus[i][j]) + '\t')
            f.write('\n')
         
    return


make_file('LATR_280_gaus_10x_zoom', size = 128, res = 0.015, data_spacing = 0.01)

