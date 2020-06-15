import numpy as np
import matplotlib.pyplot as plt
import re, os, pickle
from scipy import optimize
from scipy.optimize import curve_fit

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
    for line in open(filename, 'rb'):
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
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    arr += gaussian(*args[0:6])(x,y)
    return arr


#make_file('LATR_280_gaus_10x_zoom', size = 128, res = 0.015, data_spacing = 0.01)

data, meta = load_beam_txt('LAT_beam_1100um_center_tube1.TXT')

#Following two functions courtousy of Scipy cookbook
def gaussian(scale,y_offset, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: y_offset + scale*np.exp(
                -(((x-center_x)/width_x)**2+((y-center_y)/width_y)**2)/2)

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = [9, 9, .7, .7]
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, succes = optimize.leastsq(errorfunction, params)
    return p

#p= fitgaussian(data)

#gaus = gaussian(*p)

x = np.arange(0, 128, 1, float)
y = x[:,np.newaxis]



"""
plt.imshow(np.log(gaus(x,y)))
plt.savefig('LAT_2020_fit_gaus.pdf')
plt.show()

res = data - gaus(x,y)
plt.imshow(np.log(res))
plt.savefig('LAT_2020_residual.pdf')
plt.show()

#print(np.argmax(data)/128)
#print(np.argmax(gaus(x,y))/128)

testgaus = gaussian (10,10,1,1)
testx = np.arange(0,20,0.1, float)
testy = testx[:, np.newaxis]
ptest = fitgaussian(testgaus(testx, testy))

print(ptest)
"""
"""
X, Y = np.meshgrid(x,y)
xdata = np.vstack((X.ravel(), Y.ravel()))
print("X data")
print(xdata)
p0 = [1,65,65,.1,.1]

popt, pcov = curve_fit(_gaussian, xdata, data.ravel(),p0)
print(popt)

gaus2 = gaussian(*popt[0:5])(x,y)

plt.imshow(np.log(gaus2))
plt.savefig('LAT_2020_fit_gaus.pdf')
plt.show()

res2 = data-gaus2
plt.imshow(np.log(res2))
plt.savefig('LAT_2020_residual.pdf')
plt.show()

midpoint = int(len(res2[1])/2)

yslice = res2[midpoint]
xmin, xmax = -0.64, 0.64
xslice = np.linspace(xmin, xmax, len(y))

plt.plot(xslice, np.log(yslice))
plt.xlim(xmin, xmax)
plt.ylim(-20,0)
#plt.axvline(x = -res, color = 'r')
#plt.axvline(x = res, color = 'r')
plt.savefig('residual_central_slice.pdf')
plt.show()


datamax = np.argmax(data)
print(data.ravel()[datamax])

print(np.argmax(gaus2(x,y))/128)

testX, testY = np.meshgrid(testx, testy)

testxdata = np.vstack((testX.ravel(), testY.ravel()))

p1 = [9, 9, .7, .8]

testpopt, testpcov = curve_fit(_gaussian, testxdata, testgaus(testx, testy).ravel(), p1)
print(testpopt)

"""
print("Starting SAT Beams")
sat_beam = pickle.load(open('fitting_det0000_90_test2_wide.pkl', 'rb'), encoding = 'bytes')

plt.imshow(np.log(sat_beam[b'data']), extent = [-32.12481, 17.87519,  16.29601, -33.70399])
plt.savefig("SAT_beam.pdf")
plt.show()

X, Y = sat_beam['mesh'][0], sat_beam['mesh'][1]
xdata = np.vstack((X.ravel(), Y.ravel()))

p0 = [.5, -1,-7,-8,.5,.5]


datamax = np.argmax(sat_beam['data'])
print(sat_beam['data'].ravel()[datamax])
print(xdata.ravel()[datamax])
print("Y max: {}".format(Y.ravel()[datamax]))
print("X max: {}".format(X.ravel()[datamax]))

i,j = np.unravel_index(sat_beam['data'].argmax(), sat_beam['data'].shape)
print("Peak: {}".format(sat_beam['data'][i,j]))
#print(sat_beam['mesh'][0][i,j])
#print((-32.12481 + 17.87519)/1001*i)
#print((-33.70399 + 16.29601)/1001*j)
print("Pixel size: {}".format((-33.70399 + 16.29601)/1001))

popt, pcov = curve_fit(_gaussian, xdata, sat_beam['data'].ravel(),p0)
print(popt)

#Fit Plot
x = np.linspace(-32.12481, 17.87519, 1001)
y = np.linspace(-33.70399, 16.29601, 1001)
y = y[:, np.newaxis]
gaus2 = gaussian(*popt[0:6])(x,y)
print(gaus2.shape)
plt.imshow(np.log(gaus2), extent = [-32.12481, 17.87519,  16.29601, -33.70399])
plt.savefig('SAT_90_0000_wide_fit_gaus.pdf')
plt.show()

#Fit and Data Slice
print(sat_beam['data'].shape)
yslice_data = sat_beam['data'][i]
yslice_fit = gaus2[i]
plt.plot(x,np.log(yslice_data), label = "data")
plt.plot(x,np.log(yslice_fit), label = "fit")
plt.legend()
plt.savefig("slice_fit_data.pdf")
plt.show()

#Data Slice
plt.plot(x,np.log(yslice_data), label = "data")
#plt.plot(x,np.log(yslice_fit), label = "fit")
#plt.legend()
plt.savefig("slice_data.pdf")
plt.show()

#2D Residual Plots
res2 = sat_beam['data']-gaus2
plt.imshow(np.log(res2), extent = [-32.12481, 17.87519,  16.29601, -33.70399])
plt.savefig('SAT_90_0000_wide_residual.pdf')
plt.show()

#Midpoint Residual Plots
midpoint = int(len(res2[1])/2)

yslice = res2[midpoint]
#xmin, xmax = -0.64, 0.64
#xslice = np.linspace(xmin, xmax, len(y))

plt.plot(x, np.log(yslice))
#plt.xlim(xmin, xmax)
plt.ylim(-20,0)
#plt.axvline(x = -res, color = 'r')
#plt.axvline(x = res, color = 'r')
plt.savefig('residual_central_slice.pdf')
plt.show()

#Maxrow Residual Plots
yslice_max = res2[i]

plt.plot(x, np.log(yslice_max))
plt.ylim(-20,0)
plt.savefig('residual_central_slice_max.pdf')
plt.show()

