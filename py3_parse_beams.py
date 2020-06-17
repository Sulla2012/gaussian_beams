import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, pickle
import numpy as np
opj = os.path.join

cw = [np.array([78., 121., 165.])/255.,
np.array([241., 143., 59.])/255.,
np.array([224., 88., 91.])/255.,
np.array([119., 183., 178.])/255.,
np.array([90., 161., 85.])/255.,
np.array([237., 201., 88.])/255.,
np.array([175., 122., 160.])/255.,
np.array([254., 158., 168.])/255.,
np.array([156., 117., 97.])/255.,
np.array([186., 176., 172.])/255.]

def det_name(idx, nus):

    det = 'det{:04d}_{:d}'.format(int(idx), int(np.mean(nus)))

    return det

def degsq2srad(deg2):
    '''
    https://en.wikipedia.org/wiki/Square_degree
    '''

    return deg2/(41252.9612494/4/np.pi)

def get_da(cr, numel, mult=1):

    return ((float(cr[2])-float(cr[0]))/(mult*float(numel[0])))**2

def get_mesh(cr, numel, mult=1):

    xx, yy = np.meshgrid(np.linspace(cr[0], cr[2], mult*numel[0]),
        np.linspace(cr[1], cr[3], mult*numel[1]))

    dA = get_da(cr, numel, mult=1)

    return xx, yy, dA


def bsa(arr, dA, maxval=None, normalize=True):

    if maxval is None and normalize:
        maxval = np.max(arr.flatten())

        return dA * np.sum(arr/maxval)

    return dA * np.sum(arr)

def parse_data(nus=[90], run_str1='test1', run_str2='test1_wide',
    rdir='pkl/', idir='img_profiles/'):

    nu = np.mean(nus)

    i = 64
    det = det_name(i, nus)
    det1 = det + '_{}'.format(run_str1)
    det2 = det + '_{}'.format(run_str2)

    fname1 = det1 + '.pkl'
    fname2 = det2 + '.pkl'
    results1 = pickle.load(open(rdir+fname1, 'rb'), encoding = 'bytes')
    prop1 = pickle.load(open(rdir+fname1.replace('.pkl','_prop.pkl'), 'rb'), encoding = 'bytes')
    results2 = pickle.load(open(rdir+fname2, 'rb'), encoding = 'bytes')
    prop2 = pickle.load(open(rdir+fname2.replace('.pkl','_prop.pkl'), 'rb'), encoding = 'bytes')

    arr11 = (np.abs(results1[b'e_cx'])**2).astype('float32')
    arr12 = (np.abs(results2[b'e_cx'])**2).astype('float32')

    print(results1.keys())
    print(prop1.keys())
    
    #print(results1[b'co'])

    bsa1 = degsq2srad(bsa(arr11, get_da(results1[b'cr'], results1[b'numel'])))
    bsa2 = degsq2srad(bsa(arr12, get_da(results2[b'cr'], results2[b'numel'])))
    
    forfitting = {}
    forfitting['data'] = arr11
    forfitting['mesh'] = get_mesh(results1[b'cr'], results1[b'numel'])
    pickle.dump(forfitting, open('fitting_'+fname1, 'wb'))

    foraamir = {}
    foraamir['data'] = arr11
    len_mesh = abs(forfitting['mesh'][0][0][len(forfitting['mesh'][0][0])-1]-forfitting['mesh'][0][0][0])
    pitch_mesh = abs(forfitting['mesh'][0][0][0]-forfitting['mesh'][0][0][1])
    foraamir['size'] = [[round(len_mesh, 2), round(pitch_mesh, 3)], [len(forfitting['data'][0]), 1.0]]
    print(len(forfitting['data'][0]))
    pickle.dump(foraamir, open('aamir_'+fname1, 'wb'))
    
    """
    fg1 = 4*np.pi/bsa1
    fg2 = 4*np.pi/bsa2

    print(10*np.log10(fg1))
    print(10*np.log10(fg2))
    print(fname1)
    print(fname2)

    pco1 = prop1['pow_co']
    pco2 = prop2['pow_co']
    pcx1 = prop1['pow_cx']
    pcx2 = prop2['pow_cx']

    lstr1 = 'Det #{}'.format(i+1)
    lstr2 = 'Det #{} w/window scattering'.format(i+1)

    theta = np.linspace(prop1['theta_min'], prop1['theta_max'], prop1['ntheta'])
    plt.plot(theta, 10*np.log10(fg1*pco1[1]/np.max(pco1[1])), label=lstr1, color=cw[0])
    plt.plot(theta, 10*np.log10(fg2*pco2[1]/np.max(pco2[1])), label=lstr2, color=cw[1], ls='--')

    plt.legend(ncol=1, frameon=False)
    plt.xlabel('Theta [deg]')
    plt.ylabel('Power [dBi]')

    plt.ylim([-50, 50])
    plt.title('Co-polar beam profiles at {:d} GHz'.format(int(nu)))
    plt.savefig(opj(idir, '{}_co'.format(det)), dpi=300, bbox_inches='tight')

    plt.xlim([-20, 20])
    plt.savefig(opj(idir, '{}_co_zoom1'.format(det)), dpi=300, bbox_inches='tight')

    plt.xlim([-20, 20])
    plt.savefig(opj(idir, '{}_co_zoom1'.format(det)), dpi=300, bbox_inches='tight')

    plt.close()
    """

def main():

    parse_data()

if __name__ == '__main__':

    main()
