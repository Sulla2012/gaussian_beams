from sotodlib import core
import numpy as np
import so3g
from scipy.optimize import curve_fit
from sotodlib.data.load import load_file
import ephem
from datetime import datetime
from so3g.proj import coords, quat
from astropy.table import Table

######Parameter setting area######
radius_cut = 5. #arcmin, cut-out radius for beam fitting
p0 = [10000, 0, 0, 1.5e-4, 1.5e-4, np.pi/6] #initial parameter values, organized as
                                            #amp, xi0, eta0, fwhm_xi, fwhm_eta, phi
offset_bound = np.radians(5./60) # bound for the offset fit at 5 arcmin
fwhm_bound = 5e-4 # radians, upper bound for the fwhm fit
bounds = ((-np.inf, -offset_bound, -offset_bound, 0., 0., 0),
          (np.inf, offset_bound, offset_bound, fwhm_bound, fwhm_bound, 2*np.pi),)

#Data loading
context = core.Context('pipe-sim-metadata/pipe-s0001.yaml')
tod = load_file('/home/aamirmali/pwg-scripts/pwg-bcp/sim_sso/'
                 'genesis/CES-Atacama-LAT-Jupiter-0-0/UHF1_00000000.g3')
meta = context.get_meta({}) # you might need a timestamp or something in there!
tod.merge(meta)

#Coordinates calculation
time = tod['timestamps']
data = tod['signal']
az = tod['boresight']['az']
el = tod['boresight']['el']
csl = so3g.proj.CelestialSightLine.az_el(time, -az, el, weather='typical')
assembly = so3g.proj.Assembly.attach(csl, tod.focal_plane.quat)
ra, dec = np.transpose(csl.coords()[:, :2])

#Setup the object position information
timestamp = np.mean(time)
dt = datetime.fromtimestamp(timestamp)
dt_str = dt.strftime('%Y/%m/%d %H:%M:%S')
obj = ephem.Jupiter()
obj.compute(dt_str)
ra_obj = obj.ra
dec_obj = obj.dec

#Initial searching by peak-finding
sig_peak = []
dets = []
xi_offset = []
eta_offset = []
for det in range(data.shape[0]):
    data_det = data[det, :]
    sig_peak.append(np.max(data_det))
for det in range(data.shape[0]):
    data_det = data[det, :]
    percentile = 99.99
    if np.max(data_det) < np.mean(sig_peak):
        continue # rejecting detectors that did not see the object
    thresh = np.nanpercentile(data_det, percentile)
    idx_t = np.where(data_det > thresh)
    ra_peaks = np.mod(ra[idx_t], 2*np.pi)
    dec_peaks = dec[idx_t]
    ra_bs = np.mean(ra_peaks)
    dec_bs = np.mean(dec_peaks)
    q_obj = quat.rotation_lonlat(ra_obj, dec_obj)
    q_bs_inv = coords.quat.quat(1,0,0,0)/quat.rotation_lonlat(ra_bs, dec_bs)
    xi_t, eta_t, _ = quat.decompose_xieta(q_bs_inv * q_obj)
    xi_offset.append(xi_t)
    eta_offset.append(eta_t)
    dets.append(det)
det_table = Table([dets, xi_offset, eta_offset], 
                  names=['det_uid', 'xi', 'eta'],
                  dtype=['i4', 'f4', 'f4'])

#2D Gaussian Beam model
def gaussian2d(xieta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi):
    xi, eta = xieta
    xi_rot = xi*np.cos(phi) - eta*np.sin(phi)
    eta_rot = xi*np.sin(phi) + eta*np.cos(phi)
    factor = 2*np.sqrt(2*np.log(2))
    xi_coef = -0.5 * (xi_rot-xi0)**2/(fwhm_xi/factor)**2
    eta_coef = -0.5 * (eta_rot-eta0)**2/(fwhm_eta/factor)**2
    return(a*np.exp(xi_coef+eta_coef))

#Time-domain Beam fitting
beam_table = Table(names=['det_uid', 'amp', 'xi0', 'eta0', 'fwhm_xi', 'fwhm_eta', 'phi'],
                   dtype=('i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'),)

q_bs = quat.rotation_lonlat(ra, dec)
q_obj = quat.rotation_lonlat(ra_obj, dec_obj)

for det in det_table['det_uid']:
    print(det, end=' ')
    idx_t = np.where(det_table['det_uid'] == det)[0][0]
    xi_t = det_table['xi'][idx_t]
    eta_t = det_table['eta'][idx_t]
    q_det = quat.rotation_xieta(xi_t, eta_t)
    xi_det_center, eta_det_center, psi_det_center = quat.decompose_xieta(~q_det * ~q_bs * q_obj)
    data_det = data[det, :]
    radius = np.arcsin(np.sqrt(xi_det_center**2 + eta_det_center**2))
    idx_t = np.where(radius < np.radians(radius_cut/60))[0]
    xi_cut = xi_det_center[idx_t]
    eta_cut = eta_det_center[idx_t] 
    data_cut = data_det[idx_t]
    xieta = np.vstack((xi_cut, eta_cut))
    popt, _ = curve_fit(gaussian2d, xieta, data_cut, 
                        p0=p0, bounds=bounds,)
    q_t = quat.rotation_xieta(xi_t, eta_t)
    q_delta = quat.rotation_xieta(popt[1], popt[2]) # xi0 and eta0
    popt[1], popt[2],_ = quat.decompose_xieta(q_delta * q_t)
    beam_table.add_row([det, *popt])

beam_table.write('beam_fitting_results.txt', format='ascii')
