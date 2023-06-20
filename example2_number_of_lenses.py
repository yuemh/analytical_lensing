import os, sys
import numpy as np

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import analytical_lensing as al
import estcount

'''
First, let's define the source luminosity function.

A luminosity function should work as follows:
lumfun(M, z) -> returns source density in (Mpc^-3 mag^-1)
'''

def doublepowerlaw(M, phis, Ms, alpha, beta):

    phi =  phis / (10**(0.4*(M-Ms)*(alpha+1)) + 10**(0.4*(M-Ms)*(beta+1)))
    return phi


def lumfun_quasar_M18(M, z):
    Phis = 10.9e-9 * 10**(-0.7*(z-6))#Mpc-3
    alpha = -1.23
    beta = -2.73
    Ms = -24.9

    return doublepowerlaw(M, Phis, Ms, alpha, beta)

def lumfun_AGN_H23(M, z):
    Phis = 10**-4.87 * 10**(-0.7*(z-5.763)) * np.log(10)/2.5
    Ms = -21.23
    alpha = -2.14
    beta = -5.03

    return doublepowerlaw(M, Phis, Ms, alpha, beta)

'''
Second, let's define the deflector velocity dispersion function.

A velocity dispersion function should work as follows:
vdfun(sigma, z) -> returns deflector density in (Mpc^-3 (km/s)^-1)
'''

def vdf_yue22(sigma, z):
    v0 = 172.2*(1+z)**0.18
    phi0 = 5.86e-3*(1+z)**-1.18
    alpha = -1.15
    beta = 2.35

    x_red = sigma / v0
    t1 = x_red ** (alpha + 1)
    t2 = np.exp(1-(x_red)**beta)

    return phi0 * t1 * t2

'''
One last thing: we need to define the k-correction.

Specifically, we need a function to convert absolute magnitude
to apparent magnitude (and the other way around)
'''

def appmag_absmag_diff(redshift, band='J'):
    # returns apparent magnitude - absolute magnitude, at redshift z
    # we use the result from simulated quasars (Yue+22)

    tbl = Table.read('kcorr_mock.fits')
    zlist = np.array(tbl['z'])
    dMlist = np.array(tbl['dM%s'%band])

    return np.interp(redshift, zlist, dMlist)

'''
That is all we need!

Now we can calculate the number of lenses 
that can be detected in the Roman survey.
'''

def main():
    # we count number of luminous quasars in 6<z<9

    num = estcount.Nlens_zrange(lumfun=lumfun_quasar_M18, # the source luminosity function
                 vdffun=vdf_yue22, # the deflector velocity dispersion function
                 Pmu=al.Pmu_total, # the probability distribution of magnification
                 mfaint=26.7, # the survey limit
                 dMfunc=appmag_absmag_diff, # the fnuction that returns the difference between apparent magnitude and the absolute magnitude
                 zmin=6, # minimum source redshift 
                 zmax=9, #maximum source redshift
                 dz=0.1, # the increment of redshift list
                 cosmo=defaultcosmo # the cosmology
                      )

    print('Number of lensed quasars above the flux limit (all sky): %d'%num)

if __name__=='__main__':
    main()
