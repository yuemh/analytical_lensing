import os, sys
import numpy as np
import matplotlib.pyplot as plt

import analytical_lensing.analytical_lensing as al

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)

'''
First step:
We need to define the deflector velocity dispersion function.

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
We can now cauculate some statistics of the lensed sources
'''

def main():
    # we use source redshift of 6 as an example.
    zs = 6

    # the strong lensing optical depth
    taum_z6 = al.taum(zs, vdf_yue22, defaultcosmo)

    print('Lensing optical depth at z=%f: %f'%(zs, taum_z6))

    # now let's plot the distribution of deflector redshift
    zdlist = np.arange(0.1, 5.9, 0.1)
    deflector_redshift_prob = [al.taumdiff(zd, zs, vdf_yue22, defaultcosmo) for zd in zdlist]

    plt.plot(zdlist, deflector_redshift_prob, 'k-')
    plt.xlabel('Deflector Redshift ($z_d$)')
    plt.ylabel(r'$d\tau_m/dz_d$')
    plt.savefig('Deflector_Redshift_Distribution.pdf')
    plt.close('all')

    # now let's get the lensing separation distribution
    dthetalist = np.arange(0.1, 3, 0.1)
    dtheta_prob = [al.sep_distribution_diff(dtheta/2, zs, vdf_yue22, defaultcosmo)/taum_z6 for dtheta in dthetalist]

    plt.plot(dthetalist, dtheta_prob, 'k-')
    plt.xlabel(r'Lensing Separation ($\Delta\theta=2\theta_E$) [arcsec]')
    plt.ylabel(r'$dP/d(\Delta\theta)$')
    plt.savefig('Lensing_Separation_Distribution.pdf')


if __name__=='__main__':
    main()


