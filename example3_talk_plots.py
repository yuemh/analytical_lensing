import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from astropy.table import Table

from scipy.integrate import quad, dblquad
from scipy.interpolate import griddata
from scipy.special import gamma as gammafunc

import analytical_lensing.analytical_lensing as al
import analytical_lensing.estcount as estcount

import matplotlib
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.direction'] = 'in'

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.sans-serif': 'Computer Modern Serif'
})

defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)

'''
Define luminosity functions and velocity dispersion functions
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
We also define the apparent magnitude and absolute magnitude difference.
'''
def appmag_absmag_diff(redshift, band='J'):
    # returns apparent magnitude - absolute magnitude, at redshift z
    # we use the result from simulated quasars (Yue+22)

    tbl = Table.read('kcorr_mock.fits')
    zlist = np.array(tbl['z'])
    dMlist = np.array(tbl['dM%s'%band])

    return np.interp(redshift, zlist, dMlist)

'''
A convenient function below
'''

def get_dNdz(lumfun, vdffun, Pmu, mfaint, dMfunc,\
            zmin, zmax, dz, cosmo, area):

    dNdz_list = []

    for z in np.arange(zmin, zmax, dz):
        Mfaint = mfaint - dMfunc(z+dz/2)
        Mbright = -30
        dndz = estcount.Nlens_dz(lumfun, vdffun, Pmu,\
                        Mfaint, Mbright, z, cosmo, area)

        dNdz_list.append(dndz)

    return dNdz_list

'''
Plotting
'''

def plot_quasars():
    zlist = np.arange(5.5, 9, 0.1)

    dMfunc_z = lambda z: appmag_absmag_diff(z, 'z')
    dMfunc_J = lambda z: appmag_absmag_diff(z, 'J')

    dNdz_LSST = get_dNdz(lumfun_quasar_M18,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         26,\
                         dMfunc_z,\
                         5.5, 9, 0.1, defaultcosmo, 20000)

    dNdz_Euclid = get_dNdz(lumfun_quasar_M18,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         24,\
                         dMfunc_J,\
                         5.5, 9, 0.1, defaultcosmo, 15000)


    dNdz_Roman = get_dNdz(lumfun_quasar_M18,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         26,\
                         dMfunc_J,\
                         5.5, 9, 0.1, defaultcosmo, 1700)

    fig, ax = plt.subplots(figsize=[5.5,4])

    ax.plot(zlist, dNdz_LSST, ':', color='k', label=r'LSST')
    ax.plot(zlist, dNdz_Euclid, '--', color='darkorange',label=r'Euclid')
    ax.plot(zlist, dNdz_Roman, '-', color='darkred',  label=r'Roman')

    plt.yscale('log')

    ax.set_xlabel(r'Source Redshift', fontsize=14)
    ax.set_ylabel(r'$\mathrm{dN_{lens} / dz}$', fontsize=14)
    ax.set_ylim([1e-2, 1e3])
    ax.set_xlim([5.55,8.55])

    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('./plots/dNdz_M18.pdf')

    plt.show()

def plot_AGN():
    zlist = np.arange(5.5, 9, 0.1)

    dMfunc_z = lambda z: appmag_absmag_diff(z, 'z')
    dMfunc_J = lambda z: appmag_absmag_diff(z, 'J')

    dNdz_LSST = get_dNdz(lumfun_AGN_H23,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         26,\
                         dMfunc_z,\
                         5.5, 9, 0.1, defaultcosmo, 20000)

    dNdz_Euclid = get_dNdz(lumfun_AGN_H23,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         24,\
                         dMfunc_J,\
                         5.5, 9, 0.1, defaultcosmo, 15000)


    dNdz_Roman = get_dNdz(lumfun_AGN_H23,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         26,\
                         dMfunc_J,\
                         5.5, 9, 0.1, defaultcosmo, 1700)

    fig, ax = plt.subplots(figsize=[5.5,4])

    ax.plot(zlist, dNdz_LSST, ':', color='k', label=r'LSST')
    ax.plot(zlist, dNdz_Euclid, '--', color='darkorange',label=r'Euclid')
    ax.plot(zlist, dNdz_Roman, '-', color='darkred',  label=r'Roman')

    plt.yscale('log')

    ax.set_xlabel(r'Source Redshift', fontsize=14)
    ax.set_ylabel(r'$\mathrm{dN_{lens} / dz}$', fontsize=14)
    ax.set_ylim([1e-1, 1e5])
    ax.set_xlim([5.55,8.55])

    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('./plots/dNdz_H22.pdf')

    plt.show()

def print_numbers():
    dMfunc_z = lambda z: appmag_absmag_diff(z, 'z')
    dMfunc_J = lambda z: appmag_absmag_diff(z, 'J')

    Nagn_LSST = estcount.Nlens_zrange(lumfun_AGN_H23,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         26,\
                         dMfunc_z,\
                         6, 9, 0.1, defaultcosmo, 20000)

    Nagn_Euclid = estcount.Nlens_zrange(lumfun_AGN_H23,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         24,\
                         dMfunc_J,\
                         6, 9, 0.1, defaultcosmo, 15000)


    Nagn_Roman = estcount.Nlens_zrange(lumfun_AGN_H23,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         26,\
                         dMfunc_J,\
                         6, 9, 0.1, defaultcosmo, 1700)

    Nqso_LSST = estcount.Nlens_zrange(lumfun_quasar_M18,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         26,\
                         dMfunc_z,\
                         6, 9, 0.1, defaultcosmo, 20000)

    Nqso_Euclid = estcount.Nlens_zrange(lumfun_quasar_M18,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         24,\
                         dMfunc_J,\
                         6, 9, 0.1, defaultcosmo, 15000)


    Nqso_Roman = estcount.Nlens_zrange(lumfun_quasar_M18,\
                         vdf_yue22,\
                         al.Pmu_total,\
                         26,\
                         dMfunc_J,\
                         6, 9, 0.1, defaultcosmo, 1700)

    print('Number of lensed quasars in LSST, Euclid, and Roman (z>6): %.2f, %.2f, %.2f'%(Nqso_LSST, Nqso_Euclid, Nqso_Roman))
    print('Number of lensed AGNs in LSST, Euclid, and Roman (z>6): %.2f, %.2f, %.2f'%(Nagn_LSST, Nagn_Euclid, Nagn_Roman))


if __name__=='__main__':
#    plot_AGN()
#    plot_quasars()
    print_numbers()
