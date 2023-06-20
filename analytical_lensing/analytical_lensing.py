import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.integrate import quad, dblquad

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u


defaultcosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# taum related functions

def theta_E_func(sigma, zl, zs, cosmo):

    Ds = cosmo.angular_diameter_distance(zs)
    Dls = cosmo.angular_diameter_distance_z1z2(zl, zs)

    return 4 * np.pi * (sigma/3e5)**2 * Dls / Ds

def inverse_theta_E_func(theta_E, zl, zs, cosmo):
    Ds = cosmo.angular_diameter_distance(zs)
    Dls = cosmo.angular_diameter_distance_z1z2(zl, zs)

    sc2 = theta_E / 4 / np.pi * Ds / Dls

    return np.sqrt(sc2) * 3e5

def tau_integral(sigma, z, zs, vdf, cosmo):

    if z>zs:
        return 0

    vdfterm = vdf(sigma, z) / sigma / np.log(10) * u.Mpc**-3

    dV = cosmo.differential_comoving_volume(z)

    theta_E = theta_E_func(sigma, z, zs, cosmo)
    area  = np.pi * theta_E **2  * u.sr

    return vdfterm * dV * area

def sep_integral(theta_E, z, zs, vdf, cosmo):
    if z>zs:
        return 0

    sigma = inverse_theta_E_func(theta_E, z, zs, cosmo)

    vdfterm = vdf(sigma, z) / sigma / np.log(10) * u.Mpc**-3

    dV = cosmo.differential_comoving_volume(z)
    Ds = cosmo.angular_diameter_distance(zs)
    Dls = cosmo.angular_diameter_distance_z1z2(z, zs)

    theta_E = theta_E_func(sigma, z, zs, cosmo)
    area  = np.pi * theta_E **2  * u.sr
    additional_factor = 8 * np.pi * sigma / (3e5)**2 * Dls / Ds

    return vdfterm * dV * area / additional_factor


def taum(zs, vdf, cosmo):
    paras = [zs, vdf, cosmo]
    result = dblquad(tau_integral, 0, zs, 0, 1000, args=paras)
    return result[0]


def taumdiff(zd, zs, vdf, cosmo):
    paras = (zd, zs, vdf, cosmo)
    result = quad(tau_integral, 0, 1000, args=paras)

    return result[0]


def sep_distribution_diff(theta_E, zs, vdf, cosmo):
    theta_E_rad = theta_E / 206265
    paras = (zs,  vdf, cosmo)

    intfunc = lambda z: sep_integral(theta_E_rad, z, zs, vdf, cosmo)

    result = quad(intfunc, 0, zs)

    return result[0] / 206265

def sep_distribution(theta_E, zs, vdf, cosmo):
    theta_E_rad = theta_E / 206265
    paras = (zs,  vdf, cosmo)

    result = dblquad(sep_integral, 0, zs, 0, theta_E_rad, args=paras)

    return result[0]

# magbias related functions

def Pmu_bright(mu):
    return 2/(mu-1)**3

def Pmu_total(mu):
    return 8 / mu**3

def N_Llim(Mlim, lumfun):
    '''
    Input Parameters:
        Llim: float
            The lower boundary of the luminosity
        lumfun: callable
            Call as Phi = lumfun(L)
    '''

    N = quad(lumfun, -40, Mlim)[0]
    return N

def magbias_differential(M, mu, lumfun, Pmu):
    return lumfun(M+2.5*np.log10(mu)) * Pmu(mu)

def magbias(Mlim, lumfun, Pmu, mu_min=2, mu_max = +np.inf):
    intfunc = lambda mu: N_Llim(Mlim+2.5*np.log10(mu), lumfun) * Pmu(mu)

    B = quad(intfunc, mu_min, mu_max)[0] / N_Llim(Mlim, lumfun)

    return B

def Fmulti(tau, B):
    return tau * B / (tau * B + 1 - tau)
