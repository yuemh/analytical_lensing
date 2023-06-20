import os, sys
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.integrate import quad

import analytical_lensing.analytical_lensing as al

def doublepowerlaw(M, phis, Ms, alpha, beta):
    phi =  phis / (10**(0.4*(M-Ms)*(alpha+1)) + 10**(0.4*(M-Ms)*(beta+1)))
    return phi

def Nlens_dz(lumfun, vdffun, Pmu, Mfaint, Mbright, z, cosmo,\
            area=41253, seplim=-1):

    lumfun_thisz = lambda M: lumfun(M, z)

    if seplim<0:
        tau = al.taum(z, vdf=vdffun, cosmo=cosmo)
    else:
        tau = al.sep_distribution(seplim/2, z, vdf, cosmo)
    B = al.magbias(Mfaint, lumfun_thisz, Pmu)
    fm = al.Fmulti(tau, B)

    nqso = quad(lumfun_thisz, Mbright, Mfaint)[0] * u.Mpc **-3
    dVdz = cosmo.differential_comoving_volume(z)\
            * 4 * np.pi * u.sr * area / 41253

    nlens = nqso * dVdz * fm
    return nlens.to(1).value

def Nlens_zrange(lumfun, vdffun, Pmu, mfaint, dMfunc,\
                 zmin, zmax, dz, cosmo, area=41253, seplim=-1):

    ntotal = 0

    for z in np.arange(zmin, zmax, dz):
        Mfaint = mfaint - dMfunc(z+dz/2)
        Mbright = -30

        ntotal +=\
            Nlens_dz(lumfun, vdffun, Pmu, Mfaint, Mbright,\
                     z, cosmo, area, seplim) * dz

    return ntotal

