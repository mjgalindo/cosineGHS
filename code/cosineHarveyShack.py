# (c) Villads Egede Johansen, 2014
# Current e-mail: vej22 (a) cam.ac.uk
# Long term e-mail: villads (a) egede.com
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from numpy import cos, sin, arcsin, exp, pi, sqrt, multiply
try:
    from numpy.polynomial.legendre import leggauss
except:
    from scipy.special.orthogonal import p_roots as leggauss
# HankelLib uses precalculated coefficients of the Hankel transform found in c.mat
from HankelLib import HankelTransform
#from HankelLibSimple import HankelTransform


def chs(lam, sigmas, lc, thetai, n1, n2, N, Nf):
    ############################
    # SETUP
    phi, wphi = calcGauss(2*N, 0, pi)
    phi = phi[np.newaxis].T
    mu, wmu = calcGauss(N, 0, 1)
    rho = sqrt(1-mu**2)[::-1]

    f, wf = calcGauss(99, 0, 1/lam)
    fo = -sin(thetai)/lam
    fm = sqrt(f**2+fo**2-2*f*fo*(cos(phi)))
    PSD = pi * lc**2 * sigmas**2 * exp(-(pi*lc*fm)**2)
    sigmarel = sqrt(((PSD*f).dot(wf).dot(2*wphi)))

    sigmashat = sigmas/lam
    sigmarelhat = sigmarel/lam
    lchat = lc/lam
    gammai = cos(thetai)

    rhom = sqrt(rho**2+sin(thetai)**2
                + 2*rho*sin(thetai)*(cos(phi)))

    cosk = cos(multiply(np.arange(Nf), phi))
    sk = np.zeros((Nf, N))

    # Setup Hankel transform class
    ht = HankelTransform(Rmax=50, order=0, N=201)
    vrad, rrad = ht.v, ht.r

    ############################
    # Calculations
    for n in range(N):
        gammas = mu[::-1][n]
        A = exp(-(2*pi*(n1*gammai-n2*gammas)*sigmarelhat)**2)
        B = 1-A
        Cs = sigmashat**2 * exp(-rrad**2/lchat**2)
        G = calcG(gammai, gammas, sigmarel, sigmarelhat, sigmas, Cs, n1, n2)
        Srad = ht.fht(G)

        _rhom = rhom[:, n]
        idxNew = np.argsort(_rhom)
        rhoOrdered = _rhom[idxNew]
        idxBack = np.argsort(idxNew)
        Stot = B*np.interp(rhoOrdered, vrad, Srad)[idxBack]
        sk[:, n] = 2/pi*multiply(Stot, cosk.T).dot(wphi)

    K = 1/(pi*sk[0][::-1] * mu).dot(wmu)
    B = 1-exp(-(2*pi*(n1*gammai-n2*gammai)*sigmarelhat)**2)
    K = B*K

    return sk, K, rho, mu

def visualize(N, Nf, sk, K, rho, mu):
    ############################
    # VISUALIZATION
    Slinepos = 0.5*sk[0] + sum([sk[k]*cos(k*0)
                                for k in range(1, sk.shape[0])])
    Slineneg = 0.5*sk[0] + sum([sk[k]*cos(k*pi)
                                for k in range(1, sk.shape[0])])
    angles = arcsin(rho)
    angles = np.concatenate([-angles[::-1], angles])
    I = cos(angles) * K*np.concatenate([Slineneg[::-1], Slinepos])
    ax = plt.subplot(1, 2, 1, label='rect')
    ax.plot(angles/pi*180, I)
    ax.set_xlabel('Scattering Angle (Degrees)')
    ax.set_ylabel('Relative Intensity (1/sr)')
    ax.set_ylim([0,4.5])
    plt.xlim(-90, 90)
    plt.ylim(0)

    phifull = np.linspace(0, 2*pi, 2*N+1)[np.newaxis].T
    cosk = cos(multiply(np.arange(Nf), phifull))
    f = 0.5*sk[0] + sum([multiply(sk[k, :], cosk[:, k][np.newaxis].T)
                         for k in range(1, sk.shape[0])])
    f = mu[::-1]*K*f
    ax = plt.subplot(1, 2, 2, projection="polar", aspect=1., label='polar')
    cax = ax.pcolormesh(phifull.ravel(), rho, f.T)
    plt.gcf().colorbar(cax)


def main():
    ############################
    # INPUT PARAMETERS
    lam = 10.6e-6  # wavelength
    sigmas = 2.27e-6  # rms roughness
    lc = 20.9e-6  # roughness correlation length
    n1 = 1.0
    n2 = -1.0

    N = 1000
    Nf = 20
    plt.figure(figsize=(10,5))

    thetai = 20 / 180*pi
    sk, K, rho, mu = chs(lam, sigmas, lc, thetai, n1, n2, N, Nf)
    visualize(N, Nf, sk, K, rho, mu)

    thetai = 40 / 180*pi
    sk, K, rho, mu = chs(lam, sigmas, lc, thetai, n1, n2, N, Nf)
    visualize(N, Nf, sk, K, rho, mu)

    thetai = 60 / 180*pi
    sk, K, rho, mu = chs(lam, sigmas, lc, thetai, n1, n2, N, Nf)
    visualize(N, Nf, sk, K, rho, mu)

    thetai = 70 / 180*pi
    sk, K, rho, mu = chs(lam, sigmas, lc, thetai, n1, n2, N, Nf)
    visualize(N, Nf, sk, K, rho, mu)
    plt.show()


def main2():
    ############################
    # INPUT PARAMETERS
    LAMBDAS = 1
    lamlist = np.linspace(1050e-9, 450e-9, LAMBDAS)   # wavelength
    lamlist = [10.6e-6]
    SIGMAS = 1
    sigmalist = np.linspace(2.27e-6 / 2, 2.27e-6 * 2, SIGMAS)  # rms roughness
    sigmalist = [2.27e-6]
    lc = 2.27e-6  # roughness correlation length
    thetai = 20 / 180 * pi
    n1 = 1.0
    n2 = -1.0

    N = 100
    Nf = 20

    results = np.zeros((SIGMAS, LAMBDAS))

    for ilambda, lam in enumerate(lamlist):
        for isigma, sigmas in enumerate(sigmalist):
            sk, K, rho, mu = chs(lam, sigmas, lc, thetai, n1, n2, N, Nf)
            Slinepos = 0.5*sk[0] + sum([sk[k]*cos(k*0)
                                for k in range(1, sk.shape[0])])
            Slineneg = 0.5*sk[0] + sum([sk[k]*cos(k*pi)
                                for k in range(1, sk.shape[0])])
            angles = arcsin(rho)
            angles = np.concatenate([-angles[::-1], angles])
            I = cos(angles) * K*np.concatenate([Slineneg[::-1], Slinepos])
            
            '''
            if np.random.rand(1) > 0.5:
                plt.figure()
                plt.plot(angles, I)
                plt.title('$\lambda$ %E, $\sigma_s$ %E' % (lam, sigmas))
            '''
            results[isigma, ilambda] = I[I.shape[0] // 4]

    fig, axs = plt.subplots(1, SIGMAS)
    if SIGMAS == 1:
        axs = [axs]
    for i in range(SIGMAS):
        axs[i].plot(lamlist, results[i, :])
        axs[i].invert_xaxis()
        axs[i].set_xlabel('$\lambda$')
        axs[i].set_ylabel('$I$')

    plt.show()

def calcGauss(N, a, b):
    p, w = leggauss(N)
    p = (b-a)/2.*p + (b+a)/2.
    w = (b-a)/2.*w
    return p, w


def calcG(gammai, gammas, sigmarel, sigmarelhat, sigmas, Cs, n1, n2):
    exponent1 = (2*pi*(n1*gammai-n2*gammas)*sigmarel/sigmas)**2 * Cs
    exponent2 = (2*pi*(n1*gammai-n2*gammas)*sigmarelhat)**2
    if exponent2 > 50:
        return exp(exponent1-exponent2)
    return (exp(exponent1)-1)/(exp(exponent2)-1)


if __name__ == "__main__":
    main()
