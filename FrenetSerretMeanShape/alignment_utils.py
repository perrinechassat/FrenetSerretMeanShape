import numpy as np
from scipy.linalg import logm, svd, expm
import scipy.linalg
from sklearn.gaussian_process.kernels import Matern
import fdasrsf.utility_functions as uf
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d, UnivariateSpline
from numpy.linalg import norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
import collections
import optimum_reparamN2_C as orN2_C
import sys
from maths_utils import *
from frenet_path import *

""" Set of functions for the alignment of curvature and torsion """


def optimum_reparam_curvatures(theta1, time, theta2, lam=0.0, grid_dim=7):
    """
    calculates the warping to align theta2 to theta1

    Param:
        theta1: matrix of size 2xM (curvature, torsion)
        time: vector of size M describing the sample points
        theta2: matrix of size 2xM (curvature, torsion)
        lam: controls the amount of elasticity (default = 0.0)
        grid_dim: size of the grid (default = 7)

    Return:
        gam: describing the warping function used to align theta2 with theta1

    """

    gam = orN2_C.coptimum_reparamN2(np.ascontiguousarray(theta1), time,
                                          np.ascontiguousarray(theta2), lam, grid_dim)

    return gam



def optimum_reparam_vect_curvatures(theta1, time, theta2, lam=0.0, grid_dim=7):
    """
    calculates the warping to align theta2 to theta1

    Param:
        theta1: matrix of size 2xM (curvature, torsion)
        time: vector of size M describing the sample points
        theta2: matrix of size 2xM (curvature, torsion)
        lam: controls the amount of elasticity (default = 0.0)
        grid_dim: size of the grid, for the DP2 method only (default = 7)

    Return:
        gam: describing the warping function used to align theta2 with theta1

    """

    gam = orN2_C.coptimum_reparam_curve(np.ascontiguousarray(theta1), time,
                                         np.ascontiguousarray(theta2), lam, grid_dim)

    return gam


def align_vect_curvatures_fPCA(f, time, weights, num_comp=3, cores=-1, smoothdata=False, MaxItr=1, init_cost=0, lam=0.0):
    """
    aligns a collection of functions while extracting principal components.
    The functions are aligned to the principal components

    ...

    Param:
        f: numpy ndarray of shape (n,M,N) of N functions with M samples of 2 dimensions (kappa and tau)
        time: vector of size M describing the sample points
        weights: numpy ndarray of shape (M,N) of N functions with M samples
        num_comp: number of fPCA components
        number of cores for parallel (default = -1 (all))
        smooth_data: bool, smooth the data using a box filter (default = F)
        MaxItr: maximum number of iterations (default = 1)
        init_cost: (default = 0)
        lam: coef of alignment (default = 0)

    Return:
        fn: numpy array of aligned functions (n,M,N)
        gamf: numpy array of warping functions used to align the data (M,N)
        mfn: weighted mean of the functions algned (2,M)
        fi: aligned functions at each iterations (n,M,N,nb_itr)
        gam: estimated warping functions at each iterations (M,N,nb_itr)
        mf: estimated weighted mean at each iterations (2,M,nb_itr)
        nb_itr: number of iterations needed to align curves
        convergence: True if nb_itr < MaxItr, False otherwise

    """
    n = f.shape[0]
    M = f.shape[1]
    N = f.shape[2]
    parallel = True

    eps = np.finfo(np.double).eps

    if smoothdata:
        f_init = f
        f_smooth = np.zeros((M, N))
        for k in range(0, N):
            spar = time.shape[0] * (.025 * np.fabs(f[:, k]).max()) ** 2
            tmp_spline = UnivariateSpline(time, f[:, k], s=spar)
            f_smooth[:, k] = tmp_spline(time)
        f = f_smooth

    f0 = f
    kappa0 = f[0,:,:]
    tau0 = f[1,:,:]

    mf0 = weighted_mean_vect(f, weights)
    a = mf0.repeat(N)
    d1 = a.reshape(n, M, N)
    d = (f - d1) ** 2
    dqq = np.sqrt(d.sum(axis=1).sum(axis=0))
    min_ind = dqq.argmin()

    itr = 0
    mf = np.zeros((n, M, MaxItr + 1))
    mf_cent = np.zeros((n, M, MaxItr + 1))
    mf[:, :, itr] = f[:, :, min_ind]
    mf_cent[:, :, itr] = f[:, :, min_ind]
    # mf[:, itr] = mf0
    fi = np.zeros((n, M, N, MaxItr + 1))
    fi_cent = np.zeros((n, M, N, MaxItr + 1))
    fi[:, :, :, 0] = f
    fi_cent[:, :, :, 0] = f
    gam = np.zeros((M, N, MaxItr + 1))
    cost = np.zeros(MaxItr + 1)
    cost[itr] = init_cost

    MS_phase = (trapz(f[:, :, min_ind] ** 2, time) - trapz(mf0 ** 2, time)).mean()
    # print('MS_phase :', MS_phase)
    if np.abs(MS_phase) < 0.01:
        print('MS_phase :', MS_phase)
        print("%d functions already aligned..."
              % (N))
        mfn = mf0
        fn = f0
        gamf = np.zeros((M,N))
        for k in range(0, N):
            gamf[:, k] = time

        align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'nb_itr', 'convergence'])

        out = align_results(fn, gamf, mfn, 0, True)

        return out

    print("Aligning %d functions to %d fPCA components..."
          % (N, num_comp))

    while itr < MaxItr:
        # print("updating step: r=%d" % (itr + 1))

        # PCA Step
        fhat = np.zeros((n,M,N))
        for k in range(n):
            a = mf[k, :, itr].repeat(N)
            d1 = a.reshape(M, N)
            fhat_cent = fi[k, :, :, itr] - d1
            K = np.cov(fi[k, :, :, itr])
            if True in np.isnan(K) or True in np.isinf(K):
                mfn = mf0
                fn = f0
                gamf = np.zeros((M,N))
                for k in range(0, N):
                    gamf[:, k] = time
                align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'nb_itr', 'convergence'])
                out = align_results(f0, gamf, mfn, MaxItr, False)
                return out

            U, s, V = svd(K)

            alpha_i = np.zeros((num_comp, N))
            for ii in range(0, num_comp):
                for jj in range(0, N):
                    alpha_i[ii, jj] = trapz(fhat_cent[:, jj] * U[:, ii], time)

            U1 = U[:, 0:num_comp]
            tmp = U1.dot(alpha_i)
            fhat[k,:,:] = d1 + tmp
            # #
            # plt.figure()
            # for i in range(N):
            #     plt.plot(time, fhat[k,:,i])
            # plt.show()

        cost_init = np.zeros(N)

        # Matching Step

        if parallel:
            out = Parallel(n_jobs=cores)(delayed(optimum_reparam_vect_curvatures)(fhat[:, :, n], time, fi[:, :, n, itr], lam) for n in range(N))
            gam_t = np.array(out)
            gam[:, :, itr] = gam_t.transpose()
        else:
            for n in range(N):
                gam[:, n, itr] = optimum_reparam_vect_curvatures(fhat[:, :, n], time, fi[:, :, n, itr], lam)

        for kk in range(n):
            for k in range(0, N):
                time0 = (time[-1] - time[0]) * gam[:, k, itr] + time[0]
                fi[kk, :, k, itr + 1] = np.interp(time0, time, fi[kk, :, k, itr]) * np.gradient(gam[:, k, itr], 1 / float(M - 1))

        fi[np.isnan(fi)] = 0.0
        fi[np.isinf(fi)] = 0.0

        ftemp = fi[:, :, :, itr + 1]
        mf[:, :, itr + 1] = weighted_mean_vect(ftemp, weights)

        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, gam[:, i, itr])
        # plt.show()
        #
        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, ftemp[0, :, i])
        # plt.show()
        # plt.figure()
        # plt.plot(time, mf[0, :, itr + 1])
        # plt.show()
        #
        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, ftemp[1, :, i])
        # plt.show()
        # plt.figure()
        # plt.plot(time, mf[1, :, itr + 1])
        # plt.show()

        fi_cent[:, :, :, itr + 1], mf_cent[:, :, itr + 1] = align_and_center(np.copy(gam), np.copy(mf[:, :, itr + 1]), np.copy(ftemp), itr+1, np.copy(time))

        cost_temp = np.zeros(N)

        for ii in range(0, N):
            cost_temp[ii] = norm(fi[:,:,ii,itr] - ftemp[:,:,ii], 'fro')

        cost[itr + 1] = cost_temp.mean()

        if abs(cost[itr + 1] - cost[itr]) < 1:
            break

        itr += 1

    print("Alignment in %d iterations" % (itr))
    if itr >= MaxItr:
        itrf = MaxItr
    else:
        itrf = itr+1
    cost = cost[1:(itrf+1)]

    # Aligned data & stats
    fn = fi[:, :, :, itrf]
    mfn = mf[:, :, itrf]
    gamf = gam[:, :, 0]
    for k in range(1, itrf):
        gam_k = gam[:, :, k]
        for l in range(0, N):
            time0 = (time[-1] - time[0]) * gam_k[:, l] + time[0]
            gamf[:, l] = np.interp(time0, time, gamf[:, l])


    ## Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    time0 = (time[-1] - time[0]) * gamI + time[0]
    for kk in range(n):
        mfn[kk] = np.interp(time0, time, mfn[kk]) * gamI_dev
        for k in range(0, N):
            fn[kk, :, k] = np.interp(time0, time, fn[kk, :, k]) * gamI_dev

    for k in range(0, N):
        gamf[:, k] = np.interp(time0, time, gamf[:, k])
    #
    # plt.figure()
    # plt.plot(time, mfn[0])
    # plt.show()
    # plt.figure()
    # plt.plot(time, mfn[1])
    # plt.show()
    #
    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, gamf[:, i])
    # plt.show()
    #
    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, fn[0, :, i])
    # plt.show()
    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, fn[1, :, i])
    # plt.show()

    align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'fi', 'gam', 'mf', 'nb_itr', 'convergence'])

    if itr==MaxItr:
        out = align_results(fn, gamf, mfn, fi_cent[:,:,:,0:itrf+1], gam[:,:,0:itrf+1], mf_cent[:,:,0:itrf+1], itr, False)
    else:
        out = align_results(fn, gamf, mfn, fi_cent[:,:,:,0:itrf+1], gam[:,:,0:itrf+1], mf_cent[:,:,0:itrf+1], itr, True)

    return out


def compose(f, g, time):
    """
    Compose functions f by functions g on the grid time.
    ...

    Param:
        f: array of N functions evaluted on time (M,N)
        g: array of N warping functions evaluted on time (M,N)
        time: array of time points (M)

    Return:
        f_g: array of functions f evaluted on g(time)

    """
    N = f.shape[1]
    f_g = np.zeros((time.shape[0], N))
    for n in range(N):
        f_g[:,n] = np.interp((time[-1] - time[0]) * g[:,n] + time[0], time, f[:,n])

    return f_g


def warp_curvatures(theta, gam_fct, time, weights):
    """
    Apply warping on curvatures: theta_align = theta(gam_fct(time))*grad(gam_fct)(time)
    and compute the weighted mean of the aligned functions
    ...

    Param:
        theta: array of curvatures or torsions (M,N)
        gam_fct: functions, array of N warping functions
        time: array of time points (M)
        weights: array of weights (M)

    Return:
        theta align: array of functions theta aligned
        weighted_mean_theta: weighted mean of the aligned functions (M)

    """
    M = theta.shape[0]
    N = theta.shape[1]
    theta_align = np.zeros(theta.shape)
    gam = np.zeros((time.shape[0], N))
    for n in range(N):
        gam[:,n] = gam_fct[n](time)
        time0 = (time[-1] - time[0]) * gam[:, n] + time[0]
        theta_align[:,n] = np.interp(time0, time, theta[:,n]) * np.gradient(gam[:, n], 1 / float(M - 1))
    weighted_mean_theta = weighted_mean(theta_align, weights)

    return theta_align, weighted_mean_theta





def align_and_center(gam, mf, f, itr, time):
    """
    Utility functions for the alignment function, used to aligned functions at the end of the iterations.
    ...

    """
    n = 2
    N = gam.shape[1]
    M = gam.shape[0]
    gamf = gam[:, :, 0]
    for k in range(1, itr):
        gam_k = gam[:, :, k]
        for l in range(0, N):
            time0 = (time[-1] - time[0]) * gam_k[:, l] + time[0]
            gamf[:, l] = np.interp(time0, time, gamf[:, l])

    ## Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    time0 = (time[-1] - time[0]) * gamI + time[0]
    for kk in range(n):
        mf[kk] = np.interp(time0, time, mf[kk]) * gamI_dev
        for k in range(0, N):
            f[kk, :, k] = np.interp(time0, time, f[kk, :, k]) * gamI_dev

    return f, mf



def compute_deformation(list_Q0, thetai, gami, mthetai, S):
    M = thetai.shape[1]
    N = thetai.shape[2]
    nb_itr = thetai.shape[3]
    Xij = np.zeros((M, 3, N, nb_itr))
    mXi = np.zeros((M, 3, nb_itr))
    u, p = polar(np.mean(list_Q0, axis=0))
    for i in range(nb_itr):
        curv_fct_mi = lambda s: interp1d(S, mthetai[0,:,i])(s)
        tors_fct_mi = lambda s: interp1d(S, mthetai[1,:,i])(s)
        mQi = FrenetPath(S, S, init=u, curv=curv_fct_mi, tors=tors_fct_mi, dim=3)
        mQi.frenet_serret_solve()
        mXi[:,:,i] = mQi.data_trajectory
        for j in range(N):
            curv_fct_ij = lambda s: interp1d(S, thetai[0,:,j,i])(s)
            tors_fct_ij = lambda s: interp1d(S, thetai[1,:,j,i])(s)
            Qij = FrenetPath(S, S, init=list_Q0[j], curv=curv_fct_ij, tors=tors_fct_ij, dim=3)
            Qij.frenet_serret_solve()
            Xij[:,:,j,i] = Qij.data_trajectory
    return Xij, mXi
