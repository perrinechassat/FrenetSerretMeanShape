import numpy as np
from scipy.linalg import logm, svd, expm
import scipy.linalg
from sklearn.gaussian_process.kernels import Matern
import fdasrsf.utility_functions as uf
from scipy.integrate import trapz, cumtrapz, quad
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize, Bounds
from scipy.misc import derivative
from numpy.linalg import norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
import collections
import optimum_reparamN2_C as orN2_C
import sys
from maths_utils import *
from frenet_path import *
from visu_utils import *
import fdasrsf as fs
from numba import jit
from sympy import integrate, symbols

""" Set of functions for the alignment of curvature and torsion """

def optimum_reparam_1d(theta1, time, theta2, lam=0.0, grid_dim=7):

    gam = orN2_C.coptimum_reparam(np.ascontiguousarray(theta1), time, np.ascontiguousarray(theta2), lam, grid_dim)

    return gam


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
    theta1_norm = theta1/np.linalg.norm(theta1, 1)
    theta2_norm = theta2/np.linalg.norm(theta2, 1)

    gam = orN2_C.coptimum_reparamN2(np.ascontiguousarray(theta1_norm), time,
                                          np.ascontiguousarray(theta2_norm), lam, grid_dim)

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
    theta1_norm = theta1/np.linalg.norm(theta1, 1)
    theta2_norm = theta2/np.linalg.norm(theta2, 1)

    gam = orN2_C.coptimum_reparam_curve(np.ascontiguousarray(theta1_norm), time,
                                         np.ascontiguousarray(theta2_norm), lam, grid_dim)

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
    if len(f.shape)==2:
        f = f[np.newaxis, :, :]

    n = f.shape[0]
    M = f.shape[1]
    N = f.shape[2]
    parallel = True

    eps = np.finfo(np.double).eps

    # smoothdata = True
    # if smoothdata:
    #     f_init = f
    #     f_smooth = np.zeros((n, M, N))
    #     for j in range(n):
    #         for k in range(0, N):
    #             spar = time.shape[0] * (.025 * np.fabs(f[j, :, k]).max()) ** 2
    #             tmp_spline = UnivariateSpline(time, f[j, :, k], s=spar)
    #             f_smooth[j, :, k] = tmp_spline(time)
    # f = f_smooth

    f0 = f

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

    # plot_array_2D(time, fn[0].T, '')
    # plot_array_2D(time, fn[1].T, '')

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


def group_action_curvatures(f, gam, time, smooth=True):
    M = gam.shape[0]
    N = gam.shape[1]
    fi = np.zeros((M,N))

    if smooth==True:
        for n in range(N):
            g, g1, g2 = spline_approx(time, gam[:, n], smooth=True)
            time0 = (time[-1] - time[0]) * g + time[0]
            fi[:,n] = f(time0) * g1
    else:
        for n in range(N):
            time0 = (time[-1] - time[0]) * gam[:, n] + time[0]
            fi[:,n] = f(time0) * np.gradient(gam[:, n], 1 / float(M - 1))
    return fi


def group_action_mean_arclenght(ms, gam_g, gam_h, time, smooth=True):
    M = gam_g.shape[0]
    N = gam_g.shape[1]
    si = np.zeros((M,N))

    if smooth==True:
        for n in range(N):
            # g_h, g1_h, g2_h = spline_approx(time, gam_h[:, n], smooth=True)
            g_g, g1_g, g2_g = spline_approx(time, gam_g[:, n], smooth=True)
            time0 = (time[-1] - time[0]) * gam_h[:, n] + time[0]
            si[:,n] = np.interp((time[-1] - time[0]) * ms(time0) + time[0], time, g_g)
    else:
        for n in range(N):
            time0 = (time[-1] - time[0]) * gam_h[:, n] + time[0]
            si[:,n] = np.interp((time[-1] - time[0]) * ms(time0) + time[0], time, gam_g[:,n])
    return si



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



def warp_curvatures_bis(theta, gam, time):
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
    for n in range(N):
        time0 = (time[-1] - time[0]) * gam[:, n] + time[0]
        theta_align[:,n] = np.interp(time0, time, theta[:,n]) * np.gradient(gam[:, n], 1 / float(M - 1))
    weighted_mean_theta = np.mean(theta_align, axis=1)

    return theta_align, weighted_mean_theta


def align_and_center(gam, mf, f, itr, time):
    """
    Utility functions for the alignment function, used to aligned functions at the end of the iterations.
    ...

    """
    n = f.shape[0]
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


def invertGamma_fct(gam, t):
    gam_inv = interp1d(gam(t), t)
    return gam_inv


def invert(f, x):
    y = f(x)
    x_m = x[0]
    x_M = x[-1]
    x_bis = np.linspace(np.min(y),np.max(y),len(x))
    y_bis = np.zeros(x_bis.shape)

    def diff(t,a):
        yt = f(t)
        return (yt - a )**2

    for idx,x_value in enumerate(x_bis):
        res = minimize(diff, 0.5, args=(x_value), method='Nelder-Mead', bounds=Bounds(x_m, x_M), tol=1e-6)
        y_bis[idx] = res.x[0]

    return x_bis, y_bis


# def invert_warp_func(gam, t_in, t_out):
#     M = gam.shape[0]
#     N = gam.shape[1]
#
#     gamI = np.zeros((len(t_out), N))
#     for n in range(N):
#         g = interp1d(t_in, gam[:,n])
#         print(t_in[0], t_in[-1])
#         x, y = invert(g, t_in)
#         g_inv = interp1d(x, y)
#         gamI[:,n] = g_inv(t_out)
#
#     return gamI



def invert_warp_func(gam, t):
    """
    finds the inverse of the diffeomorphism gammas

    """
    M = gam.shape[0]
    N = gam.shape[1]

    x = np.linspace(0,1,M)
    gamI = np.zeros((M,N))
    gamI_fct = np.empty((N), dtype=object)
    for n in range(N):
        g = interp1d((gam[:,n] - gam[:,n][0]) / (gam[:,n][-1] - gam[:,n][0]), t)
        gam_i = g(t)
        gamI_fct[n] = g
        gamI[:,n] = (gam_i - gam_i[0]) / (gam_i[-1] - gam_i[0])
    return gamI, gamI_fct



def warp_value_gamma(t, f, gam):
    M = gam.size
    N = f.shape[0]
    f_temp = np.zeros(f.shape)
    for n in range(N):
        f_temp[n] = np.interp((t[-1] - t[0]) * gam + t[0], t, f[n])
    return f_temp

def warp_norm_gamma(t, f, gam):
    M = gam.size
    N = f.shape[0]
    f_temp = np.zeros(f.shape)
    gam_dev = np.sqrt(np.gradient(gam, 1 / np.double(M - 1)))
    for n in range(N):
        f_temp[n] = np.interp((t[-1] - t[0]) * gam + t[0], t, f[n]) * gam_dev
    return f_temp

def warp_area_gamma(t, f, gam):
    M = gam.size
    N = f.shape[0]
    f_temp = np.zeros(f.shape)
    gam_dev = np.gradient(gam, 1 / np.double(M - 1))
    for n in range(N):
        f_temp[n] = np.interp((t[-1] - t[0]) * gam + t[0], t, f[n]) * gam_dev
    return f_temp

def dist_phase_amp(f1, f2, time, gam):
    Dy_L2 = np.sqrt(trapz((f2 - f1) ** 2, time))
    Dy_L1 = trapz(abs(f2 - f1), time)
    M = time.shape[0]
    time1 = np.linspace(0,1,M)
    binsize = np.mean(np.diff(time))
    psi = np.sqrt(np.gradient(gam,binsize))
    f1dotf2 = trapz(psi, time1)
    if f1dotf2 > 1:
        f1dotf2 = 1
    elif f1dotf2 < -1:
        f1dotf2 = -1
    Dx = np.real(np.arccos(f1dotf2))
    return Dy_L1, Dy_L2, Dx

def dist_value_warp(f1, f2, time, method="DP", lam=0.0):
    if len(f1.shape)==1:
        f1 = f1/np.max(abs(f1))
        f2 = f2/np.max(abs(f2))
        f, g, g2 = fs.utility_functions.gradient_spline(t, f1, False)
        q1 = g / np.sqrt(abs(g) + np.finfo(np.double).eps)
        f, g, g2 = fs.utility_functions.gradient_spline(t, f2, False)
        q2 = g / np.sqrt(abs(g) + np.finfo(np.double).eps)
        gam = fs.utility_functions.optimum_reparam(q1, time, q2, method, lam)
    else:
        f1 = f1/np.max(abs(f1), axis=-1)[:,None]
        f2 = f2/np.max(abs(f2), axis=-1)[:,None]
        q1 = fs.curve_functions.curve_to_q(f1, 'C')[0]
        q2 = fs.curve_functions.curve_to_q(f2, 'C')[0]
        gam = fs.curve_functions.optimum_reparam_curve(q1, q2, lam=lam)
    # if gam.ndim > 1:
    #     gam = gam[0,:].squeeze()
    fw = warp_value_gamma(time, f2, gam)
    Dy_L1, Dy_L2, Dx = dist_phase_amp(f1, fw, time, gam)
    return Dy_L1, Dy_L2, Dx

def dist_norm_warp(f1, f2, time, method="DP", lam=0.0):
    f1 = f1/np.linalg.norm(f1, axis=-1)[:,None]
    f2 = f2/np.linalg.norm(f2, axis=-1)[:,None]
    # gam = fs.utility_functions.optimum_reparam(f1.T, time, f2.T, method, lam)
    gam = fs.curve_functions.optimum_reparam_curve(f1, f2, lam=lam)
    # if gam.ndim > 1:
    #     gam = gam[:,0].squeeze()
    fw = warp_norm_gamma(time, f2, gam)
    Dy_L1, Dy_L2, Dx = dist_phase_amp(f1, fw, time, gam)
    return Dy_L1, Dy_L2, Dx

def dist_area_warp(f1, f2, time, lam=0.0):
    f1 = f1/np.linalg.norm(f1, axis=-1, ord=1)[:,None]
    f2 = f2/np.linalg.norm(f2, axis=-1, ord=1)[:,None]
    gam = optimum_reparam_vect_curvatures(f1, time, f2, lam)
    fw = warp_area_gamma(time, f2, gam)
    Dy_L1, Dy_L2, Dx = dist_phase_amp(f1, fw, time, gam)
    return Dy_L1, Dy_L2, Dx


def dist_area_warp_bis(f1, f2, time, lam=0.0):
    f1 = f1/np.linalg.norm(f1, axis=-1, ord=1)[:,None]
    f2 = f2/np.linalg.norm(f2, axis=-1, ord=1)[:,None]
    sqrt_f1 = np.sqrt(abs(f1))
    sqrt_f2 = np.sqrt(abs(f2))
    # gam = fs.utility_functions.optimum_reparam(sqrt_f1.T, time, sqrt_f2.T, lam=lam)
    gam = fs.curve_functions.optimum_reparam_curve(sqrt_f1, sqrt_f2, lam=lam)
    # if gam.ndim > 1:
    #     gam = gam[:,0].squeeze()
    fw = warp_area_gamma(time, f2, gam)
    Dy_L1, Dy_L2, Dx = dist_phase_amp(f1, fw, time, gam)
    return Dy_L1, Dy_L2, Dx


# def partial_alignement(q1, q2, grid1, grid2, grid_a):
#     c1 = grid1[-1]
#     c2 = grid2[-1]
#     interp_q1 = interp1d(grid1, q1)
#     interp_q2 = interp1d(grid2, q2)
#     def q1_f(t):
#         if t<=c1:
#             return interp_q1(t)
#         else:
#             return 0
#     def q2_f(t):
#         if t<=c2:
#             return interp_q2(t)
#         else:
#             return 0
#     # q1_f = lambda t: interp_q1(t)*(t<=c1) + 0*(t>c1)
#     # q2_f = lambda t: interp_q2(t)*(t<=c2) + 0*(t>c2)
#     n = np.max([len(grid1),len(grid2)])
#     J = len(grid_a)
#     grid_b = np.zeros((J))
#     grid_gam = np.zeros((J,n))
#     grid_E = np.zeros((J))
#     for j in range(J):
#         q2_tilde_f = lambda t: np.sqrt(grid_a[j])*q2_f(grid_a[j]*t)*(t<=c2/grid_a[j]) + 0*(t>c2/grid_a[j])
#         grid_b[j] = np.min([c1, c2/grid_a[j]])
#         time0 = np.linspace(0,grid_b[j],n)
#         q1tilde = np.sqrt(1/grid_b[j])*np.array([q1_f(time0[k]) for k in range(n)])
#         q2tilde = np.sqrt(1/grid_b[j])*np.array([q2_tilde_f(time0[k]) for k in range(n)])
#         grid_gam[j] = fs.utility_functions.optimum_reparam(q1tilde, np.linspace(0,1,n), q2tilde)
#
#         E = np.trapz(y=np.array([(q1_f(time0[k])-q2_f(grid_a[j]*grid_b[j]*grid_gam[j][k])*np.sqrt(grid_a[j]*np.gradient(grid_gam[j], 1 / np.double(grid_gam[j].size - 1))[k]))**2 for k in range(n)]), x=time0)
#         func = lambda t: (q1_f(t) - q2_f(grid_a[j]*t)*np.sqrt(grid_a[j]))**2
#         E += quad(func, grid_b[j], np.inf)[0]
#         grid_E[j] = E
#
#     i = np.argmin(E)
#     ai = grid_a[i]
#     ds = np.sqrt(grid_E[i])
#     bi = grid_b[i]
#     gami = grid_gam[i]
#     return ai, bi, gami, ds

'''___________________________________________ Partial Alignement _______________________________________________'''



def align_subparts_gam(a,b,c,d):
    f = lambda s: ((c-d)/(a-b))*(s-a) + c
    return f

def compute_n(a, D):
    # f goes from [0,1] to [a,b]
    # f_inv from [a,b] to [0,1]
    f = lambda s: (D*s + a)*(a<=(D*s + a)<=a+D) + (D+a)*((D*s + a)>a+D) + a*((D*s + a)<a)
    f_inv = lambda s: ((s-a)/D)*(0<=((s-a)/D)<=1) + 1*(((s-a)/D)>1) + 0*(((s-a)/D)<0)
    # f = lambda s: (D*s + a)
    # f_inv = lambda s: ((s-a)/D)
    return f, f_inv

def compute_f(alpha, beta):
    # f = lambda s: np.round(alpha*s + beta, decimals=8)
    # f_inv = lambda s: np.round((s-beta)/alpha, decimals=8)
    # f = lambda s: (alpha*s + beta)*(beta<=(alpha*s + beta)<=alpha+beta) + (alpha+beta)*((alpha*s + beta)>alpha+beta) + alpha*((alpha*s + beta)<alpha)
    # f_inv = lambda s: ((s-beta)/alpha)*(0<=((s-beta)/alpha)<=1) + 1*(((s-beta)/alpha)>1) + 0*(((s-beta)/alpha)<0)
    f = lambda s: (alpha*s + beta)
    f_inv = lambda s: ((s-beta)/alpha)
    return f, f_inv

def poly(order, gamma_spl, a):
    if order==0:
        poly = lambda s: gamma_spl(a) + 0*s
        poly_dev = lambda s: 0*s
    elif order==1:
        poly = lambda s: gamma_spl(a) + (s-a)*gamma_spl.derivative(1)(a)
        poly_dev = lambda s: gamma_spl.derivative(1)(a) + 0*s
    elif order==2:
        poly = lambda s: gamma_spl(a) + (s-a)*gamma_spl.derivative()(a) + ((s-a)**2)*gamma_spl.derivative(2)(a)/2
        poly_dev = lambda s:  gamma_spl.derivative()(a) + (s-a)*gamma_spl.derivative(2)(a)
    elif order==3:
        poly = lambda s: gamma_spl(a) + (s-a)*gamma_spl.derivative()(a) + ((s-a)**2)*gamma_spl.derivative(2)(a)/2 + ((s-a)**3)*gamma_spl.derivative(3)(a)/(2*3)
        poly_dev = lambda s:  gamma_spl.derivative()(a) + (s-a)*gamma_spl.derivative(2)(a) + ((s-a)**2)*gamma_spl.derivative(3)(a)/(2)
    elif order==4:
        poly = lambda s: gamma_spl(a) + (s-a)*gamma_spl.derivative()(a) + ((s-a)**2)*gamma_spl.derivative(2)(a)/2 + ((s-a)**3)*gamma_spl.derivative(3)(a)/(2*3) + ((s-a)**4)*gamma_spl.derivative(4)(a)/(2*3*4)
        poly_dev = lambda s:  gamma_spl.derivative()(a) + (s-a)*gamma_spl.derivative(2)(a) + ((s-a)**2)*gamma_spl.derivative(3)(a)/(2) + ((s-a)**3)*gamma_spl.derivative(4)(a)/(2*3)
    else:
        raise ValueError("order must be <= 4")
    return poly, poly_dev

def partial_align_1d_fromInt(X1, X2, init_ind2, sub_int1, sub_int2, lbda=0.0):
    a, b, c, d = sub_int1[0], sub_int1[1], sub_int2[0], sub_int2[1]
    alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
    out = partial_align_1d(X1, X2, init_ind2, a, D, alpha, beta, lbda)
    return out


def partial_align_1d(X1, X2, init_ind2, a, D, alpha, beta, lbda=0):
    # step 1:
    b = a+D
    f, f_inv = compute_f(alpha, beta)
    X2tilde = lambda s: alpha*X2(f(s))

    t = np.linspace(0,1,200)
    n, n_inv = compute_n(a, D)

    x1 = np.array([D*X1(n(t_)) for t_ in t])
    x2 = np.array([D*X2tilde(n(t_)) for t_ in t])
    gam = optimum_reparam_1d(x1, t, x2, lam=lbda)
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    gam_inv = fs.utility_functions.invertGamma(gam)
    gamma = interp1d(t, gam, fill_value=([gam[0]], [gam[-1]]), bounds_error=False)
    gamma_inv = interp1d(t, gam_inv, fill_value=([gam_inv[0]], [gam_inv[-1]]), bounds_error=False)
    gamma_tilde = UnivariateSpline(np.linspace(a,b,200), np.array([n(gamma(n_inv(j))) for j in np.linspace(a,b,200)]), k=5)
    gamma_tilde_prime = gamma_tilde.derivative()
    gamma_tilde_inv = interp1d(np.linspace(a,b,200), np.array([n(gamma_inv(n_inv(t_))) for t_ in np.linspace(a,b,200)]), fill_value=([a], [b]), bounds_error=False)

    # step 3:
    # q1 = np.sqrt(x1)
    # q2 = np.sqrt(x2)
    # gam = fs.utility_functions.optimum_reparam(q1, t, q2, lam=lbda)
    # gam = optimum_reparam_1d(x1, t, x2, lam=lbda)
    # gam = (gam - gam[0]) / (gam[-1] - gam[0])
    # gam_inv = fs.utility_functions.invertGamma(gam)
    # gamma = interp1d(t, gam, fill_value=([gam[0]], [gam[-1]]), bounds_error=False)
    # gamma_inv = interp1d(t, gam_inv, fill_value=([gam_inv[0]], [gam_inv[-1]]), bounds_error=False)
    # gam_dev = np.gradient(gam, 1/np.double(200 - 1))
    # gamma_prime = interp1d(t, gam_dev, fill_value=([gam_dev[0]], [gam_dev[-1]]), bounds_error=False)
    # step 4:
    # gamma_tilde = lambda s : n(gamma(n_inv(s)))
    # gamma_tilde_inv = lambda s : n(gamma_inv(n_inv(s)))
    # gamma_tilde_interp = UnivariateSpline(np.linspace(0,1,200), np.array([gamma(j) for j in np.linspace(0,1,200)]), k=5)
    # gamma_tilde_prime = gamma_tilde_interp.derivative()
    # gamma_tilde = lambda s : n(gamma_tilde_interp(n_inv(s)))
    # gamma_tilde_inv = lambda s : n(gamma_inv(n_inv(s)))
    # gamma_tilde = UnivariateSpline(np.linspace(a,b,200), np.array([n(gamma(n_inv(j))) for j in np.linspace(a,b,200)]), k=5)
    # gamma_tilde_prime = gamma_tilde_interp.derivative()
    # gamma_tilde_inv = interp1d(gamma_tilde(np.linspace(a,b,200)), np.linspace(a,b,200), fill_value=([a], [b]), bounds_error=False)

    # step 5:

    def g(s):
        if a<=s<=b:
            return gamma_tilde(s)
        elif s<a:
            return gamma_tilde_prime(a)*(s-a) + gamma_tilde(a)
        elif s>b:
            return gamma_tilde_prime(b)*(s-b) + gamma_tilde(b)

    def g_prime(s):
        if a<=s<=b:
            return gamma_tilde_prime(s)
        elif s<a:
            return gamma_tilde_prime(a)
        elif s>b:
            return gamma_tilde_prime(b)

    def g_inv(s):
        if a<=s<=b:
            return gamma_tilde_inv(s)
        elif s<a:
            return (1/gamma_tilde_prime(a))*(s-gamma_tilde(a)) + a
        elif s>b:
            return (1/gamma_tilde_prime(b))*(s-gamma_tilde(b)) + b

    A, B = g_inv(f_inv(init_ind2[0])), g_inv(f_inv(init_ind2[1]))
    g_tilde = UnivariateSpline(np.linspace(A,B,400), np.array([g(j) for j in np.linspace(A,B,400)]), k=4, s=0.001)
    g_tilde_prime = g_tilde.derivative()

    def X2new(s):
        return X2tilde(g_tilde(s))*g_tilde_prime(s)

    partial_align_results = collections.namedtuple('partial_align', ['X2new', 'A', 'B', 'g', 'g_prime', 'g_inv', 'gamma', 'gamma_inv', 'gamma_prime', 'gam'])
    out = partial_align_results(X2new, A, B, g_tilde, g_tilde_prime, g_inv, gamma_tilde, gamma_tilde_inv, gamma_tilde_prime, gam)
    return out


def extend_X(X, int):
    A,B = int[0], int[1]
    def X_tilde(s):
        if A<=s<=B:
            return X(s)
        elif s<A:
            return X(A)
        else:
            return X(B)
    return X_tilde

def H(a, D, X1, X2new, int1, int2new, ratio):
    min_int = np.min([int1[0],int2new[0]])
    max_int = np.max([int1[1],int2new[1]])
    x1_tilde = extend_X(X1, int1)
    x2_tilde = extend_X(X2new, int2new)

    # grid1 = np.linspace(min_int, a, int(abs(min_int-a)/ratio)+1) #, decimals=6)
    # grid2 = np.linspace(a, a+D, int(abs(D)/ratio)+1)
    # grid3 = np.linspace(a+D, max_int, int(abs(max_int-a-D)/ratio)+1) #, decimals=6)
    # dist = np.array([x1_tilde(grid2[i])-x2_tilde(grid2[i]) for i in range(len(grid2))])
    # dist_shape = np.trapz(dist**2, grid2)
    # # dist_shape = np.trapz(abs(dist), grid2)
    # p1 = np.array([x1_tilde(t)-x2_tilde(t) for t in grid1])
    # p2 = np.array([x1_tilde(t)-x2_tilde(t) for t in grid3])
    # pen = np.trapz(p1**2,grid1)+ np.trapz(p2**2,grid3)
    # # pen = np.trapz(abs(p1),grid1)+ np.trapz(abs(p2),grid3)
    # print('dist_shape 2', dist_shape, 'dist_shape 1', np.trapz(abs(dist), grid2), 'pen 2', pen, 'pen 1', np.trapz(abs(p1),grid1)+ np.trapz(abs(p2),grid3) )
    # # print('L1', dist_shape+pen, 'L2', np.trapz(dist**2, grid2)+np.trapz(p1**2,grid1)+ np.trapz(p2**2,grid3))
    #
    grid = np.linspace(min_int, max_int, int(abs(max_int-min_int)/ratio)+1)
    tot_dist = np.trapz(np.array([x1_tilde(grid[i])-x2_tilde(grid[i]) for i in range(len(grid))])**2, grid)
    # print('tot_dist', tot_dist)
    return tot_dist


def cost_gridsearch(X1, X2, init_ind1, init_ind2, lbda):
    def func(x):
        a, b, c, d = x[0], x[1], x[2], x[3]
        alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
        out = partial_align_1d(X1, X2, init_ind2, a, D, alpha, beta, lbda)
        cost = H(a, D, X1, out.X2new, init_ind1, [out.A, out.B], 1/1000)
        return cost
    return func


def partial_align_1d_bis(X1, X2, init_ind2, a, D, alpha, beta, lbda=0):
    # step 1:
    b = a+D
    f, f_inv = compute_f(alpha, beta)
    X2tilde = lambda s: alpha*X2(f(s))
    # X2tilde = lambda s: X2(f(s))

    t = np.linspace(0,1,200)
    n, n_inv = compute_n(a, D)

    x1 = np.array([D*X1(n(t_)) for t_ in t])
    x2 = np.array([D*X2tilde(n(t_)) for t_ in t])
    gam = optimum_reparam_1d(x1, t, x2, lam=lbda)
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    # gam_inv = fs.utility_functions.invertGamma(gam)
    n_gam = np.array([n(g_) for g_ in gam])
    gamma = interp1d(np.array([n(t_) for t_ in t]), n_gam, fill_value=([n_gam[0]], [n_gam[-1]]), bounds_error=False)
    # gamma_inv = interp1d(t, gam_inv, fill_value=([gam_inv[0]], [gam_inv[-1]]), bounds_error=False)
    gamma_tilde = UnivariateSpline(np.linspace(a,b,200), np.array([gamma(j) for j in np.linspace(a,b,200)]), k=5)
    gamma_tilde_prime = gamma_tilde.derivative()
    #
    def X2new(s):
        return X2tilde(gamma(s))*gamma_tilde_prime(s)

    partial_align_results = collections.namedtuple('partial_align', ['X2new', 'gamma', 'gamma_smooth', 'gamma_prime', 'gam', 'a', 'b'])
    out = partial_align_results(X2new, gamma, gamma_tilde, gamma_tilde_prime, gam, a, b)
    # partial_align_results = collections.namedtuple('partial_align', ['gamma', 'gam'])
    # out = partial_align_results(gamma, gam)

    return out


def partial_align_1d_v3(X1, X2, init_ind2, a, D, alpha, beta, lbda=0):
    # step 1:
    b = a+D
    f, f_inv = compute_f(alpha, beta)
    X2tilde = lambda s: alpha*X2(f(s))
    # X2tilde = lambda s: X2(f(s))

    t = np.linspace(0,1,200)
    n, n_inv = compute_n(a, D)

    x1 = np.array([D*X1(n(t_)) for t_ in t])
    x2 = np.array([D*X2tilde(n(t_)) for t_ in t])
    gam = optimum_reparam_1d(x2, t, x1, lam=lbda)
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    # gam_inv = fs.utility_functions.invertGamma(gam)
    n_gam = np.array([n(g_) for g_ in gam])
    gamma = interp1d(np.array([n(t_) for t_ in t]), n_gam, fill_value=([n_gam[0]], [n_gam[-1]]), bounds_error=False)
    # gamma_inv = interp1d(t, gam_inv, fill_value=([gam_inv[0]], [gam_inv[-1]]), bounds_error=False)
    gamma_tilde = UnivariateSpline(np.linspace(a,b,200), np.array([gamma(j) for j in np.linspace(a,b,200)]), k=5)
    gamma_tilde_prime = gamma_tilde.derivative()
    #
    def X1new(s):
        return X1(gamma_tilde(s))*gamma_tilde_prime(s)

    partial_align_results = collections.namedtuple('partial_align', ['X1new', 'X2new', 'gamma', 'gamma_smooth', 'gamma_prime', 'gam', 'a', 'b'])
    out = partial_align_results(X1new, X2tilde, gamma, gamma_tilde, gamma_tilde_prime, gam, a, b)
    # partial_align_results = collections.namedtuple('partial_align', ['gamma', 'gam'])
    # out = partial_align_results(gamma, gam)

    return out


def cost_bis(X1, X2, init_ind1, init_ind2, lbda):
    def func(x):
        a, b, c, d = x[0], x[1], x[2], x[3]
        alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
        out = partial_align_1d_bis(X1, X2, init_ind2, a, D, alpha, beta, lbda)
        grid = np.linspace(a,b,200)
        cost = np.power(init_ind1[1]/(b-a), 3)*np.power(init_ind2[1]/(d-c), 3)*np.trapz((X1(grid)-out.X2new(grid))**2, grid) + lbda*np.trapz((np.ones(200,)-out.gamma_prime(grid))**2, grid)
        # print('shape dist:', np.trapz((X1(grid)-out.X2new(grid))**2, grid), 'weighted shape dist:', np.power(init_ind1[1]/(b-a), 3)*np.power(init_ind2[1]/(d-c), 3)*np.trapz((X1(grid)-out.X2new(grid))**2, grid), 'pen:', lbda*np.trapz((np.ones(200,)-out.gamma_prime(grid))**2, grid))
        return cost
    return func

def make_grid(list1, list2, dist=0):
    c1 = np.concatenate(np.stack(np.meshgrid(list1,list1), axis=-1))
    L1 =  []
    for x in c1:
        if x[0]<x[1]:
            if x[1]-x[0]>=dist:
                L1.append(x)
    L1 = np.unique(L1, axis=0)
    c2 = np.concatenate(np.stack(np.meshgrid(list2,list2), axis=-1))
    L2 =  []
    for x in c2:
        if x[0]<x[1]:
            if x[1]-x[0]>=dist:
                L2.append(x)
    L2 = np.unique(L2, axis=0)
    L_final = np.empty((len(L1), len(L2)), dtype=object)
    for i in range(len(L1)):
        for j in range(len(L2)):
            L_final[i,j] = np.concatenate([L1[i], L2[j]])
    return np.concatenate(L_final)


''' Gradient descent '''

# def param_to_p(a, b, c, d, gamma_prime, int1):
#     alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
#     p = np.empty((5), dtype=object)
#     p[0], p[1], p[2], p[3] = a, np.log(D/int1[1]), np.log(alpha), beta
#     gam = np.sqrt(gamma_prime(np.linspace(0,1,200)))
#     # p[4] = (gam - gam[0]) / (gam[-1] - gam[0])
#     p[4] = gam
#     return p

def param_to_p(a, b, c, d, gam, int1):
    alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
    p = np.empty((5), dtype=object)
    p[0], p[1], p[2], p[3] = a, np.log(D/int1[1]), np.log(alpha), beta
    psi = np.sqrt(np.gradient(gam))
    # p[4] = (gam - gam[0]) / (gam[-1] - gam[0])
    p[4] = psi
    return p

def p_to_param(p, init_ind1):
    a, D, alpha, beta, psi = p[0], init_ind1[1]*np.exp(p[1]), np.exp(p[2]), p[3], p[4]
    b = a+D
    d = beta + alpha*b
    c = d - alpha*D
    return a, b, c, d

def compose_Xk_p(p, Xk, indk, init_ind1):
    ''' p = (a, delta, mu, beta, psi) '''
    a, D, alpha, beta, psi = p[0], init_ind1[1]*np.exp(p[1]), np.exp(p[2]), p[3], p[4]
    b = a+D
    f, f_inv = compute_f(alpha, beta)
    n, n_inv = compute_n(a, D)

    time = np.linspace(0,1,200)
    gam_prime = psi**2
    # gam_prime = (gam_prime - gam_prime[0]) / (gam_prime[-1] - gam_prime[0])
    gamma_prime = interp1d(time, gam_prime)
    gam = cumtrapz(psi**2, time, initial=0)
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    gamma = interp1d(time, gam)

    # step 4:
    gamma_tilde = lambda s : n(gamma(n_inv(s)))
    gamma_tilde_interp = UnivariateSpline(np.linspace(a,b,200), gamma_tilde(np.linspace(a,b,200)), k=3)
    gamma_tilde_prime = gamma_tilde_interp.derivative()
    # step 5:
    def g(s):
        if a<=s<=b:
            return f(gamma_tilde(s))
        else:
            return f(s)
    def g_prime(s):
        if a<=s<=b:
            return alpha*gamma_tilde_prime(s)
        else:
            return alpha
    A, B = f_inv(indk[0]), f_inv(indk[1])
    def Xknew(s):
        return Xk(g(s))*g_prime(s)

    compose_res = collections.namedtuple('compose_results', ['Xnew', 'A', 'B', 'g', 'g_prime', 'params'])
    out = compose_res(Xknew, A, B, g, g_prime, [a, D, alpha, beta])
    return out


def compute_Ek(p, X1, X2, init_ind1, init_ind2, lbda, ratio):
    out = compose_Xk_p(p, X2, init_ind2, init_ind1)
    X2new = out.Xnew
    A, B = out.A, out.B
    # print(A, B)
    cost = E(out.params[0], out.params[1], X1, X2new, init_ind1, [A,B], ratio)
    Ek_results = collections.namedtuple('Ek_results', ['X2new', 'A', 'B', 'g', 'g_prime', 'cost'])
    out = Ek_results(X2new, A, B, out.g, out.g_prime, cost)
    return out


def compute_Ek_bis(p, X1, X2, init_ind1, init_ind2, lbda, ratio):
    a, D, alpha, beta, psi = p[0], init_ind1[1]*np.exp(p[1]), np.exp(p[2]), p[3], p[4]
    out = partial_align_1d(X1, X2, init_ind2, a, D, alpha, beta)
    X2new = out.X2new
    A, B = out.A, out.B
    # print(A, B)
    cost = E(a, D, X1, X2new, init_ind1, [A,B], ratio)
    Ek_results = collections.namedtuple('Ek_results', ['X2new', 'A', 'B', 'g', 'g_prime', 'cost'])
    out = Ek_results(X2new, A, B, out.g, out.g_prime, cost)
    return out


def compute_dev(f, time, smooth):
    if smooth:
        spar = time.shape[0] * (.025 * fabs(np.array([f(t) for t in time])).max()) ** 2
    else:
        spar = 0
    tmp_spline = UnivariateSpline(time, np.array([f(t) for t in time]), s=spar)
    return tmp_spline.derivative()

# def compute_grad_Ek_id(X1, X2_k, init_ind1, int2_k, lbda):
#     L1 = init_ind1[1]
#     X2_k_dev = compute_dev(X2_k, np.linspace(int2_k[0], int2_k[1], int(abs(int2_k[1]-int2_k[0])/(1/1000))+1), False)
#     # X2_k_dev = lambda s: derivative(X2_k, s, dx=0.0001)
#     # print(X2_k_dev(int2_k[0]+0.0001))
#     x1_tilde = extend_X(X1, init_ind1)
#     x2_tilde = extend_X(X2_k, int2_k)
#     x2_dev_tilde = extend_X(X2_k_dev, int2_k)
#
#     DEk_a = (lbda-1)*abs(x1_tilde(0)-x2_tilde(0)) + (1-lbda)*abs(x1_tilde(L1)-x2_tilde(L1))*L1
#     DEk_delta = (1-lbda)*abs(x1_tilde(L1)-x2_tilde(L1))*L1
#
#     def f_xi(s):
#         return -np.sign(x1_tilde(s)-x2_tilde(s))*(s*x2_dev_tilde(s)+x2_tilde(s))
#     DEk_xi = quad(f_xi, 0, L1, epsabs=1e-4, limit=100)[0] + lbda*(quad(f_xi, -np.inf, 0, epsabs=1e-4, limit=100)[0] + quad(f_xi, L1, np.inf, epsabs=1e-4, limit=100)[0])
#
#     def f_beta(s):
#         return -np.sign(x1_tilde(s)-x2_tilde(s))*x2_dev_tilde(s)
#     DEk_beta = quad(f_beta, 0, L1, epsabs=1e-4, limit=100)[0] + lbda*(quad(f_beta, -np.inf, 0, epsabs=1e-4, limit=100)[0] + quad(f_beta, L1, np.inf, epsabs=1e-4, limit=100)[0])
#
#     print(DEk_a, DEk_delta, DEk_xi, DEk_beta)
#
#
#     def w_k(x):
#         return 2*L1*quad(f_beta, 0, L1*x, epsabs=1e-4, limit=100)[0] - 2*L1*np.sign(x1_tilde(L1*x)-x2_tilde(L1*x))*x2_tilde(L1*x)
#     # w_k = lambda x: 2*L1*quad(f_beta, 0, L1*x)[0] - 2*L1*np.sign(x1_tilde(L1*x)-x2_tilde(L1*x))*x2_tilde(L1*x)
#     int_w_k = quad(w_k, 0, 1, epsabs=1e-4, limit=100)[0]
#     print(int_w_k)
#     DEk_psi = lambda x: w_k(x) + int_w_k
#     scalprod_psi = quad(lambda s: DEk_psi(s)**2, 0, 1, epsabs=1e-4, limit=100)[0]
#
#     norm = np.sqrt(DEk_a**2 + DEk_delta**2 + DEk_xi**2 + DEk_beta**2 + scalprod_psi)
#
#     res = np.empty((5), dtype=object)
#     res[0], res[1], res[2], res[3] = DEk_a, DEk_delta, DEk_xi, DEk_beta
#     res[4] = DEk_psi
#     return res, norm, scalprod_psi


def compute_grad_Ek_id(X1, X2_k, init_ind1, int2_k, lbda):
    ratio = 1/1000
    L1 = init_ind1[1]
    min_int = np.min([init_ind1[0],int2_k[0]])
    max_int = np.max([init_ind1[1],int2_k[1]])

    X2_k_dev = compute_dev(X2_k, np.linspace(int2_k[0], int2_k[1], int(abs(int2_k[1]-int2_k[0])/ratio)+1), False)
    x1_tilde = extend_X(X1, init_ind1)
    x2_tilde = extend_X(X2_k, int2_k)
    x2_dev_tilde = extend_X(X2_k_dev, int2_k)

    DEk_a = (lbda-1)*abs(x1_tilde(0)-x2_tilde(0)) + (1-lbda)*abs(x1_tilde(L1)-x2_tilde(L1))*L1
    DEk_delta = (1-lbda)*abs(x1_tilde(L1)-x2_tilde(L1))*L1

    f_xi = lambda s: -np.sign(x1_tilde(s)-x2_tilde(s))*(s*x2_dev_tilde(s)+x2_tilde(s))
    f_beta = lambda s: -np.sign(x1_tilde(s)-x2_tilde(s))*x2_dev_tilde(s)

    grid2 = np.linspace(init_ind1[0], L1, int(abs(L1-init_ind1[0])/ratio)+1)
    DEk_xi = np.trapz(y=np.array([f_xi(t) for t in grid2]), x=grid2)
    DEk_beta = np.trapz(y=np.array([f_beta(t) for t in grid2]), x=grid2)
    if min_int < init_ind1[0]:
        grid1 = np.linspace(min_int, init_ind1[0], int(abs(init_ind1[0]-min_int)/ratio)+1)
        DEk_xi += lbda*np.trapz(y=np.array([f_xi(t) for t in grid1]), x=grid1)
        DEk_beta += lbda*np.trapz(y=np.array([f_beta(t) for t in grid1]), x=grid1)
    if max_int > L1:
        grid3 = np.linspace(L1, max_int, int(abs(max_int-L1)/ratio)+1)
        DEk_xi += lbda*np.trapz(y=np.array([f_xi(t) for t in grid3]), x=grid3)
        DEk_beta += lbda*np.trapz(y=np.array([f_beta(t) for t in grid3]), x=grid3)
    # print(DEk_a, DEk_delta, DEk_xi, DEk_beta)

    time = np.linspace(0, 1, 200)
    grid = np.linspace(0, L1, 200)
    y = np.array([f_beta(t) for t in grid])
    int_x_wk = cumtrapz(y, grid, initial=0)
    wk = np.array([(2*L1*int_x_wk[i] - 2*L1*np.sign(x1_tilde(L1*time[i])-x2_tilde(L1*time[i]))*x2_tilde(L1*time[i])) for i in range(len(time))])
    int_w_k = np.trapz(wk, x=time)
    DEk_psi = wk + int_w_k

    norm = np.sqrt(DEk_a**2 + DEk_delta**2 + DEk_xi**2 + DEk_beta**2 + np.trapz(DEk_psi**2, x=time))

    res = np.empty((5), dtype=object)
    res[0], res[1], res[2], res[3] = DEk_a, DEk_delta, DEk_xi, DEk_beta
    res[4] = DEk_psi
    return res, norm


def update_p_id(v, delta):
    res = np.empty((5), dtype=object)
    res[0], res[1], res[2], res[3] = -delta*v[0], -delta*v[1], -delta*v[2], -delta*v[3]
    norm_v4 = np.sqrt(np.trapz(v[4]**2, np.linspace(0,1,200)))
    res[4] = -delta*(np.sin(norm_v4)/norm_v4)*v[4] - delta*np.cos(norm_v4)
    return res

def group_action_psi(psi1, psi2):
    time = np.linspace(0,1,200)
    int_p2 = cumtrapz(psi2**2, time, initial=0)
    int_p2 = (int_p2 - int_p2[0]) / (int_p2[-1] - int_p2[0])
    p1_temp = np.interp((time[-1] - time[0]) * int_p2 + time[0], time, psi1)
    return p1_temp*psi2

def group_operation_P(p1, p2):
    res = np.empty((5), dtype=object)
    res[0], res[1], res[2], res[3] = p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]
    # res[4] = lambda t: p1[4](quad(lambda s: p2[4](s)**2, 0, t, limit=100)[0])*p2[4](t)
    res[4] = group_action_psi(p1[4], p2[4])
    return res


# def gradient_descent_E(X1, X2_0, ind1, ind2_0, p0, beta, tau, step_size, tol, max_iter, lbda):
#     k = 0
#     p_id = np.empty((5), dtype=object)
#     p_id[0], p_id[1], p_id[2], p_id[3] = 0, 0, 0, 0
#     p_id[4] = np.ones((200))
#     print('pid', p_to_param(p_id, ind1))
#     X2k = X2_0
#     pk = p0
#     print('p0', p_to_param(pk, ind1))
#     ind2k = ind2_0
#     out_Ek = compute_Ek(p_id, X1, X2k, ind1, ind2k, lbda, 1/1000)
#     Ek_id = out_Ek.cost
#     print('Cost 0', Ek_id)
#     DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#     print('Norm grad 0', norm_k)
#     print("grad :", DEk_id[0], DEk_id[1], DEk_id[2], DEk_id[3])
#     while norm_k>tol and k<max_iter:
#         delta_k = step_size
#         p = update_p_id(DEk_id, delta_k)
#         print('update', p_to_param(p, ind1))
#         print('new p', p_to_param(group_operation_P(pk, p), ind1))
#         out_Ek_p = compute_Ek(p, X1, X2k, ind1, ind2k, lbda, 1/1000)
#         Ek_p = out_Ek_p.cost
#         print('Cost p ',k, Ek_p)
#         # psik_psi = np.array([pk[4](quad(lambda s: p[4](s)**2, 0, t, limit=100)[0])*p[4](t) for t in np.linspace(0,1,200)])
#         psik_psi = group_action_psi(pk[4], p[4])
#         while Ek_p > Ek_id - beta*step_size*norm_k or (psik_psi<0).any():
#             print(Ek_p, Ek_id - beta*step_size*norm_k, (psik_psi<0).any())
#             delta_k = tau*delta_k
#             p = update_p_id(DEk_id, delta_k)
#             print(p_to_param(p, ind1))
#             out_Ek_p = compute_Ek(p, X1, X2k, ind1, ind2k, lbda, 1/1000)
#             Ek_p = out_Ek_p.cost
#             psik_psi = group_action_psi(pk[4], p[4])
#
#         pk = group_operation_P(pk, p)
#         out_Xk = compose_Xk_p(p, X2k, ind2k, ind1)
#         X2k = out_Xk.Xnew
#         ind2k = [out_Xk.A, out_Xk.B]
#         out_Ek = compute_Ek(p_id, X1, X2k, ind1, ind2k, lbda, 1/1000)
#         Ek_id = out_Ek.cost
#         DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#         k = k+1
#         print(k)
#
#     grad_desc = collections.namedtuple('Gradient_descent_results', ['X2new', 'int2new', 'p_opt', 'g', 'g_prime', 'cost'])
#     out = grad_desc(X2k, ind2k, pk, out_Xk.g, out_Xk.g_prime, Ek_id)
#     return out

# def gradient_descent_E(X1, X2, ind1, ind2, p0, beta, tau, step_size, tol, max_iter, lbda):
#     k = 0
#     p_id = np.empty((5), dtype=object)
#     p_id[0], p_id[1], p_id[2], p_id[3] = 0, 0, 0, 0
#     p_id[4] = np.ones((200))
#     print('pid', p_to_param(p_id, ind1))
#     pk = p0
#     print('p0', p_to_param(pk, ind1))
#     out_Xk = compose_Xk_p(pk, X2, ind2, ind1)
#     X2k = out_Xk.Xnew
#     ind2k = [out_Xk.A, out_Xk.B]
#     out_Ek = compute_Ek(p_id, X1, X2k, ind1, ind2k, lbda, 1/1000)
#     Ek_id = out_Ek.cost
#     print('Cost 0', Ek_id)
#     DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#     print('Norm grad 0', norm_k)
#     print("grad :", DEk_id[0], DEk_id[1], DEk_id[2], DEk_id[3])
#     while norm_k>tol and k<max_iter:
#         delta_k = step_size
#         p = group_operation_P(pk, update_p_id(DEk_id, delta_k))
#         print('update', p_to_param(update_p_id(DEk_id, delta_k), ind1))
#         print('new p', p_to_param(group_operation_P(pk, update_p_id(DEk_id, delta_k)), ind1))
#         out_Ek_p = compute_Ek(p, X1, X2, ind1, ind2, lbda, 1/1000)
#         Ek_p = out_Ek_p.cost
#         print('Cost p ',k, Ek_p)
#         # psik_psi = np.array([pk[4](quad(lambda s: p[4](s)**2, 0, t, limit=100)[0])*p[4](t) for t in np.linspace(0,1,200)])
#         psik_psi = group_action_psi(pk[4], p[4])
#         while Ek_p > Ek_id - beta*step_size*norm_k or (psik_psi<0).any():
#             print(Ek_p, Ek_id - beta*step_size*norm_k, (psik_psi<0).any())
#             delta_k = tau*delta_k
#             # p = update_p_id(DEk_id, delta_k)
#             # print(p_to_param(p, ind1))
#             delta_p = update_p_id(DEk_id, delta_k)
#             p = group_operation_P(pk, delta_p)
#             print('update', p_to_param(delta_p, ind1))
#             print('new p', p_to_param(p, ind1))
#             out_Ek_p = compute_Ek(p, X1, X2, ind1, ind2, lbda, 1/1000)
#             Ek_p = out_Ek_p.cost
#             psik_psi = group_action_psi(pk[4], delta_p[4])
#
#         pk = p
#         out_Xk = compose_Xk_p(p, X2, ind2, ind1)
#         X2k = out_Xk.Xnew
#         ind2k = [out_Xk.A, out_Xk.B]
#         out_Ek = compute_Ek(p_id, X1, X2k, ind1, ind2k, lbda, 1/1000)
#         Ek_id = out_Ek.cost
#         DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#         k = k+1
#         print(k)
#
#     grad_desc = collections.namedtuple('Gradient_descent_results', ['X2new', 'int2new', 'p_opt', 'g', 'g_prime', 'cost'])
#     out = grad_desc(X2k, ind2k, pk, out_Xk.g, out_Xk.g_prime, Ek_id)
#     return out

def gradient_descent_E(X1, X2, ind1, ind2, p0, beta, tau, step_size, tol, max_iter, lbda):
    k = 0
    p_id = np.empty((5), dtype=object)
    p_id[0], p_id[1], p_id[2], p_id[3] = 0, 0, 0, 0
    p_id[4] = np.ones((200))
    print('pid', p_to_param(p_id, ind1))
    pk = p0
    print('p0', p_to_param(pk, ind1))
    out_Ek = compute_Ek_bis(pk, X1, X2, ind1, ind2, lbda, 1/1000)
    X2k = out_Ek.X2new
    ind2k = [out_Ek.A, out_Ek.B]
    Ek_id = out_Ek.cost
    print('Cost 0', Ek_id)
    DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
    print('Norm grad 0', norm_k)
    print("grad :", DEk_id[0], DEk_id[1], DEk_id[2], DEk_id[3])
    while norm_k>tol and k<max_iter:
        delta_k = step_size
        p = group_operation_P(pk, update_p_id(DEk_id, delta_k))
        print('update', p_to_param(update_p_id(DEk_id, delta_k), ind1))
        print('new p', p_to_param(group_operation_P(pk, update_p_id(DEk_id, delta_k)), ind1))
        out_Ek_p = compute_Ek_bis(p, X1, X2, ind1, ind2, lbda, 1/1000)
        Ek_p = out_Ek_p.cost
        print('Cost p ',k, Ek_p)
        # psik_psi = np.array([pk[4](quad(lambda s: p[4](s)**2, 0, t, limit=100)[0])*p[4](t) for t in np.linspace(0,1,200)])
        psik_psi = group_action_psi(pk[4], p[4])
        while Ek_p > Ek_id - beta*step_size*norm_k or (psik_psi<0).any():
            print(Ek_p, Ek_id - beta*step_size*norm_k, (psik_psi<0).any())
            delta_k = tau*delta_k
            # p = update_p_id(DEk_id, delta_k)
            # print(p_to_param(p, ind1))
            delta_p = update_p_id(DEk_id, delta_k)
            p = group_operation_P(pk, delta_p)
            print('update', p_to_param(delta_p, ind1))
            print('new p', p_to_param(p, ind1))
            out_Ek_p = compute_Ek_bis(p, X1, X2, ind1, ind2, lbda, 1/1000)
            Ek_p = out_Ek_p.cost
            psik_psi = group_action_psi(pk[4], delta_p[4])

        pk = p
        out_Ek = compute_Ek_bis(p, X1, X2, ind1, ind2, lbda, 1/1000)
        X2k = out_Ek.Xnew
        ind2k = [out_Ek.A, out_Ek.B]
        Ek_id = out_Ek.cost
        DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
        k = k+1
        print(k)

    grad_desc = collections.namedtuple('Gradient_descent_results', ['X2new', 'int2new', 'p_opt', 'g', 'g_prime', 'cost'])
    out = grad_desc(X2k, ind2k, pk, out_Xk.g, out_Xk.g_prime, Ek_id)
    return out

# def n(a, D):
#     # f goes from [0,1] to [a,b]
#     # f_inv from [a,b] to [0,1]
#     f = lambda s: D*s + a
#     f_inv = lambda s: (s-a)/D
#     return f, f_inv
#
# def align_subparts_gam(a,b,c,d):
#     f = lambda s: ((c-d)/(a-b))*(s-a) + c
#     return f
#
# def align_subintervals(X1, X2, int1, int2, n):
#     '''
#     Function to align X2, restrict to the interval 'int2', to X1, restrict to the interval 'int1'
#     ...
#
#     Param:
#         X1: function object, 1d curve
#         X2: function object, 1d curve to be aligned to X1 on the respective intervals
#         int1: subinterval to which we restrict X1
#         int2: subinterval to which we restrict X2
#         n: number of discretization points to do the alignement
#
#     Return:
#         warp_func: warping function to align the subcurves of X2 to the one of X1
#         warp_func_inv: the inverse warping function
#         dist: distance between the align subcurves.
#
#     '''
#     f1, f1_inv = scale_gam(int1[0], int1[1])
#     f2, f2_inv = scale_gam(int2[0], int2[1])
#     t = np.linspace(0,1,n)
#     x1 = X1(f1(t))*(int1[1]-int1[0])
#     x2 = X2(f2(t))*(int2[1]-int2[0])
#     q1 = np.sqrt(x1)
#     q2 = np.sqrt(x2)
#     gam = fs.utility_functions.optimum_reparam(q1, t, q2)
#     gam_inv = fs.utility_functions.invertGamma(gam)
#     q2new = fs.utility_functions.warp_q_gamma(t, q2, gam)
#     dist = np.linalg.norm((x1-q2new**2),ord=1)
#     g = interp1d(t, gam)
#     g_inv = interp1d(t, gam_inv)
#     warp_func = lambda s: f2(g(f1_inv(s)))
#     warp_func_inv = lambda s: f1(g_inv(f2_inv(s)))
#     return warp_func, warp_func_inv, dist
#
#
# def extend_gam(g, g_inv, int_g, int_final):
#     '''
#     Function to extend the warping function found on the subintervals to the whole intervals.
#     ...
#
#     Param:
#         g: function object, warping function on the subinterval int_g
#         g_inv: function object, inverse wapring function
#         int_g: interval of definiton f the warping function g
#         int_final: final interval to which we extend g
#
#     Return:
#         gamma: extended warping function define on [A,B]
#         gamma_prime: first derivative of gamma
#         gamma_inv: inverse warping function define on int_final
#         [A,B]: interval of definiton of gamma
#
#     '''
#     a, b = int_g[0][0], int_g[0][1]
#     c, d = int_g[1][0], int_g[1][1]
#     g_interp = UnivariateSpline(np.linspace(a,b,200), g(np.linspace(a,b,200)), k=3)
#     gprime = g_interp.derivative()
#     gprime_a = gprime(a)
#     gprime_b = gprime(b)
#     A = (int_final[0] - g(a))/gprime_a + a
#     B = (int_final[1] - g(b))/gprime_b + b
#     def gamma(s):
#         if A<=s<a:
#             return gprime_a*(s-a) + g(a)
#         elif a<=s<=b:
#             return g(s)
#         elif b<s<=B:
#             return gprime_b*(s-b) + g(b)
#     def gamma_prime(s):
#         if A<=s<a:
#             return gprime_a
#         elif a<=s<=b:
#             return gprime(s)
#         elif b<s<=B:
#             return gprime_b
#     def gamma_inv(s):
#         if int_final[0]<=s<c:
#             return (s-c)/gprime_a + a
#         elif c<=s<=d:
#             return g_inv(s)
#         elif d<s<=int_final[1]:
#             return (s-d)/gprime_b + b
#     return gamma, gamma_prime, gamma_inv, [A,B]

# def partial_align_1d(X1, X2, init_ind1, init_ind2, sub_int1, sub_int2):
#     g, g_inv, dist = align_subintervals(X1, X2, sub_int1, sub_int2, 100)
#     ext_g, ext_g_prime, ext_g_inv, [A,B] = g2gamma(g, g_inv, sub_int1, init_ind2)
#     int_AB = np.linspace(A, B, 100)
#     def Xnew(s):
#         return X(ext_g(s))*ext_g_prime(s)
#     gam = np.array([np.round(ext_g(s), decimals=10) for s in int_AB])
#     gam_prime = np.array([ext_g_prime(s) for s in int_AB])
#     return X(gam)*gam_prime, Xnew

# def opt_grid(C1, C2, int1, int2, N):
    # grid1 = np.linspace(int1[0], int1[1], N)
    # grid2 = np.linspace(int2[0], int2[1], N)
    # xs, ys = np.meshgrid(grid1, grid1, sparse=False)
    # int_grid1 = np.unique(np.concatenate(np.stack((np.tril(xs),np.tril(ys)), axis=-1), axis=0), axis=0)
    # xs, ys = np.meshgrid(grid2, grid2, sparse=False)
    # int_grid2 = np.unique(np.concatenate(np.stack((np.tril(xs),np.tril(ys)), axis=-1), axis=0), axis=0)

# def opt_grid(C1, C2, int_grid1, int_grid2):
#     n1 = len(int_grid1)
#     n2 = len(int_grid2)
#     cost = np.zeros((n1,n2))
#     for i in range(n1):
#         for j in range(n2):
#             I1 = int_grid1[i]
#             I2 = int_grid2[j]
#             g, g_inv, dist = align_part(I1, I2, C1, C2, 100)
#             gamma, gamma_prime, gamma_inv, [A,B] = g2gamma(g, g_inv, I1, [0,1], True)
#             if A>B:
#                 cost[i,j] = None
#             else:
#                 c2new_grid, c2new = final_warp(C2, gamma, gamma_prime, A, B)
#                 # c2new = final_warp(C2, gamma, gamma_prime, A, B)
#                 cost[i,j] = cost_func(C1, c2new, [0,1], [A,B], 400)
#     return cost

# def opt_grid_bis(C1, C2, int_grid1, int_grid2, lbda):
#     n1 = len(int_grid1)
#     n2 = len(int_grid2)
#     cost_tot = np.zeros((n1,n2))
#     cost = np.zeros((n1,n2))
#     pen = np.zeros((n1,n2))
#     for i in range(n1):
#         for j in range(n2):
#             I1 = int_grid1[i]
#             I2 = int_grid2[j]
#             g, g_inv, dist = align_part(I1, I2, C1, C2, 100)
#             gamma, gamma_prime, gamma_inv, [A,B] = g2gamma(g, g_inv, I1, [0,1], True)
#             if A>B:
#                 cost_tot[i,j], cost[i,j], pen[i,j] = None, None, None
#             else:
#                 c2new_grid, c2new = final_warp(C2, gamma, gamma_prime, A, B)
#                 # c2new = final_warp(C2, gamma, gamma_prime, A, B)
#                 cost_tot[i,j], cost[i,j], pen[i,j] = cost_func_bis(C1, c2new, [0,1], [A,B], I1, 0.01, lbda)
#     return cost_tot, cost, pen
