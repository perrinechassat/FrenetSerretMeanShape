import sys
from frenet_path import *
from trajectory import *
from model_curvatures import *
from maths_utils import *
from optimization_utils import *
from alignment_utils import *
from tracking_utils import *
from smoothing_frenet_path import *
from visu_utils import *

import numpy as np
from scipy.linalg import expm, polar, logm
from scipy.integrate import cumtrapz
from scipy.interpolate import splrep, splder, sproot, splev, interp1d
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.matrices import Matrices
import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.riemannian_metric import RiemannianMetric

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.registration import ElasticRegistration, ShiftRegistration, landmark_registration_warping
from skfda.preprocessing.registration.elastic import elastic_mean
from skfda.misc import metrics
import fdasrsf as fs
from joblib import Parallel, delayed
from timeit import default_timer as timer
import torch

from numba.experimental import jitclass
from numba import int32, float64, cuda, float32, objmode, njit, prange
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


""" Computing the raw curvatures estimates """


@njit
def compute_sort_unique_val(S, Omega, Kappa, Tau):
    """
    Step of function Compute Raw Curvature, compute the re-ordering of the data.
    ...
    """
    uniqueS = np.unique(S)
    nb_unique_val = len(uniqueS)
    mOmega = np.zeros(nb_unique_val)
    mKappa = np.zeros(nb_unique_val)
    mTau   = np.zeros(nb_unique_val)
    for ijq in range(nb_unique_val):
        id_ijq      = np.where(S==uniqueS[ijq])[0]
        Omega_ijq   = Omega[id_ijq]
        Kappa_ijq   = Kappa[id_ijq]
        Tau_ijq     = Tau[id_ijq]
        mOmega[ijq] = np.sum(Omega_ijq)
        if mOmega[ijq]>0:
            mKappa[ijq] = (np.ascontiguousarray(Omega_ijq[np.where(Omega_ijq>0)]) @ np.ascontiguousarray(np.transpose(Kappa_ijq[np.where(Omega_ijq>0)])))/mOmega[ijq]
            mTau[ijq]   = (np.ascontiguousarray(Omega_ijq[np.where(Omega_ijq>0)]) @ np.ascontiguousarray(np.transpose(Tau_ijq[np.where(Omega_ijq>0)])))/mOmega[ijq]
        else:
            mKappa[ijq] = 0
            mTau[ijq]   = 0

    return uniqueS, mOmega, mKappa, mTau


@njit
def compute_Rq_boucle(dim, N_q, Obs_q, data, u_q, q, nb_grid):
    """
    Step of function Compute Raw Curvature
    ...
    """
    R_q = np.zeros((dim,dim,N_q))

    for j in range(N_q):
        if (q!=0 or j!=0) and (q!=nb_grid-1 or j!=N_q-1):
            R_q[:,:,j] = -my_log_M3(np.transpose(np.ascontiguousarray(data))@np.ascontiguousarray(Obs_q[:,:,j]))/u_q[j]

    return R_q


def compute_Rq(q, FrenetPath, SmoothFrenetPath):
    """
    Step of function Compute Raw Curvature
    ...
    """
    N_q = len(FrenetPath.neighbor_obs[q])
    Obs_q = FrenetPath.data[:,:,FrenetPath.neighbor_obs[q]]
    w_q = FrenetPath.weight[q]
    u_q = np.copy(FrenetPath.delta[q])
    omega_q = np.multiply(w_q,np.power(u_q,2))

    if q!=0 and q!=FrenetPath.nb_grid_eval-1:
        v_q = np.where(u_q==0)[0]
        u_q[u_q==0] = 1

    R_q = compute_Rq_boucle(FrenetPath.dim, N_q, Obs_q, SmoothFrenetPath.data[:,:,q], u_q, q, FrenetPath.nb_grid_eval)

    if q!=0 and q!=FrenetPath.nb_grid_eval-1:
        R_q[:,:,v_q] = np.abs(0*R_q[:,:,v_q])

    kappa = np.squeeze(R_q[1,0,:])
    tau = np.squeeze(R_q[2,1,:])

    return omega_q.tolist(), kappa.tolist(), tau.tolist()


def compute_raw_curvatures_without_alignement(PopulationFrenetPath, h, PopulationSmoothFrenetPath):
    """
    Compute the weighted instantaneous rate of change of the Frenet frames without alignment between samples.
    They are noisy and often needs to be smoothed by splines
    ...
    """

    N_samples = PopulationFrenetPath.nb_samples
    PopulationFrenetPath.compute_neighbors(h)
    if N_samples==1:
        Omega, S, Kappa, Tau = [], [], [], []
        for q in range(PopulationFrenetPath.nb_grid_eval):
            if q==0:
                # s = np.zeros(len(PopulationFrenetPath.neighbor_obs[q]))
                s = PopulationFrenetPath.grid_obs[0]*np.ones(len(PopulationFrenetPath.neighbor_obs[q]))
            elif q==PopulationFrenetPath.nb_grid_eval-1:
                # s = PopulationFrenetPath.length*np.ones(len(PopulationFrenetPath.neighbor_obs[q]))
                s = PopulationFrenetPath.grid_obs[-1]*np.ones(len(PopulationFrenetPath.neighbor_obs[q]))
            else:
                s = PopulationFrenetPath.grid_double[q]
            S += list(s)
            omega_q, kappa, tau = compute_Rq(q, PopulationFrenetPath, PopulationSmoothFrenetPath)
            Omega = np.append(Omega, omega_q)
            Kappa = np.append(Kappa, kappa)
            Tau = np.append(Tau, tau)
    else:
        Omega, S, Kappa, Tau = [], [], [], []
        for i in range(N_samples):
            for q in range(PopulationFrenetPath.frenet_paths[i].nb_grid_eval):
                if q==0:
                    s = np.zeros(len(PopulationFrenetPath.frenet_paths[i].neighbor_obs[q]))
                elif q==PopulationFrenetPath.frenet_paths[i].nb_grid_eval-1:
                    s = PopulationFrenetPath.frenet_paths[i].length*np.ones(len(PopulationFrenetPath.frenet_paths[i].neighbor_obs[q]))
                else:
                    s = PopulationFrenetPath.frenet_paths[i].grid_double[q]
                S += list(s)
                omega_q, kappa, tau = compute_Rq(q, PopulationFrenetPath.frenet_paths[i], PopulationSmoothFrenetPath.frenet_paths[i])
                Omega = np.append(Omega, omega_q)
                Kappa = np.append(Kappa, kappa)
                Tau = np.append(Tau, tau) # Kappa and Tau contains the empirical estimation of the curvature and torsion.

    Ms, Momega, Mkappa, Mtau = compute_sort_unique_val(np.around(S, 8), Omega, Kappa, Tau)

    # Test pour enlever les valeurs à zeros.
    Momega = np.asarray(Momega)
    ind_nozero = np.where(Momega!=0.)
    Momega = np.squeeze(Momega[ind_nozero])
    Mkappa = np.squeeze(np.asarray(Mkappa)[ind_nozero])
    Mtau = np.squeeze(np.asarray(Mtau)[ind_nozero])
    Ms = Ms[ind_nozero]

    return Mkappa, Mtau, Ms, Momega


def compute_raw_curvatures_i(SingleFrenetPath, SmoothFrenetPath):
    omega_i = []
    kappa_i = []
    s_i = []
    tau_i = []
    for q in range(SingleFrenetPath.nb_grid_eval):
        if q==0:
            s = np.zeros(len(SingleFrenetPath.neighbor_obs[q]))
        elif q==SingleFrenetPath.nb_grid_eval-1:
            s = SingleFrenetPath.length*np.ones(len(SingleFrenetPath.neighbor_obs[q]))
        else:
            s = SingleFrenetPath.grid_double[q]
        s_i = np.append(s_i, s.tolist())

        omega_q, kappa, tau = compute_Rq(q, SingleFrenetPath, SmoothFrenetPath)
        omega_i = np.append(omega_i, omega_q)
        kappa_i = np.append(kappa_i, kappa)
        tau_i = np.append(tau_i, tau)

    Ms_i, Momega_i, Mkappa_i, Mtau_i = compute_sort_unique_val(np.around(s_i, 5), omega_i, kappa_i, tau_i)
    return Ms_i, Momega_i, Mkappa_i, Mtau_i


def compute_raw_curvatures_alignement(PopulationFrenetPath, h, PopulationSmoothFrenetPath, lam=0.0,  gam={"flag" : False, "value" : None}):
    """
    Compute the weighted instantaneous rate of change of the Frenet frames with alignment between samples.
    They are noisy and often needs to be smoothed by splines.
    ...
    """
    N_samples = PopulationFrenetPath.nb_samples
    PopulationFrenetPath.compute_neighbors(h)
    gam_res = gam.copy()

    if N_samples==1:
        Ms_i, Momega_i, Mkappa_i, Mtau_i = compute_raw_curvatures_i(PopulationFrenetPath, PopulationSmoothFrenetPath)
        return Mkappa_i, Mtau_i, Ms_i, Momega_i

    else:
        Omega, S, Kappa, Tau = [], [], [], []
        """for alignement all FrenetPath needs to have the same grid"""

        for i in range(N_samples):
            Ms_i, Momega_i, Mkappa_i, Mtau_i = compute_raw_curvatures_i(PopulationFrenetPath.frenet_paths[i], PopulationSmoothFrenetPath.frenet_paths[i])
            S.append(Ms_i)
            Omega.append(Momega_i)
            Kappa.append(Mkappa_i)
            Tau.append(Mtau_i)

        # Alignment of curves
        Omega = np.asarray(Omega)
        sum_omega = np.sum(Omega, axis=0)
        ind_nozero = np.where(sum_omega!=0.)
        sum_omega = sum_omega[ind_nozero]
        Omega = np.squeeze(Omega[:,ind_nozero])
        Kappa = np.squeeze(np.asarray(Kappa)[:,ind_nozero])
        Tau = np.squeeze(np.asarray(Tau)[:,ind_nozero])
        S = S[0][ind_nozero]

        if gam["flag"]==False:
            theta = np.stack((np.transpose(Kappa), np.abs(np.transpose(Tau))))
            theta[np.isnan(theta)] = 0.0
            theta[np.isinf(theta)] = 0.0

            res = align_vect_curvatures_fPCA(theta, np.squeeze(S), np.transpose(Omega), num_comp=3, cores=-1, smoothdata=False, MaxItr=20, lam=lam)
            theta_align, gam_theta, weighted_mean_theta = res.fn, res.gamf, res.mfn
            ind_conv = res.convergence

            gam_fct = np.empty((gam_theta.shape[1]), dtype=object)
            for i in range(gam_theta.shape[1]):
                gam_fct[i] = interp1d(np.squeeze(S), gam_theta[:,i])
            gam_res["value"] = gam_fct
            gam_res["flag"] = True

            weighted_mean_kappa = np.transpose(weighted_mean_theta[0])
            tau_align, weighted_mean_tau = warp_curvatures(np.transpose(Tau), gam_fct, np.squeeze(S), np.transpose(Omega))

        else:
            # WARP KAPPA BY THE PREVIOUS GAM Functions
            kappa_align, weighted_mean_kappa = warp_curvatures(np.transpose(Kappa), gam["value"], np.squeeze(S), np.transpose(Omega))
            tau_align, weighted_mean_tau = warp_curvatures(np.transpose(Tau), gam["value"], np.squeeze(S), np.transpose(Omega))
            weighted_mean_kappa = np.transpose(weighted_mean_kappa)
            weighted_mean_tau = np.transpose(weighted_mean_tau)
            ind_conv = True

        return weighted_mean_kappa, weighted_mean_tau, S, sum_omega, gam_res, ind_conv



def compute_raw_curvatures(PopFrenetPath, h, SmoothPopFrenetPath, alignment=False, lam=0.0, gam={"flag" : False, "value" : None}):
    if alignment==True:
        weighted_mean_kappa, weighted_mean_tau, S, sum_omega, gam_res, ind_conv = compute_raw_curvatures_alignement(PopFrenetPath, h, SmoothPopFrenetPath, lam,  gam)
    else:
        weighted_mean_kappa, weighted_mean_tau, S, sum_omega = compute_raw_curvatures_without_alignement(PopFrenetPath, h, SmoothPopFrenetPath)

        # plt.figure()
        # plt.plot(S, weighted_mean_kappa)
        # plt.show()
        # plt.figure()
        # plt.plot(S, weighted_mean_tau)
        # plt.show()
        # plot_2D(S, weighted_mean_kappa)
        # plot_2D(S, weighted_mean_tau)

        gam_res = {"flag" : False, "value" : None}
        ind_conv = True
    return weighted_mean_kappa, weighted_mean_tau, S, sum_omega, gam_res, ind_conv




def estimation(PopFrenetPath, Model, x, smoothing={"flag":False, "method":"karcher_mean"}, alignment=False, lam=0.0, gam={"flag" : False, "value" : None}):

    N_samples = PopFrenetPath.nb_samples
    PopFrenetPath.compute_neighbors(x[0])

    if smoothing["flag"]==True:
        SmoothPopFrenetPath0 = frenet_path_smoother(PopFrenetPath, Model, x, smoothing["method"])
    else:
        SmoothPopFrenetPath0 = PopFrenetPath

    mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(PopFrenetPath, x[0], SmoothPopFrenetPath0, alignment, lam, gam)
    # mean_kappa = np.mean(mKappa)
    # print(mean_kappa)
    # Model_theta.curv.function = lambda s: s*0 + mean_kappa
    try:
        theta_curv = Model.curv.smoothing(mS, mKappa, mOmega, x[1])
        theta_torsion = Model.tors.smoothing(mS, mTau, mOmega, x[2])
    except:
        Model.curv.reinitialize()
        Model.tors.reinitialize()
        ind_conv = False

    # plt.figure()
    # plt.plot(mS, Model.curv.function(mS))
    # plt.show()
    # plt.figure()
    # plt.plot(mS, Model.tors.function(mS))
    # plt.show()
    # plot_2D(mS, Model.curv.function(mS))
    # plot_2D(mS, Model.tors.function(mS))

    if smoothing["flag"]==True:
        epsilon = 10e-3
        theta0 = np.concatenate((theta_curv, theta_torsion))
        theta  = theta0
        Dtheta = theta
        k=0
        # Resmooth data with model information
        while np.linalg.norm(Dtheta)>=epsilon*np.linalg.norm(theta) and k<31:

            SmoothPopFrenetPath = frenet_path_smoother(PopFrenetPath, Model, x, smoothing["method"])

            mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(PopFrenetPath, x[0], SmoothPopFrenetPath, alignment, lam, gam)

            try:
                theta_curv = Model.curv.smoothing(mS, mKappa, mOmega, x[1])
                theta_torsion = Model.tors.smoothing(mS, mTau, mOmega, x[2])
            except:
                Model.curv.reinitialize()
                Model.tors.reinitialize()
                ind_conv = False

            thetakm1 = theta
            theta = np.concatenate((theta_curv, theta_torsion))
            Dtheta = theta - thetakm1
            k += 1

        if k==31:
            ind_conv=False
    else:
        SmoothPopFrenetPath = SmoothPopFrenetPath0
        # ind_conv = True

    if N_samples!=1 and alignment==True:
        Model.set_gam_functions(gam["value"])

    return SmoothPopFrenetPath, Model, ind_conv


def global_estimation(PopFrenetPath, param_model, smoothing={"flag":False, "method":"karcher_mean"}, hyperparam=None, opt=False, param_bayopt=None, alignment=False, lam=0.0, parallel=False):

    N_samples = PopFrenetPath.nb_samples
    curv_smoother = BasisSmoother(domain_range=param_model["domain_range"], nb_basis=param_model["nb_basis"])
    tors_smoother = BasisSmoother(domain_range=param_model["domain_range"], nb_basis=param_model["nb_basis"])

    # curv_smoother = BasisSmoother_scipy()
    # tors_smoother = BasisSmoother_scipy()
    # curv_smoother = BasisSmoother(domain_range=param_model["domain_range"])
    # tors_smoother = BasisSmoother(domain_range=param_model["domain_range"])

    if opt==True:
        Opt_fun = lambda x: objective_function(param_bayopt["n_splits"], PopFrenetPath, curv_smoother, tors_smoother, x, smoothing, alignment, lam, parallel)
        if smoothing["flag"]==True and smoothing["method"]=="tracking":
            hyperparam_bounds = [param_bayopt["bounds_h"], param_bayopt["bounds_lcurv"], param_bayopt["bounds_ltors"], param_bayopt["bounds_ltrack"]]
        else:
            hyperparam_bounds = [param_bayopt["bounds_h"], param_bayopt["bounds_lcurv"], param_bayopt["bounds_ltors"]]
        x = bayesian_optimisation(Opt_fun, param_bayopt["n_calls"], hyperparam_bounds)
        curv_smoother.reinitialize()
        tors_smoother.reinitialize()
    else:
        x = hyperparam

    Model_theta = Model(curv_smoother, tors_smoother)
    SmoothPopFrenetPath_fin, Model_fin, ind_conv = estimation(PopFrenetPath, Model_theta, x, smoothing, alignment, lam)

    return SmoothPopFrenetPath_fin, Model_fin, [x, ind_conv]


# """ Estimate and compute dist """
#
# def eval_estimation(true_model, PopFrenetPath, Model, x, smoothing={"flag":False, "method":"karcher_mean"}, alignment=False, lam=0.0, gam={"flag" : False, "value" : None}):
#
#     s0 = true_model["s0"]
#     list_error_Q = [mean_geodesic_dist(true_model["PopFP"], PopFrenetPath)]
#     list_error_kappa = [(np.linalg.norm((Model.curv.function(s0) - true_model["curv"](s0)))**2)/len(s0)]
#     list_error_tau = [(np.linalg.norm((Model.curv.function(s0) - true_model["tors"](s0)))**2)/len(s0)]
#
#     N_samples = PopFrenetPath.nb_samples
#     PopFrenetPath.compute_neighbors(x[0])
#
#     if smoothing["flag"]==True:
#         SmoothPopFrenetPath0 = frenet_path_smoother(PopFrenetPath, Model, x, smoothing["method"])
#     else:
#         SmoothPopFrenetPath0 = PopFrenetPath
#
#     mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(PopFrenetPath, x[0], SmoothPopFrenetPath0, alignment, lam, gam)
#     # mean_kappa = np.mean(mKappa)
#     # print(mean_kappa)
#     # Model_theta.curv.function = lambda s: s*0 + mean_kappa
#     theta_curv = Model.curv.smoothing(mS, mKappa, mOmega, x[1])
#     theta_torsion = Model.tors.smoothing(mS, mTau, mOmega, x[2])
#
#     list_error_Q.append(mean_geodesic_dist(true_model["PopFP"], SmoothPopFrenetPath0))
#     list_error_kappa.append((np.linalg.norm((Model.curv.function(s0) - true_model["curv"](s0)))**2)/len(s0))
#     list_error_tau.append((np.linalg.norm((Model.curv.function(s0) - true_model["tors"](s0)))**2)/len(s0))
#
#     if smoothing["flag"]==True:
#         epsilon = 10e-3
#         theta0 = np.concatenate((theta_curv, theta_torsion))
#         theta  = theta0
#         Dtheta = theta
#         k=0
#         # Resmooth data with model information
#         while np.linalg.norm(Dtheta)>=epsilon*np.linalg.norm(theta) and k<31:
#
#             SmoothPopFrenetPath = frenet_path_smoother(PopFrenetPath, Model, x, smoothing["method"])
#
#             mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(PopFrenetPath, x[0], SmoothPopFrenetPath, alignment, lam, gam)
#             theta_curv = Model.curv.smoothing(mS, mKappa, mOmega, x[1])
#             theta_torsion = Model.tors.smoothing(mS, mTau, mOmega, x[2])
#
#             list_error_Q.append(mean_geodesic_dist(true_model["PopFP"], SmoothPopFrenetPath))
#             list_error_kappa.append((np.linalg.norm((Model.curv.function(s0) - true_model["curv"](s0)))**2)/len(s0))
#             list_error_tau.append((np.linalg.norm((Model.curv.function(s0) - true_model["tors"](s0)))**2)/len(s0))
#
#             thetakm1 = theta
#             theta = np.concatenate((theta_curv, theta_torsion))
#             Dtheta = theta - thetakm1
#             k += 1
#
#         if k==31:
#             ind_conv=False
#     else:
#         SmoothPopFrenetPath = SmoothPopFrenetPath0
#         ind_conv = True
#
#     SmoothPopFrenetPath.set_estimate_theta(Model.curv.function, Model.tors.function)
#     if N_samples!=1 and alignment==True:
#         SmoothPopFrenetPath.set_gam_functions(gam["value"])
#
#     return SmoothPopFrenetPath, ind_conv, list_error_Q, list_error_kappa, list_error_tau
#
#
# def eval_global_estimation(true_model, PopFrenetPath, param_model, smoothing={"flag":False, "method":"karcher_mean"}, hyperparam=None, opt=False, param_bayopt=None, alignment=False, lam=0.0):
#
#     N_samples = PopFrenetPath.nb_samples
#     curv_smoother = BasisSmoother(domain_range=param_model["domain_range"], nb_basis=param_model["nb_basis"])
#     tors_smoother = BasisSmoother(domain_range=param_model["domain_range"], nb_basis=param_model["nb_basis"])
#
#     if opt==True:
#         Opt_fun = lambda x: objective_function(param_bayopt["n_splits"], PopFrenetPath, curv_smoother, tors_smoother, x, smoothing, alignment, lam)
#         if smoothing["flag"]==True and smoothing["method"]=="tracking":
#             hyperparam_bounds = [param_bayopt["bounds_h"], param_bayopt["bounds_lcurv"], param_bayopt["bounds_ltors"], param_bayopt["bounds_ltrack"]]
#         else:
#             hyperparam_bounds = [param_bayopt["bounds_h"], param_bayopt["bounds_lcurv"], param_bayopt["bounds_ltors"]]
#         x = bayesian_optimisation(Opt_fun, param_bayopt["n_calls"], hyperparam_bounds)
#         curv_smoother.reinitialize()
#         tors_smoother.reinitialize()
#     else:
#         x = hyperparam
#
#     Model_theta = Model(curv_smoother, tors_smoother)
#     SmoothPopFrenetPath_fin, ind_conv, list_error_Q, list_error_kappa, list_error_tau = eval_estimation(true_model, PopFrenetPath, Model_theta, x, smoothing, alignment, lam)
#
#     return SmoothPopFrenetPath_fin, [x, ind_conv], list_error_Q, list_error_kappa, list_error_tau


""" Functions for the optimization of hyperparameters """


def step_cross_val(curv_smoother, tors_smoother, test_index, train_index, PopFrenetPath, hyperparam, smoothing={"flag":False, "method":"karcher_mean"}, alignment=False, lam=0.0, gam={"flag" : False, "value" : None}):
    """
    Step of cross validation. The error is computed on Q.
    ...
    """
    N_samples = PopFrenetPath.nb_samples
    if N_samples==1:
        t_train = PopFrenetPath.grid_obs[train_index]
        t_test = PopFrenetPath.grid_obs[test_index]
        data_train = PopFrenetPath.data[:,:,train_index]
        # if smoothing["flag"]==True and smoothing["method"]=='tracking':
        #     t_eval = t_train
        # else:
        # t_eval = PopFrenetPath.grid_obs
        train_PopFP = FrenetPath(t_train, t_train, data=data_train)
    else:
        train_PopFP_data = []
        for i in range(N_samples):
            # if smoothing["flag"]==True and smoothing["method"]=='tracking':
            #     t_eval = PopFrenetPath.grids_obs[i][train_index]
            # else:
            #     t_eval = PopFrenetPath.grids_obs[i]
            train_PopFP_data.append(FrenetPath(PopFrenetPath.grids_obs[i][train_index], PopFrenetPath.grids_obs[i][train_index], data=np.copy(PopFrenetPath.data[i][:,:,train_index])))
        train_PopFP = PopulationFrenetPath(train_PopFP_data)

    curv_smoother.reinitialize()
    tors_smoother.reinitialize()

    Model_test = Model(curv_smoother, tors_smoother)
    pred_PopFP, pred_Model, ind_conv = estimation(train_PopFP, Model_test, hyperparam, smoothing, alignment, lam, gam)


    if ind_conv==True:
        if N_samples==1:

            # temp_FrenetPath_Q0 = FrenetPath(PopFrenetPath.grid_obs, PopFrenetPath.grid_eval, init=pred_PopFP.data[:,:,0], curv=pred_Model.curv.function, tors=pred_Model.tors.function)
            temp_FrenetPath_Q0 = FrenetPath(t_test, t_test, init=PopFrenetPath.data[:,:,test_index[0]], curv=pred_Model.curv.function, tors=pred_Model.tors.function)
            try:
                temp_FrenetPath_Q0.frenet_serret_solve()
                # dist = geodesic_dist(np.rollaxis(PopFrenetPath.data[:,:,test_index], 2), np.rollaxis(temp_FrenetPath_Q0.data[:,:,test_index], 2))
                dist = geodesic_dist(np.rollaxis(PopFrenetPath.data[:,:,test_index], 2), np.rollaxis(temp_FrenetPath_Q0.data, 2))

            except:
                ind_conv = False
                dist = 100
            return dist

        else:
            Q0 = mean_Q0(pred_PopFP)
            temp_FrenetPath_Q0 = FrenetPath(PopFrenetPath.grids_obs[0], PopFrenetPath.grids_obs[0], init=Q0, curv=pred_Model.curv.function, tors=pred_Model.tors.function)
            temp_FrenetPath_Q0.frenet_serret_solve()
            dist = np.zeros(N_samples)
            for i in range(N_samples):
                dist[i] = geodesic_dist(np.rollaxis(PopFrenetPath.data[i][:,:,test_index], 2), np.rollaxis(temp_FrenetPath_Q0.data[:,:,test_index], 2))
            return dist.mean()

    else:
        return 100


def objective_function(n_splits, PopFrenetPath, curv_smoother, tors_smoother, hyperparam, smoothing, alignment, lam, parallel):
    """
    Objective function that do the cross validation.
    ...
    """
    print(hyperparam)
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    kf = KFold(n_splits=n_splits, shuffle=False)

    # k = 0
    curv_smoother.reinitialize()
    tors_smoother.reinitialize()

    N_samples = PopFrenetPath.nb_samples
    # print('begin parallel cross val')
    # start = timer()
    if alignment==True:
        Model_theta = Model(curv_smoother, tors_smoother)
        pred_PopFP, pred_Model, ind_conv = estimation(PopFrenetPath, Model_theta, hyperparam, {"flag":False}, alignment, lam)
        gam = {"flag" : True, "value" : pred_Model.gam}
        if ind_conv==False:
            return 100
    else:
        gam = {"flag" : False, "value" : None}
        Model_theta = Model(curv_smoother, tors_smoother)
        pred_PopFP, pred_Model, ind_conv = estimation(PopFrenetPath, Model_theta, hyperparam)

    if N_samples==1:
        grid_split = PopFrenetPath.grid_obs[1:-1]
        # grid_split = PopFrenetPath.grid_obs
    else:
        # grid_split = PopFrenetPath.frenet_paths[0].grid_obs[1:-1]
        grid_split = PopFrenetPath.frenet_paths[0].grid_obs

    if parallel==True:
        err = Parallel(n_jobs=10)(delayed(step_cross_val)(curv_smoother, tors_smoother, test_index, train_index, PopFrenetPath, hyperparam, smoothing, alignment, lam, gam)
                for train_index, test_index in kf.split(grid_split))
        for i in range(len(err)):
            if np.isnan(err[i]):
                print('Error NaN value in cross validation')
                err[i] = 100

    else:
        err = []
        # k=0
        for train_index, test_index in kf.split(grid_split):
            # print('------- step ', k, ' cross validation --------')
            train_index = train_index+1
            test_index = test_index+1
            train_index = np.concatenate((np.array([0]), train_index, np.array([len(grid_split)+1])))

            dist = step_cross_val(curv_smoother, tors_smoother, test_index, train_index, PopFrenetPath, hyperparam, smoothing, alignment, lam, gam)

            if np.isnan(dist):
                print('Error NaN value in cross validation')
                return 100
            else:
                err.append(dist)
            # k += 1

    # duration = timer() - start
    # print('cross val', duration)
    return np.mean(np.array(err))
