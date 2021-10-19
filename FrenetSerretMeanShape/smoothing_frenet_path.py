import sys
from frenet_path import *
from trajectory import *
from model_curvatures import *
from maths_utils import *
from optimization_utils import *
from alignment_utils import *
from tracking_utils import *

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


def compute_A(theta):
    """
    Compute the matrix 3x3 A_theta
    ...
    """
    return np.array([[0, -theta[0], 0], [theta[0], 0, -theta[1]], [0, theta[1], 0]])


""" Smoothing with Tracking method """


def tracking_smoother_q(SingleFrenetPath, Model, lbda, p, q):
    h = SingleFrenetPath.grid_eval[q+1] - SingleFrenetPath.grid_eval[q]
    kappa_q = Model.curv.function(SingleFrenetPath.grid_eval[q])
    tau_q = Model.tors.function(SingleFrenetPath.grid_eval[q])
    A_q = compute_A([kappa_q, tau_q])
    # A_q_troch = torch.from_numpy(A_q)
    # expA_q = torch.matrix_exp(A_q_troch)
    # expA_q = expA_q.cpu().detach().numpy()
    expA_q = exp_matrix(h*A_q)
    extend_A = np.concatenate((np.concatenate((h*A_q, np.eye(3)), axis=1), np.zeros((3,6))), axis=0)
    phi_A = expm(extend_A)[:p,p:]
    # B = h * np.array([[0,-1,0],[1,0,-1],[0,1,0]]) @ phi_A
    B = h * phi_A
    M_tilde_q = np.concatenate((np.concatenate((expA_q,np.zeros((p,p))),axis=1), np.concatenate((np.zeros((p,p)), np.eye(p)),axis=1)),axis=0)
    B_tilde_q = np.concatenate((B, np.zeros((p,p))), axis=0)
    R_q = lbda*h*np.eye(p)

    return M_tilde_q, B_tilde_q, R_q


def tracking_smoother_i(SingleFrenetPath, Model, lbda, p):

    Q = np.zeros((2*p,2*p,SingleFrenetPath.nb_grid_eval))
    M_tilde = np.zeros((2*p,2*p,SingleFrenetPath.nb_grid_eval))
    B_tilde = np.zeros((2*p,p,SingleFrenetPath.nb_grid_eval))
    R = np.zeros((p,p,SingleFrenetPath.nb_grid_eval))

    for q in range(SingleFrenetPath.nb_grid_eval):
        Y_q = SingleFrenetPath.data[:,:,q]
        Q[:,:,q] = np.concatenate((np.concatenate((np.eye(p),-Y_q),axis=1), np.concatenate((-Y_q.T, Y_q.T @ Y_q),axis=1)),axis=0)

    for q in range(SingleFrenetPath.nb_grid_eval-1):
        M_tilde[:,:,q], B_tilde[:,:,q], R[:,:,q] = tracking_smoother_q(SingleFrenetPath, Model, lbda, p, q)

    Q0 = SingleFrenetPath.data[:,:,0]
    U, Z, K, P = tracking(Q0, Q, R, M_tilde, B_tilde, SingleFrenetPath.nb_grid_eval-1, p)
    Q_hat = np.moveaxis(Z[:,:p,:],0, -1)
    SmoothFrenetPath = FrenetPath(grid_obs=SingleFrenetPath.grid_eval, grid_eval=SingleFrenetPath.grid_eval, data=Q_hat)


def tracking_smoother(PopFrenetPath, Model, lbda):

    if isinstance(PopFrenetPath, FrenetPath):
        N_samples = 1
    else:
        N_samples = PopFrenetPath.nb_samples

    p = PopFrenetPath.dim

    if N_samples==1:
        return tracking_smoother_i(PopFrenetPath, Model, lbda, p)
    else:
        data_smoothfrenetpath = []
        for i in range(N_samples):
            data_smoothfrenetpath.append(tracking_smoother_i(PopFrenetPath.frenet_paths[i], Model, p))

        return PopulationFrenetPath(data_smoothfrenetpath)


""" Smoothing with Karcher Mean method """

def lie_smoother_q(q, SingleFrenetPath, Model, p):
    """
    Step of function Lie smoother.
    ...
    """

    SO3 = SpecialOrthogonal(3)
    Obs_q = SingleFrenetPath.data[:,:,SingleFrenetPath.neighbor_obs[q]]
    Obs_q = np.rollaxis(Obs_q, 2)

    kappa_q = np.apply_along_axis(Model.curv.function, 0, SingleFrenetPath.grid_double[q])
    tau_q = np.apply_along_axis(Model.tors.function, 0, SingleFrenetPath.grid_double[q])
    theta_q = np.stack((SingleFrenetPath.delta[q]*kappa_q, SingleFrenetPath.delta[q]*tau_q), axis=1)

    A_q = np.apply_along_axis(compute_A, 1, theta_q)
    A_q = torch.from_numpy(A_q)
    exp_A_q = torch.matrix_exp(A_q)
    Obs_q_torch = torch.from_numpy(Obs_q)
    mat_product = Obs_q_torch
    ind_norm = torch.where(torch.linalg.matrix_norm(A_q) > 0.)[0]
    mat_product[ind_norm,:,:] = torch.matmul(Obs_q_torch[ind_norm,:,:].double(), exp_A_q[ind_norm,:,:].double())

    mean = FrechetMean(metric=SO3.metric, max_iter=100, point_type='matrix', verbose=True)
    mean.fit(mat_product.cpu().detach().numpy(), SingleFrenetPath.weight[q])
    mu_q = mean.estimate_

    return mu_q
    # mu_q_proj = SO3.projection(mu_q)
    # return mu_q_proj

def lie_smoother_i(SingleFrenetPath, Model, p):
    """
    Step of function Lie smoother.
    ...
    """
    M = np.zeros((p,p,SingleFrenetPath.nb_grid_eval))

    for q in range(SingleFrenetPath.nb_grid_eval):
        M[:,:,q] = lie_smoother_q(q, SingleFrenetPath, Model, p)

    SmoothFrenetPath = FrenetPath(grid_obs=SingleFrenetPath.grid_eval, grid_eval=SingleFrenetPath.grid_eval, data=M)
    return SmoothFrenetPath


def lie_smoother(PopFrenetPath, Model):
    """
    LieSmoother
    Smoothing by local averaging based on the geodesic distance in SO(p),
    i.e. we compute local Karcher mean. It can use the information on the curvature and torsion defined in Model for a better estimation.
    ...
    """

    if isinstance(PopFrenetPath, FrenetPath):
        N_samples = 1
    else:
        N_samples = PopFrenetPath.nb_samples

    p = PopFrenetPath.dim

    if N_samples==1:
        return lie_smoother_i(PopFrenetPath, Model, p)
    else:
        data_smoothfrenetpath = []
        for i in range(N_samples):
            data_smoothfrenetpath.append(lie_smoother_i(PopFrenetPath.frenet_paths[i], Model, p))

        return PopulationFrenetPath(data_smoothfrenetpath)


def frenet_path_smoother(PopFrenetPath, Model, x, method):
    if method=='tracking':
        return tracking_smoother(PopFrenetPath, Model, x[3])
    elif method=='karcher_mean':
        return lie_smoother(PopFrenetPath, Model)
    else:
        raise Exception('Invalid Smoothing Method')
