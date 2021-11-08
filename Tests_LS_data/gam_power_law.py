import sys
import os.path
sys.path.insert(1, '../FrenetSerretMeanShape')
from frenet_path import *
from trajectory import *
from model_curvatures import *
from estimation_algo_utils import *
from maths_utils import *
from simu_utils import *
from visu_utils import *
from pre_process_Mocaplab_data import take_numpy_subset, barycenter_from_3ptsHand
import numpy as np
from pickle import *
import dill as pickle
from scipy.interpolate import interp1d
from scipy.misc import derivative
from pygam import LinearGAM, s


def process_and_gam(path):
    hand_barycentre = barycenter_from_3ptsHand(path, plot=False)
    data_traj = take_numpy_subset(hand_barycentre, 0, len(hand_barycentre.index))
    t = np.linspace(0,1,len(data_traj))

    # Estimation des dérivées, de s(t) et de Q(t)
    t_new = np.linspace(0,1,3000)
    X = Trajectory(data_traj, t)
    X.loc_poly_estimation(X.t, 5, 0.01)
    X.compute_S(scale=True)
    Q_GS = X.TNB_GramSchmidt(t_new)

    # Estimate raw curvature and torsion
    h = 0.002
    Q_GS.compute_neighbors(h)
    mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(Q_GS, h, Q_GS, False)

    # Smooth them
    s_lim = [0.02, 0.97]
    ind_bornes = np.intersect1d(np.where(mS>s_lim[0]), np.where(mS<s_lim[1]))
    mS_cut = mS[ind_bornes]
    mKappa_cut = mKappa[ind_bornes]
    mTau_cut = mTau[ind_bornes]
    lbda1 = 1e-10
    lbda2 = 1e-10
    curv_smoother = BasisSmoother(domain_range=(s_lim[0],s_lim[1]), nb_basis=1000)
    tors_smoother = BasisSmoother(domain_range=(s_lim[0],s_lim[1]), nb_basis=1000)
    Model_theta = Model(curv_smoother, tors_smoother)
    theta_curv = Model_theta.curv.smoothing(mS_cut, mKappa_cut, mOmega[ind_bornes], lbda1)
    theta_torsion = Model_theta.tors.smoothing(mS_cut, mTau_cut, mOmega[ind_bornes], lbda2)
    mKappa_cut_bis = Model_theta.curv.function(mS_cut)
    mTau_cut_bis = Model_theta.tors.function(mS_cut)

    invS = interp1d(X.S(t_new), t_new)
    mt_cut = invS(mS_cut)
    y = np.log(X.Sdot(mt_cut))
    x = np.stack((mKappa_cut_bis, mTau_cut_bis), 1)
    gam = LinearGAM(s(0) + s(1))
    gam.gridsearch(x, y)

    return gam


def process_and_gam_raw(path):
    hand_barycentre = barycenter_from_3ptsHand(path, plot=False)
    data_traj = take_numpy_subset(hand_barycentre, 0, len(hand_barycentre.index))
    t = np.linspace(0,1,len(data_traj))

    # Estimation des dérivées, de s(t) et de Q(t)
    t_new = np.linspace(0,1,8000)
    X = Trajectory(data_traj, t)
    X.loc_poly_estimation(X.t, 5, 0.01)
    X.compute_S(scale=True)
    Q_GS = X.TNB_GramSchmidt(t_new)

    # Estimate raw curvature and torsion
    h = 0.002
    Q_GS.compute_neighbors(h)
    mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(Q_GS, h, Q_GS, False)

    # Smooth them
    s_lim = [0.02, 0.97]
    ind_bornes = np.intersect1d(np.where(mS>s_lim[0]), np.where(mS<s_lim[1]))
    mS_cut = mS[ind_bornes]
    mKappa_cut = mKappa[ind_bornes]
    mTau_cut = np.abs(mTau[ind_bornes])

    invS = interp1d(X.S(t_new), t_new)
    mt_cut = invS(mS_cut)
    y = np.log(X.Sdot(mt_cut))
    x = np.stack((mKappa_cut, mTau_cut), 1)
    gam = LinearGAM(s(0) + s(1))
    gam.gridsearch(x, y)

    return gam


def min_max(path):
    hand_barycentre = barycenter_from_3ptsHand(path, plot=False)
    data_traj = take_numpy_subset(hand_barycentre, 0, len(hand_barycentre.index))
    t = np.linspace(0,1,len(data_traj))

    # Estimation des dérivées, de s(t) et de Q(t)
    t_new = np.linspace(0,1,3000)
    X = Trajectory(data_traj, t)
    X.loc_poly_estimation(X.t, 5, 0.01)
    X.compute_S(scale=True)
    Q_GS = X.TNB_GramSchmidt(t_new)

    print('debut compute raw curv')
    # Estimate raw curvature and torsion
    h = 0.002
    Q_GS.compute_neighbors(h)
    mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(Q_GS, h, Q_GS, False)

    # Smooth them
    s_lim = [0.02, 0.97]
    ind_bornes = np.intersect1d(np.where(mS>s_lim[0]), np.where(mS<s_lim[1]))
    mS_cut = mS[ind_bornes]
    mKappa_cut = mKappa[ind_bornes]
    mTau_cut = np.abs(mTau[ind_bornes])
    return mKappa_cut, mTau_cut



path_dir = r"/home/pchassat/Documents/data/LSFtraj/"
files = os.listdir(path_dir)
N = len(files)

# for i in range(N):
#     print(i)
#     print(files[i])
#     hand_barycentre = barycenter_from_3ptsHand(path_dir+files[i], plot=False)
#
# out = Parallel(n_jobs=-1)(delayed(process_and_gam_raw)(path_dir+files[i]) for i in range(N))
#
# filename = "gam_LSFtraj_raw_abs"
# dic = {"gam_array" : out}
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()


out = Parallel(n_jobs=-1)(delayed(min_max)(path_dir+files[i]) for i in range(N))

filename = "gam_LSFtraj_raw_curv_tors"
dic = {"raw_array" : out}
if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()
