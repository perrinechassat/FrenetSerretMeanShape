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

# path_dir = r"/home/pchassat/Documents/data/LSFtraj/"
# files = os.listdir(path_dir)
# N = len(files)

# array_X = np.empty((N), dtype=object)
# array_Q_GS = np.empty((N), dtype=object)
#
# def preprocess(file):
#     hand_barycentre = barycenter_from_3ptsHand(path_dir+file, plot=False)
#     data_traj = take_numpy_subset(hand_barycentre, 0, len(hand_barycentre.index))
#     t = np.linspace(0,1,len(data_traj))
#
#     # Estimation des dérivées, de s(t) et de Q(t)
#     t_new = np.linspace(0,1,3000)
#     X = Trajectory(data_traj, t)
#     h_opt = opti_loc_poly_traj(X.data, X.t, 0.005, 0.02, 50)
#     X.loc_poly_estimation(X.t, 5, h_opt)
#     X.compute_S(scale=True)
#     s_lim = [0.02, 0.98]
#     ind_bornes = np.intersect1d(np.where(X.S(t_new)>s_lim[0]), np.where(X.S(t_new)<s_lim[1]))
#     Q_GS = X.TNB_GramSchmidt(t_new[ind_bornes])
#
#     return X, Q_GS
#
# out = Parallel(n_jobs=-1)(delayed(preprocess)(files[i]) for i in range(N))
#
# for i in range(N):
#     array_X[i] = out[i][0]
#     array_Q_GS[i] = out[i][1]
#
# filename = "LSFtraj_data_preprocess"
# dic = {"array_X" : array_X, "array_Q_GS" : array_Q_GS}
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()

filename = "LSFtraj_data_preprocess"
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_X = dic["array_X"]
array_Q_GS = dic["array_Q_GS"]
N = array_X.shape[0]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.001, 0.003), "bounds_lcurv" : (1e-13, 1e-8), "bounds_ltors" :  (1e-13, 1e-8)}
param_model = {"nb_basis" : 1000, "domain_range": (0.02, 0.98)}

print("Individual estimations...")

array_SmoothFP = np.empty((N), dtype=object)
array_resOpt = np.empty((N), dtype=object)
array_SmoothThetaFP = np.empty((N), dtype=object)

out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_Q_GS[i], param_model, opt=True, param_bayopt=param_bayopt) for i in range(N))

for i in range(N):
    array_SmoothFP[i] = out[i][0]
    array_resOpt[i] = out[i][1]
    if array_resOpt[i][1]==True:
        array_SmoothThetaFP[i] = FrenetPath(array_SmoothFP[i].grid_obs, array_SmoothFP[i].grid_obs, init=array_SmoothFP[i].data[:,:,0], curv=array_SmoothFP[i].curv, tors=array_SmoothFP[i].tors, dim=3)
        array_SmoothThetaFP[i].frenet_serret_solve()


filename = "curv_tors_estim_LSFtraj_data"
dic = {"array_X" : array_X, "array_Q_GS" : array_Q_GS, "array_SmoothFP" : array_SmoothFP, "array_resOpt" : array_resOpt, "array_SmoothThetaFP" : array_SmoothThetaFP, "param_bayopt" : param_bayopt, "param_model" : param_model}
if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()
