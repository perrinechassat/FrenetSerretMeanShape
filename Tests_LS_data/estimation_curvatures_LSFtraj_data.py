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
from signal_utils import *
from pre_process_Mocaplab_data import *
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

# filename = "LSFtraj_data_preprocess"
# fil = open(filename,"rb")
# dic = pickle.load(fil)
# fil.close()
#
# # array_X = dic["array_X"]
# array_Q_GS = dic["array_Q_GS"]
# # N = array_Q_GS.shape[0]
# N = 2
# print(N)

# param_bayopt = {"n_splits":  10, "n_calls" : 2, "bounds_h" : (0.001, 0.003), "bounds_lcurv" : (1e-13, 1e-8), "bounds_ltors" :  (1e-13, 1e-8)}
# param_model = {"nb_basis" : 1000, "domain_range": (0.02, 0.98)}
# hyperparam = [0.001, 1e-10, 1e-10]
#
# print("Individual estimations...")
# #
# # array_SmoothFP = np.empty((N), dtype=object)
# # array_resOpt = np.empty((N), dtype=object)
# # array_SmoothThetaFP = np.empty((N), dtype=object)
#
# out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_Q_GS[i], param_model, opt=True, param_bayopt=param_bayopt) for i in range(N))
# # out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_Q_GS[i], param_model, opt=False, hyperparam=hyperparam, param_bayopt=param_bayopt) for i in range(N))
#
# print("fin")
#
# for i in range(N):
#     array_SmoothFP = out[i][0]
#     array_resOpt = out[i][1]
#     # if array_resOpt[1]==True:
#     #     array_SmoothThetaFP = FrenetPath(array_SmoothFP.grid_obs, array_SmoothFP.grid_obs, init=array_SmoothFP.data[:,:,0], curv=array_SmoothFP.curv, tors=array_SmoothFP.tors, dim=3)
#         # array_SmoothThetaFP.frenet_serret_solve()
#     filename = "results/curv_tors_estim_LSFtraj_data_"+str(i)
#     # dic = {"array_resOpt" : array_resOpt, "array_SmoothThetaFP" : array_SmoothThetaFP}
#     dic = {"array_resOpt" : array_resOpt, "curv" : array_SmoothFP.curv, "tors" : array_SmoothFP.tors}
#     if os.path.isfile(filename):
#         print("Le fichier ", filename, " existe déjà.")
#     fil = open(filename,"xb")
#     pickle.dump(dic,fil)
#     fil.close()


# array_X = np.empty((N), dtype=object)
# array_Q_GS = np.empty((N), dtype=object)

# def preprocess_peaks_parts(file):
#     hand_barycentre = barycenter_from_3ptsHand(path_dir+file, plot=False)
#     data_traj = take_numpy_subset(hand_barycentre, 0, len(hand_barycentre.index))
#     t = np.linspace(0,1,len(data_traj))
#     ind_t = np.intersect1d(np.where(t>=0.05), np.where(t<=0.95))
#     data_traj = data_traj[ind_t,:]
#     t = np.linspace(0,1,len(data_traj))
#
#     # Estimation des dérivées, de s(t) et de Q(t)
#     X = Trajectory(data_traj, t)
#     h_opt = opti_loc_poly_traj(X.data, X.t, 5*(1/len(data_traj)), 10*(1/len(data_traj)), 50)
#     X.loc_poly_estimation(X.t, 5, h_opt)
#     X.compute_S(scale=True)
#     Q_GS = X.TNB_GramSchmidt(t)
#
#     h = 0.002
#     Q_GS.compute_neighbors(h)
#     mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(Q_GS, h, Q_GS, False)
#
#     bornes_s = bornes_peaks(mS, mKappa, 6)
#     n = len(bornes_s)
#     s = X.S(t)
#     list_ind = []
#     if len(np.where(s<=bornes_s[0])[0])!=0:
#         list_ind.append(np.where(s<=bornes_s[0])[0])
#     for i in range(n-1):
#         ind = np.intersect1d(np.where(s>=bornes_s[i]), np.where(s<=bornes_s[i+1]))
#         if len(ind)!=0:
#             list_ind.append(ind)
#     if len(np.where(s>=bornes_s[-1])[0])!=0:
#         list_ind.append(np.where(s>=bornes_s[-1])[0])
#
#     nb_parts = len(list_ind)
#     list_t = []
#     parts = []
#     list_Q_GS = []
#     for i in range(nb_parts):
#         parts.append(data_traj[list_ind[i],:])
#         list_t.append(t[list_ind[i]])
#         n_i = len(list_t[i])
#         list_Q_GS.append(X.TNB_GramSchmidt(np.linspace(list_t[i][0], list_t[i][-1], 3*n_i)))
#
#     return X, parts, list_Q_GS
#
#
# def preprocess_regular_parts(file):
#     hand_barycentre = barycenter_from_3ptsHand(path_dir+file, plot=False)
#     data_traj = take_numpy_subset(hand_barycentre, 0, len(hand_barycentre.index))
#     t = np.linspace(0,1,len(data_traj))
#     ind_t = np.intersect1d(np.where(t>=0.05), np.where(t<=0.95))
#     data_traj = data_traj[ind_t,:]
#     t = np.linspace(0,1,len(data_traj))
#
#     # Estimation des dérivées, de s(t) et de Q(t)
#     X = Trajectory(data_traj, t)
#     h_opt = opti_loc_poly_traj(X.data, X.t, 5*(1/len(data_traj)), 10*(1/len(data_traj)), 50)
#     X.loc_poly_estimation(X.t, 5, h_opt)
#     X.compute_S(scale=True)
#
#     # parts_t = cut_regular_parts(t, 200)
#     parts = cut_regular_parts(data_traj, 100)
#
#     Q_GS = X.TNB_GramSchmidt(np.linspace(0, 1, 3*len(data_traj)))
#     data = Q_GS.data.transpose()
#     parts_Q = cut_regular_parts(data, 300)
#     parts_grid = cut_regular_parts(Q_GS.grid_obs, 300)
#     nb_parts = len(parts_grid)
#     list_Q_GS = []
#     for i in range(nb_parts):
#         list_Q_GS.append(FrenetPath(grid_obs=parts_grid[i], grid_eval=parts_grid[i], data=parts_Q[i].transpose()))
#     return X, parts, list_Q_GS
#
#
# array_X = np.empty((N), dtype=object)
# array_parts = np.empty((N), dtype=object)
# array_Q_GS = np.empty((N), dtype=object)
# #
# out = Parallel(n_jobs=-1)(delayed(preprocess_regular_parts)(files[i]) for i in range(N))
# #
# for i in range(N):
#     array_X[i] = out[i][0]
#     array_parts[i] = out[i][1]
#     array_Q_GS[i] = out[i][2]


filename = "LSFtraj_data_preprocess_parts_regular_100"
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_Q_GS = dic["array_Q_GS"]
N = array_Q_GS.shape[0]

def estim_file(list_Q_GS, param_bayopt, hyperparam):
    n = len(list_Q_GS)

    array_resOpt = np.empty((n), dtype=object)
    array_SmoothFP = np.empty((n), dtype=object)

    for j in range(n):
        array_SmoothFP[j], array_resOpt[j] = global_estimation(list_Q_GS[j], param_model={"nb_basis" : int(len(list_Q_GS[j].grid_obs)/2), "domain_range": (list_Q_GS[j].grid_obs[0], list_Q_GS[j].grid_obs[-1])},
                            opt=True, param_bayopt=param_bayopt)

    return array_SmoothFP, array_resOpt


array_SmoothFPIndiv = np.empty((N), dtype=object)
array_resOptIndiv = np.empty((N), dtype=object)


param_bayopt={"n_splits":  10, "n_calls" : 40, "bounds_h" : (int(3), int(9)), "bounds_lcurv" : (0.00001,0.1), "bounds_ltors" :  (0.00001,0.1)}
hyperparam = [5, 0.001, 0.001]

out = Parallel(n_jobs=-1)(delayed(estim_file)(array_Q_GS[i], param_bayopt, hyperparam) for i in range(N))


for i in range(N):
    array_SmoothFPIndiv[i] = out[i][0]
    array_resOptIndiv[i] = out[i][1]

print('Save file')
filename = "results/curv_tors_estim_LSFtraj_data_cut_regular_100_n_calls_"+str(param_bayopt["n_calls"])
dic = {"array_resOpt" : array_resOptIndiv, "array_SmoothFP" : array_SmoothFPIndiv}
# dic = {"array_resOpt" : out[i][0], "array_curv" : out[i][1], "array_tors" : out[i][2]}
if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

#
# for i in range(N):
#     print('---------------------------------------------------- estimation file',i,'----------------------------------------------------------------')
#     # array_X[i], array_parts[i], array_Q_GS[i] = preprocess_regular_parts(files[i])
#     n = len(array_Q_GS[i])
#     print('--------', n, 'parts --------')
#     # for j in range(n):
#     #     print(parts[j].shape[0])
#     array_resOpt = np.empty((n), dtype=object)
#     array_curv = np.empty((n), dtype=object)
#     array_tors = np.empty((n), dtype=object)
#
#     hyperparam = [0.002, 1e-40, 1e-40]
#
#     for j in range(n):
#         out, array_resOpt[j] = global_estimation(array_Q_GS[i][j], param_model={"nb_basis" : int(len(array_Q_GS[i][j].grid_obs)/2), "domain_range": (array_Q_GS[i][j].grid_obs[0], array_Q_GS[i][j].grid_obs[-1])},
#                             opt=False, hyperparam=hyperparam)
#
#         array_curv[j] = out.curv
#         array_tors[j] = out.tors
#         print('ok for',j)
#
#     # out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_Q_GS[i][j], param_model={"nb_basis" : int(len(array_Q_GS[i][j].grid_obs)/2), "domain_range": (array_Q_GS[i][j].grid_obs[0], array_Q_GS[i][j].grid_obs[-1])},
#     #                     opt=True, param_bayopt={"n_splits":  10, "n_calls" : 20, "bounds_h" : (0.0015, 0.0025), "bounds_lcurv" : (1e-40, 1e-30), "bounds_ltors" :  (1e-40, 1e-30)}) for j in range(n))
#
#     # array_resOpt = np.empty((n), dtype=object)
#     # array_curv = np.empty((n), dtype=object)
#     # array_tors = np.empty((n), dtype=object)
#
#     # for j in range(n):
#     #     array_resOpt[j] = out[j][1]
#     #     array_curv[j] = out[j][0].curv
#     #     array_tors[j] = out[j][0].tors
#     #     print('ok for',j)
#
#     print('before save')
#     filename = "results/curv_tors_estim_LSFtraj_data_cut_without_opti"+str(i)
#     # dic = {"array_resOpt" : array_resOpt, "array_SmoothThetaFP" : array_SmoothThetaFP, "list_Q_GS" : list_Q_GS, "X" : X, "parts" : parts}
#     dic = {"array_resOpt" : array_resOpt, "array_curv" : array_curv, "array_tors" : array_tors}
#     if os.path.isfile(filename):
#         print("Le fichier ", filename, " existe déjà.")
#     fil = open(filename,"xb")
#     pickle.dump(dic,fil)
#     fil.close()
#
#     del array_resOpt
#     del array_curv
#     del array_tors

    # array_SmoothFP = np.empty((n), dtype=object)
    # array_resOpt = np.empty((n), dtype=object)
    # array_SmoothThetaFP = np.empty((n), dtype=object)

    # for j in range(n):
    #     array_SmoothFP[j], array_resOpt[j] = global_estimation(list_Q_GS[j],
    #                     param_model={"nb_basis" : int(len(list_Q_GS[j].grid_obs)/2), "domain_range": (list_Q_GS[j].grid_obs[0], list_Q_GS[j].grid_obs[-1])},
    #                     opt=True,
    #                     param_bayopt={"n_splits":  10, "n_calls" : 2, "bounds_h" : (0.001, 0.002), "bounds_lcurv" : (1e-45, 1e-30), "bounds_ltors" :  (1e-45, 1e-30)},
    #                     parallel=True)
    #     if array_resOpt[j][1]==True:
    #         array_SmoothThetaFP[j] = FrenetPath(array_SmoothFP[j].grid_obs, array_SmoothFP[j].grid_obs, init=array_SmoothFP[j].data[:,:,0], curv=array_SmoothFP[j].curv, tors=array_SmoothFP[j].tors, dim=3)
    #         array_SmoothThetaFP[j].frenet_serret_solve()
    #     print('ok for',j)
    # #
    # for j in range(n):
    #     array_SmoothFP[j] = out[j][0]
    #     array_resOpt[j] = out[j][1]
    #     if array_resOpt[j][1]==True:
    #         array_SmoothThetaFP[j] = FrenetPath(out[j][0].grid_obs, out[j][0].grid_obs, init=out[j][0].data[:,:,0], curv=out[j][0].curv, tors=out[j][0].tors, dim=3)
    #         array_SmoothThetaFP[j].frenet_serret_solve()
    #     print('ok for',j)


# filename = "LSFtraj_data_preprocess_parts_regular_100"
# dic = {"array_X" : array_X, "array_Q_GS" : array_Q_GS, "array_parts" : array_parts}
# # dic = {"array_resOpt" : array_resOpt, "curv" : array_SmoothFP.curv, "tors" : array_SmoothFP.tors}
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
