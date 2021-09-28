import sys
import os.path
sys.path.insert(1, '../FrenetSerretMeanShape')
from frenet_path import *
from trajectory import *
from model_curvatures import *
from estimation_algo_utils import *
from maths_utils import *
from simu_utils import *
from optimization_utils import opti_loc_poly_traj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import cumtrapz
from skopt import gp_minimize
from skopt.plots import plot_convergence
from pickle import *
import dill as pickle
from timeit import default_timer as timer
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


""" Without smoothing of Q for conditions 3,7,11,15. """

### Construction des données et de la liste des trajectoires.

pathFile = r"armDat.csv"
df = pd.read_csv(pathFile,sep=',')
#
n_cond = df.cond.nunique()
n_subj = df.subj.nunique()
n_rept = df.rept.nunique()
# n_cond = 16
# n_subj = 10
# n_rept = 10
print('Nombre de conditions: ', n_cond, ' , de sujets: ', n_subj, ' , de repetitions: ', n_rept)

# list_cond = [3,7,11,15]
# n_cond = 4
array_traj = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)

k = 0
for cond in range(n_cond):
    data_cond = df[df.cond.eq(cond+1)]
    for subj in range(n_subj):
        data_subj = data_cond[data_cond.subj.eq(subj+1)]
        for rept in range(n_rept):
            data_rept = data_subj[data_subj.rept.eq(rept+1)]
            traj = data_rept[['y1', 'y2', 'y3']]
            # time = np.linspace(0, 100, len(data_rept['time'].to_numpy()[10:-15]))
            # time = time/np.max(time)
            time = np.linspace(0, 1, len(data_rept['time'].to_numpy()))
            ind = np.intersect1d(np.where(0.1 < time), np.where(0.8 > time))
            time = time[ind]
            time = time - time[0]
            time = time/np.max(time)
            traj_np = traj.to_numpy()[ind]
            array_traj[k,subj,rept] = Trajectory(data=traj_np,t=time)
    k+=1

"""Pre-processing"""

n_resamples = 200
param_loc_poly_deriv = { "h_min" : 0.1, "h_max" : 0.2, "nb_h" : 20}
param_loc_poly_TNB = {"h" : 20, "p" : 3, "iflag": [1,1], "ibound" : 0}

array_X = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
array_Xnew = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
array_Xnew_scale = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
array_Q_LP = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
array_Q_GS = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
array_ThetaExtrins = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
array_echec_flag = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)

for i in range(n_cond):
    for j in range(n_subj):
        res = Parallel(n_jobs=-1)(delayed(preprocess_raket)(array_traj[i,j,k].data, array_traj[i,j,k].t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True) for k in range(n_rept))
        for k in range(n_rept):
            array_Xnew[i,j,k], array_X[i,j,k], array_Q_LP[i,j,k], array_echec_flag[i,j,k] = res[k][0], res[k][1], res[k][2], res[k][3]
            if array_echec_flag[i,j,k]==True:
                print('echec :', i, j, k)
            array_Xnew_scale[i,j,k] = array_Xnew[i,j,k]/array_X[i,j,k].L


filename = "Raket_data_preprocessed_SingleEstim_4"
dic = {"array_X" : array_X, "array_Xnew" : array_Xnew, "array_Xnew_scale" : array_Xnew_scale, "ThetaExtrins" : array_ThetaExtrins, "array_Q_LP" : array_Q_LP, "array_Q_GS" : array_Q_GS,
"param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



# """Load data"""
#
# filename = "Raket_data_preprocessed_SingleEstim_2"
# fil = open(filename,"rb")
# dic = pickle.load(fil)
# fil.close()
#
# array_X, array_Xnew, array_Xnew_scale, array_Q_LP = dic["array_X"], dic["array_Xnew"], dic["array_Xnew_scale"], dic["array_Q_LP"]
#
#
# print('End of preprocessing.')
#
# ### Individual estimation of kappa and tau
#
# """Estimation of mean at subject and condition fixed"""
#
# print('Estimation of means at subject and condition fixed...')
#
# #
# # SmoothCondSubjMeanFP = np.empty((n_cond,n_subj), dtype=object)
# array_SmoothCondSubjPopFP = np.empty((n_cond, n_subj), dtype=object)
# array_SmoothCondSubjThetaFP = np.empty((n_cond, n_subj), dtype=object)
# array_CondSubjresOpt = np.empty((n_cond, n_subj), dtype=object)
#
# hyperparam = [0.011, 0.000000001, 0.0000001]
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.015, 0.1), "bounds_lcurv" : (1e-12, 1e-8), "bounds_ltors" : (1e-10, 1e-6)}  # Pas utilisé
# nb_knots = 100
# new_s = np.linspace(0,1,n_resamples)
# domain_range = (0.0,1.0)
#
# for i in range(n_cond):
#     TruePopCondSubjFP = np.empty((n_subj), dtype=object)
#     for j in range(n_subj):
#         TruePopCondSubjFP[j] = PopulationFrenetPath(list(array_Q_LP[i,j]))
#         list_Q0 = []
#         list_X0 = []
#         for k in range(TruePopCondSubjFP[j].nb_samples):
#             list_X0.append(array_Q_LP[i,j,k].data_trajectory[0,:])
#
#     # Estimate the mean curvatures from the population of true frenet path
#     print('estimation mean condition: ', i, 'all subject.')
#
#     out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(TruePopCondSubjFP[j], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                                 for j in range(n_subj))
#
#     for j in range(n_subj):
#         array_SmoothCondSubjPopFP[i,j] = out[j][0]
#         array_CondSubjresOpt[i,j] = out[j][1]
#         if array_CondSubjresOpt[i,j][1]==True:
#             array_SmoothCondSubjThetaFP[i,j] = FrenetPath(array_SmoothCondSubjPopFP[i,j].grids_obs[0], array_SmoothCondSubjPopFP[i,j].grids_obs[0], init=mean_Q0(array_SmoothCondSubjPopFP[i,j]), curv=array_SmoothCondSubjPopFP[i,j].mean_curv, tors=array_SmoothCondSubjPopFP[i,j].mean_tors, dim=3)
#             array_SmoothCondSubjThetaFP[i,j].frenet_serret_solve()
#             array_SmoothCondSubjThetaFP[i,j].data_trajectory = array_SmoothCondSubjThetaFP[i,j].data_trajectory + np.mean(list_X0, 0)
#
# filename = "Raket_data_means_SubjCondFixed_SingleEstim"
# dic = {"SmoothCondSubjPopFP" : array_SmoothCondSubjPopFP, "SmoothCondSubjThetaFP" : array_SmoothCondSubjThetaFP, "CondSubjresOpt" : array_CondSubjresOpt}
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()



#
# """Estimation of mean at condition fixed"""
#
# print('Estimation of means at condition fixed...')
#
# hyperparam = [0.011, 0.000000001, 0.0000001]
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.015, 0.1), "bounds_lcurv" : (1e-12, 1e-8), "bounds_ltors" : (1e-10, 1e-6)}  # Pas utilisé
# nb_knots = 100
# domain_range = (0.0,1.0)
#
#
# print("Mean estimations Frenet Serret...")
#
# array_TruePopFP = np.empty((n_cond), dtype=object)
# array_X_srvf = np.empty((n_cond), dtype=object)
# array_X_Arithm = np.empty((n_cond), dtype=object)
#
# for i in range(n_cond):
#     list_Q_LP = list(np.concatenate(array_Q_LP[i]))
#     list_X_srvf = list(np.concatenate(array_Xnew[i]))
#     list_X_arithm = list(np.concatenate(array_Xnew_scale[i]))
#     array_TruePopFP[i] = PopulationFrenetPath(list_Q_LP)
#     array_X_srvf[i] = list_X_srvf
#     array_X_Arithm[i] = list_X_arithm
#
# out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_TruePopFP[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                             for i in range(n_cond))
#
# array_SmoothPopFP = np.empty((n_cond), dtype=object)
# array_SmoothThetaFP = np.empty((n_cond), dtype=object)
# array_resOpt = np.empty((n_cond), dtype=object)
#
# for k in range(n_cond):
#     array_SmoothPopFP[k] = out[k][0]
#     array_resOpt[k] = out[k][1]
#     if array_resOpt[k][1]==True:
#         array_SmoothThetaFP[k] = FrenetPath(array_SmoothPopFP[k].grids_obs[0], array_SmoothPopFP[k].grids_obs[0], init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
#         array_SmoothThetaFP[k].frenet_serret_solve()
#
# list_X_LP = np.empty((n_cond), dtype=object)
# list_Xnew = np.empty((n_cond), dtype=object)
# list_Xnew_scale = np.empty((n_cond), dtype=object)
# for i in range(n_cond):
#     X_LP = []
#     Xnew = []
#     Xnew_scale = []
#     for j in range(n_subj):
#         for k in range(n_rept):
#             X_LP.append(array_Q_LP[i,j,k].data_trajectory)
#             Xnew.append(array_Xnew[i,j,k])
#             Xnew_scale.append(array_Xnew_scale[i,j,k])
#     list_X_LP[i], list_Xnew[i], list_Xnew_scale[i] = X_LP, Xnew, Xnew_scale
#
# print("Mean estimations SRVF LP...")
# t = np.linspace(0,1,n_resamples)
# array_SRVF_mean_LP = np.empty((n_cond), dtype=object)
# array_SRVF_gam_LP = np.empty((n_cond), dtype=object)
# for i in range(n_cond):
#     array_SRVF_mean_LP[i], array_SRVF_gam_LP[i] = compute_mean_SRVF(list_X_LP[i], t)
#
# print("Mean estimations SRVF...")
# array_SRVF_mean = np.empty((n_cond), dtype=object)
# array_SRVF_gam = np.empty((n_cond), dtype=object)
# for i in range(n_cond):
#     array_SRVF_mean[i], array_SRVF_gam[i] = compute_mean_SRVF(list_Xnew[i], t)
#
# print("Mean estimations Arithmetic...")
# array_Arithmetic_mean = Parallel(n_jobs=-1)(delayed(compute_mean_Arithmetic)(list_Xnew_scale[i]) for i in range(n_cond))
#
# print("Mean estimations Arithmetic...")
# array_Arithmetic_mean_LP = Parallel(n_jobs=-1)(delayed(compute_mean_Arithmetic)(list_X_LP[i]) for i in range(n_cond))
#
#
# ### Sauvegarde des données
#
# print('Saving the data...')
#
# filename = "Raket_data_means_per_cond_only4cond"
# dic = {"TruePopFP":array_TruePopFP, "SmoothPopFP":array_SmoothPopFP, "SmoothThetaFP":array_SmoothThetaFP, "resOpt":array_resOpt, "SRVF_mean" :array_SRVF_mean, "SRVF_gam" : array_SRVF_gam, "SRVF_mean_LP" :array_SRVF_mean_LP, "SRVF_gam_LP" : array_SRVF_gam_LP, "Arithmetic_mean" : array_Arithmetic_mean,
# "Arithmetic_mean_LP" : array_Arithmetic_mean_LP}
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
#
# print('END !')


""" Complete estimation with smoothing of Q """
#
# ### Construction des données et de la liste des trajectoires.
#
# pathFile = r"/home/pchassat/Documents/frenet-serret-smoothing/data/Racket_ArmData/armDat.csv"
# df = pd.read_csv(pathFile,sep=',')
#
# n_cond = df.cond.nunique()
# n_subj = df.subj.nunique()
# n_rept = df.rept.nunique()
# print('Nombre de conditions: ', n_cond, ' , de sujets: ', n_subj, ' , de repetitions: ', n_rept)
# #
# # array_traj = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
# #
# # for cond in range(n_cond):
# #     data_cond = df[df.cond.eq(cond+1)]
# #     for subj in range(n_subj):
# #         data_subj = data_cond[data_cond.subj.eq(subj+1)]
# #         for rept in range(n_rept):
# #             data_rept = data_subj[data_subj.rept.eq(rept+1)]
# #             traj = data_rept[['y1', 'y2', 'y3']]
# #             # time = np.linspace(0, 100, len(data_rept['time'].to_numpy()[10:-15]))
# #             # time = time/np.max(time)
# #             time = np.linspace(0, 1, len(data_rept['time'].to_numpy()))
# #             ind = np.intersect1d(np.where(0.1 < time), np.where(0.8 > time))
# #             time = time[ind]
# #             time = time - time[0]
# #             time = time/np.max(time)
# #             traj_np = traj.to_numpy()[ind]
# #             array_traj[cond,subj,rept] = Trajectory(data=traj_np,t=time)
# #
# #
# #
# # """Pre-processing"""
# #
# n_resamples = 200
# # param_loc_poly_deriv = { "h_min" : 0.1, "h_max" : 0.2, "nb_h" : 20}
# # param_loc_poly_TNB = {"h" : 20, "p" : 3, "iflag": [1,1], "ibound" : 0}
# #
# # array_X = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
# # array_Xnew = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
# # array_Xnew_scale = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
# # array_Q_LP = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
# # array_Q_GS = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
# # array_ThetaExtrins = np.ndarray((n_cond,n_subj,n_rept,),dtype=np.object_)
# #
# # for i in range(n_cond):
# #     for j in range(n_subj):
# #         res = Parallel(n_jobs=-1)(delayed(pre_process_data)(array_traj[i,j,k].data, array_traj[i,j,k].t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True) for k in range(n_rept))
# #         for k in range(n_rept):
# #             # array_Xnew[i,j,k], array_X[i,j,k], array_Q_LP[i,j,k], array_Q_GS[i,j,k], array_ThetaExtrins[i,j,k], successLocPoly = pre_process_data(array_traj[i,j,k].data, array_traj[i,j,k].t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True)
# #             array_Xnew[i,j,k], array_X[i,j,k], array_Q_LP[i,j,k], array_Q_GS[i,j,k], array_ThetaExtrins[i,j,k], successLocPoly = res[k][0], res[k][1], res[k][2], res[k][3], res[k][4], res[k][5]
# #             print(successLocPoly)
# #             array_Xnew_scale[i,j,k] = array_Xnew[i,j,k]/array_X[i,j,k].L
# #
# #
# # filename = "Raket_data_preprocessed"
# # dic = {"array_X" : array_X, "array_Xnew" : array_Xnew, "array_Xnew_scale" : array_Xnew_scale, "ThetaExtrins" : array_ThetaExtrins, "array_Q_LP" : array_Q_LP, "array_Q_GS" : array_Q_GS,
# # "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB}
# #
# # if os.path.isfile(filename):
# #     print("Le fichier ", filename, " existe déjà.")
# # fil = open(filename,"xb")
# # pickle.dump(dic,fil)
# # fil.close()
# #
# # print('End of preprocessing.')
#
# ### Individual estimation of kappa and tau
#
# ### Estimation of mean at subject and condition fixed
#
# ### Estimation of mean at condition fixed


# filename = "Raket_data_preprocessed"
# fil = open(filename,"rb")
# dic = pickle.load(fil)
# fil.close()
#
# array_X, array_Xnew, array_Xnew_scale = dic["array_X"], dic["array_Xnew"], dic["array_Xnew_scale"]
# array_Q_LP, array_Q_GS = dic["array_Q_LP"], dic["array_Q_GS"]
s0 = array_Q_LP[0,0,0].grid_obs
# ThetaExtrins = dic["ThetaExtrins"]

for i in range(n_cond):
    for j in range(n_subj):
        for k in range(n_rept):
            if array_echec_flag[i,j,k]==True:
                del array_Q_LP[i,j,k]

print('Estimation of means at condition fixed...')


hyperparam = [0.011, 0.000000001, 0.0000001]
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.015, 0.1), "bounds_lcurv" : (1e-12, 1e-8), "bounds_ltors" : (1e-10, 1e-6)}  # Pas utilisé
nb_knots = 100
domain_range = (0.0,1.0)


print("Mean estimations Frenet Serret...")

array_TruePopFP = np.empty((n_cond), dtype=object)
array_X_srvf = np.empty((n_cond), dtype=object)
array_X_Arithm = np.empty((n_cond), dtype=object)

for i in range(n_cond):
    # if i==4:
    #     list_Q_LP = list(np.concatenate(array_Q_LP[i]))
    #     list_X_srvf = list(np.concatenate(array_Xnew[i]))
    #     list_X_arithm = list(np.concatenate(array_Xnew_scale[i]))
    #     del list_Q_LP[16]
    #     del list_X_srvf[16]
    #     del list_X_arithm[16]
    # else:
    list_Q_LP = list(np.concatenate(array_Q_LP[i]))
    list_X_srvf = list(np.concatenate(array_Xnew[i]))
    list_X_arithm = list(np.concatenate(array_Xnew_scale[i]))
    array_TruePopFP[i] = PopulationFrenetPath(list_Q_LP)
    array_X_srvf[i] = list_X_srvf
    array_X_Arithm[i] = list_X_arithm

# out = Parallel(n_jobs=-1)(delayed(adaptative_estimation)(array_TruePopFP[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                             for i in range(n_cond))
out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_TruePopFP[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
                            for i in range(n_cond))

array_SmoothPopFP = np.empty((n_cond), dtype=object)
array_SmoothThetaFP = np.empty((n_cond), dtype=object)
array_resOpt = np.empty((n_cond), dtype=object)

for k in range(n_cond):
    array_SmoothPopFP[k] = out[k][0]
    array_resOpt[k] = out[k][1]
    if array_resOpt[k][1]==True:
        array_SmoothThetaFP[k] = FrenetPath(array_SmoothPopFP[k].grids_obs[0], array_SmoothPopFP[k].grids_obs[0], init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
        array_SmoothThetaFP[k].frenet_serret_solve()

duration = timer() - start1
print('total time', duration)


list_X_LP = np.empty((n_cond), dtype=object)
list_Xnew = np.empty((n_cond), dtype=object)
list_Xnew_scale = np.empty((n_cond), dtype=object)
for i in range(n_cond):
    X_LP = []
    Xnew = []
    Xnew_scale = []
    for j in range(n_subj):
        for k in range(n_rept):
            X_LP.append(array_Q_LP[i,j,k].data_trajectory)
            Xnew.append(array_Xnew[i,j,k])
            Xnew_scale.append(array_Xnew_scale[i,j,k])
    list_X_LP[i], list_Xnew[i], list_Xnew_scale[i] = X_LP, Xnew, Xnew_scale

print("Mean estimations SRVF...")

t = np.linspace(0,1,n_resamples)

array_SRVF_mean_LP = np.empty((n_cond), dtype=object)
array_SRVF_gam_LP = np.empty((n_cond), dtype=object)

for i in range(n_cond):
    array_SRVF_mean_LP[i], array_SRVF_gam_LP[i] = compute_mean_SRVF(list_X_LP[i], t)

print("Mean estimations SRVF...")

array_SRVF_mean = np.empty((n_cond), dtype=object)
array_SRVF_gam = np.empty((n_cond), dtype=object)

for i in range(n_cond):
    array_SRVF_mean[i], array_SRVF_gam[i] = compute_mean_SRVF(list_Xnew[i], t)


print("Mean estimations Arithmetic...")

array_Arithmetic_mean = Parallel(n_jobs=-1)(delayed(compute_mean_Arithmetic)(list_Xnew_scale[i]) for i in range(n_cond))

print("Mean estimations Arithmetic...")

array_Arithmetic_mean_LP = Parallel(n_jobs=-1)(delayed(compute_mean_Arithmetic)(list_X_LP[i]) for i in range(n_cond))


### Sauvegarde des données

print('Saving the data...')

filename = "Raket_data_means_per_cond_SingleEstim"
dic = {"TruePopFP":array_TruePopFP, "SmoothPopFP":array_SmoothPopFP, "SmoothThetaFP":array_SmoothThetaFP, "resOpt":array_resOpt, "SRVF_mean" :array_SRVF_mean,
"SRVF_gam" : array_SRVF_gam, "SRVF_mean_LP" :array_SRVF_mean_LP, "SRVF_gam_LP" : array_SRVF_gam_LP, "Arithmetic_mean" : array_Arithmetic_mean,
"Arithmetic_mean_LP" : array_Arithmetic_mean_LP}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')
