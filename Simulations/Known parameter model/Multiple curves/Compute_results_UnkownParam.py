import sys
import os.path
sys.path.insert(1, '../../../FrenetSerretMeanShape')
sys.path.insert(1, '../../Sphere')
import numpy as np
from pickle import *
import dill as pickle
from estimation_algo_utils import *
from frenet_path import *
from trajectory import *
from model_curvatures import *
from simu_utils import *
from scipy import interpolate

""" Load file """

n_MC = 90
n_curves = 25
nb_S = 100
n_calls = 80
sigma_e = 0
sigma_p = 0.05

# dic = {"N_curves": n_curves, "L" : L0, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots, "n_MC" : n_MC, "resOpt" : array_resOpt, "SmoothPopFP" : array_SmoothPopFP,
# "SmoothThetaFP" : array_SmoothThetaFP, "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "sigma_p" : sigma_p,
# "PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam,"Arithmetic_mean" : array_Arithmetic_mean, "array_Phi" : array_Phi, "array_Xmean" : array_Xmean, "array_Qmean" : array_Qmean, "ThetaMeanExtrins" : array_ThetaMeanExtrins,
# "ThetaMeanTrue" : array_ThetaMeanTrue, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
#

filename = "MultipleEstimationUnknownModel_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_e_"+str(sigma_e)+"_sigma_p_"+str(sigma_p)+'_nMC_'+str(n_MC)+'_nCalls_'+str(n_calls)
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

L0 = 5
phi_ref = (1,0.9,0.8)
f = lambda t,phi: np.array([np.cos(phi[0]*t), np.sin(phi[1]*t), phi[2]*t]).transpose()
df = lambda t,phi: np.array([-phi[0]*np.sin(phi[0]*t), phi[1]*np.cos(phi[1]*t), phi[2]])
ddf = lambda t,phi: np.array([-(phi[0]**2)*np.cos(phi[0]*t), -(phi[1]**2)*np.sin(phi[1]*t), 0])
dddf = lambda t,phi: np.array([np.power(phi[0],3)*np.sin(phi[0]*t), -np.power(phi[1],3)*np.cos(phi[1]*t), 0])

t = np.linspace(0,1,100)
s0 = np.linspace(0,L0,nb_S)

def true_curv(t,phi):
    return np.linalg.norm(np.cross(df(t,phi),ddf(t,phi)))/np.power(np.linalg.norm(df(t,phi)),3)
def true_torsion(t,phi):
    return np.dot(np.cross(df(t,phi),ddf(t,phi)), dddf(t,phi))/np.power(np.linalg.norm(np.cross(df(t,phi),ddf(t,phi))),2)

true_curv0 = lambda s: np.array([true_curv(s_,phi_ref) for s_ in s])
true_tors0 = lambda s: np.array([true_torsion(s_,phi_ref) for s_ in s])


resOpt = dic["resOpt"]
SmoothPopFP = dic["SmoothPopFP"]
SmoothThetaFP = dic["SmoothThetaFP"]
PopTraj = dic["PopTraj"]
SRVF_mean = dic["SRVF_mean"]
ThetaExtrins = dic["ThetaExtrins"]
Arithmetic_mean = dic["Arithmetic_mean"]
SmoothFPIndiv = dic["SmoothFPIndiv"]
resOptIndiv = dic["resOptIndiv"]
SmoothThetaFPIndiv = dic["SmoothThetaFPIndiv"]
array_Phi = dic["array_Phi"]
array_Xmean = dic["array_Xmean"]

""" Compute distances """

# d_ThetaFS = np.zeros(n_MC)
d_kappa = np.zeros(n_MC)
d_tau = np.zeros(n_MC)
d_fr_X_FS = np.zeros(n_MC)
d_fr_X_SRVF = np.zeros(n_MC)
d_fr_X_Arith = np.zeros(n_MC)
d_l2_X_FS = np.zeros(n_MC)
d_l2_X_SRVF = np.zeros(n_MC)
d_l2_X_Arith = np.zeros(n_MC)
d_kappa_ext = np.zeros(n_MC)
d_tau_ext = np.zeros(n_MC)
d_kappa_ind = np.zeros(n_MC)
d_tau_ind = np.zeros(n_MC)

L_mean = 0
for k in range(n_MC):
    L_mean += array_Xmean[k].L
L_mean = L_mean/n_MC

for k in range(n_MC):

    """ delta FS Theta """
    # d_ThetaFS[k] = geodesic_dist(SmoothThetaFP[k].data, Q0.data)

    """ delta kappa ext """
    mean_kappa_ext = np.zeros(nb_S)
    # for i in range(n_curves):
    #     mean_kappa_ext += (np.linalg.norm((ThetaExtrins[k][i][0] - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S
    # d_kappa_ext[k] = (np.linalg.norm((mean_kappa_ext/n_curves - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S
    for i in range(n_curves):
        kappa_ext = interpolate.interp1d(PopTraj[k][i].S(t), ThetaExtrins[k][i][0])(s0/L0)
        mean_kappa_ext += (np.linalg.norm((kappa_ext - true_curv0(s0)))**2)/nb_S
    d_kappa_ext[k] = (np.linalg.norm((mean_kappa_ext/n_curves - true_curv0(s0)))**2)/nb_S

    """ delta tau ext """
    mean_tau_ext = np.zeros(nb_S)
    # for i in range(n_curves):
    #     mean_tau_ext += (np.linalg.norm((ThetaExtrins[k][i][1] - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S
    # d_tau_ext[k] = (np.linalg.norm((mean_tau_ext/n_curves - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S
    for i in range(n_curves):
        tau_ext = interpolate.interp1d(PopTraj[k][i].S(t), ThetaExtrins[k][i][1])(s0/L0)
        mean_tau_ext += (np.linalg.norm((tau_ext - true_tors0(s0)))**2)/nb_S
    d_tau_ext[k] = (np.linalg.norm((mean_tau_ext/n_curves - true_tors0(s0)))**2)/nb_S

    """ delta kappa ind """
    mean_kappa_ind = np.zeros(nb_S)
    for i in range(n_curves):
        mean_kappa_ind += SmoothThetaFPIndiv[k,i].curv(s0/L0)
    d_kappa_ind[k] = (np.linalg.norm((mean_kappa_ind/n_curves - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta tau ind"""
    mean_tau_ind = np.zeros(nb_S)
    for i in range(n_curves):
        mean_tau_ind += SmoothThetaFPIndiv[k,i].tors(s0/L0)
    d_tau_ind[k] = (np.linalg.norm((mean_tau_ind/n_curves - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta kappa """
    d_kappa[k] = (np.linalg.norm((SmoothThetaFP[k].curv(s0/L0) - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta tau """
    d_tau[k] = (np.linalg.norm((SmoothThetaFP[k].tors(s0/L0) - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S

    phi_mean = array_Phi[k].mean(axis=0)
    """ delta X FS """
    d_l2_X_FS[k] = L2_dist(f(s0,phi_mean)/L0, SmoothThetaFP[k].data_trajectory, s0)
    d_fr_X_FS[k] = FisherRao_dist(f(s0,phi_mean)/L0, SmoothThetaFP[k].data_trajectory, s0)

    """ delta X SRVF """
    d_l2_X_SRVF[k] = L2_dist(f(s0,phi_mean)/L0, SRVF_mean[k], s0)
    d_fr_X_SRVF[k] = FisherRao_dist(f(s0,phi_mean)/L0, SRVF_mean[k], s0)

    """ delta X Arithmetic """
    d_l2_X_Arith[k] = L2_dist(f(s0,phi_mean)/L0, Arithmetic_mean[k], s0)
    d_fr_X_Arith[k] = FisherRao_dist(f(s0,phi_mean)/L0, Arithmetic_mean[k], s0)

# print(resOpt)

""" Print results """

print('Results of simulation on multiple curves Unknown model from X with sigma_p ='+str(sigma_p)+' and sigma_e ='+str(sigma_e))
# print('Delta_ThetaFS : ', d_ThetaFS.mean(), d_ThetaFS.std())
print('Delta_kappa_ext : ', d_kappa_ext.mean(), d_kappa_ext.std())
print('Delta_kappa_ind : ', d_kappa_ind.mean(), d_kappa_ind.std())
print('Delta_kappa : ', d_kappa.mean(), d_kappa.std())
print('Delta_tau_ext : ', d_tau_ext.mean(), d_tau_ext.std())
print('Delta_tau_ind : ', d_tau_ind.mean(), d_tau_ind.std())
print('Delta_tau : ', d_tau.mean(), d_tau.std())
print('Delta_fr_X_FS : ', d_fr_X_FS.mean(), d_fr_X_FS.std())
print('Delta_fr_X_SRVF : ', d_fr_X_SRVF.mean(), d_fr_X_SRVF.std())
print('Delta_fr_X_Arithmetic : ', d_fr_X_Arith.mean(), d_fr_X_Arith.std())
print('Delta_l2_X_FS : ', d_l2_X_FS.mean(), d_l2_X_FS.std())
print('Delta_l2_X_SRVF : ', d_l2_X_SRVF.mean(), d_l2_X_SRVF.std())
print('Delta_l2_X_Arithmetic : ', d_l2_X_Arith.mean(), d_l2_X_Arith.std())



""" Load file """

n_MC = 90
n_curves = 25
nb_S = 100
n_calls = 80
sigma_e = 0.03
sigma_p = 0.02

# dic = {"N_curves": n_curves, "L" : L0, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots, "n_MC" : n_MC, "resOpt" : array_resOpt, "SmoothPopFP" : array_SmoothPopFP,
# "SmoothThetaFP" : array_SmoothThetaFP, "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "sigma_p" : sigma_p,
# "PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam,"Arithmetic_mean" : array_Arithmetic_mean, "array_Phi" : array_Phi, "array_Xmean" : array_Xmean, "array_Qmean" : array_Qmean, "ThetaMeanExtrins" : array_ThetaMeanExtrins,
# "ThetaMeanTrue" : array_ThetaMeanTrue, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
#

filename = "MultipleEstimationUnknownModel_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_e_"+str(sigma_e)+"_sigma_p_"+str(sigma_p)+'_nMC_'+str(n_MC)+'_nCalls_'+str(n_calls)
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

L0 = 5
phi_ref = (1,0.9,0.8)
f = lambda t,phi: np.array([np.cos(phi[0]*t), np.sin(phi[1]*t), phi[2]*t]).transpose()
df = lambda t,phi: np.array([-phi[0]*np.sin(phi[0]*t), phi[1]*np.cos(phi[1]*t), phi[2]])
ddf = lambda t,phi: np.array([-(phi[0]**2)*np.cos(phi[0]*t), -(phi[1]**2)*np.sin(phi[1]*t), 0])
dddf = lambda t,phi: np.array([np.power(phi[0],3)*np.sin(phi[0]*t), -np.power(phi[1],3)*np.cos(phi[1]*t), 0])

s0 = np.linspace(0,L0,nb_S)

def true_curv(t,phi):
    return np.linalg.norm(np.cross(df(t,phi),ddf(t,phi)))/np.power(np.linalg.norm(df(t,phi)),3)
def true_torsion(t,phi):
    return np.dot(np.cross(df(t,phi),ddf(t,phi)), dddf(t,phi))/np.power(np.linalg.norm(np.cross(df(t,phi),ddf(t,phi))),2)

true_curv0 = lambda s: np.array([true_curv(s_,phi_ref) for s_ in s])
true_tors0 = lambda s: np.array([true_torsion(s_,phi_ref) for s_ in s])


resOpt = dic["resOpt"]
SmoothPopFP = dic["SmoothPopFP"]
SmoothThetaFP = dic["SmoothThetaFP"]
PopTraj = dic["PopTraj"]
SRVF_mean = dic["SRVF_mean"]
ThetaExtrins = dic["ThetaExtrins"]
Arithmetic_mean = dic["Arithmetic_mean"]
SmoothFPIndiv = dic["SmoothFPIndiv"]
resOptIndiv = dic["resOptIndiv"]
SmoothThetaFPIndiv = dic["SmoothThetaFPIndiv"]
array_Phi = dic["array_Phi"]
array_Xmean = dic["array_Xmean"]

""" Compute distances """

# d_ThetaFS = np.zeros(n_MC)
d_kappa = np.zeros(n_MC)
d_tau = np.zeros(n_MC)
d_fr_X_FS = np.zeros(n_MC)
d_fr_X_SRVF = np.zeros(n_MC)
d_fr_X_Arith = np.zeros(n_MC)
d_l2_X_FS = np.zeros(n_MC)
d_l2_X_SRVF = np.zeros(n_MC)
d_l2_X_Arith = np.zeros(n_MC)
d_kappa_ext = np.zeros(n_MC)
d_tau_ext = np.zeros(n_MC)
d_kappa_ind = np.zeros(n_MC)
d_tau_ind = np.zeros(n_MC)

L_mean = 0
for k in range(n_MC):
    L_mean += array_Xmean[k].L
L_mean = L_mean/n_MC

for k in range(n_MC):

    """ delta FS Theta """
    # d_ThetaFS[k] = geodesic_dist(SmoothThetaFP[k].data, Q0.data)

    """ delta kappa ext """
    mean_kappa_ext = np.zeros(nb_S)
    # for i in range(n_curves):
    #     mean_kappa_ext += (np.linalg.norm((ThetaExtrins[k][i][0] - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S
    # d_kappa_ext[k] = (np.linalg.norm((mean_kappa_ext/n_curves - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S
    mean_kappa_ext = np.zeros(nb_S)
    for i in range(n_curves):
        mean_kappa_ext += (np.linalg.norm((ThetaExtrins[k][i][0] - true_curv0(s0)))**2)/nb_S
    d_kappa_ext[k] = (np.linalg.norm((mean_kappa_ext/n_curves - true_curv0(s0)))**2)/nb_S

    """ delta tau ext """
    mean_tau_ext = np.zeros(nb_S)
    # for i in range(n_curves):
    #     mean_tau_ext += (np.linalg.norm((ThetaExtrins[k][i][1] - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S
    # d_tau_ext[k] = (np.linalg.norm((mean_tau_ext/n_curves - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S
    for i in range(n_curves):
        mean_tau_ext += (np.linalg.norm((ThetaExtrins[k][i][1] - true_tors0(s0)))**2)/nb_S
    d_tau_ext[k] = (np.linalg.norm((mean_tau_ext/n_curves - true_tors0(s0)))**2)/nb_S

    """ delta kappa ind """
    mean_kappa_ind = np.zeros(nb_S)
    for i in range(n_curves):
        mean_kappa_ind += SmoothThetaFPIndiv[k,i].curv(s0/L0)
    d_kappa_ind[k] = (np.linalg.norm((mean_kappa_ind/n_curves - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta tau ind"""
    mean_tau_ind = np.zeros(nb_S)
    for i in range(n_curves):
        mean_tau_ind += SmoothThetaFPIndiv[k,i].tors(s0/L0)
    d_tau_ind[k] = (np.linalg.norm((mean_tau_ind/n_curves - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta kappa """
    d_kappa[k] = (np.linalg.norm((SmoothThetaFP[k].curv(s0/L0) - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta tau """
    d_tau[k] = (np.linalg.norm((SmoothThetaFP[k].tors(s0/L0) - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S

    phi_mean = array_Phi[k].mean(axis=0)
    """ delta X FS """
    d_l2_X_FS[k] = L2_dist(f(s0,phi_mean)/L0, SmoothThetaFP[k].data_trajectory, s0)
    d_fr_X_FS[k] = FisherRao_dist(f(s0,phi_mean)/L0, SmoothThetaFP[k].data_trajectory, s0)

    """ delta X SRVF """
    d_l2_X_SRVF[k] = L2_dist(f(s0,phi_mean)/L0, SRVF_mean[k], s0)
    d_fr_X_SRVF[k] = FisherRao_dist(f(s0,phi_mean)/L0, SRVF_mean[k], s0)

    """ delta X Arithmetic """
    d_l2_X_Arith[k] = L2_dist(f(s0,phi_mean)/L0, Arithmetic_mean[k], s0)
    d_fr_X_Arith[k] = FisherRao_dist(f(s0,phi_mean)/L0, Arithmetic_mean[k], s0)

# print(resOpt)

""" Print results """

print('Results of simulation on multiple curves Unknown model from X with sigma_p ='+str(sigma_p)+' and sigma_e ='+str(sigma_e))
# print('Delta_ThetaFS : ', d_ThetaFS.mean(), d_ThetaFS.std())
print('Delta_kappa_ext : ', d_kappa_ext.mean(), d_kappa_ext.std())
print('Delta_kappa_ind : ', d_kappa_ind.mean(), d_kappa_ind.std())
print('Delta_kappa : ', d_kappa.mean(), d_kappa.std())
print('Delta_tau_ext : ', d_tau_ext.mean(), d_tau_ext.std())
print('Delta_tau_ind : ', d_tau_ind.mean(), d_tau_ind.std())
print('Delta_tau : ', d_tau.mean(), d_tau.std())
print('Delta_fr_X_FS : ', d_fr_X_FS.mean(), d_fr_X_FS.std())
print('Delta_fr_X_SRVF : ', d_fr_X_SRVF.mean(), d_fr_X_SRVF.std())
print('Delta_fr_X_Arithmetic : ', d_fr_X_Arith.mean(), d_fr_X_Arith.std())
print('Delta_l2_X_FS : ', d_l2_X_FS.mean(), d_l2_X_FS.std())
print('Delta_l2_X_SRVF : ', d_l2_X_SRVF.mean(), d_l2_X_SRVF.std())
print('Delta_l2_X_Arithmetic : ', d_l2_X_Arith.mean(), d_l2_X_Arith.std())



""" Load file """

n_MC = 90
n_curves = 25
nb_S = 100
n_calls = 80
sigma_e = 0
sigma_p = 0.05

# dic = {"N_curves": n_curves, "L" : L0, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots, "n_MC" : n_MC, "resOpt" : array_resOpt, "SmoothPopFP" : array_SmoothPopFP,
# "SmoothThetaFP" : array_SmoothThetaFP, "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "sigma_p" : sigma_p,
# "PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam,"Arithmetic_mean" : array_Arithmetic_mean, "array_Phi" : array_Phi, "array_Xmean" : array_Xmean, "array_Qmean" : array_Qmean, "ThetaMeanExtrins" : array_ThetaMeanExtrins,
# "ThetaMeanTrue" : array_ThetaMeanTrue, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
#

filename = "MultipleEstimationUnknownModel_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_e_"+str(sigma_e)+"_sigma_p_"+str(sigma_p)+'_nMC_'+str(n_MC)+'_nCalls_'+str(n_calls)
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

L0 = 5
phi_ref = (1,0.9,0.8)
f = lambda t,phi: np.array([np.cos(phi[0]*t), np.sin(phi[1]*t), phi[2]*t]).transpose()
df = lambda t,phi: np.array([-phi[0]*np.sin(phi[0]*t), phi[1]*np.cos(phi[1]*t), phi[2]])
ddf = lambda t,phi: np.array([-(phi[0]**2)*np.cos(phi[0]*t), -(phi[1]**2)*np.sin(phi[1]*t), 0])
dddf = lambda t,phi: np.array([np.power(phi[0],3)*np.sin(phi[0]*t), -np.power(phi[1],3)*np.cos(phi[1]*t), 0])

s0 = np.linspace(0,L0,nb_S)

def true_curv(t,phi):
    return np.linalg.norm(np.cross(df(t,phi),ddf(t,phi)))/np.power(np.linalg.norm(df(t,phi)),3)
def true_torsion(t,phi):
    return np.dot(np.cross(df(t,phi),ddf(t,phi)), dddf(t,phi))/np.power(np.linalg.norm(np.cross(df(t,phi),ddf(t,phi))),2)

true_curv0 = lambda s: np.array([true_curv(s_,phi_ref) for s_ in s])
true_tors0 = lambda s: np.array([true_torsion(s_,phi_ref) for s_ in s])


resOpt = dic["resOpt"]
SmoothPopFP = dic["SmoothPopFP"]
SmoothThetaFP = dic["SmoothThetaFP"]
PopTraj = dic["PopTraj"]
SRVF_mean = dic["SRVF_mean"]
ThetaExtrins = dic["ThetaExtrins"]
Arithmetic_mean = dic["Arithmetic_mean"]
SmoothFPIndiv = dic["SmoothFPIndiv"]
resOptIndiv = dic["resOptIndiv"]
SmoothThetaFPIndiv = dic["SmoothThetaFPIndiv"]
array_Phi = dic["array_Phi"]
array_Xmean = dic["array_Xmean"]

""" Compute distances """

# d_ThetaFS = np.zeros(n_MC)
d_kappa = np.zeros(n_MC)
d_tau = np.zeros(n_MC)
d_fr_X_FS = np.zeros(n_MC)
d_fr_X_SRVF = np.zeros(n_MC)
d_fr_X_Arith = np.zeros(n_MC)
d_l2_X_FS = np.zeros(n_MC)
d_l2_X_SRVF = np.zeros(n_MC)
d_l2_X_Arith = np.zeros(n_MC)
d_kappa_ext = np.zeros(n_MC)
d_tau_ext = np.zeros(n_MC)
d_kappa_ind = np.zeros(n_MC)
d_tau_ind = np.zeros(n_MC)

L_mean = 0
for k in range(n_MC):
    L_mean += array_Xmean[k].L
L_mean = L_mean/n_MC

for k in range(n_MC):

    """ delta FS Theta """
    # d_ThetaFS[k] = geodesic_dist(SmoothThetaFP[k].data, Q0.data)

    """ delta kappa ext """
    mean_kappa_ext = np.zeros(nb_S)
    # for i in range(n_curves):
    #     mean_kappa_ext += (np.linalg.norm((ThetaExtrins[k][i][0] - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S
    # d_kappa_ext[k] = (np.linalg.norm((mean_kappa_ext/n_curves - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S
    mean_kappa_ext = np.zeros(nb_S)
    for i in range(n_curves):
        mean_kappa_ext += (np.linalg.norm((ThetaExtrins[k][i][0] - true_curv0(s0)))**2)/nb_S
    d_kappa_ext[k] = (np.linalg.norm((mean_kappa_ext/n_curves - true_curv0(s0)))**2)/nb_S

    """ delta tau ext """
    mean_tau_ext = np.zeros(nb_S)
    # for i in range(n_curves):
    #     mean_tau_ext += (np.linalg.norm((ThetaExtrins[k][i][1] - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S
    # d_tau_ext[k] = (np.linalg.norm((mean_tau_ext/n_curves - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S
    for i in range(n_curves):
        mean_tau_ext += (np.linalg.norm((ThetaExtrins[k][i][1] - true_tors0(s0)))**2)/nb_S
    d_tau_ext[k] = (np.linalg.norm((mean_tau_ext/n_curves - true_tors0(s0)))**2)/nb_S

    """ delta kappa ind """
    mean_kappa_ind = np.zeros(nb_S)
    for i in range(n_curves):
        mean_kappa_ind += SmoothThetaFPIndiv[k,i].curv(s0/L0)
    d_kappa_ind[k] = (np.linalg.norm((mean_kappa_ind/n_curves - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta tau ind"""
    mean_tau_ind = np.zeros(nb_S)
    for i in range(n_curves):
        mean_tau_ind += SmoothThetaFPIndiv[k,i].tors(s0/L0)
    d_tau_ind[k] = (np.linalg.norm((mean_tau_ind/n_curves - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta kappa """
    d_kappa[k] = (np.linalg.norm((SmoothThetaFP[k].curv(s0/L0) - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta tau """
    d_tau[k] = (np.linalg.norm((SmoothThetaFP[k].tors(s0/L0) - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S

    phi_mean = array_Phi[k].mean(axis=0)
    """ delta X FS """
    d_l2_X_FS[k] = L2_dist(f(s0,phi_mean)/L0, SmoothThetaFP[k].data_trajectory, s0)
    d_fr_X_FS[k] = FisherRao_dist(f(s0,phi_mean)/L0, SmoothThetaFP[k].data_trajectory, s0)

    """ delta X SRVF """
    d_l2_X_SRVF[k] = L2_dist(f(s0,phi_mean)/L0, SRVF_mean[k], s0)
    d_fr_X_SRVF[k] = FisherRao_dist(f(s0,phi_mean)/L0, SRVF_mean[k], s0)

    """ delta X Arithmetic """
    d_l2_X_Arith[k] = L2_dist(f(s0,phi_mean)/L0, Arithmetic_mean[k], s0)
    d_fr_X_Arith[k] = FisherRao_dist(f(s0,phi_mean)/L0, Arithmetic_mean[k], s0)

# print(resOpt)

""" Print results """

print('Results of simulation on multiple curves Unknown model from X with sigma_p ='+str(sigma_p)+' and sigma_e ='+str(sigma_e))
# print('Delta_ThetaFS : ', d_ThetaFS.mean(), d_ThetaFS.std())
print('Delta_kappa_ext : ', d_kappa_ext.mean(), d_kappa_ext.std())
print('Delta_kappa_ind : ', d_kappa_ind.mean(), d_kappa_ind.std())
print('Delta_kappa : ', d_kappa.mean(), d_kappa.std())
print('Delta_tau_ext : ', d_tau_ext.mean(), d_tau_ext.std())
print('Delta_tau_ind : ', d_tau_ind.mean(), d_tau_ind.std())
print('Delta_tau : ', d_tau.mean(), d_tau.std())
print('Delta_fr_X_FS : ', d_fr_X_FS.mean(), d_fr_X_FS.std())
print('Delta_fr_X_SRVF : ', d_fr_X_SRVF.mean(), d_fr_X_SRVF.std())
print('Delta_fr_X_Arithmetic : ', d_fr_X_Arith.mean(), d_fr_X_Arith.std())
print('Delta_l2_X_FS : ', d_l2_X_FS.mean(), d_l2_X_FS.std())
print('Delta_l2_X_SRVF : ', d_l2_X_SRVF.mean(), d_l2_X_SRVF.std())
print('Delta_l2_X_Arithmetic : ', d_l2_X_Arith.mean(), d_l2_X_Arith.std())



""" Load file """

n_MC = 90
n_curves = 25
nb_S = 100
n_calls = 80
sigma_e = 0
sigma_p = 0.02

# dic = {"N_curves": n_curves, "L" : L0, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots, "n_MC" : n_MC, "resOpt" : array_resOpt, "SmoothPopFP" : array_SmoothPopFP,
# "SmoothThetaFP" : array_SmoothThetaFP, "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "sigma_p" : sigma_p,
# "PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam,"Arithmetic_mean" : array_Arithmetic_mean, "array_Phi" : array_Phi, "array_Xmean" : array_Xmean, "array_Qmean" : array_Qmean, "ThetaMeanExtrins" : array_ThetaMeanExtrins,
# "ThetaMeanTrue" : array_ThetaMeanTrue, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
#

filename = "MultipleEstimationUnknownModel_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_e_"+str(sigma_e)+"_sigma_p_"+str(sigma_p)+'_nMC_'+str(n_MC)+'_nCalls_'+str(n_calls)
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

L0 = 5
phi_ref = (1,0.9,0.8)
f = lambda t,phi: np.array([np.cos(phi[0]*t), np.sin(phi[1]*t), phi[2]*t]).transpose()
df = lambda t,phi: np.array([-phi[0]*np.sin(phi[0]*t), phi[1]*np.cos(phi[1]*t), phi[2]])
ddf = lambda t,phi: np.array([-(phi[0]**2)*np.cos(phi[0]*t), -(phi[1]**2)*np.sin(phi[1]*t), 0])
dddf = lambda t,phi: np.array([np.power(phi[0],3)*np.sin(phi[0]*t), -np.power(phi[1],3)*np.cos(phi[1]*t), 0])

s0 = np.linspace(0,L0,nb_S)

def true_curv(t,phi):
    return np.linalg.norm(np.cross(df(t,phi),ddf(t,phi)))/np.power(np.linalg.norm(df(t,phi)),3)
def true_torsion(t,phi):
    return np.dot(np.cross(df(t,phi),ddf(t,phi)), dddf(t,phi))/np.power(np.linalg.norm(np.cross(df(t,phi),ddf(t,phi))),2)

true_curv0 = lambda s: np.array([true_curv(s_,phi_ref) for s_ in s])
true_tors0 = lambda s: np.array([true_torsion(s_,phi_ref) for s_ in s])


resOpt = dic["resOpt"]
SmoothPopFP = dic["SmoothPopFP"]
SmoothThetaFP = dic["SmoothThetaFP"]
PopTraj = dic["PopTraj"]
SRVF_mean = dic["SRVF_mean"]
ThetaExtrins = dic["ThetaExtrins"]
Arithmetic_mean = dic["Arithmetic_mean"]
SmoothFPIndiv = dic["SmoothFPIndiv"]
resOptIndiv = dic["resOptIndiv"]
SmoothThetaFPIndiv = dic["SmoothThetaFPIndiv"]
array_Phi = dic["array_Phi"]
array_Xmean = dic["array_Xmean"]

""" Compute distances """

# d_ThetaFS = np.zeros(n_MC)
d_kappa = np.zeros(n_MC)
d_tau = np.zeros(n_MC)
d_fr_X_FS = np.zeros(n_MC)
d_fr_X_SRVF = np.zeros(n_MC)
d_fr_X_Arith = np.zeros(n_MC)
d_l2_X_FS = np.zeros(n_MC)
d_l2_X_SRVF = np.zeros(n_MC)
d_l2_X_Arith = np.zeros(n_MC)
d_kappa_ext = np.zeros(n_MC)
d_tau_ext = np.zeros(n_MC)
d_kappa_ind = np.zeros(n_MC)
d_tau_ind = np.zeros(n_MC)

L_mean = 0
for k in range(n_MC):
    L_mean += array_Xmean[k].L
L_mean = L_mean/n_MC

for k in range(n_MC):

    """ delta FS Theta """
    # d_ThetaFS[k] = geodesic_dist(SmoothThetaFP[k].data, Q0.data)

    """ delta kappa ext """
    mean_kappa_ext = np.zeros(nb_S)
    # for i in range(n_curves):
    #     mean_kappa_ext += (np.linalg.norm((ThetaExtrins[k][i][0] - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S
    # d_kappa_ext[k] = (np.linalg.norm((mean_kappa_ext/n_curves - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S
    mean_kappa_ext = np.zeros(nb_S)
    for i in range(n_curves):
        mean_kappa_ext += (np.linalg.norm((ThetaExtrins[k][i][0] - true_curv0(s0)))**2)/nb_S
    d_kappa_ext[k] = (np.linalg.norm((mean_kappa_ext/n_curves - true_curv0(s0)))**2)/nb_S

    """ delta tau ext """
    mean_tau_ext = np.zeros(nb_S)
    # for i in range(n_curves):
    #     mean_tau_ext += (np.linalg.norm((ThetaExtrins[k][i][1] - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S
    # d_tau_ext[k] = (np.linalg.norm((mean_tau_ext/n_curves - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S
    for i in range(n_curves):
        mean_tau_ext += (np.linalg.norm((ThetaExtrins[k][i][1] - true_tors0(s0)))**2)/nb_S
    d_tau_ext[k] = (np.linalg.norm((mean_tau_ext/n_curves - true_tors0(s0)))**2)/nb_S

    """ delta kappa ind """
    mean_kappa_ind = np.zeros(nb_S)
    for i in range(n_curves):
        mean_kappa_ind += SmoothThetaFPIndiv[k,i].curv(s0/L0)
    d_kappa_ind[k] = (np.linalg.norm((mean_kappa_ind/n_curves - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta tau ind"""
    mean_tau_ind = np.zeros(nb_S)
    for i in range(n_curves):
        mean_tau_ind += SmoothThetaFPIndiv[k,i].tors(s0/L0)
    d_tau_ind[k] = (np.linalg.norm((mean_tau_ind/n_curves - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta kappa """
    d_kappa[k] = (np.linalg.norm((SmoothThetaFP[k].curv(s0/L0) - true_curv0(s0)*array_Xmean[k].L))**2)/nb_S

    """ delta tau """
    d_tau[k] = (np.linalg.norm((SmoothThetaFP[k].tors(s0/L0) - true_tors0(s0)*array_Xmean[k].L))**2)/nb_S

    phi_mean = array_Phi[k].mean(axis=0)
    """ delta X FS """
    d_l2_X_FS[k] = L2_dist(f(s0,phi_mean)/L0, SmoothThetaFP[k].data_trajectory, s0)
    d_fr_X_FS[k] = FisherRao_dist(f(s0,phi_mean)/L0, SmoothThetaFP[k].data_trajectory, s0)

    """ delta X SRVF """
    d_l2_X_SRVF[k] = L2_dist(f(s0,phi_mean)/L0, SRVF_mean[k], s0)
    d_fr_X_SRVF[k] = FisherRao_dist(f(s0,phi_mean)/L0, SRVF_mean[k], s0)

    """ delta X Arithmetic """
    d_l2_X_Arith[k] = L2_dist(f(s0,phi_mean)/L0, Arithmetic_mean[k], s0)
    d_fr_X_Arith[k] = FisherRao_dist(f(s0,phi_mean)/L0, Arithmetic_mean[k], s0)

# print(resOpt)

""" Print results """

print('Results of simulation on multiple curves Unknown model from X with sigma_p ='+str(sigma_p)+' and sigma_e ='+str(sigma_e))
# print('Delta_ThetaFS : ', d_ThetaFS.mean(), d_ThetaFS.std())
print('Delta_kappa_ext : ', d_kappa_ext.mean(), d_kappa_ext.std())
print('Delta_kappa_ind : ', d_kappa_ind.mean(), d_kappa_ind.std())
print('Delta_kappa : ', d_kappa.mean(), d_kappa.std())
print('Delta_tau_ext : ', d_tau_ext.mean(), d_tau_ext.std())
print('Delta_tau_ind : ', d_tau_ind.mean(), d_tau_ind.std())
print('Delta_tau : ', d_tau.mean(), d_tau.std())
print('Delta_fr_X_FS : ', d_fr_X_FS.mean(), d_fr_X_FS.std())
print('Delta_fr_X_SRVF : ', d_fr_X_SRVF.mean(), d_fr_X_SRVF.std())
print('Delta_fr_X_Arithmetic : ', d_fr_X_Arith.mean(), d_fr_X_Arith.std())
print('Delta_l2_X_FS : ', d_l2_X_FS.mean(), d_l2_X_FS.std())
print('Delta_l2_X_SRVF : ', d_l2_X_SRVF.mean(), d_l2_X_SRVF.std())
print('Delta_l2_X_Arithmetic : ', d_l2_X_Arith.mean(), d_l2_X_Arith.std())
