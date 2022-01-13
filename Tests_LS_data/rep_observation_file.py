import sys
import os.path
sys.path.insert(1, '../FrenetSerretMeanShape')
from pre_process_Mocaplab_data import *
import numpy as np
from scipy import interpolate
from visu_utils import *
from pickle import *
import dill as pickle
from research_law_utils import *
import fdasrsf as fs
from scipy.integrate import cumtrapz
from alignment_utils import *

def preprocess_X(data, h_min, h_max, nb_h, t_new):
    t = np.linspace(0, 1, len(data))
    X = Trajectory(data, t)
    h_opt = opti_loc_poly_traj(X.data, X.t, h_min, h_max, nb_h, n_splits=5)
    X.loc_poly_estimation(X.t, 5, h_opt)
    X.compute_S(scale=False)
    return X
    # Q_GS = X.TNB_GramSchmidt(t_new)
    # return X, Q_GS


"""___________________________________________________________________________ importation of the data ___________________________________________________________________________"""

# # import data "Washington is a mess" to add to the previous gloses
# path_dir = r"/home/pchassat/Documents/frenet-serret-smoothing/data/Repetitions_LSF/"
# files = ['SMI2_X0077_0.csv', 'SMI2_X0077_1.csv', 'SMI2_X0077_2.csv'] #, 'SMI2_X0077_3.csv'] the last one is "Washington DC is a mess"
# data_WDC = []
# for i in range(3):
#     data_WDC.append(load_Mocap_data_hand(path_dir+files[i], plot=False, hand='Right'))
# data_WDC = np.array(data_WDC)

# import cut repeated glose from annotation
filename = "Repeated_glose_LSFtraj_data_to_be_preprocessed"
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()
rep_data_collections = collections.namedtuple('rep_data', ['data', 'glose', 'nb_rep'])

# we sort the repetition in decreasing order of number of rep, we select only the glose repeated at least 3 times, we remove "dactylologie" and "??"
ind1 = np.argsort(-dic["nb_rep"])
ind2 = ind1[np.where(dic["nb_rep"][ind1]>2)]
ind3 = ind2[np.where(ind2!=np.where(dic["glose"]=='??')[0])]
index_to_keep = [2, 4, 5, 6, 8, 11, 12, 16, 17, 18, 21, 23, 29, 31]
index_to_pop = [[16, 15, 13, 8, 7, 5, 4, 3], [16, 11, 6, 3], [13, 12, 10, 9, 1], [14, 11, 7, 5, 3, 2], [9], [6, 3], [4, 2, 1], [2], [3, 1], [], [], [], [3], []]

glose = dic["glose"][ind3][1:][index_to_keep]
nb_rep = []
data = dic["rep_data"][ind3][1:][index_to_keep]
for i in range(len(data)):
    data[i] = np.delete(data[i], index_to_pop[i])
    nb_rep.append(len(data[i]))
ind4 = np.argsort(-np.array(nb_rep))

# put all together
# rep_data = rep_data_collections(dic["rep_data"][ind3][1:], dic["glose"][ind3][1:], [int(dic["nb_rep"][ind3][1:][j]) for j in range(len(dic["nb_rep"][ind3][1:]))], dic["ind_init_curve"][ind3][1:])
rep_data = rep_data_collections(data[ind4], np.array(glose)[ind4], np.array(nb_rep)[ind4])
N = len(rep_data.data)
print(f"We have {N} gloses repeated at least 4 times:")
print(' ')
for i in range(N):
    print(f"'{rep_data.glose[i]}': repeated {rep_data.nb_rep[i]} times")
print(' ')


"""___________________________________________________________________________ estimation of parameters ___________________________________________________________________________"""

# estimation of s_dot, kappa, tau and determinant function for each sample
print("estimation of s_dot, kappa, tau and determinant function for each sample... \n")

h_min, h_max, nb_h = 0.1, 0.2, 20
t_new = np.linspace(0, 1, 200)
hyperparam = [11, 1e-07, 1e-07]
param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (7, 13), "bounds_lcurv" : (1e-09, 1e-06), "bounds_ltors" :  (1e-09, 1e-06)}


# estimation of trajectories
print("estimation of trajectories... \n")

trajs = np.empty((N), dtype=object)
for i in range(N):
    ni = rep_data.nb_rep[i]
    trajs[i] = np.empty((ni), dtype=object)
    for j in range(ni):
        trajs[i][j] = preprocess_X(rep_data.data[i][j], h_min, h_max, nb_h, t_new)
    for j in range(ni):
        trajs[i][j].scale(factor=1)


# # estimation of time warping functions
# print("estimation of time warping functions... \n")
#
# warp_h_func = np.empty((N), dtype=object)
# sdot_mean = np.zeros((N,len(t_new)))
# s_mean = np.zeros((N,len(t_new)))
# for i in range(N):
#     ni = rep_data.nb_rep[i]
#     s_i = np.zeros((ni, len(t_new)))
#     for j in range(ni):
#         s_i[j,:] = trajs[i][j].S(t_new)
#     obj = fs.fdawarp(s_i.T,t_new)
#     obj.srsf_align(parallel=True, lam=0.01)
#     s_mean[i,:] = obj.fmean - obj.fmean[0]
#     warp_h_func[i] = obj.gam # ou obj.gaml (inverse warping functions)
#     _, sdot_mean[i,:] = warp_curvatures_bis(np.array([trajs[i][j].Sdot(t_new) for j in range(ni)]).T, obj.gam, t_new)
#     # _, sdot_mean[i,:], _ = fs.utility_functions.gradient_spline(t_new, s_mean[i,:], smooth=False)


# estimation of frenet paths
print("estimation of frenet paths... \n")

frenet_paths = np.empty((N), dtype=object)
new_s = np.linspace(0, 1, 200)
for i in range(N):
    ni = rep_data.nb_rep[i]
    frenet_paths[i] = np.empty((ni), dtype=object)
    for j in range(ni):
        s_init = trajs[i][j].S(trajs[i][j].t)
        s_init[-1] = np.round(s_init[-1], decimals=8)
        inv_s_fct = interpolate.interp1d(s_init, trajs[i][j].t)
        frenet_paths[i][j] = trajs[i][j].TNB_GramSchmidt(inv_s_fct(new_s))
        frenet_paths[i][j].grid_obs = new_s
        frenet_paths[i][j].grid_eval = new_s


# estimation of individual curvatures and torsions
print("estimation of individual curvatures and torsions... \n")

models = np.empty((N), dtype=object)
for i in range(N):
    ni = rep_data.nb_rep[i]
    models[i] = np.empty((ni), dtype=object)

    # for j in range(ni):
    #     _, models[i][j], _ = global_estimation(frenet_paths[i][j], param_model={"nb_basis" : None,
    #             "domain_range": (np.round(frenet_paths[i][j].grid_obs[0], decimals=8), np.round(frenet_paths[i][j].grid_obs[-1], decimals=8))}, opt=True, hyperparam=hyperparam, param_bayopt=param_bayopt,
    #             adaptive_h=True)


    out = Parallel(n_jobs=-1)(delayed(global_estimation)(frenet_paths[i][j], param_model={"nb_basis" : None,
            "domain_range": (np.round(frenet_paths[i][j].grid_obs[0], decimals=8), np.round(frenet_paths[i][j].grid_obs[-1], decimals=8))}, opt=True, hyperparam=hyperparam, param_bayopt=param_bayopt,
            adaptive_h=True) for j in range(ni))

    for j in range(ni):
        models[i][j] = out[j][1]



# save first part of the data
print("first saving of the data...")

filename = "rep_observations_estimates_part1"
dic = {"trajs" : trajs, "frenet_paths" : frenet_paths, "models" : models}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



# estimation of mean curvatures and torsions
print("estimation of mean curvatures and torsions... \n")

models_mean = np.empty((N), dtype=object)

# for i in range(N):
#     _, models_mean[i], res_opt = global_estimation(PopulationFrenetPath(frenet_paths[i]), param_model={"nb_basis" : None,
#                 "domain_range": (np.round(frenet_paths[i][j].grid_obs[0], decimals=8), np.round(frenet_paths[i][j].grid_obs[-1], decimals=8))}, opt=True, hyperparam=hyperparam, param_bayopt=param_bayopt,
#                 alignment=True, lam=100.0, adaptive_h=True)

out = Parallel(n_jobs=-1)(delayed(global_estimation)(PopulationFrenetPath(frenet_paths[i]), param_model={"nb_basis" : None,
            "domain_range": (np.round(frenet_paths[i][j].grid_obs[0], decimals=8), np.round(frenet_paths[i][j].grid_obs[-1], decimals=8))}, opt=True, hyperparam=hyperparam, param_bayopt=param_bayopt,
            alignment=True, lam=100.0, adaptive_h=True) for i in range(N))

for i in range(N):
    models_mean[i] = out[i][1]



# estimation of gamma and omega warping functions
print("estimation of gamma and omega warping functions... \n")

warp_gamma_func = np.empty((N), dtype=object)
warp_omega_func = np.empty((N), dtype=object)
for i in range(N):
    warp_gamma_func[i] = models_mean[i].gam
    warp_omega_func[i] = np.empty((len(models_mean[i].gam)), dtype=object)
    for j in range(len(models_mean[i].gam)):
        warp_omega_func[i][j] = invertGamma_fct(warp_gamma_func[i][j], np.linspace(0, 1, 500))



# estimation of mean s and the time warping functions h_i
print("estimation of mean s and the time warping functions h_i... \n")

warp_h_func = np.empty((N), dtype=object)
s_mean = np.zeros((N, len(t_new)))
for i in range(N):
    ni = rep_data.nb_rep[i]
    w_s_i = np.zeros((ni, len(t_new)))
    for j in range(ni):
        s_ij = trajs[i][j].S(t_new)
        s_ij[-1] = np.round(s_ij[-1], decimals=0)
        s_ij[-2] = np.round(s_ij[-2], decimals=3)
        w_s_i[j,:] = warp_omega_func[i][j](s_ij)
    obj = fs.fdawarp(w_s_i.T,t_new)
    obj.srsf_align(parallel=True, lam=0)
    s_mean[i,:] = obj.fmean - obj.fmean[0]
    warp_h_func[i] = obj.gam



# estimation of the mean sdot
print("estimation of the mean sdot... \n")

sdot_mean = np.zeros((N, len(t_new)))
for i in range(N):
    sdot_mean[i,:] = interpolate.UnivariateSpline(t_new, s_mean[i,:], s=0.0001)(t_new, 1)



"""___________________________________________________________________________ computation of relevant covariates ___________________________________________________________________________"""

sdot_arr = np.empty((N), dtype=object)
curv_arr = np.empty((N), dtype=object)
tors_arr = np.empty((N), dtype=object)
det_arr = np.empty((N), dtype=object)
sdot_mean_arr = sdot_mean
curv_mean_arr = np.zeros((N, len(t_new)))
tors_mean_arr = np.zeros((N, len(t_new)))
det_mean_arr = np.zeros((N, len(t_new)))

for i in range(N):
    ni = rep_data.nb_rep[i]
    sdot_arr[i] = np.zeros((ni, len(t_new)))
    curv_arr[i] = np.zeros((ni, len(t_new)))
    tors_arr[i] = np.zeros((ni, len(t_new)))
    det_arr[i] = np.zeros((ni, len(t_new)))
    for j in range(ni):
        sdot_arr[i][j] = trajs[i][j].Sdot(t_new)
        curv_arr[i][j] = models[i][j].curv.function(trajs[i][j].S(t_new))
        tors_arr[i][j] = models[i][j].tors.function(trajs[i][j].S(t_new))
        det_arr[i][j] = compute_determinant(sdot_arr[i][j], curv_arr[i][j], tors_arr[i][j], abs=True, normalized=True)

    curv_mean_arr[i] = models_mean[i].curv.function(s_mean[i])
    tors_mean_arr[i] = models_mean[i].tors.function(s_mean[i])
    det_mean_arr[i] = compute_determinant(sdot_mean_arr[i], curv_mean_arr[i], tors_mean_arr[i], abs=True, normalized=True)





"""___________________________________________________________________________ save estimates ___________________________________________________________________________"""


filename = "rep_observations_estimates_part2"
dic = {"sdot_arr" : sdot_arr, "curv_arr" : curv_arr, "tors_arr" : tors_arr, "det_arr" : det_arr, "sdot_mean_arr" : sdot_mean_arr, "curv_mean_arr" : curv_mean_arr, "tors_mean_arr" : tors_mean_arr,
"det_mean_arr" : det_mean_arr, "s_mean" : s_mean, "trajs" : trajs, "frenet_paths" : frenet_paths, "warp_h_func" : warp_h_func, "models" : models, "models_mean" : models_mean,
"warp_gamma_func" : warp_gamma_func, "warp_omega_func" : warp_omega_func}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('the algorithm is ended.')
