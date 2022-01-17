import sys
import os.path
sys.path.insert(1, '../FrenetSerretMeanShape')
import numpy as np
from scipy import interpolate
from visu_utils import *
from research_law_utils import *
import dill as pickle


'''
_________________________________________________________________________________________________________________________________________________________________________________________________
    On load le fichier de données. Il est crée en segmentant, à partir des annotations données, les gloses dans chaque phrases.
    On crée un object "gloses" qui contient 3 paramètres :
    - gloses.name = tableau de taille N qui contient le nom des N gloses differentes se trouvant dans les phrases initiales (exemple: "bonjour", "manger", "20", etc...)
    - gloses.nb_rep = tableau de taille N qui contient le nombre de répétition de chaque glose dans les phrases initiales : n_i repétition pour la glose d'indice i.
    - gloses.data = tableau de taille N. Pour chaque indice i (i=1,...,N), gloses.data[i] est un tableau de taille n_i qui contient les points dans R^3 en fonction du temps
        des trajectoires de chaque répétition de la glose i.
_________________________________________________________________________________________________________________________________________________________________________________________________
'''

filename = "isolated_gloses_ROSETTA_T01"
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()
gloses_collections = collections.namedtuple('gloses', ['data', 'name', 'nb_rep'])
gloses = gloses_collections(dic["data"], dic["glose"], dic["n_occurrences"])

N = len(gloses.data)
print(f"We have {N} different gloses:")
print(' ')
for i in range(N):
    print(f"'{gloses.name[i]}': repeated {gloses.nb_rep[i]} times")
print(' ')





'''
_________________________________________________________________________________________________________________________________________________________________________________________________
    PREMIERE ETAPE : le pre-processing des données, c'est à dire :
    - estimation des dérivées 1ere, 2nd et 3ieme de la trajectoire par une regression polynomiale locale.
    - estimation de la fonction arc-lenght s(t) et de sa dérivées sdot(t) qui correspond à la vitesse curvilinéaire.
    - mise à l'échelle des données : on rescale chaque trajectoire pour qu'elle est une longueur de 1.

    On va construire un tableau "trajs" qui contient à son tour N tableaux de taille n_i (i=1,...,N).
    Chaque élément i,j de trajs (i=1,...,N, j=1,...,n_i) contient une instance de la classe Trajectory (cf fichier trajectory.py).
_________________________________________________________________________________________________________________________________________________________________________________________________
'''

print("estimation of trajectories... \n")

h_min, h_max, nb_h = 0.2, 0.3, 20
trajs = np.empty((N), dtype=object)
for i in range(N):
    ni = gloses.nb_rep[i]
    trajs[i] = np.empty((ni), dtype=object)
    for j in range(ni):
        # on crée pour chaque trajectoire un object "Trajectory" avec comme 1er argument les données et 2ieme le temps.
        trajs[i][j] = Trajectory(gloses.data[i][j], np.linspace(0, 1, len(gloses.data[i][j])))
        # éventuellement, on optimize avec une cross validation le paramètre pour la regression polynomiale locale pour l'estimation les dérivées
        h_opt = opti_loc_poly_traj(trajs[i][j].data, trajs[i][j].t, h_min, h_max, nb_h, n_splits=5)
        # on estime les dérivées avec le paramètre optimal
        trajs[i][j].loc_poly_estimation(trajs[i][j].t, 5, h_opt)
        # on calcul s(t), sdot(t) et on rescale
        trajs[i][j].compute_S(scale=True, factor=1)




'''
_________________________________________________________________________________________________________________________________________________________________________________________________
    DEUXIEME ETAPE : le calcul des repères de Frenet :

    On va construire un tableau "frenet_paths" qui contient à son tour N tableaux de taille n_i (i=1,...,N).
    Chaque élément i,j de frenet_paths (i=1,...,N, j=1,...,n_i) contient une instance de la classe FrenetPath (cf fichier frenet_path.py).

    (Nous avons implémenté plusieurs méthodes pour ça (voir le fichier frenet_path.py et le papier). Ici j'utilise la méthode d'orthonormalization de Gram Schmidt,
     mais je calcul la réciproque de la fonction s_ij(t) avant pour que tous les frenet paths soient évalués sur la même grille (nécessaire pour calculer la moyenne)).
_________________________________________________________________________________________________________________________________________________________________________________________________
'''

print("estimation of frenet paths... \n")

frenet_paths = np.empty((N), dtype=object)
new_s = np.linspace(0, 1, 200)
for i in range(N):
    ni = gloses.nb_rep[i]
    frenet_paths[i] = np.empty((ni), dtype=object)
    for j in range(ni):
        # pour chaque courbe on calcul la fonction réciproque de s(t)
        s_init = trajs[i][j].S(trajs[i][j].t)
        s_init = s_init/s_init[-1]
        inv_s_fct = interpolate.interp1d(s_init, trajs[i][j].t)
        # on calcul le répère de Frenet Serret en fonction de s : Q(s(t))
        frenet_paths[i][j] = trajs[i][j].TNB_GramSchmidt(inv_s_fct(new_s))
        frenet_paths[i][j].grid_obs = new_s
        frenet_paths[i][j].grid_eval = new_s





'''
_________________________________________________________________________________________________________________________________________________________________________________________________
    TROISIEME ETAPE : estimation des paramètres courbure et torsion

    On va construire deux tableaux :

    - "models" qui contient à son tour N tableaux de taille n_i (i=1,...,N).
    Chaque élément i,j de models (i=1,...,N, j=1,...,n_i) contient une instance de la classe Model (cf fichier model_curvatures.py) avec l'estimation de la fonction de courbure
    et de la fonction torsion de la trajectoire i,j.

    - "models_mean" un tableau de taille N dans lequel chaque élément contient une instance de la classe Model correspondant à la courbure moyenne et à la torsion moyenne des
    trajectoires i (j=1,...,n_i).
_________________________________________________________________________________________________________________________________________________________________________________________________
'''


hyperparam = [11, 1e-07, 1e-07]
param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (7, 13), "bounds_lcurv" : (1e-08, 1e-06), "bounds_ltors" :  (1e-08, 1e-06)}
flag_opti = False

print("estimation of curvatures and torsions... \n")

models = np.empty((N), dtype=object)
models_mean = np.empty((N), dtype=object)

for i in range(N):

    # on estime individuellement la courbure et la torsion de chaque courbe
    ni = gloses.nb_rep[i]
    models[i] = np.empty((ni), dtype=object)
    # *
    for j in range(ni):
        _, models[i][j], _ = global_estimation(frenet_paths[i][j], param_model={"nb_basis" : None,
                "domain_range": (np.round(frenet_paths[i][j].grid_obs[0], decimals=8), np.round(frenet_paths[i][j].grid_obs[-1], decimals=8))},
                opt=flag_opti, hyperparam=hyperparam, param_bayopt=param_bayopt, adaptive_h=True)
    # *

    """ ici en commentaire: une version parallélisée de ce qui est au dessus entre les *. """
    # out = Parallel(n_jobs=-1)(delayed(global_estimation)(frenet_paths[i][j], param_model={"nb_basis" : None,
    #         "domain_range": (np.round(frenet_paths[i][j].grid_obs[0], decimals=8), np.round(frenet_paths[i][j].grid_obs[-1], decimals=8))},
    #         opt=flag_opti, hyperparam=hyperparam, param_bayopt=param_bayopt, adaptive_h=True) for j in range(ni))
    # for j in range(ni):
    #     models[i][j] = out[j][1]

    # on estime les courbures et torsions moyennes s'il y a plus d'une répétition de la glose.
    # **
    if ni > 1:
        _, models_mean[i], res_opt = global_estimation(PopulationFrenetPath(frenet_paths[i]), param_model={"nb_basis" : None,
                    "domain_range": (np.round(frenet_paths[i][j].grid_obs[0], decimals=8), np.round(frenet_paths[i][j].grid_obs[-1], decimals=8))},
                    opt=flag_opti, hyperparam=hyperparam, param_bayopt=param_bayopt, alignment=True, lam=100.0, adaptive_h=True)
    else:
        models_mean[i] = models[i][0]
    # **


""" ici en commentaire: une version parallélisée de ce qui est au dessus entre les **. """
# ind_mult = np.where((gloses.nb_rep > 1))
# out = Parallel(n_jobs=-1)(delayed(global_estimation)(PopulationFrenetPath(frenet_paths[i]), param_model={"nb_basis" : None,
#             "domain_range": (np.round(frenet_paths[i][j].grid_obs[0], decimals=8), np.round(frenet_paths[i][j].grid_obs[-1], decimals=8))},
#             opt=flag_opti, hyperparam=hyperparam, param_bayopt=param_bayopt, alignment=True, lam=100.0, adaptive_h=True) for i in ind_mult)
# k = 0
# for i in range(N):
#     if gloses.nb_rep[i]>1:
#         models_mean[i] = out[k][1]
#         k += 1
#     else:
#         models_mean[i] = models[i][0]




'''
_________________________________________________________________________________________________________________________________________________________________________________________________
    QUATRIEME ETAPE : affichage et sauvegarde des données.

    (dans le fichier visu_utils.py il existe pleins d'autres fonctions pour faire des affichages, notament pour afficher les trajectoires en 3D.)
_________________________________________________________________________________________________________________________________________________________________________________________________
'''

# on sauvegarde les données dans un dictionnaire
res_filename = "res_demo_estim_mean_geometry_Mocaplab_data"
dic = {"names" : gloses.name, "nb_rep" : gloses.nb_rep, "trajs" : trajs, "frenet_paths" : frenet_paths, "models" : models, "models_mean" : models_mean}
if os.path.isfile(res_filename):
    print("Le fichier ", res_filename, " existe déjà, les données n'ont pas pu être sauvegardées.")
else:
    fil = open(res_filename,"xb")
    pickle.dump(dic,fil)
    fil.close()


# on affiche le résultat des estimations des courbures et torsions moyennes
s = np.linspace(0, 1, 200)
curv_mean_arr = np.array([models_mean[i].curv.function(s) for i in range(N)])
tors_mean_arr = np.array([models_mean[i].tors.function(s) for i in range(N)])
plot_array_2D(s, curv_mean_arr, '', legend={"index":True, "title": 'mean curvatures', "x axis": 's', "y axis": 'kappa(s)'})
plot_array_2D(s, tors_mean_arr, '', legend={"index":True, "title": 'mean torsions', "x axis": 's', "y axis": 'tau(s)'})
