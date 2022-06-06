import numpy as np
from trajectory import *
from maths_utils import *
from frenet_path import *
from trajectory import *
from model_curvatures import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.plots import plot_convergence
from joblib import Parallel, delayed
from timeit import default_timer as timer
from tqdm import tqdm


def opti_loc_poly_traj(data_traj, t, minh, maxh, nb_h, n_splits=10):
    """
    Find the optimal parameter h to estimate the derivatives with local polynomial regression
    ...
    """
    HH = np.linspace(minh,maxh,nb_h)
    err_h = np.zeros(len(HH))
    kf = KFold(n_splits=n_splits, shuffle=False)
    for j in range(len(HH)):
        err = []
        for train_index, test_index in kf.split(t):
            t_train, t_test = t[train_index], t[test_index]
            data_train, data_test = data_traj[train_index,:], data_traj[test_index,:]
            X_train = Trajectory(data_train, t_train)
            X_train.loc_poly_estimation(t_test, 5, HH[j])
            dX0 = X_train.derivatives[:,0:3]
            diff = dX0 - data_test
            err.append(np.linalg.norm(diff)**2)
        err_h[j] = np.mean(err)
    if isinstance(np.where(err_h==np.min(err_h)), int):
        h_opt = HH[np.where(err_h==np.min(err_h))]
    else:
        h_opt = HH[np.where(err_h==np.min(err_h))][0]
    return h_opt


def bayesian_optimisation(func, n_call, hyperparam_bounds, plot=True):
    """
    Do a bayesian optimisation and return the optimal parameter (h, lambda1, lambda2)
    ...
    """
    res = gp_minimize(func,               # the function to minimize
                    hyperparam_bounds,    # the bounds on each dimension of x
                    acq_func="EI",        # the acquisition function
                    n_calls=n_call,       # the number of evaluations of f
                    n_random_starts=2,    # the number of random initialization points
                    random_state=1,       # the random seed
                    n_jobs=-1,            # use all the cores for parallel calculation
                    verbose=True)
    x = res.x
    #if plot==True:
       # figure = plot_convergence(res).figure
        # figure.show()
    print('the optimal hyperparameters selected are: ', x)
    return x


def gridsearch_optimisation(func, hyperparam_list):

    # grid = create_hyperparam_grid(hyperparam_list)
    grid = hyperparam_list
    n_grid = grid.shape[0]

    print('Begin grid search optimisation with', n_grid, 'combinations of parameters...')

    # out = []
    # for i in range(n_grid):
    #     print('Iteration :', i, 'with parameters ', grid[i])
    #     out.append(func(grid[i]))
    #     print('Score :', out[i])

    def parallel_func(f, param, i):
        # print('Iteration :', i, 'with parameters ', param)
        cv = f(param)
        # print('Cross validation score :', cv)
        return cv

    out = Parallel(n_jobs=-1)(delayed(parallel_func)(func, grid[i], i) for i in tqdm(range(n_grid)))

    ind = np.where([out[i]==np.min(out, axis=0) for i in range(n_grid)])[0]
    if len(ind)!=1:
        ind = ind[0]

    res = grid[ind]
    print('End of grid search optimisation. The optimal parameters are :', res)

    return res
