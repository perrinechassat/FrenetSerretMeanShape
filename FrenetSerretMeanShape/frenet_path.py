import numpy as np
from scipy.integrate import solve_ivp
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.riemannian_metric import RiemannianMetric
import time
import multiprocessing
import functools

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try:
                return async_result.get(timeout)
            except multiprocessing.TimeoutError:
                return
        return inner
    return decorator


class FrenetPath:

    """
    A class used to represent a Frenet Path T,N,B.
    Could be initialized with grids and data or with grids, initial condition Q0, curvature function and torsion function.

    ...

    Attributes
    ----------
    grid_obs : numpy array of shape (N1) that contained the observation grid
    grid_eval : numpy array of shape (N2) that contained the evaluation grid
    nb_grid_eval : number of point in grid_eval
    init : numpy array (3,3), initial condition matrix
    data : numpy array of shape (N,3,3) of TNB
    length : max of grid obs
    curv : function, curvature
    tors : function, torsion
    nb_samples : 1, number of frenet path
    neighbor_obs : array, index of observations in the neighborhood of grid_eval
    weight : array, weight at grid_eval
    grid_double : array
    delta : array
    data_trajectory : numpy array of shape (N,3) of the corresponding trajectory

    Methods
    -------
    set_estimate_theta(curv, tors):
        set the parameter curv and tors to the values in argument

    compute_neighbors(h):
        compute the value neighbor_obs, weight, grid_double, delta

    frenet_serret_solve(Q0=None, t_span=None, t_eval=None):
        solve the Frenet Serret ODE with the initial condition Q0.
    """

    def __init__(self, grid_obs, grid_eval, init=None, data=None, curv=None, tors=None, dim=3):
        self.dim = dim
        self.length = np.max(grid_obs)
        self.nb_data = len(grid_obs)
        if data is None:
            self.data = np.zeros((dim,dim,self.nb_data))
        else:
            self.data = data
        self.grid_obs = grid_obs
        self.grid_eval = grid_eval
        self.nb_grid_eval = len(grid_eval)
        self.curv = curv # function
        self.tors = tors # function
        self.init = init
        self.nb_samples = 1

    def set_estimate_theta(self, curv, tors):
        self.curv = curv
        self.tors = tors

    def compute_neighbors(self, h):
        Kern = lambda x: (3/4)*(1-np.power(x,2))

        neighbor_obs = []
        weight = []
        grid_double = []
        delta = []
        val_min = np.min(self.grid_obs)
        val_max = np.max(self.grid_obs)
        for q in range(self.nb_grid_eval):
            t_q = self.grid_eval[q]
            if t_q-val_min < h and q!=0:
                h_bis = np.abs(t_q-val_min) + 10e-10
                neighbor_obs.append(np.where(abs(self.grid_obs - t_q) <= h_bis)[0])
                weight.append((1/h)*Kern((t_q - self.grid_obs[neighbor_obs[q]])/h))
                grid_double.append((t_q + self.grid_obs[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - self.grid_obs[neighbor_obs[q]])
            elif val_max-t_q < h and q!=self.nb_grid_eval-1:
                h_bis = np.abs(val_max-t_q) + 10e-10
                neighbor_obs.append(np.where(abs(self.grid_obs - t_q) <= h_bis)[0])
                weight.append((1/h)*Kern((t_q - self.grid_obs[neighbor_obs[q]])/h))
                grid_double.append((t_q + self.grid_obs[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - self.grid_obs[neighbor_obs[q]])
            elif q==0:
                neighbor_obs.append(np.array([0,1]))
                weight.append((1/h)*Kern((t_q - self.grid_obs[neighbor_obs[q]])/h))
                grid_double.append((t_q + self.grid_obs[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - self.grid_obs[neighbor_obs[q]])
            elif q==self.nb_grid_eval-1:
                neighbor_obs.append(np.array([len(self.grid_obs)-2,len(self.grid_obs)-1]))
                weight.append((1/h)*Kern((t_q - self.grid_obs[neighbor_obs[q]])/h))
                grid_double.append((t_q + self.grid_obs[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - self.grid_obs[neighbor_obs[q]])
            else:
                neighbor_obs.append(np.where(abs(self.grid_obs - t_q) <= h)[0]) # index of observations in the neighborhood of t_q
                weight.append((1/h)*Kern((t_q - self.grid_obs[neighbor_obs[q]])/h)) # K_h(t_q-s_j)
                grid_double.append((t_q + self.grid_obs[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - self.grid_obs[neighbor_obs[q]])  # t_q-s_j
        self.neighbor_obs = np.squeeze(neighbor_obs)
        self.weight = np.squeeze(np.asarray(weight))
        self.grid_double = np.squeeze(np.asarray(grid_double))
        self.delta = np.squeeze(np.asarray(delta))


    # @with_timeout(300)
    def frenet_serret_solve(self, Q0=None, t_span=None, t_eval=None):
        """
        FrenetSerretSolve
        Solve Serret-Frenet ODE and compute the Path of Frenet frame and the curve corresponding
        to a given curvature (curv) and torsion (torsion) functions.
        U : 3*3*N ([TNB])
        X : N*3 (State = Integrated Tangent)
        """
        if Q0 is None:
            Q0 = self.init
        if t_span is None:
            t_span = (self.grid_obs[0], self.grid_obs[-1])
        if t_eval is None:
            t_eval = self.grid_eval
        p = np.shape(Q0)[0]
        if p!=self.dim:
            raise ValueError("Wrong dimension of the initial condition.")
        if self.curv==None or self.tors==None:
            raise ValueError("Set first the function curv and torsion.")

        SO3 = SpecialOrthogonal(3)

        h = lambda t: [self.curv(t), self.tors(t)]
        F = lambda t: np.diag(h(t),1) - np.diag(h(t),-1)
        A22 = lambda t: np.kron(F(t), np.eye(self.dim))
        A11 = np.zeros((self.dim,self.dim))
        A21 = np.zeros((self.dim*self.dim,self.dim))
        A12 = np.concatenate((np.eye(self.dim), np.zeros((self.dim,self.dim*(self.dim-1)))), axis=1)
        Az  = lambda t: np.concatenate((np.concatenate((A11, A12), axis=1), np.concatenate((A21, A22(t)), axis=1)))

        X0  = [0,0,0]
        Z0  = np.concatenate((X0, Q0[:,0], Q0[:,1], Q0[:,2]))
        ode_func = lambda t,z: np.matmul(Az(t),z)
        sol = solve_ivp(ode_func, t_span=t_span, y0=Z0, t_eval=t_eval)
        Z = sol.y
        X = Z[0:p,:] # Integration of tangent X(t)=X0+int_0^t T(s)ds
        self.data_trajectory = np.transpose(X)
        self.data[:,0,:] = Z[p:2*p,:]   # Tangent
        self.data[:,1,:] = Z[2*p:3*p,:] # Normal
        self.data[:,2,:] = Z[3*p:4*p,:] # Binormal



class PopulationFrenetPath:
    """
    A class used to represent a Population of Frenet Paths.

    ...

    Attributes
    ----------
    grids_obs : list of numpy array, list of the observation grid of each Frenet path.
    grids_eval : list of numpy array, list of the evaluation grid of each Frenet path.
    frenet_paths : list of instance of FrenetPath
    data : list of numpy array of shape (nb_samples,N_i,3,3) of TNB
    nb_samples : number of frenet paths
    mean_curv : function, mean curvature of all frenet paths
    mean_tors : function, mean torsion of all frenet paths
    gam : array of functions, warping functions between each curvatures and torsions of the different frenet paths

    Methods
    -------
    compute_neighbors(h):
        compute neighbors of each frenet path in the Population

    set_estimate_theta(mean_curv, mean_tors):
        set the parameter mean_curv and mean_tors to the values in argument

    set_gam_functions(gam):
        set the parameter gam to the value in argument

    """

    def __init__(self, popFrenetPaths):
        self.nb_samples = len(popFrenetPaths)
        self.frenet_paths = popFrenetPaths
        self.data = [popFrenetPaths[i].data for i in range(self.nb_samples)]
        grids_obs = [popFrenetPaths[i].grid_obs for i in range(self.nb_samples)]
        self.grids_obs = grids_obs
        grids_eval = [popFrenetPaths[i].grid_eval for i in range(self.nb_samples)]
        self.grids_eval = grids_eval
        self.dim = popFrenetPaths[0].dim

    def compute_neighbors(self, h):
        for i in range(self.nb_samples):
            self.frenet_paths[i].compute_neighbors(h)

    def set_estimate_theta(self, mean_curv, mean_tors):
        self.mean_curv = mean_curv
        self.mean_tors = mean_tors

    def set_gam_functions(self, gam):
        self.gam = gam
