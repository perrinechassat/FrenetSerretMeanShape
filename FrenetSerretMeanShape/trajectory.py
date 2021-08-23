import numpy as np
from scipy import interpolate, optimize
from scipy.integrate import cumtrapz
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from frenet_path import *

Kern = lambda x: (3/4)*(1-np.power(x,2))*(np.abs(x)<1)
Kern_bis = lambda x,delta: np.power((1 - np.power((np.abs(x)/delta),3)), 3)


class Trajectory:
    """
    A class used to represent a 3D curve.

    ...

    Attributes
    ----------
    data : numpy array of shape (N,3) that contained the coordinates of the curve
    t : numpy array of shape N, time of each point in data, supposed to be croissant
    dim : 3
    t0 : float, initial time value
    tmax : float, final time value
    scale : Boolean, True if the curve is scale, False otherwise
    dX1 : function, estimated first derivative
    dX2 : function, estimated second derivative
    dX3 : function, estimated third derivative
    S : function, estimated of the arclenght function
    Sdot : function, estimated derivative of the arclenght function
    L : float, estimated lenght of the curve
    curv_extrins : function, extrinsic estimates of the curvature
    tors_extrins : function, extrinsic estimates of the torsion

    Methods
    -------
    loc_poly_estimation(t_out, deg, h):
        estimation of derivatives using local polynomial regression with parameters "h" and degree "deg" evaluted of the grid "t_out"

    compute_S(scale=False):
        compute the arclenght function, the lenght of the curve and scale it if "scale" equals True.

    scale():
        scale the curve, needs to have run compute_S before.

    TNB_GramSchmidt(t):
        compute the T,N,B frame of the curve from the pointwise estimated derivatives (of Higher Order : 1,2,3) by Gram-Schmidt Orthonormalization on t
        return: instance of class FrenetPath

    theta_extrinsic_formula(t):
        compute the curvature and torsion functions from the pointwise estimated derivatives (of Higher Order : 1,2,3) computed by the classical formulas
        BECAREFUL very unstable (numerically ill-posed).
        return: pointwise estimate of curvature, pointwise estimate of torsion

    TNB_locPolyReg(grid_in, grid_out, h, p=3, iflag=[1,1], ibound=0, local=True):
        TNB estimates based on constrained local polynomial regression |T|=1, <T,N>=0
        b0 + b1(t-t_0)+b2(t-t0)^2/2 + b3(t-t0)^3/6 + ... + bp(t-t0)^p/p!, |b1|=1, <b1,b2>=0
        minimize (Y-XB)'W(Y-XB) -la*(|b1|^2-1) - mu(2*<b1,b2>)
        inputs:
           grid_in - input grid
           grid_out - output grid
           h     - scalar
           p     - degree of polynomial (defaul = 3)
           iflag - [1,1] for both constraints, [1,0] for |b1|=1, [0,1] for <b1,b2>=0
           ibound - 1 for boundary correction, 0 by default
           local - True for local version, False for regular version
        return:
            Q - instance of class FrenetPath
            kappa   - [kappa, kappap, tau]
            Param - estimates with constraints
            Param0 - estimates without constraints
            vparam  - [la, mu, vla, vmu] tuning parameters
                      [la, mu]: optimal values amongst vla, and vmu
            success - True if a solution was found for all point, False otherwise
    """

    def __init__(self, data, t):
        self.t = t
        self.data = data
        self.dim = data.shape[1]
        self.t0 = np.min(t)
        self.tmax = np.max(t)
        self.scale = False

    def loc_poly_estimation(self, t_out, deg, h):
        pre_process = PolynomialFeatures(degree=deg)
        deriv_estim = np.zeros((len(t_out),(deg+1)*self.dim))
        for i in range(len(t_out)):
            T = self.t - t_out[i]
            # print(T)
            W = Kern(T/h)
            # print(W)
            T_poly = pre_process.fit_transform(T.reshape(-1,1))
            for j in range(deg+1):
                T_poly[:,j] = T_poly[:,j]/np.math.factorial(j)
            pr_model = LinearRegression(fit_intercept = False)
            pr_model.fit(T_poly, self.data, W)
            B = pr_model.coef_
            deriv_estim[i,:] = B.reshape(1,(deg+1)*self.dim, order='F')
        self.derivatives = deriv_estim
        def dx1(t): return interpolate.griddata(self.t, deriv_estim[:,3:6], t, method='cubic')
        self.dX1 = dx1
        def dx2(t): return interpolate.griddata(self.t, deriv_estim[:,6:9], t, method='cubic')
        self.dX2 = dx2
        def dx3(t): return interpolate.griddata(self.t, deriv_estim[:,9:12], t, method='cubic')
        self.dX3 = dx3

    def compute_S(self, scale=False):
        def Sdot_fun(t): return np.linalg.norm(self.dX1(t), axis=1)
        self.Sdot = Sdot_fun
        def S_fun(t): return cumtrapz(self.Sdot(t), t, initial=0)
        self.L = S_fun(self.t)[-1]
        # print(self.L)
        if scale==True:
            self.scale = True
            def S_fun_scale(t): return cumtrapz(self.Sdot(t), t, initial=0)/self.L
            self.S = S_fun_scale
            self.data = self.data/self.L
        else:
            self.S = S_fun

    def scale(self):
        self.scale = True
        def S_fun_scale(t): return cumtrapz(self.Sdot(t), t, initial=0)/self.L
        self.S = S_fun_scale
        self.data = self.data/self.L


    def TNB_GramSchmidt(self, t_grid):

        def GramSchmidt(DX1, DX2, DX3):
            normdX1 = np.linalg.norm(DX1)
            normdX2 = np.linalg.norm(DX2)
            normdX3 = np.linalg.norm(DX3)
            T = DX1/normdX1
            N = DX2 - np.dot(np.transpose(T),DX2)*T
            N = N/np.linalg.norm(N)
            B = DX3 - np.dot(np.transpose(N),DX3)*N - np.dot(np.transpose(T),DX3)*T
            B = B/np.linalg.norm(B)
            Q = np.stack((T, N, B))
            if np.linalg.det(Q)<0:
                B = -B
                Q = np.stack((T, N, B))
            return np.transpose(Q)

        dX1 = self.dX1(t_grid)
        dX2 = self.dX2(t_grid)
        dX3 = self.dX3(t_grid)
        nb_t = len(t_grid)
        Q = np.zeros((self.dim, self.dim, nb_t))

        for i in range(nb_t):
            Qi = GramSchmidt(dX1[i,:],dX2[i,:],dX3[i,:])
            Q[:,:,i]= Qi
        Q_fin = FrenetPath(self.S(t_grid), self.S(t_grid), data=Q)
        return Q_fin


    def theta_extrinsic_formula(self, t_grid):

        dX1 = self.dX1(t_grid)
        dX2 = self.dX2(t_grid)
        dX3 = self.dX3(t_grid)
        nb_t = len(t_grid)

        crossvect = np.zeros(dX1.shape)
        norm_crossvect = np.zeros(nb_t)
        curv = np.zeros(nb_t)
        tors = np.zeros(nb_t)
        for t in range(nb_t):
            crossvect[t,:]    = np.cross(dX1[t,:],dX2[t,:])
            norm_crossvect[t] = np.linalg.norm(crossvect[t,:],1)
            curv[t]= norm_crossvect[t]/np.power(np.linalg.norm(dX1[t,:]),3)
            tors[t]= (np.dot(crossvect[t,:],np.transpose(dX3[t,:])))/(norm_crossvect[t]**2)

        if self.scale==True:
            curv = curv*self.L
            tors = tors*self.L

        def curv_extrins_fct(s): return interpolate.interp1d(self.S(t_grid), curv)(s)
        def tors_extrins_fct(s): return interpolate.interp1d(self.S(t_grid), tors)(s)

        self.curv_extrins = curv_extrins_fct
        self.tors_extrins = tors_extrins_fct

        return curv, tors


    def TNB_locPolyReg(self, grid_in, grid_out, h, p=3, iflag=[1,1], ibound=0, local=True):

        (n,d) = self.data.shape
        nout = len(grid_out)
        s0 = np.min(grid_in)
        smax = np.max(grid_in)

        if ibound>0:
            # bandwidth correction at the boundary
            hvec = h + np.maximum(np.maximum(s0 - (grid_out-h), (grid_out+h) - smax),np.zeros(nout))
        else:
            hvec = h*np.ones(nout)

        Param0 = np.zeros((nout,(p+1)*self.dim))
        Param = np.zeros((nout,(p+1)*self.dim))
        vparam = np.zeros((nout,2))

        U = np.zeros((p+1,p+1))
        U[1,1] = 1
        V = np.zeros((p+1,p+1))
        V[1,2] = 1
        V[2,1] = 1
        list_error = []

        for i in range(nout):
            t_out = grid_out[i]
            if local==True:
                ik = np.sort(np.argsort(abs(grid_in - t_out))[:h])
            else:
                h = hvec[i]
                lo = np.maximum(s0, grid_out[i]-h)
                up = np.minimum(grid_out[i] + h, smax)
                ik = np.intersect1d(np.where((grid_in>=lo)), np.where((grid_in<=up)))

            tti = grid_in[ik]
            ni = len(tti)
            Yi = self.data[ik,:]   # ni x 3

            if local==True:
                delta = 1.0001*np.maximum(tti[-1]-grid_out[i], grid_out[i]-tti[0])
                K = Kern_bis(tti-grid_out[i],delta)
            else:
                K = Kern((tti-grid_out[i])/h)
            Wi = np.diag(K)
            Xi = np.ones((ni,p+1))
            # Ci = [1]
            for ip in range(1,p+1):
                Xi[:,ip] = np.power((tti-grid_out[i]),ip)/np.math.factorial(ip)  # ni x (p+1)
                # Ci += [1/np.math.factorial(ip)]

            Si = Xi.T @ Wi @ Xi # p+1 x p+1
            Ti = Xi.T @ Wi @ Yi # p+1 x 3

            # estimates without constraints
            # B0i = np.linalg.solve(Si,Ti) # (p+1) x 3
            B0i = np.linalg.inv(Si) @ Ti
            # B0i = np.diag(Ci) @ B0i
            Param0[i,:] = np.reshape(B0i,(1,(p+1)*d))

            # estimates with constraints
            if p==1: # local linear
                tb0 = np.array([-Si[0,1], Si[0,0]]) @ Ti
                la_m = (np.linalg.det(Si) - np.linalg.norm(tb0))/Si[0,0]
                vparam[i,:] = np.array([la_m,0])
                Param[i,:] = np.reshape(np.linalg.solve(Si-la_m*np.array([[0,0],[0,1]]), Ti),(1,(p+1)*d))

            elif p>1:
                la0 = 0
                mu0 = 0
                # tol = 1e-4
                param0 = np.array([la0,mu0])

                res = optimize.root(fun=GetLocParam, x0=param0, args=(Si, Ti), method='hybr')
                parami = res.x
                itr = 0
                epsilon_vect = np.array([10e-6,10e-6])
                while res.success==False and itr<30:
                    parami += epsilon_vect
                    res = optimize.root(fun=GetLocParam, x0=parami, args=(Si, Ti), method='hybr')
                    parami = res.x
                    itr += 1
                if res.success==False:
                    list_error.append(i)

                la0 = parami[0]
                mu0 = parami[1]
                Bi = np.linalg.inv(Si-la0*U-mu0*V) @ Ti
                vparam[i,:] = parami
                Param[i,:] = np.reshape(Bi,(1,(p+1)*d))


        # output
        Gamma = Param[:,:3]
        T = Param[:,3:6]
        if (p>1):
            Ntilde = Param[:,6:9]
            kappa = np.sqrt(np.sum(np.power(Ntilde,2),1))
            N = np.diag(1/kappa)@Ntilde
            Bi = np.cross(T, N)
        if (p>2):
            kappap = np.empty((nout))
            kappap[:] = np.nan
            tau = np.empty((nout))
            tau[:] = np.nan
            for i in range(nout):
                x = np.linalg.solve([T[i,:].T, N[i,:].T, Bi[i,:].T],Param[i,9:12].T)
                # theoretically : x(1) = -kappa^2 ; x(2)= kappap; x(3)= kappa*tau;
                kappap[i] = x[1]
                tau[i] = x[2]/kappa[i]

        vkappa = [kappa, kappap, tau]

        Q = np.zeros((self.dim, self.dim, nout))
        Q[:,0,:] = np.transpose(T)
        Q[:,1,:] = np.transpose(N)
        Q[:,2,:] = np.transpose(Bi)
        Q_fin = FrenetPath(grid_out, grid_out, data=Q)
        Q_fin.data_trajectory = Gamma

        success = True
        if len(list_error) > 0:
            print(list_error)
            success = False

        return Q_fin, vkappa, Param, Param0, vparam, success



    def TNB_locPolyReg_2(self, grid_in, grid_out, h, p=3, iflag=[1,1], ibound=0, local=True):
        ''' Does: TNB estimates based on constrained local polynomial regression |T|=1, <T,N>=0
                 b0 + b1(t-t_0)+b2(t-t0)^2/2 + b3(t-t0)^3/6 + ... + bp(t-t0)^p/p!
                 |b1|=1, <b1,b2>=0
             minimize (Y-XB)'W(Y-XB) -la*(|b1|^2-1) - mu(2*<b1,b2>)
             Inputs:
               vtout - output grid, length(vtout)=nout
               Y     - J x 3 matrix
               vt    - input grid
               h     - scalar
               p     - degree of polynomial (defaul = 3)
               iflag - [1,1] for both constraints
                   [1,0] for |b1|=1
                   [0,1] for <b1,b2>=0
               ibound - 1 for boundary correction
                        0 by default
               t_rng - [t0, tmax]
            Outputs:
               tvecout - output grid on [0,1]
               Gamma   - Shape (function - order 0) : nout x 3
               T       - Tangent                    : nout x 3
               N       - Normal                     : nout x 3
               Bi      - Binormal                   : nout x 3
               kappa   - [kappa, kappap, tau]       : nout x 3
               Param - estimates with constraints
               Param0 - estimates without constraints
               --------
               < outputs from OrthoNormCon.m >
               vparam  - [la, mu, vla, vmu] tuning parameters: nout x 6
                         [la, mu]: optimal values amongst vla, and vmu
        '''

        (n,d) = self.data.shape
        nout = len(grid_out)
        s0 = np.min(grid_in)
        smax = np.max(grid_in)

        if ibound>0:
            # bandwidth correction at the boundary
            hvec = h + np.maximum(np.maximum(s0 - (grid_out-h), (grid_out+h) - smax),np.zeros(nout))
        else:
            hvec = h*np.ones(nout)

        Param0 = np.zeros((nout,(p+1)*self.dim))
        Param = np.zeros((nout,(p+1)*self.dim))

        U = np.zeros((p+1,p+1))
        U[0,0] = 1
        V = np.zeros((p+1,p+1))
        V[0,1] = 1
        V[1,0] = 1
        W = np.zeros((p+1,p+1))
        W[0,2] = 1
        W[2,0] = 1
        W[1,1] = 1
        list_error = []

        for i in range(nout):
            t_out = grid_out[i]
            if local==True:
                ik = np.sort(np.argsort(abs(grid_in - t_out))[:h])
            else:
                h = hvec[i]
                lo = np.maximum(s0, grid_out[i]-h)
                up = np.minimum(grid_out[i] + h, smax)
                ik = np.intersect1d(np.where((grid_in>=lo)), np.where((grid_in<=up)))

            tti = grid_in[ik]
            ni = len(tti)
            Yi = self.data[ik,:]   # ni x 3

            if local==True:
                delta = 1.0001*np.maximum(tti[-1]-grid_out[i], grid_out[i]-tti[0])
                K = Kern_bis(tti-grid_out[i],delta)
            else:
                K = Kern((tti-grid_out[i])/h)
            Wi = np.diag(K)
            Xi = np.ones((ni,p+1))
            for ip in range(1,p+1):
                Xi[:,ip-1] = np.power((tti-grid_out[i]),ip)/np.math.factorial(ip)  # ni x (p+1)

            Si = Xi.T @ Wi @ Xi # p+1 x p+1
            Ti = Xi.T @ Wi @ Yi # p+1 x 3

            # estimates without constraints
            B0i_1 = np.linalg.solve(Si,Ti) # (p+1) x 3
            B0i = np.zeros(B0i_1.shape)
            B0i[0,:] = B0i_1[-1,:]
            B0i[1:,:] = B0i[:-1,:]
            Param0[i,:] = np.reshape(B0i,(1,(p+1)*d))

            # estimates with constraints
            if p==1: # local linear
                tb0 = np.array([-Si[0,1], Si[0,0]]) @ Ti
                la_p = (np.linalg.det(Si) + np.linalg.norm(tb0))/Si[0,0]
                la_m = (np.linalg.det(Si) - np.linalg.norm(tb0))/Si[0,0]
                Param[i,:] = np.reshape(np.linalg.solve(Si-la_m*np.array([[0,0],[0,1]]), Ti),(1,(p+1)*d))

            elif p>1:
                param0 = np.array([0,0,0])

                res = optimize.root(fun=GetLocParam3, x0=param0, args=(Si, Ti), method='hybr')
                parami = res.x
                itr = 0
                # epsilon_vect = np.array([10e-5,10e-5,10e-5])
                while res.success==False and itr<30:
                    # parami += epsilon_vect
                    res = optimize.root(fun=GetLocParam3, x0=parami, args=(Si, Ti), method='hybr')
                    parami = res.x
                    itr += 1
                if itr!=0:
                    print('LocPolyTNB 2')
                    print(itr)
                    print(res.success)

                la0 = parami[0]
                mu0 = parami[1]

                Bi_1 = np.linalg.inv(Si-parami[0]*U-parami[1]*V-parami[2]*W) @ Ti
                Bi = np.zeros(Bi_1.shape)
                Bi[0,:] = Bi_1[-1,:]
                Bi[1:,:] = Bi_1[:-1,:]
                Param[i,:] = np.reshape(Bi,(1,(p+1)*d))


        # output
        Gamma = Param[:,:3]
        T = Param[:,3:6]
        if (p>1):
            Ntilde = Param[:,6:9]
            kappa = np.sqrt(np.sum(np.power(Ntilde,2),1))
            N = np.diag(1/kappa)@Ntilde
            Bi = np.cross(T, N)
        if (p>2):
            kappap = np.empty((nout))
            kappap[:] = np.nan
            tau = np.empty((nout))
            tau[:] = np.nan
            for i in range(nout):
                x = np.linalg.solve([T[i,:].T, N[i,:].T, Bi[i,:].T],Param[i,9:12].T)
                kappap[i] = x[1]
                tau[i] = x[2]/kappa[i]

        vkappa = [kappa, kappap, tau]

        Q = np.zeros((self.dim, self.dim, nout))
        Q[:,0,:] = np.transpose(T)
        Q[:,1,:] = np.transpose(N)
        Q[:,2,:] = np.transpose(Bi)
        Q_fin = FrenetPath(grid_out, grid_out, data=Q)
        Q_fin.data_trajectory = Gamma

        if len(list_error) > 0:
            print(list_error)

        return Q_fin, vkappa, Param, Param0, Param


def GetLocParam(param,S,T):
    #  param - 1 x 2 vector
    #  S  - pp x pp
    #  T  - pp x d

    pp = S.shape[0]
    U = np.zeros((pp,pp))
    U[1,1] = 1
    V = np.zeros((pp,pp))
    V[1,2] = 1
    V[2,1] = 1
    B = np.linalg.inv(S-param[0]*U-param[1]*V) @ T
    # B = np.linalg.solve(S-param[0]*U-param[1]*V,T)
    out = [B[1,:] @ B[1,:].T - 1, B[1,:] @ B[2,:].T]
    return out

def GetLocParam3(param,S,T):
#      param - 1 x 3 vector
#      S  - pp x pp
#      T  - pp x d
#
    pp = S.shape[0]
    U = np.zeros((pp,pp))
    U[0,0] = 1
    V = np.zeros((pp,pp))
    V[0,1] = 1
    V[1,0] = 1
    W = np.zeros((pp,pp))
    W[0,2] = 1
    W[2,0] = 1
    W[1,1] = 2
    B = np.linalg.inv(S-param[0]*U-param[1]*V-param[2]*W) @ T
    out = [B[0,:]@B[0,:].T - 1, B[0,:]@B[1,:].T, B[0,:]@B[2,:].T + B[1,:]@B[1,:].T]
    return out

def laInv(S,T,mu):
#  constrained 3-dim inverse problem
# (S3 - lam*U3 - mu*V3)B3 = T3

    U = np.zeros((3,3))
    U[1,1] = 1
    V = np.zeros((3,3))
    V[1,2] = 1
    V[2,1] = 1
    # given mu
    P = S - mu*V
    Del = np.linalg.det(P)
    aP = adj(P)
    B0 = aP @ T
    b0 = B0[1,:]
    lap = (Del + np.linalg.norm(b0))/aP[1,1]
    lam = (Del - np.linalg.norm(b0))/aP[1,1]
    la = [lap, lam]
    return la


def adj(matrix):
    return (np.linalg.inv(matrix).T * np.linalg.det(matrix)).transpose()
