import numpy as np

def get_X0(P0, dim):
    K = P0[:dim,:dim]
    a = P0[dim,:dim]
    c = P0[:dim,dim]
    b = P0[dim,dim]
    return -np.linalg.inv(K+K.T) @ (a+c.T)


def state_space_model(A, z_t_minus_1, B, u_t_minus_1):
    """
    Calculates the state at time t given the state at time t-1 and
    the control inputs applied at time t-1
    """
    state_estimate_t = (A @ z_t_minus_1) + (B @ u_t_minus_1)
    return state_estimate_t


def tracking(X0, Q, R, M, B, N, dim):
    P = [None] * (N + 1)
    Qf = Q[:,:,N]
    P[N] = Qf
    for i in range(N, 0, -1):
        # Discrete-time Algebraic Riccati equation to calculate the optimal state cost matrix
        P[i-1] = Q[:,:,i-1] + M[:,:,i-1].T @ P[i] @ M[:,:,i-1] - (M[:,:,i-1].T @ P[i] @ B[:,:,i-1]) @ np.linalg.inv(R[:,:,i-1] + B[:,:,i-1].T @ P[i] @ B[:,:,i-1]) @ (B[:,:,i-1].T @ P[i] @ M[:,:,i-1])

    # Create a list of N elements
    K = [None] * N
    u = [None] * N
    z = [None] * (N+1)
    if X0 is None:
        X0 = get_X0(P[0],3)
        z[0] = np.concatenate((X0,np.squeeze(np.eye(dim),axis=0)),axis=0)
        # print(X0)
    else:
        z[0] = np.concatenate((X0,np.eye(dim)),axis=0)
    # z[0] = np.concatenate((X0,np.array([1])))
    for i in range(0,N):
        # Calculate the optimal feedback gain K
        K[i] = np.linalg.inv(R[:,:,i] + B[:,:,i].T @ P[i+1] @ B[:,:,i]) @ B[:,:,i].T @ P[i+1] @ M[:,:,i]
        u[i] = -K[i] @ z[i]
        # print(u[i])
        z[i+1] = state_space_model(M[:,:,i], z[i], B[:,:,i], u[i])

    Z = np.array(z)
    return np.array(u), Z, np.array(K), np.array(P)
