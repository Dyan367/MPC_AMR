import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from utils import gen_prediction_matrices, gen_cost_matrices, gen_constraint_matrices
from qpsolvers import solve_qp
import polytope as pc
import scipy.linalg as la

def ellipsoid_to_polytope(P, c, num_directions=100):
    """
    Converts an ellipsoidal terminal set {x | xᵀ P x <= c} into a polyhedral approximation.
    Returns a polytope in H-representation: {x | A x <= b}
    """
    n = P.shape[0]
    directions = np.random.randn(num_directions, n)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    A = []
    b = []
    sqrtP_inv = la.sqrtm(la.inv(P))

    for d in directions:
        x_boundary = np.sqrt(c) * la.solve(sqrtP_inv, d)
        A.append(d)
        b.append(np.dot(d, x_boundary))

    A = np.array(A)
    b = np.array(b)
    return pc.Polytope(A, b)

def find_terminal_set_outer_approx(P, K, u_bounds, num_samples=5000):
    """
    Outer ellipsoidal approximation: find max c s.t. u = -Kx lies in input bounds
    """
    n = P.shape[0]
    dim_u = K.shape[0]

    X_unit = np.random.randn(num_samples, n)
    X_unit = X_unit / np.linalg.norm(X_unit, axis=1, keepdims=True)  # on unit sphere

    c_list = []
    for x in X_unit:
        Kx = K @ x
        scaling_factors = []

        for i in range(dim_u):
            if Kx[i] != 0:
                lb = u_bounds[0] / Kx[i] if Kx[i] > 0 else u_bounds[1] / Kx[i]
                ub = u_bounds[1] / Kx[i] if Kx[i] > 0 else u_bounds[0] / Kx[i]
                s = min(abs(lb), abs(ub))
                scaling_factors.append(s)
            else:
                scaling_factors.append(np.inf)

        s_min = min(scaling_factors)
        x_scaled = x * s_min
        c_val = x_scaled.T @ P @ x_scaled
        c_list.append(c_val)

    c_max = min(c_list)
    return c_max

def solve_condensed_mpc(
    x0, A, B, Q, R, P, N,
    u_lb, u_ub,
    D=None, c_lb=None, c_ub=None,
    with_terminal_constraint=False,
    terminal_set=None
):
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    print("\n--- Solving MPC Problem ---")
    print(f"Initial state x0: {x0}")

    T, S = gen_prediction_matrices(A, B, N)
    H, h = gen_cost_matrices(Q, R, P, T, S, x0, N)
    print(f"Cost matrices: H shape = {H.shape}, h shape = {h.shape}")

    G, g = gen_constraint_matrices(x0, A, B, T, S, N, u_lb, u_ub, D, c_lb, c_ub)
    print(f"Stage constraints: G shape = {G.shape}, g shape = {g.shape}")

    Gf = None
    gf = None

    if with_terminal_constraint:
        print("Terminal constraint is enabled.")
        if terminal_set is not None:
            try:
                F = terminal_set.A
                f = terminal_set.b

                T_N = T[-dim_x:, :]
                S_N = S[-dim_x:, :]

                Gf = F @ S_N
                gf = f - F @ (T_N @ x0)

                print("Terminal set constraint applied.")
                print(f"Terminal set: F shape = {F.shape}, f shape = {f.shape}")
                print(f"Transformed terminal constraint: Gf shape = {Gf.shape}, gf shape = {gf.shape}")
            except Exception as e:
                print(f"[Warning] Failed to apply terminal set: {e}")
        else:
            Gf = S[-dim_x:, :]
            gf = -T[-dim_x:, :] @ x0
            print("No terminal set provided → using default terminal constraint to origin.")

    try:
        H += 1e-6 * np.eye(H.shape[0])  # Numerical stability

        if Gf is not None:
            G_total = np.vstack([G, Gf])
            g_total = np.hstack([g, gf])
            print(f"Final QP constraints: G_total shape = {G_total.shape}, g_total shape = {g_total.shape}")
        else:
            G_total = G
            g_total = g

        print("Calling QP solver...")
        u_bar = solve_qp(H, h, G=G_total, h=g_total, solver='quadprog')

        if u_bar is None:
            raise ValueError("QP solver failed to find a solution")

        x_bar = T @ x0 + S @ u_bar
        x_bar = x_bar.reshape((N + 1, dim_x))
        u_bar = u_bar.reshape((N, dim_u))

        print("MPC solve successful.")
        return x_bar, u_bar

    except Exception as e:
        print(f"[ERROR] MPC solve failed: {e}")
        raise




## Initialize Linear State Space #############################################################################################################################
#chapter 3 of "Vehicle Dynamics and Control" by Rajesh Rajamani page 48 has parameters for car
m = 1573
Iz = 2873
lf = 1.1
lr = 1.58
Caf = 80000
Car = 80000
Vx = 30.0  # set number

# state space vector is [e1,e1dot,e2,e2dot] where e1 is the lateral error and e2 is the heading error
dim_x = 4
dim_u = 1 
A = np.zeros((4, 4))
A[0, 1] = 1
A[1, 1] = -(2*Caf + 2*Car)/(m*Vx)
A[1, 2] = (2*Caf + 2*Car)/m
A[1, 3] = (-2*Caf*lf + 2*Car*lr)/(m*Vx)
A[2, 3] = 1
A[3, 1] = (-2*Caf*lf + 2*Car*lr)/(Iz*Vx)
A[3, 2] = (-2*Caf*lf + 2*Car*lr)/Iz
A[3, 3] = (-2*Caf*lf**2 - 2*Car*lr**2)/(Iz*Vx)

# B matrix for steer angle (delta)
B1 = np.array([
    [0],
    [2*Caf/m],
    [0],
    [2*Caf*lf/Iz]
])

## Discretize the linear system #######################################################################################################################################
Ts = 0.1  
A_c = A
B_c = B1 
C_c = np.eye(A.shape[0]) 
D_c = np.zeros((A.shape[0], B1.shape[1]))

sys_d = cont2discrete((A_c, B_c, C_c, D_c), Ts)
Ad, Bd, Cd, Dd, _ = sys_d

## Compute terminal cost using discrete riccati equation ############################################################################################################
N = 20
Q = np.diag([100, 10, 100, 10])
R = np.array([[0.1]])  # Make R a matrix for proper matrix operations
P = solve_discrete_are(Ad, Bd, Q, R)
K = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
print("Terminal cost matrix P:")
print(P)
print("Optimal gain matrix K:")
print(K)

u_lb = np.array([-0.5])  # input constraints [delta_min]
u_ub = np.array([0.5])   # input constraints [delta_max]
c_terminal = find_terminal_set_outer_approx(P, K, u_bounds=[u_lb, u_ub])
print("Maximal terminal set constant c =", c_terminal)
terminal_set = ellipsoid_to_polytope(P, c_terminal, num_directions=200)

def plot_terminal_set_2d(terminal_set, dims=(0, 2), title="Terminal Set in (e1, e2)"):
    """
    Plots 2D projection of terminal set polytope onto selected dimensions (e.g., e1 vs e2).

    Args:
        terminal_set: polytope.Polytope object
        dims: tuple of indices for projection (e.g., (0, 2) for e1 vs e2)
        title: plot title
    """
    proj_set = pc.projection(terminal_set, dims)

    fig, ax = plt.subplots()
    proj_set.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    ax.grid(True)
    ax.axis('equal')
    plt.show()

plot_terminal_set_2d(terminal_set, dims=(0, 2), title="Terminal Set: e1 vs e2")
from polytope import is_inside

D = np.array([
    [1, 0, 0, 0],  # e1
    [0, 0, 1, 0],  # e2
])
c_lb = np.array([-0.5, -0.1])
c_ub = np.array([0.5, 0.1])


# Run MPC without terminal constraints first to ensure it works
print("\nRunning MPC with terminal constraints...")
x0 = np.array([0.2, 0.1, 0.05, 0.1])  # initial state

#Predict final state from current x0 and nominal zero control
x_N = Ad @ x0  # Rough idea (not accurate for N steps, but just to test)

# Check if x_N lies in the terminal set
print("xN in terminal set:", is_inside(terminal_set, x_N))

# First try without terminal constraints
x_bar, u_bar = solve_condensed_mpc(
    x0=x0, A=Ad, B=Bd, Q=Q, R=R, P=P, N=N,
    u_lb=u_lb, u_ub=u_ub,
    D=D, c_lb=c_lb, c_ub=c_ub,
    with_terminal_constraint=True, terminal_set=terminal_set
)

time = np.arange(N + 1) * Ts

plt.figure(figsize=(10, 8))
plt.subplot(3,1,1)
plt.plot(time, x_bar[:, 0], 'b-', label='e1 (lateral error)')
plt.plot(time, x_bar[:, 2], 'r-', label='e2 (heading error)')
plt.grid(True)
plt.ylabel("Error [m, rad]")
plt.title("State Trajectories (Without Terminal Constraints)")
plt.legend()

plt.subplot(3,1,2)
plt.plot(time, x_bar[:, 1], 'b:', label='e1_dot')
plt.plot(time, x_bar[:, 3], 'r:', label='e2_dot')
plt.grid(True)
plt.ylabel("Error rates")
plt.legend()

plt.subplot(3,1,3)
plt.plot(time[:-1], u_bar[:, 0], 'k-', label='Steering angle δ')
plt.grid(True)
plt.ylabel("Steering [rad]")
plt.xlabel("Time [s]")
plt.legend()

plt.tight_layout()
plt.show()

# Phase portrait
plt.figure(figsize=(8, 6))

# Plot trajectory
plt.plot(x_bar[:, 0], x_bar[:, 2], 'b-', linewidth=1.5, label='MPC trajectory')
plt.scatter(x_bar[0, 0], x_bar[0, 2], c='g', s=100, label='Initial state')
plt.scatter(x_bar[-1, 0], x_bar[-1, 2], c='r', s=100, label='Final state')

# Plot state constraints
e1_min, e1_max = -0.5, 0.5
e2_min, e2_max = -0.1, 0.1
plt.plot([e1_min, e1_max, e1_max, e1_min, e1_min], 
         [e2_min, e2_min, e2_max, e2_max, e2_min], 
         'r--', label='State Constraints')

plt.grid(True)
plt.xlabel("e1 (lateral error) [m]")
plt.ylabel("e2 (heading error) [rad]")
plt.title("Phase Portrait: Lateral vs Heading Error")
plt.legend()
plt.axis('equal')
plt.show()

def plot_terminal_ellipsoid_2D(P, c, indices=(0, 2)):
    idx1, idx2 = indices
    P_sub = P[np.ix_([idx1, idx2], [idx1, idx2])]
    eigvals, eigvecs = np.linalg.eigh(P_sub)
    width, height = 2 * np.sqrt(c / eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    fig, ax = plt.subplots()
    ellipse = plt.Ellipse((0, 0), width, height, angle, edgecolor='r', facecolor='none')
    ax.add_patch(ellipse)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f"Ellipsoidal Terminal Set (e1 vs e2), c={c:.4f}")
    ax.set_xlabel("e1")
    ax.set_ylabel("e2")
    plt.grid(True)
    plt.show()



