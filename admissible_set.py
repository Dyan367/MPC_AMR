import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from utils import gen_prediction_matrices, gen_cost_matrices, gen_constraint_matrices
from qpsolvers import solve_qp
import polytope as pc
import scipy.linalg as la

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

def find_terminal_set_outer_approx(P, K, u_bounds, x_bounds=None, num_samples=20000):
    """
    Finds the largest c such that the ellipsoid {x | xᵀ P x <= c} lies within input and (optional) state constraints.

    Args:
        P (ndarray): Terminal cost matrix.
        K (ndarray): Feedback gain matrix.
        u_bounds (tuple or list): (u_lb, u_ub) input bounds as scalars or vectors.
        x_bounds (tuple or list, optional): (x_lb, x_ub) state bounds as vectors.
        num_samples (int): Number of direction vectors to sample.

    Returns:
        c_max (float): Maximum allowable c for terminal set ellipsoid.
    """
    n = P.shape[0]
    dim_u = K.shape[0]

    u_lb, u_ub = u_bounds
    if x_bounds is not None:
        x_lb, x_ub = x_bounds

    # Sample directions on unit sphere
    X_unit = np.random.randn(num_samples, n)
    X_unit /= np.linalg.norm(X_unit, axis=1, keepdims=True)

    c_list = []

    for x in X_unit:
        Kx = K @ x
        scaling_factors = []

        for i in range(dim_u):
            if Kx[i] != 0:
                lb = u_lb[i] / Kx[i] if Kx[i] > 0 else u_ub[i] / Kx[i]
                ub = u_ub[i] / Kx[i] if Kx[i] > 0 else u_lb[i] / Kx[i]
                s = min(abs(lb), abs(ub))
                scaling_factors.append(s)
            else:
                scaling_factors.append(np.inf)

        s_min = min(scaling_factors)

        # Candidate scaled state
        x_scaled = x * s_min

        # Check state constraints (if provided)
        if x_bounds is not None:
            if not np.all(x_scaled >= x_lb) or not np.all(x_scaled <= x_ub):
                continue  # reject this direction due to state violation

        # Evaluate c value
        c_val = x_scaled.T @ P @ x_scaled
        c_list.append(c_val)

    if len(c_list) == 0:
        raise ValueError("No valid directions found that satisfy input and state constraints")

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

D = np.array([
    [1, 0, 0, 0],  # e1
    [0, 0, 1, 0],  # e2
])
c_lb = np.array([-0.5, -0.2])
c_ub = np.array([0.5, 0.2])

# Initialize full bounds with infinities
c_lb_full = -np.inf * np.ones(dim_x)
c_ub_full = np.inf * np.ones(dim_x)

c_lb_full[0] = c_lb[0]  # e1 lower bound
c_lb_full[2] = c_lb[1]  # e2 lower bound

c_ub_full[0] = c_ub[0]  # e1 upper bound
c_ub_full[2] = c_ub[1]  # e2 upper bound

margin = 0.95
x_lb_shrunk = c_lb_full #* margin
x_ub_shrunk = c_ub_full # * margin

# Add this section at the end of your code to compute admissible sets X_N

import polytope as pc

def compute_Oinf(A, H, h, max_iter=100, tol=1e-6):
    """
    Compute the maximal positive invariant set O_∞ for the system x_{k+1} = A x_k
    subject to constraints H x <= h.
    
    Args:
        A: System dynamics matrix
        H: Constraint matrix
        h: Constraint vector
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        
    Returns:
        polytope.Polytope representing O_∞
    """
    # Initialize with constraint set
    H_i = H.copy()
    h_i = h.copy()
    
    for i in range(max_iter):
        # Compute new constraint: H A^i x <= h
        H_next = H @ np.linalg.matrix_power(A, i+1)
        h_next = h.copy()
        
        # Append new constraints
        H_i = np.vstack([H_i, H_next])
        h_i = np.hstack([h_i, h_next])
        
        # Create polytopes for checking convergence
        P_i = pc.Polytope(H_i, h_i)
        
        # Check if current set is invariant
        # For invariance: x ∈ P_i ⇒ A x ∈ P_i
        # This is true when P_i ⊆ P_{i+1}
        if i > 0:
            # Check if constraints haven't changed significantly
            if np.allclose(H_i[-H.shape[0]:, :], H_i[-2*H.shape[0]:-H.shape[0], :], atol=tol) and \
               np.allclose(h_i[-H.shape[0]:], h_i[-2*H.shape[0]:-H.shape[0]], atol=tol):
                print(f"Maximal invariant set converged in {i} iterations")
                break
        
        if i == max_iter - 1:
            print(f"Warning: Maximal invariant set computation reached max iterations ({max_iter})")
    
    # Try to remove redundant constraints if minimize() method is available
    try:
        P_i.minimize()
    except (AttributeError, TypeError) as e:
        print(f"Could not minimize polytope: {e}")
        # Try alternative approach to remove redundant constraints
        try:
            # Some implementations have reduce() instead of minimize()
            P_i.reduce()
        except (AttributeError, TypeError):
            print("Could not remove redundant constraints, returning as is.")
            # If neither method is available, just keep the polytope as is
            pass
    
    return P_i

def compute_admissible_sets(A, B, K, N, x_lb, x_ub, u_lb, u_ub):
    """
    Backward recursive computation of admissible sets X_N.
    Returns a list of polytopes X_0, X_1, ..., X_N.
    """
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    # Step 1: Terminal invariant set under u = Kx
    A_U = np.vstack([np.eye(dim_u), -np.eye(dim_u)])
    b_U = np.hstack([u_ub, -u_lb])
    A_X = np.vstack([np.eye(dim_x), -np.eye(dim_x)])
    b_X = np.hstack([x_ub, -x_lb])

    # Create state constraint polytope
    state_poly = pc.Polytope(A_X, b_X)
    
    A_lqr = A_U @ K
    b_lqr = b_U

    A_con = np.vstack([A_lqr, A_X])
    b_con = np.hstack([b_lqr, b_X])

    A_cl = A + B @ K
    
    # Use our custom implementation instead of pc.compute_Oinf
    Xf = compute_Oinf(A_cl, A_con, b_con)
    Xn = [Xf]

    # Step 2: Backward recursion from Xf to X0
    for i in range(N):
        # Predecessor set: X_{n+1} under system dynamics
        print(f"Computing admissible set X_{i}")
        
        H_prev = Xn[-1].A
        h_prev = Xn[-1].b
        
        # For the predecessor set, we need to find all states that can reach X_{n+1}
        # using an admissible input u
        
        # Create a polytope with extended state-input space
        # [ I   0 ] [x] ≤ [x_ub]
        # [-I   0 ] [u]   [-x_lb]
        # [ 0   I ]      ≤ [u_ub]
        # [ 0  -I ]        [-u_lb]
        # [ H*A H*B ]      ≤ [h]  (reach next set condition)
        
        # Define state-input constraints
        H_xu_state = np.hstack([np.eye(dim_x), np.zeros((dim_x, dim_u))]) 
        H_xu_state = np.vstack([H_xu_state, -H_xu_state])
        h_xu_state = np.hstack([x_ub, -x_lb])
        
        H_xu_input = np.hstack([np.zeros((dim_u, dim_x)), np.eye(dim_u)])
        H_xu_input = np.vstack([H_xu_input, -H_xu_input])
        h_xu_input = np.hstack([u_ub, -u_lb])
        
        H_xu_next = np.hstack([H_prev @ A, H_prev @ B])
        h_xu_next = h_prev
        
        # Stack all constraints
        H_xu = np.vstack([H_xu_state, H_xu_input, H_xu_next])
        h_xu = np.hstack([h_xu_state, h_xu_input, h_xu_next])
        
        # Project polytope to state space by eliminating inputs
        try:
            # First try the standard projection
            X_pre = pc.projection(pc.Polytope(H_xu, h_xu), list(range(dim_x)))
            
            # Check if dimensions match before attempting intersection
            if X_pre.dim != state_poly.dim:
                print(f"Warning: Dimension mismatch during projection: X_pre.dim={X_pre.dim}, state_poly.dim={state_poly.dim}")
                # Try to handle dimension mismatch
                if X_pre.dim < state_poly.dim:
                    # If projection resulted in lower dimension, we need to manually create a full-dimensional polytope
                    print("Attempting to fix dimension mismatch...")
                    X_pre_A = np.zeros((X_pre.A.shape[0], dim_x))
                    X_pre_A[:, :X_pre.dim] = X_pre.A
                    X_pre = pc.Polytope(X_pre_A, X_pre.b)
                else:
                    # If somehow projection has higher dimension, something is wrong
                    raise ValueError(f"Projection resulted in higher dimension than state space: {X_pre.dim} > {state_poly.dim}")
            
            # Now intersection should work
            X_pre = X_pre.intersect(state_poly)
            
        except Exception as e:
            print(f"Error during projection or intersection: {e}")
            # Fallback approach: Use only state constraints as a conservative approximation
            print("Using fallback: State constraints only")
            X_pre = state_poly
        
        Xn.append(X_pre)
        print(f"Admissible set X_{i} computed: {X_pre.A.shape[0]} constraints")

    return Xn[::-1]  # reverse to get X0 to XN

# Call function after computing K
X_admissible = compute_admissible_sets(Ad, Bd, -K, N, x_lb_shrunk, x_ub_shrunk, u_lb, u_ub)

print("\nPlotting admissible sets...")

# Function to plot 2D projection of a 4D polytope
def plot_2d_projection(poly, indices=(0, 2), title="Polytope Projection", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # Check if polytope is empty
    if poly.volume <= 0:
        print(f"Warning: Cannot plot empty polytope in {title}")
        return ax
    
    # For 4D polytopes, we need to project to 2D for visualization
    try:
        # Direct projection attempt
        proj_poly = pc.projection(poly, list(indices))
        
        if proj_poly.dim == 2:
            # If successful projection to 2D, plot with polytope method
            proj_poly.plot(ax=ax, color='blue', alpha=0.5)
        else:
            print(f"Warning: Projection resulted in {proj_poly.dim}D polytope, expected 2D")
            # Try manual approach
            raise ValueError("Incorrect projection dimension")
            
    except Exception as e:
        print(f"Projection error: {e}, trying manual approach")
        
        # Manual plotting approach - extract vertices if available
        try:
            vertices = poly.vertices
            if vertices is not None and len(vertices) > 0:
                # Extract only the relevant dimensions
                idx1, idx2 = indices
                x_coords = [v[idx1] for v in vertices]
                y_coords = [v[idx2] for v in vertices]
                
                # Compute convex hull for proper rendering
                from scipy.spatial import ConvexHull
                if len(x_coords) > 2:
                    points = np.column_stack((x_coords, y_coords))
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    ax.fill(hull_points[:, 0], hull_points[:, 1], 
                            alpha=0.5, edgecolor='blue', facecolor='blue', 
                            label=title)
                else:
                    # Just plot the points if too few for convex hull
                    ax.scatter(x_coords, y_coords, c='blue')
            else:
                # Fall back to bounding box
                lb, ub = poly.bounding_box
                idx1, idx2 = indices
                x_min, x_max = lb[idx1], ub[idx1]
                y_min, y_max = lb[idx2], ub[idx2]
                
                # Create a rectangle for the bounding box
                from matplotlib.patches import Rectangle
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                fill=True, alpha=0.3, edgecolor='red', 
                                facecolor='blue', linewidth=2)
                ax.add_patch(rect)
                
        except Exception as e2:
            print(f"Failed to plot polytope manually: {e2}")
            # Just plot the state constraints as a fallback
            x_min, x_max = -0.5, 0.5
            y_min, y_max = -0.2, 0.2
            ax.plot([x_min, x_max, x_max, x_min, x_min], 
                    [y_min, y_min, y_max, y_max, y_min],
                    'r--', label='State Constraints')
    
    ax.set_xlabel(f"x[{indices[0]}]")
    ax.set_ylabel(f"x[{indices[1]}]")
    ax.set_title(title)
    ax.grid(True)
    return ax

# Plot individual admissible sets (a few key ones)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plot_indices = [0, N//3, 2*N//3, N]  # Plot sets at different time points

for i, idx in enumerate(plot_indices):
    if idx >= len(X_admissible):
        continue
    ax = axs[i//2, i%2]
    plot_2d_projection(X_admissible[idx], indices=(0, 2), 
                      title=f"Admissible Set X_{idx}: e1 vs e2", ax=ax)
    ax.axis('equal')

plt.tight_layout()
plt.show()

# Plot the comparison of different sets
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Plot state constraints as reference
x_min, x_max = -0.5, 0.5
y_min, y_max = -0.2, 0.2
ax.plot([x_min, x_max, x_max, x_min, x_min], 
        [y_min, y_min, y_max, y_max, y_min],
        'k--', linewidth=2, label='State Constraints')

# Plot the terminal set
try:
    plot_2d_projection(X_admissible[-1], indices=(0, 2), 
                      title="Terminal Set", ax=ax)
except Exception as e:
    print(f"Failed to plot terminal set: {e}")

# Plot the initial admissible set with different color
try:
    first_set = X_admissible[0]
    # Try to manually plot vertices
    try:
        vertices = first_set.vertices
        if vertices is not None and len(vertices) > 0:
            x_coords = [v[0] for v in vertices]
            y_coords = [v[2] for v in vertices]  # Using index 2 for e2
            
            from scipy.spatial import ConvexHull
            if len(x_coords) > 2:
                points = np.column_stack((x_coords, y_coords))
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1], 
                        alpha=0.3, edgecolor='red', facecolor='red',
                        label='Initial Admissible Set X₀')
    except Exception as e:
        print(f"Failed to plot initial set: {e}")
        # Try alternative methods here if needed
except Exception as e:
    print(f"Failed to access initial set: {e}")

plt.xlabel('e1 (lateral error) [m]')
plt.ylabel('e2 (heading error) [rad]')
plt.title('Comparison of Admissible Sets for MPC')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()