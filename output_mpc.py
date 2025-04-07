import numpy as np
from scipy.signal import cont2discrete, place_poles
from scipy.linalg import solve_discrete_are

## Initialize Linear State Space #############################################################################################################################
#chapter 3 of "Vehicle Dynamics and Control" by Rajesh Rajamani page 48 has parameters for car
m = 1573
Iz = 2873
lf = 1.1
lr = 1.58
Caf = 80000
Car = 80000
Vx = 15.0  # set number

# state space vector is [e1,e1dot,e2,e2dot] where e1 is the lateral error and e2 is the heading error
dim_x = 4
dim_u = 2 
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

# B2: input for psi_des (desired heading rate)
B2 = np.array([
    [0],
    [(-2*Caf*lf + 2*Car*lr - m*Vx)/(m*Vx)],
    [0],
    [(2*Caf*lf**2 + 2*Car*lr**2)/(Iz*Vx)]
])
# print(f"eigenvalues: {np.linalg.eigvals(A)}")
# print("A matrix:\n", A)
# print("\nB1 (for delta):\n", B1)
# print("\nB2 (for psi_des):\n", B2)

## Discretize the linear system #######################################################################################################################################
Ts = 0.01  
A_c = A
B_c = B1 
C_c = np.eye(A.shape[0]) 
D_c = np.zeros((A.shape[0], B1.shape[1]))

sys_d = cont2discrete((A_c, B_c, C_c, D_c), Ts)
Ad, Bd, Cd, Dd, _ = sys_d

## Compute terminal cost using discrete riccati equation ############################################################################################################
Q = np.eye(4)
R = np.eye(1) * 0.1
P = solve_discrete_are(Ad, Bd, Q, R)

# Calculate the optimal gain matrix K
K = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
print("Terminal cost matrix P:")
print(P)
print("Optimal gain matrix K:")
print(K)

# Compute closed-loop system matrix
A_cl = Ad - Bd @ K
print("Closed-loop eigenvalues:", np.linalg.eigvals(A_cl))

## Augment the state space to add disturbance #################################################################################################################
n = Ad.shape[0]   # 4 states
m = Bd.shape[1]   # 1 (input)
nd = 1            # disturbance dimension

# from lecture 5 augmented state space
A_aug = np.block([
    [Ad, np.zeros((n, nd)) ],               
    [np.zeros((nd, n)), np.eye(nd)]  
])

B_aug = np.vstack([
    Bd,               # B
    np.zeros((nd, m)) # 0
])

p = 4  # number of outputs
n = Ad.shape[0]  # number of states = 4

C = np.eye(4)

Cdisturbance= np.zeros((p, nd))  
Cdisturbance[0, 0] = 1.0         
C_aug = np.hstack([C, Cdisturbance])

print("C matrix:\n", C)
print("Cd_aug matrix:\n", Cdisturbance)
## Check for controlability and observability of the augmented system ############################################################################################################
# Define functions to check controllability and observability
def check_controllability(A, B):
    n = A.shape[0]
    ctrb_matrix = B
    for i in range(1, n):
        ctrb_matrix = np.hstack((ctrb_matrix, np.linalg.matrix_power(A, i) @ B))
    rank = np.linalg.matrix_rank(ctrb_matrix, tol=1e-10)
    return rank, rank == n


def check_observability(A, C):
    n = A.shape[0]
    obsv_matrix = C
    for i in range(1, n):
        obsv_matrix = np.vstack((obsv_matrix, C @ np.linalg.matrix_power(A, i)))
    rank = np.linalg.matrix_rank(obsv_matrix, tol=1e-10)
    return rank, rank == n


# Check controllability of the original system
rank_ctrl, is_ctrl = check_controllability(Ad, Bd)
print(f"Original system controllability: Rank {rank_ctrl}/{n} - {'Controllable' if is_ctrl else 'NOT Controllable'}")

# Check observability of the original system
rank_obs, is_obs = check_observability(Ad, Cd)
print(f"Original system observability: Rank {rank_obs}/{n} - {'Observable' if is_obs else 'NOT Observable'}")

n_aug = A_aug.shape[0]  # this should be 5

rank_aug_ctrl, is_aug_ctrl = check_controllability(A_aug, B_aug)
print(f"Augmented system controllability: Rank {rank_aug_ctrl}/{n_aug} - {'Controllable' if is_aug_ctrl else 'NOT Controllable'}")

# augmented system is not controllable because the disturbance is not controllable

# we can prove observability by using lemma
upper_block = np.hstack([np.eye(n) - Ad, -np.zeros((n, nd))])
lower_block = np.hstack([C, Cdisturbance])
stacked_matrix = np.vstack([upper_block, lower_block])
rank = np.linalg.matrix_rank(stacked_matrix, tol=1e-10)
full_rank = rank == n + nd
print(f"Augmented system observability: Rank {'Observable' if full_rank else 'NOT Observable'}")

## Design the luenberger observer ######################################################################################################################################## Observer dynamics matrix
# desired_poles = [0.1, 0.2, 0.3, 0.4, 0.5]

# # Compute observer gain L
# L = place_poles(A_aug.T, C_aug.T, desired_poles,method="YT").gain_matrix.T

# A_obs = A_aug - L @ C_aug

# # Compute eigenvalues
# eigvals = np.linalg.eigvals(A_obs)

# print("Observer eigenvalues:", eigvals)

# # Check if all inside unit circle
# stable = np.all(np.abs(eigvals) < 1)
# print("Observer is stable:", stable)

# import matplotlib.pyplot as plt

# def plot_observer_poles(eigvals, desired_poles=None):
#     fig, ax = plt.subplots()
#     unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', label='Unit Circle')
#     ax.add_artist(unit_circle)

#     ax.plot(np.real(eigvals), np.imag(eigvals), 'rx', label='Actual Observer Poles')
#     if desired_poles is not None:
#         ax.plot(np.real(desired_poles), np.imag(desired_poles), 'go', label='Desired Poles')

#     ax.set_title('Observer Pole Locations')
#     ax.set_xlabel('Real')
#     ax.set_ylabel('Imaginary')
#     ax.grid(True)
#     ax.set_aspect('equal', adjustable='datalim')
#     ax.legend()
#     plt.show()

# # Example usage
# eigvals = np.linalg.eigvals(A_aug - L @ C_aug)
# plot_observer_poles(eigvals, desired_poles)

def check_aug_obs_lemma(A, Bd, C, Cd):
    n = A.shape[0]
    nd = Bd.shape[1]
    upper_block = np.hstack([np.eye(n) - A, -Bd])
    lower_block = np.hstack([C, Cd])
    stacked_matrix = np.vstack([upper_block, lower_block])
    rank = np.linalg.matrix_rank(stacked_matrix,tol=1e-10)
    return rank, rank == n + nd


upper_block = np.hstack([np.eye(n) - Ad, -np.zeros((n, nd))])
lower_block = np.hstack([C, Cdisturbance])
stacked_matrix = np.vstack([upper_block, lower_block])
rank = np.linalg.matrix_rank(stacked_matrix, tol=1e-10)
full_rank = rank == n + nd
print("Rank of the stacked matrix:", rank)
print("Full rank condition met:", full_rank)
print(f"Augmented system observability: Rank {'Observable' if full_rank else 'NOT Observable'}")

rank, ok = check_aug_obs_lemma(Ad, np.zeros((n, nd)), C, Cdisturbance)
print(f"Observability rank: {rank} → {'OK' if ok else 'NOT OBSERVABLE'}")
