import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, expm
from utils import solve_condensed_mpc

## Initialize Linear State Space #############################################################################################################################
#chapter 3 of "Vehicle Dynamics and Control" by Rajesh Rajamani page 48 has parameters for car
m = 1573
Iz = 2873
lf = 1.1
lr = 1.58
Caf = 80000
Car = 80000
Vx = 15.0  # set number

# state space vector is [e1,e2,e1dot,e2dot] where e1 is the lateral error and e2 is the heading error
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
print(f"eigenvalues: {np.linalg.eigvals(A)}")
print("A matrix:\n", A)
print("\nB1 (for delta):\n", B1)
print("\nB2 (for psi_des):\n", B2)

## Discretize the linear system #######################################################################################################################################
# Discretize the dynamics
dt = 0.1
# Trick to compute the matrix exponential
ABc = np.zeros((dim_x + dim_u, dim_x + dim_u))
ABc[:dim_x, :dim_x] = A
# need to combine B1 and B2 for discretization trick
B = np.hstack((B1, B2))
ABc[:dim_x, dim_x:] = B
expm_ABc = expm(ABc*dt)
Ad = expm_ABc[:dim_x, :dim_x]
Bd = expm_ABc[:dim_x, dim_x:]

print(f"Ad:\n{Ad}")
print(f"Bd:\n{Bd}")

## Formulate MPC #######################################################################################################################################=
N=25
Q = np.eye(dim_x)*10
R = np.eye(dim_u)
P = solve_discrete_are(Ad, Bd, Q, R) # discrete algebraic Riccati equation

u_lb = np.array([-0.5, -0.5])  # input constraints [delta_min, psi_dot_des_min]
u_ub = np.array([ 0.5,  0.5])  # input constraints[delta_max, psi_dot_des_max]

D = None
c_lb = None
c_ub = None


x0 = np.array([0.2, 0.1, 0.0, 0.0]) # initial state

x_bar, u_bar = solve_condensed_mpc(
    x0=x0, A=Ad, B=Bd, Q=Q, R=R, P=P, N=N,  # Use Bd instead of B
    u_lb=u_lb, u_ub=u_ub,
    D=D, c_lb=c_lb, c_ub=c_ub,
    with_terminal_constraint=False
)

time = np.arange(N + 1) * dt

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(time, x_bar[:, 0], label='Lateral error (e1)')
plt.plot(time, x_bar[:, 1], label='Heading error (e2)')
plt.grid(True)
plt.title("Predicted States")
plt.legend()

plt.subplot(2,1,2)
plt.plot(time[:-1], u_bar[:, 0], label='Steering angle δ')
plt.plot(time[:-1], u_bar[:, 1], label='Desired heading rate ψ̇_des')
plt.grid(True)
plt.title("Optimal Control Inputs")
plt.legend()
plt.xlabel("Time (s)")

plt.tight_layout()
plt.show()
