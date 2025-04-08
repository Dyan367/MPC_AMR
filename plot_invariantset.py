import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

# Zorg dat pypoman geïnstalleerd is: pip install pypoman
try:
    from pypoman import compute_polytope_vertices
except ImportError:
    raise ImportError("Installeer pypoman met: pip install pypoman")

# =============================================================================
# 1. Systeemdefinitie en discretisatie
# =============================================================================
# Gegeven parameters:
m   = 1573      # massa
Iz  = 2873      # yaw traagheidsmoment
lf  = 1.1       # afstand voorwielen tot zwaartepunt
lr  = 1.58      # afstand achterwielen tot zwaartepunt
Caf = 80000     # voorwielkoppelveerconstante
Car = 80000     # achterwielkoppelveerconstante
Vx  = 30.0      # snelheid (m/s)

# Staat: [e1, e1_dot, e2, e2_dot] waarbij:
# e1 = laterale fout, e2 = yaw fout
dim_x = 4
dim_u = 1

# Continue systeemmatrix A (4x4)
A_cont = np.zeros((4,4))
A_cont[0,1] = 1
A_cont[1,1] = -(2*Caf+2*Car)/(m*Vx)
A_cont[1,2] = (2*Caf+2*Car)/m
A_cont[1,3] = (-2*Caf*lf+2*Car*lr)/(m*Vx)
A_cont[2,3] = 1
A_cont[3,1] = (-2*Caf*lf+2*Car*lr)/(Iz*Vx)
A_cont[3,2] = (-2*Caf*lf+2*Car*lr)/Iz
A_cont[3,3] = (-2*Caf*lf**2-2*Car*lr**2)/(Iz*Vx)

# Invoermatrix B voor stuurhoek (delta)
B_cont = np.array([
    [0],
    [2*Caf/m],
    [0],
    [2*Caf*lf/Iz]
])

# Discretisatie met sampletijd Ts = 0.1
Ts = 0.1
sys_d = cont2discrete((A_cont, B_cont, np.eye(dim_x), np.zeros((dim_x, dim_u))), Ts)
Ad, Bd = sys_d[0], sys_d[1]

# =============================================================================
# 2. Terminal kosten en LQR regelaar
# =============================================================================
# Kies kostenmatrices:
Q = np.diag([100, 10, 100, 10])
R = np.array([[0.1]])

# Bereken de terminal kostenmatrix (via discrete ARE)
P = solve_discrete_are(Ad, Bd, Q, R)
# Bereken LQR feedback gain (let op: tekenconventie)
K = -np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
print("LQR feedback gain K =\n", K)

# Gesloten-lus matrix:
F = Ad + Bd @ K

# =============================================================================
# 3. Terminal set constraints
# =============================================================================
# a) Output beperking: Enkel de laterale fout (e1) en yaw fout (e2)
D_mat = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
c_lb = np.array([-0.5, -0.1])
c_ub = np.array([ 0.5,  0.1])
A_terminal = np.vstack(( D_mat, -D_mat ))
b_terminal = np.hstack(( c_ub, -c_lb ))

# b) Invoer beperking: u = K*x moet binnen [-0.5, 0.5] liggen
u_lb = -0.5
u_ub =  0.5
A_input = np.vstack(( K, -K ))
b_input = np.hstack(( np.array([u_ub]), np.array([-u_lb]) ))

# Combineer beide sets:
A_term = np.vstack((A_terminal, A_input))
b_term = np.hstack((b_terminal, b_input))

# =============================================================================
# 4. Berekening van de terminal invariant set (maximale invariant set)
# =============================================================================
def compute_maximal_admissible_set(F, A_con, b_con, max_iter=100):
    """
    Berekent iteratief de maximaal toegestane set voor het gesloten-lus systeem:
         x⁺ = F x
    onder de constraints A_con x <= b_con.
    """
    n_constraints = A_con.shape[0]
    A_inf = A_con.copy()
    b_inf = b_con.copy()
    for t in range(max_iter):
        stop_flag = True
        for i in range(n_constraints):
            res = linprog(-A_con[i], A_ub=A_inf, b_ub=b_inf, method="highs")
            if res.success:
                x_opt = res.x
                if A_con[i] @ x_opt > b_con[i] + 1e-6:
                    stop_flag = False
                    break
            else:
                stop_flag = False
                break
        if stop_flag:
            break
        # Voeg nieuwe constraints toe: A_con * F^(t+1)
        new_constraints = A_con @ np.linalg.matrix_power(F, t+1)
        A_inf = np.vstack((A_inf, new_constraints))
        b_inf = np.hstack((b_inf, b_con))
    return A_inf, b_inf

def remove_redundant_constraints(A, b, tol=1e-6):
    """
    Verwijdert redundante constraints uit de H-representatie A x <= b.
    """
    keep = []
    n = A.shape[0]
    for i in range(n):
        A_others = np.delete(A, i, axis=0)
        b_others = np.delete(b, i, axis=0)
        res = linprog(-A[i], A_ub=A_others, b_ub=b_others, method='highs')
        if res.success:
            optimum = -res.fun
            if optimum > b[i] + tol:
                keep.append(i)
        else:
            keep.append(i)
    return A[keep], b[keep]

A_inv, b_inv = compute_maximal_admissible_set(F, A_term, b_term, max_iter=100)
A_inv, b_inv = remove_redundant_constraints(A_inv, b_inv)
print("Aantal constraints in de terminal invariant set =", A_inv.shape[0])

# =============================================================================
# 5. Finite Horizon MPC probleem met terminal penalty en terminal set constraint
# =============================================================================
N = 20  # horizon
x0 = np.array([0.5, 0, 0.1, 0])  # initiële toestand

# Optimalisatievariabelen: x[0..N] en u[0..N-1]
x = cp.Variable((dim_x, N+1))
u = cp.Variable((dim_u, N))

cost = 0
constraints = [x[:,0] == x0]
for k in range(N):
    cost += cp.quad_form(x[:,k], Q) + cp.quad_form(u[:,k], R)
    constraints += [x[:,k+1] == Ad @ x[:,k] + Bd @ u[:,k],
                    u[:,k] <= u_ub,
                    u[:,k] >= u_lb]
# Terminal penalty:
cost += cp.quad_form(x[:,N], P)
# Terminal set constraint: x_N moet binnen de terminal invariant set liggen
constraints += [A_inv @ x[:,N] <= b_inv]

# Stel het MPC probleem op en los op:
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver=cp.OSQP)

print("Optimale cost:", prob.value)
print("Optimale invoersequentie:\n", u.value)
print("Optimale toestandsreeks:\n", x.value)

# =============================================================================
# 6. Plot de laterale positie fout, laterale snelheid, heading fout, yaw rate en stuurhoek over tijd
# =============================================================================
# Tijd vectoren voor toestanden (N+1 stappen) en invoer (N stappen)
time_state = np.arange(N+1) * Ts
time_input = np.arange(N) * Ts

plt.figure(figsize=(10,12))

plt.subplot(5,1,1)
plt.plot(time_state, x.value[0,:], 'bo-')
plt.ylabel('e1 (laterale fout)')
plt.grid(True)

plt.subplot(5,1,2)
plt.plot(time_state, x.value[1,:], 'bo-')
plt.ylabel('e1_dot (laterale snelheid)')
plt.grid(True)

plt.subplot(5,1,3)
plt.plot(time_state, x.value[2,:], 'bo-')
plt.ylabel('e2 (heading fout)')
plt.grid(True)

plt.subplot(5,1,4)
plt.plot(time_state, x.value[3,:], 'bo-')
plt.ylabel('e2_dot (yaw rate)')
plt.grid(True)

plt.subplot(5,1,5)
plt.step(time_input, u.value[0,:], 'ro-', where='post')
plt.xlabel('Tijd (s)')
plt.ylabel('Stuurhoek (u)')
plt.grid(True)

plt.tight_layout()
plt.show()
