import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

# =============================================================================
# 1. Fysische parameters en systeemdefinitie
# =============================================================================
m = 1573       # massa
Iz = 2873      # yaw traagheidsmoment
lf = 1.1       # afstand voorwiel tot zwaartepunt
lr = 1.58      # afstand achterwiel tot zwaartepunt
Caf = 80000    # voorwielkoppelveerconstante
Car = 80000    # achterwielkoppelveerconstante
Vx = 30.0      # rij-snelheid

# Staat: [e1, e1_dot, e2, e2_dot] waarbij
# e1 = laterale afwijking en e2 = yaw (heading) afwijking

dim_x = 4
dim_u = 1

# Continue systeemmatrices (state-space) bouwen
A = np.zeros((4, 4))
A[0, 1] = 1
A[1, 1] = -(2*Caf + 2*Car)/(m * Vx)
A[1, 2] = (2*Caf + 2*Car)/m
A[1, 3] = (-2*Caf*lf + 2*Car*lr)/(m * Vx)
A[2, 3] = 1
A[3, 1] = (-2*Caf*lf + 2*Car*lr)/(Iz * Vx)
A[3, 2] = (-2*Caf*lf + 2*Car*lr)/Iz
A[3, 3] = (-2*Caf*lf**2 - 2*Car*lr**2)/(Iz * Vx)

# B matrix: invoer is de stuurhoek (delta)
B = np.array([
    [0],
    [2 * Caf / m],
    [0],
    [2 * Caf * lf / Iz]
])

# =============================================================================
# 2. Discretisatie van het continue systeem
# =============================================================================
Ts = 0.1  # sampletijd in seconden

# Voor de complete toestand (C = I, D = 0)
C_cont = np.eye(dim_x)
D_cont = np.zeros((dim_x, dim_u))

# Gebruik scipy.signal.cont2discrete om het systeem te discretiseren
sys_d = cont2discrete((A, B, C_cont, D_cont), Ts)
Ad, Bd, Cd, Dd, _ = sys_d

print("Discretisatie voltooid:")
print("Ad =", Ad)
print("Bd =", Bd)

# =============================================================================
# 3. Terminale kostenmatrix berekenen via de discrete Riccati-vergelijking
# =============================================================================
# Stel de kostenmatrices in (Q voor de toestand, R voor de input)
Q = np.diag([100, 10, 100, 10])
R = np.array([[0.1]])

# Oplossen van de discrete algebraïsche Riccati-vergelijking
P = solve_discrete_are(Ad, Bd, Q, R)
print("\nTerminale kostenmatrix P:")
print(P)

# Bereken de LQR-gain:
K = -np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
print("\nLQR feedback gain K:")
print(K)

# =============================================================================
# 4. Definieer constraints
# =============================================================================
# Input constraints voor de stuurhoek (delta)
u_lb = np.array([-0.5])
u_ub = np.array([0.5])

# Constraint op de projectie van de toestand in de (e1, e2)-ruimte:
# Bijvoorbeeld, een beperking op laterale afwijking e1 en yaw afwijking e2:
D_proj = np.array([
    [1, 0, 0, 0],   # selecteer e1
    [0, 0, 1, 0]    # selecteer e2
])
c_lb = np.array([-0.5, -0.1])
c_ub = np.array([0.5, 0.1])

# =============================================================================
# 5. Initialisatie van de toestand
# =============================================================================
x0 = np.array([0.2, 0.1, 0.05, 0.1])
print("\nInitiële toestand x0:")
print(x0)

# =============================================================================
# 6. (Optioneel) Visualisatie van de terminale set (niveau-set van V_f(x) = 0.5*x^T*P*x)
# =============================================================================
# Kies een waarde 'a' voor de niveau-set: { x | 0.5*x^T P x <= a }
a = 1.0

# Omdat het systeem 4-dimensionaal is, bekijken we hier een projectie op de (e1,e2)-ruimte.
# We berekenen de niveau-set in de vorm van een ellips:
# 0.5*x^T*P*x <= a  =>  x^T*P*x <= 2a.
# In de (e1, e2)-ruimte extraheren we de relevante submatrix:
P_proj = P[[0,2], :][:, [0,2]]

# Voor het plotten van een ellipse hebben we de eigenwaarden/ -vectoren van P_proj:
eigvals, eigvecs = np.linalg.eig(P_proj)
theta = np.linspace(0, 2*np.pi, 100)
ellipse = np.array([np.cos(theta), np.sin(theta)])

# Schaal de ellipse zodat x^T*P_proj*x = 2a
# Afstand langs de hoofdrichtingen is sqrt(2a/eigenvalue)
scaling = np.array([np.sqrt(2*a/val) for val in eigvals])
ellipse_scaled = eigvecs @ np.diag(scaling) @ ellipse

plt.figure(figsize=(6,6))
plt.plot(ellipse_scaled[0, :], ellipse_scaled[1, :], 'b-', label='Terminale set (projectie)')
plt.xlabel('e1')
plt.ylabel('e2')
plt.title('Projectie van de terminale set op (e1, e2)')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# =============================================================================
# 7. Extra: Controleer of de input-constraint voor de LQR-regelaar wordt voldaan
# =============================================================================
# Voor de gesloten-lus dynamica: x+ = (Ad + Bd*K)x.
A_cl = Ad + Bd @ K

# Om te controleren of de regelaar voldoet aan de input-constraint, evalueren we u = K*x voor
# de "extremen" (bijvoorbeeld vertices) van de terminale set (of een gekozen polytope).
# Hier gebruiken we de hoekpunten van de geplotte ellipse (bij benadering) als testpunten.
# We selecteren enkele punten op de ellipse:
test_points = ellipse_scaled.T  # (100 x 2) punten in (e1, e2)

# Omdat de volledige toestand 4-dimensionaal is, nemen we aan dat de overige toestandscomponenten nul zijn.
violations = []
for pt in test_points:
    x_test = np.array([pt[0], 0, pt[1], 0])
    u_test = K @ x_test
    if not (u_lb[0] <= u_test <= u_ub[0]):
        violations.append((x_test, u_test))

if violations:
    print("\nSommige testpunten schenden de input constraints:")
    for v in violations:
        print("x =", v[0], "-> u =", v[1])
else:
    print("\nAlle testpunten voldoen aan de input constraints voor de LQR-regelaar.")
