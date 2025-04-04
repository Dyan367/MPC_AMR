clc;
clear all;
close all;

%% Vehicle Parameters
m = 1573;        % vehicle mass [kg]
Iz = 2873;       % yaw moment of inertia [kg*m^2]
lf = 1.1;        % distance from CG to front axle [m]
lr = 1.58;       % distance from CG to rear axle [m]
Caf = 80000;     % cornering stiffness front [N/rad]
Car = 80000;     % cornering stiffness rear [N/rad]
Vx = 30.0;       % longitudinal velocity [m/s]

%% State-space model (continuous time)
% States: [e1; e1_dot; e2; e2_dot]
% Inputs: [steering angle delta]

A = zeros(4,4);
A(1,2) = 1;
A(2,2) = -(2*Caf + 2*Car)/(m*Vx);
A(2,3) = (2*Caf + 2*Car)/m;
A(2,4) = (-2*Caf*lf + 2*Car*lr)/(m*Vx);
A(3,4) = 1;
A(4,2) = (-2*Caf*lf + 2*Car*lr)/(Iz*Vx);
A(4,3) = (-2*Caf*lf + 2*Car*lr)/Iz;
A(4,4) = (-2*Caf*lf^2 - 2*Car*lr^2)/(Iz*Vx);

B = [0;
     2*Caf/m;
     0;
     2*Caf*lf/Iz];

C = eye(4);
D = zeros(4,1);

%% Create state-space object
vehicle_ss = ss(A,B,C,D);

%% Discretize system
Ts = 0.01;
vehicle_ss_d = c2d(vehicle_ss, Ts, 'zoh');
[A_d, B_d, C_d, D_d] = ssdata(vehicle_ss_d);

%% Initial conditions for simulation
x0 = [0.5; 0; 0.1; 0];  % Initial lateral error and heading error

%% LQR Controller
Q = diag([10 1 10 1]);  % weight lateral + heading error
R = 1;

[P,K,~] = idare(A_d, B_d, Q, R);
K = -K;

%% Print Eigenvalues (for stability check)
disp("Eigenvalues of closed-loop system:");
disp(eig(A_d + B_d*K));  % since K already has the minus sign

%% Xf invariant set

% Make sure MPT3 is initialized
mpt_init;

% Create the closed-loop system matrix
Acl = A_d + B_d * K;

% Define state and input constraints
x_bnd  = [0.5; 2; 0.3; 1.5];  % e1, e1_dot, e2, e2_dot
x_lb = -x_bnd;
x_ub = x_bnd;

u_bnd = deg2rad(30);  % steering angle bound in radians
u_lb = -u_bnd;
u_ub = u_bnd;

% Construct constraint polyhedron: Fx <= b
Fx = [eye(4); -eye(4)];
bx = [x_ub; -x_lb];

Fu = [K; -K];
bu = [u_ub; -u_lb];

% Combine state and input constraints under control law u = Kx
F_total = [Fx; Fu];
b_total = [bx; bu];

% Closed-loop invariant set computation
Xf = Polyhedron('A', F_total, 'b', b_total);
Xf.minHRep();  % simplify

% Iterate to find maximal invariant set
max_iter = 100;
for i = 1:max_iter
    preXf = Polyhedron('A', Xf.A * Acl, 'b', Xf.b);
    newXf = Xf & preXf;  % intersection
    if newXf == Xf
        disp(['Invariant set converged at iteration ', num2str(i)]);
        break;
    end
    Xf = newXf;
end

%% Plotting the invariant set projection
figure;
plot(Xf.projection([1,3]), 'color', 'b');
xlabel('Lateral error e_1');
ylabel('Heading error e_2');
title('Projection of LQR Invariant Set (e_1 vs e_2)');
grid on;

%% Setup Dimensions and Horizon
dim.nx = 4;    % State dimension
dim.nu = 1;    % Input dimension
dim.N = 20;    % Prediction horizon
T_sim = 150;   % Simulation steps

% Store system data
LTI.A = A_d;
LTI.B = B_d;
LTI.x0 = x0;

%% Prediction Matrices
T = zeros(dim.nx*(dim.N+1), dim.nx);
for k = 0:dim.N
    T(k*dim.nx+1:(k+1)*dim.nx, :) = A_d^k;
end

S = zeros(dim.nx*(dim.N+1), dim.nu*dim.N);
for k = 1:dim.N
    for i = 0:k-1
        S(k*dim.nx+1:(k+1)*dim.nx, i*dim.nu+1:(i+1)*dim.nu) = A_d^(k-1-i)*B_d;
    end
end

predmod.T = T;
predmod.S = S;

%% Cost Function Setup
Qbar = blkdiag(kron(eye(dim.N), Q), P);  % Extended Q + terminal cost
H = predmod.S' * Qbar * predmod.S + kron(eye(dim.N), R);
h = predmod.S' * Qbar * predmod.T;

%% Simulation Arrays
x = zeros(dim.nx, T_sim+1);
u_rec = zeros(dim.nu, T_sim);
x(:,1) = LTI.x0;
% Generate N-step controllable sets (X_0, ..., X_N)
%[xn_sets, ~] = Xn_gen(A_d, B_d, K, dim.N, x_lb, x_ub, u_lb, u_ub, 'lqr');

%% Simulation Loop (Receding Horizon)
for k = 1:T_sim
    x_0 = x(:,k); 
    
    % Optimization variable: sequence of control inputs
    U = sdpvar(dim.nu * dim.N, 1);

    % Terminal constraint: final state must be inside Xf
    xN = predmod.S(end-dim.nx+1:end,:) * U + predmod.T(end-dim.nx+1:end,:) * x_0;
    constraints = [Xf.A * xN <= Xf.b];
    
    % Cost
    objective = 0.5 * U' * H * U + (h * x_0)' * U;
    
    % Solve QP
    ops = sdpsettings('solver','quadprog','verbose',0);
    diagnostics = optimize(constraints, objective, ops);

    if diagnostics.problem ~= 0
        warning("QP failed to solve at step %d", k);
        break;
    end

    U_opt = value(U);
    u_rec(k) = U_opt(1);  % Apply first control

    % State update
    x(:,k+1) = A_d * x_0 + B_d * u_rec(k);
end

u_rec = [0, u_rec];  % To align input with time

%% Plotting Results
time = (0:T_sim)*Ts;

figure;
subplot(5,1,1);
plot(time, x(1,:));
ylabel('e_1 (m)');
title('Lateral Position Error');

subplot(5,1,2);
plot(time, x(2,:));
ylabel('e_1 dot (m/s)');
title('Lateral Velocity');

subplot(5,1,3);
plot(time, x(3,:));
ylabel('e_2 (rad)');
title('Heading Angle Error');

subplot(5,1,4);
plot(time, x(4,:));
ylabel('e_2 dot (rad/s)');
title('Yaw Rate');

subplot(5,1,5);
plot(time, u_rec);
ylabel('u (rad)');
xlabel('Time (s)');
title('Steering Input');

%% Optional: Assumption 2.14 (a) Validation
l = zeros(1, T_sim);
V = zeros(1, T_sim);

for i = 1:T_sim
    l(i) = 0.5 * x(:,i+1)' * Q * x(:,i+1) + 0.5 * u_rec(i)^2 * R;
    V(i) = x(:,i+1)' * P * x(:,i+1);
end

v_minus = V(2:end) - V(1:end-1);

figure;
plot(-l(1:end-1), 'DisplayName', '-l');
hold on;
plot(v_minus, 'DisplayName', 'V(k+1) - V(k)');
hold off;
legend('show');
xlabel('Time step');
ylabel('Cost variation');
title('Lyapunov decrease');



