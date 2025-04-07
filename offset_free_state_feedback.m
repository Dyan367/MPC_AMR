clc;
clear all;
close all;

%% Car params
m = 1573;
Iz = 2873;
lf = 1.1;
lr = 1.58;
Caf = 80000;
Car = 80000;
Vx = 30.0;

%% Linear CT state space
A = zeros(4,4);
A(1,2) = 1;
A(2,2) = -(2*Caf + 2*Car)/(m*Vx);
A(2,3) = (2*Caf + 2*Car)/m;
A(2,4) = (-2*Caf*lf + 2*Car*lr)/(m*Vx);
A(3,4) = 1;
A(4,2) = (-2*Caf*lf + 2*Car*lr)/(Iz*Vx);
A(4,3) = (-2*Caf*lf + 2*Car*lr)/Iz;
A(4,4) = (-2*Caf*lf^2 - 2*Car*lr^2)/(Iz*Vx);

B = [0; 2*Caf/m; 0; 2*Caf*lf/Iz];
C = eye(4);
D = zeros(4,1);
vehicle_ss = ss(A,B,C,D);

%% Discretize state space
Ts = 0.01;
vehicle_ss_d = c2d(vehicle_ss, Ts);
[A_d, B_d, C_d, ~] = ssdata(vehicle_ss_d);

%% Dimensions
dim.nx = 4; 
dim.nu = 1;
dim.N = 20;
T_sim = 400;

%% Initial Conditions
x0 = [0.5; 0; 0.1; 0];

%% LQR
Q = diag([10 1 10 1]); R = 1;
[P, K, ~] = idare(A_d, B_d, Q, R);
K = -K;

%% Invariant Set Calculation (Xf)
mpt_init;
Acl = A_d + B_d*K;

x_bnd = [1.5; 3; 0.5; 2.5];
u_bnd = deg2rad(30);
Fx = [eye(4); -eye(4)];
bx = [x_bnd; x_bnd];
Fu = [K; -K]; bu = [u_bnd; u_bnd];
F_total = [Fx; Fu];
b_total = [bx; bu];

Xf = Polyhedron('A', F_total, 'b', b_total);
Xf.minHRep();
for i = 1:100
    preXf = Polyhedron('A', Xf.A * Acl, 'b', Xf.b);
    newXf = Xf & preXf;
    if newXf == Xf
        break;
    end
    Xf = newXf;
end
fprintf("Computed Xf\n");

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

predmod.T = T; predmod.S = S;
Qbar = blkdiag(kron(eye(dim.N), Q), P);
H = predmod.S' * Qbar * predmod.S + kron(eye(dim.N), R);
h = predmod.S' * Qbar * predmod.T;
fprintf("Created prediction matrices\n");

%% Reference Trajectory (Lane Switch)
yref_traj = zeros(4, T_sim);
for k = 1:T_sim
    t = k*Ts;
    if t < 2
        yref_traj(1,k) = 0;
    elseif t < 6
        yref_traj(1,k) = 1.0;
    elseif t < 10
        yref_traj(1,k) = 0;
    end
end
fprintf("Created Reference Trajectory\n");

%% Disturbance Observer Initialization
d_hat = 0;             % scalar disturbance affecting only e1
L_d = 0.05;             % observer gain
d_hat_hist = zeros(1, T_sim);

%% Simulation
x = zeros(dim.nx, T_sim+1);
u_rec = zeros(dim.nu, T_sim);
x(:,1) = x0;
x_pred = x0;  % for model-based prediction

disturbance = [0.01; 0; 0; 0];  % constant disturbance added to true system

fprintf("Start Sim\n");
for k = 1:T_sim
    x0_k = x(:,k);
    yref = yref_traj(:,k);

    % Predict state using model (without disturbance)
    if k > 1
        x_pred = A_d * x(:,k-1) + B_d * u_rec(k-1);
    end

    % Disturbance estimation (only e1)
    y1_meas = x0_k(1);
    y1_pred = x_pred(1);
    alpha = 0.95;  % try values like 0.90 to 0.99
    d_hat = alpha * d_hat + (1 - alpha) * (y1_meas - y1_pred);

    d_hat_hist(k) = d_hat;

    % Solve optimal steady-state (xr, ur)
    eq_A = [eye(dim.nx) - A_d, -B_d; C_d, zeros(dim.nx, dim.nu)];
    eq_b = [zeros(dim.nx,1); yref - [d_hat; 0; 0; 0]];
    eqconstraints.A = eq_A; eqconstraints.b = eq_b;
    [xr, ur] = optimalss(struct('A', A_d, 'B', B_d, 'x0', x0_k), dim, [], [], eqconstraints);

    % MPC Optimization
    U = sdpvar(dim.nu * dim.N, 1);
    xN = predmod.S(end-dim.nx+1:end,:) * U + predmod.T(end-dim.nx+1:end,:) * x0_k;
    constraints = [Xf.A * xN <= Xf.b];

    Ur = repmat(ur, dim.N, 1);
    objective = 0.5 * (U - Ur)' * H * (U - Ur) + (h * (x0_k - xr))' * (U - Ur);
    
    ops = sdpsettings('solver','quadprog','verbose',0);
    diagnostics = optimize(constraints, objective, ops);

    if diagnostics.problem ~= 0
        warning("QP failed at step %d", k);
        break;
    end

    U_opt = value(U);
    u_rec(k) = U_opt(1);
    x(:,k+1) = A_d * x0_k + B_d * u_rec(k) + disturbance;
end
fprintf("End Sim\n");

% Final alignments
u_rec = [u_rec, u_rec(end)];
yref_traj = [yref_traj, yref_traj(:,end)];
time = (0:T_sim)*Ts;

%% State and Input Plots
figure;
subplot(5,1,1);
plot(time, x(1,:), 'b'); hold on;
plot(time, yref_traj(1,:), 'r--');
ylabel('e_1 (m)');
legend('Actual', 'Reference');
title('Lateral Position Tracking');

subplot(5,1,2);
plot(time, x(2,:)); ylabel('e_1 dot');

subplot(5,1,3);
plot(time, x(3,:)); ylabel('e_2 (rad)');

subplot(5,1,4);
plot(time, x(4,:)); ylabel('e_2 dot');

subplot(5,1,5);
plot(time, u_rec); ylabel('Steering (rad)');
xlabel('Time (s)');
title('Control Input');

%% Disturbance Estimate Plot
figure;
plot((0:T_sim-1)*Ts, d_hat_hist, 'LineWidth', 1.5);
yline(disturbance(1), 'r--', 'True Disturbance');
xlabel('Time (s)');
ylabel('Estimated d̂');
title('Disturbance Estimate Convergence');
legend('Estimated d̂','True d');
grid on;

%% Car Path Visualization
lane_width = 5;
car_length = 4.5;
car_width = 2.0;
X_pos = 15 * time;
Y_pos = x(1,:);

figure;
hold on;
grid on;
axis equal;
xlim([0, 120]);
ylim([-5, 5]);

plot([0 max(X_pos)], [ lane_width/2  lane_width/2], 'k--', 'LineWidth', 1);
plot([0 max(X_pos)], [-lane_width/2 -lane_width/2], 'k--', 'LineWidth', 1);
plot([0 max(X_pos)], [0 0], 'k--', 'LineWidth', 2);
plot(X_pos, Y_pos, 'r-', 'LineWidth', 1.5);

for i = 1:100:T_sim+1
    rectangle('Position', [X_pos(i) - car_length/2, Y_pos(i) - car_width/2, car_length, car_width], ...
              'FaceColor', 'r', 'EdgeColor', 'k');
end

xlabel('Longitudinal position X [m]');
ylabel('Lateral position Y [m]');
title('Car Lane Change Path (Scaled View)');
legend('Left Lane', 'Right Lane', 'Center Line', 'Car Path');
