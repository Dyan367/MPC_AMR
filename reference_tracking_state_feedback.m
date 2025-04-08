clc;
clear all;
close all;

%% Car params
m = 1573;        % vehicle mass [kg]
Iz = 2873;       % yaw moment of inertia [kg*m^2]
lf = 1.1;        % distance from CG to front axle [m]
lr = 1.58;       % distance from CG to rear axle [m]
Caf = 80000;     % cornering stiffness front [N/rad]
Car = 80000;     % cornering stiffness rear [N/rad]
Vx = 30.0;       % longitudinal velocity [m/s]

%% Linear CT state space
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
vehicle_ss = ss(A,B,C,D);
%% Discretize state space
Ts = 0.01;
vehicle_ss_d = c2d(vehicle_ss, Ts);
[A_d, B_d, C_d, ~] = ssdata(vehicle_ss_d);

%% Dimensions
dim.nx = 4; 
dim.nu = 1;
dim.N = 20;
T_sim = 1000;

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

fprintf("Computed Xf")
%% Plotting the invariant set projection
figure;
plot(Xf.projection([1,3]), 'color', 'b');
xlabel('Lateral error e_1');
ylabel('Heading error e_2');
title('Projection of LQR Invariant Set (e_1 vs e_2)');
grid on;
fprintf("Plotted Xf")
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
fprintf("Created prediction matricies")
%% Reference Trajectory (Lane Switch)
yref_traj = zeros(4, T_sim);
for k = 1:T_sim
    t = k*Ts;
    if t < 2
        yref_traj(1,k) = 0;
    elseif t < 6
        yref_traj(1,k) = 1.0;  % switch right
    elseif t < 10
        yref_traj(1,k) = 0;    % switch back
    end
end
fprintf("Created Reference Trajectory")
%% Simulation
x = zeros(dim.nx, T_sim+1); u_rec = zeros(dim.nu, T_sim);
x(:,1) = x0;
fprintf("Start Sim")
for k = 1:T_sim
    x0_k = x(:,k);
    yref = yref_traj(:,k);
    
    % Solve optimal steady-state (xr, ur)
    eq_A = [eye(dim.nx) - A_d, -B_d; C_d, zeros(dim.nx, dim.nu)];
    eq_b = [zeros(dim.nx,1); yref];
    eqconstraints.A = eq_A; eqconstraints.b = eq_b;
    [xr, ur] = OTS(struct('A', A_d, 'B', B_d, 'x0', x0), dim, [], [], eqconstraints);
    
    % Define Optimization Problem
    U = sdpvar(dim.nu * dim.N, 1);
    xN = predmod.S(end-dim.nx+1:end,:) * U + predmod.T(end-dim.nx+1:end,:) * x0_k;
    constraints = [Xf.A * xN <= Xf.b];
    
    % Tracking Cost Function
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
    x(:,k+1) = A_d * x0_k + B_d * u_rec(k);
end
fprintf("End Sim")
u_rec = [0, u_rec];  % align time steps

%% Plots
yref_traj = [yref_traj, yref_traj(:,end)];
time = (0:T_sim)*Ts;

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

%% Visualization of car path on road
% Road parameters
lane_width = 3.5;
road_width = 2 * lane_width;
car_length = 4.5;
car_width = 2.0;

% Generate longitudinal position based on constant Vx
X_pos = Vx * time;          % time already defined: (0:T_sim)*Ts
Y_pos = x(1,:);             % e1: lateral deviation

% Road boundaries
left_lane_edge  = lane_width;
right_lane_edge = -lane_width;

% Plot setup
figure;
hold on;
axis equal;
xlim([0, max(X_pos)]);
ylim([-1.5, 1.5]);

% Draw lanes
fill([0 max(X_pos) max(X_pos) 0], [left_lane_edge left_lane_edge lane_width lane_width], [0.9 0.9 0.9], 'EdgeColor', 'none');  % left lane
fill([0 max(X_pos) max(X_pos) 0], [-lane_width -lane_width -left_lane_edge -left_lane_edge], [0.9 0.9 0.9], 'EdgeColor', 'none');  % right lane
plot([0 max(X_pos)], [0 0], 'k--', 'LineWidth', 1);  % center dashed line

% Car shape
for i = 1:50:T_sim+1
    % Draw a red rectangle to represent the car
    rectangle('Position', [X_pos(i) - car_length/2, Y_pos(i) - car_width/2, car_length, car_width], ...
              'FaceColor', 'r', 'EdgeColor', 'k');
end

xlabel('Longitudinal position X [m]');
ylabel('Lateral position Y [m]');
title('Car Lane Change Path');
legend('Center Line','Car Path');
grid on;




