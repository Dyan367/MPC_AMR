%% SC42125 Model Predictive Control Donal Paddy Ryan (6284922), Tomoya Martynowicz (5118441)
% === MPC Reference tracking ===
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
%% Linear state space
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
%% Discretize
Ts = 0.1;
vehicle_ss_d = c2d(vehicle_ss, Ts);
[A_d, B_d, C_d, ~] = ssdata(vehicle_ss_d);

dim.nx = 4; 
dim.nu = 1;
dim.N = 20;
T_sim = 10/Ts;

LTI.A = A_d;
LTI.B = B_d;
%% Initial Conditions
x0 = [-1; 0; 0.1; 0];
LTI.x0 = x0;

%% LQR
Q = diag([10 1 10 1]); 
R = 1;
[P, K, ~] = idare(A_d, B_d, Q, R);
K = -K;

%% Weights
weight.Q=Q;
weight.R=R;
weight.P=P;
%% Invariant Set Xf
mpt_init;
Acl = A_d + B_d*K;

x_bnd = [10; 10; 2; 6];
x_bnd = [2; 3.5; 1; 3];
x_lb = -x_bnd;
x_ub = x_bnd;

Fx = [eye(4); -eye(4)];
bx = [x_bnd; x_bnd];

u_bnd = deg2rad(50);
u_lb = -u_bnd;
u_ub = u_bnd;

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

fprintf("Computed Xf \n")
%% Plotting the invariant set Xf
figure;
plot(Xf.projection([1,3]), 'color', 'b');
xlabel('Lateral error e_1');
ylabel('Heading error e_2');
title('Projection of LQR Invariant Set (e_1 vs e_2)');
grid on;
fprintf("Plotted Xf")
%% Prediction Matrices and costs

predmod = predmodgen(LTI,dim);
Qbar = blkdiag(kron(eye(dim.N), Q), P);
H = predmod.S' * Qbar * predmod.S + kron(eye(dim.N), R);
h = predmod.S' * Qbar * predmod.T;
%% Reference Trajectory
% simulate a lane switch
yref_traj = zeros(4, T_sim);
for k = 1:T_sim
    t = k*Ts;
    if t < 2
        yref_traj(1,k) = -1;
    elseif t < 6
        yref_traj(1,k) = 2.0; 
    elseif t < 10
        yref_traj(1,k) = -1; 
    end
end
fprintf("Created Reference Trajectory \n")
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
    [xr, ur] = optimalss(struct('A', A_d, 'B', B_d, 'x0', x0), dim, [], [], eqconstraints);
    
    U = sdpvar(dim.nu * dim.N, 1);
    xN = predmod.S(end-dim.nx+1:end,:) * U + predmod.T(end-dim.nx+1:end,:) * x0_k;
    constraints = [];
    for i = 1:dim.N
        x_i = predmod.S((i-1)*dim.nx+1:i*dim.nx,:) * U + predmod.T((i-1)*dim.nx+1:i*dim.nx,:) * x0_k;
        constraints = [constraints;
                       Fx * x_i <= bx]; 
    end
    for i = 1:dim.N
        u_i = U((i-1)*dim.nu+1:i*dim.nu);
        constraints = [constraints;
                       u_lb <= u_i <= u_ub];
    end
    constraints = [constraints;
                   Xf.A * xN <= Xf.b];    
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
u_rec = [0, u_rec]; 

%% Plots
yref_traj = [yref_traj, yref_traj(:,end)];
time = (0:T_sim)*Ts;

state_labels = {'e_1 (m)', 'e_1 dot (m/s)', 'e_2 (rad)', 'e_2 dot (rad/s)'};
x_ub = x_bnd;
x_lb = -x_bnd;

for i = 1:4
    figure;
    stairs(time, x(i,:), 'b', 'LineWidth', 1.5); hold on;
    stairs(time, yref_traj(i,:), 'r--', 'LineWidth', 1.2);

    yline(x_ub(i), 'k--', 'Max Constraint');
    yline(x_lb(i), 'k--', 'Min Constraint');

    xlabel('Time (s)');
    ylabel(state_labels{i});
    legend('Actual', 'Reference', 'Location', 'best');
    title(['State ', num2str(i), ': ', state_labels{i}]);
    grid on;
end

% input plot
figure;
stairs(time, u_rec, 'b', 'LineWidth', 1.5); hold on;
yline(u_bnd, 'k--', 'Max Constraint');
yline(-u_bnd, 'k--', 'Min Constraint');
xlabel('Time (s)');
ylabel('Steering Input u (rad)');
title('Control Input');
grid on;
legend('u', 'Location', 'best');
%%
figure;
for i = 1:4
    subplot(4,1,i);
    stairs(time, x(i,:), 'b', 'LineWidth', 1.5); hold on;
    stairs(time, yref_traj(i,:), 'r--', 'LineWidth', 1.2);
    yline(x_bnd(i), 'k--', 'Max');
    yline(-x_bnd(i), 'k--', 'Min');

    ylabel(state_labels{i});
    if i == 1
        title('MPC Reference Tracking with State Constraints');
        legend('Actual', 'Reference', 'Max/Min', 'Location', 'best');
    end
    grid on;
end
xlabel('Time (s)');


%% car lane change visualization

lane_width = 3.5;
road_width = 2 * lane_width;
car_length = 4.5;
car_width = 2.0;

% scaled down because X axis is too long
X_pos = 5 * time;     
Y_pos = x(1,:);        

left_edge = lane_width;
right_edge = -lane_width;

figure;
hold on;
axis equal;
xlim([0, max(X_pos)+5]);
ylim([-5, 5]);  
pbaspect([3 1 1]);  

fill([0 max(X_pos)+5 max(X_pos)+5 0], [right_edge left_edge left_edge right_edge], ...
     [0.95 0.95 0.95], 'EdgeColor', 'none');

plot([0 max(X_pos)+5], [left_edge left_edge], 'k-', 'LineWidth', 2);   
plot([0 max(X_pos)+5], [right_edge right_edge], 'k-', 'LineWidth', 2); 
plot([0 max(X_pos)+5], [0 0], 'k--', 'LineWidth', 1.5);                

% car trail
N_draw = 10;
idx = round(linspace(1, length(X_pos), N_draw));
alpha_vals = linspace(0.2, 1, N_draw); % transparency fade
car_color = [1 0 0]; 

for i = 1:N_draw
    xi = X_pos(idx(i));
    yi = Y_pos(idx(i));
    alpha_i = alpha_vals(i);

    rectangle('Position', [xi - car_length/2, yi - car_width/2, car_length, car_width], ...
              'FaceColor', car_color, ...
              'EdgeColor', 'none', ...
              'FaceAlpha', alpha_i);
end

% car end pos
rectangle('Position', [X_pos(end) - car_length/2, Y_pos(end) - car_width/2, car_length, car_width], ...
          'FaceColor', car_color, ...
          'EdgeColor', 'k', ...
          'LineWidth', 1.5);

% traj
plot(X_pos, Y_pos, 'b--', 'LineWidth', 2);

xlabel('Longitudinal Position X [m]', 'FontSize', 12);
ylabel('Lateral Position Y [m]', 'FontSize', 12);
title('Lane Change Maneuver Visualization', 'FontSize', 14);
set(gca, 'FontSize', 12);
grid on;




