%% SC42125 Model Predictive Control Donal Paddy Ryan (6284922), Tomoya Martynowicz (5118441)
% === MPC Regularization ===
clc;
clear all;
close all;

%% System Parameters
m = 1573;        % vehicle mass [kg]
Iz = 2873;       % yaw moment of inertia [kg*m^2]
lf = 1.1;        % distance from CG to front axle [m]
lr = 1.58;       % distance from CG to rear axle [m]
Caf = 80000;     % cornering stiffness front [N/rad]
Car = 80000;     % cornering stiffness rear [N/rad]
Vx = 30.0;       % longitudinal velocity [m/s]

%% State space
% States: [e1; e1_dot; e2; e2_dot] 
% e1 = lateral error, e2 = heading error
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

%% Discretize system 
Ts = 0.1;
vehicle_ss_d = c2d(vehicle_ss, Ts, 'zoh'); % ZOH
[A_d, B_d, C_d, D_d] = ssdata(vehicle_ss_d);

%% Initial conditions for simulation
x0 = [0.5; 0; 0.2; 0];

%% LQR 
Q = diag([100 1 100 1]);
R = 0.1;
[P,K,~] = idare(A_d, B_d, Q, R);
K = -K;

%% Check if LQR gain can make system stable
disp("Eigenvalues of closed-loop system:\n");
disp(eig(A_d + B_d*K)); 
%% Setup MPC params
dim.nx = 4;    % state dim
dim.nu = 1;    % input dim
dim.N = 30;    % horizon
T_sim = 30;   % sim time

LTI.A = A_d;
LTI.B = B_d;
LTI.x0 = x0;
%% Weights
weight.Q=Q;
weight.R=R;
weight.P=P;

%% Prediction Matrices
predmod = predmodgen(LTI,dim); % from exercises code

%% Cost generation
Qbar=blkdiag(kron(eye(dim.N),weight.Q),weight.P);
Rbar=kron(eye(dim.N),weight.R);
H=predmod.S'*Qbar*predmod.S+Rbar;   
h=predmod.S'*Qbar*predmod.T;

%% Xf invariant set
% computed using MPT3 toolbox
mpt_init;

Acl = A_d + B_d * K;
x_bnd  = [0.5; 2; 0.3; 1.5];  % e1, e1_dot, e2, e2_dot
x_lb = -x_bnd;
x_ub = x_bnd;

u_bnd = deg2rad(30);  % steering angle bound in radians
u_lb = -u_bnd;
u_ub = u_bnd;

% state constraints
Fx = [eye(4); -eye(4)];
bx = [x_ub; -x_lb];
% input constraints
Fu = [K; -K];
bu = [u_ub; -u_lb];

% state and input constraints under control law u = Kx
F_total = [Fx; Fu];
b_total = [bx; bu];

% Closed-loop invariant set computation
Xf = Polyhedron('A', F_total, 'b', b_total);
Xf.minHRep();  % simplify

max_iter = 100;
for i = 1:max_iter
    preXf = Polyhedron('A', Xf.A * Acl, 'b', Xf.b);
    newXf = Xf & preXf; 
    if newXf == Xf
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
%% Sim params
x = zeros(dim.nx, T_sim+1);
u_rec = zeros(dim.nu, T_sim);
x(:,1) = LTI.x0;
% Generate N-step controllable sets (X_0, ..., X_N)
%[xn_sets, ~] = Xn_gen(A_d, B_d, K, dim.N, x_lb, x_ub, u_lb, u_ub, 'lqr');


%% Sim loop
x_bnd  = [0.5; 2; 0.3; 1.5];  % e1, e1_dot, e2, e2_dot
x_lb = -x_bnd;
x_ub = x_bnd;

% state constraints
Fx = [eye(4); -eye(4)];
bx = [x_ub; -x_lb];
for k = 1:T_sim
    x_0 = x(:,k); 
    
    U = sdpvar(dim.nu * dim.N, 1);
    xN = predmod.S(end-dim.nx+1:end,:) * U + predmod.T(end-dim.nx+1:end,:) * x_0;

    constraints = [];
    for i = 1:dim.N
        x_i = predmod.S((i-1)*dim.nx+1:i*dim.nx,:) * U + predmod.T((i-1)*dim.nx+1:i*dim.nx,:) * x_0;
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

    objective = 0.5 * U' * H * U + (h * x_0)' * U;
    ops = sdpsettings('solver','quadprog','verbose',0);
    diagnostics = optimize(constraints, objective, ops);
    if diagnostics.problem ~= 0
        warning("QP failed to solve at step %d", k);
        break;
    end

    U_opt = value(U); % vector of optimal control inputs over horizon
    u_rec(k) = U_opt(1); % take first input 

    % update state
    x(:,k+1) = A_d * x_0 + B_d * u_rec(k);
end

u_rec = [0, u_rec];

%% Plots
time = (0:T_sim)*Ts;

figure;
hold on;
grid on;
stairs(time, x(1,:),'LineWidth', 1.5,'Color', 'r', 'DisplayName', 'e1 (m)');
stairs(time, x(2,:),'LineWidth', 1.5,'Color', 'g', 'DisplayName', 'e1 dot (m/s)');
stairs(time, x(3,:),'LineWidth', 1.5,'Color',  'b', 'DisplayName', 'e2 (rad)');
stairs(time, x(4,:),'LineWidth', 1.5,'Color',  'm', 'DisplayName', 'e2 dot (rad/s)');
hold off;

xlabel('Time (s)');
ylabel('States');
title('MPC Regularisation');
legend();



% input over horizon
figure;
grid on;
stairs(time, u_rec, 'Color','b','LineWidth', 1.5 );
ylabel('u (rad)');
xlabel('Time (s)');
title('Steering Input');

%% Lyapunov decrease
l = zeros(1, T_sim);
V = zeros(1, T_sim);

for i = 1:T_sim
    l(i) = 0.5 * x(:,i+1)' * Q * x(:,i+1) + 0.5 * u_rec(i)^2 * R;
    V(i) = x(:,i+1)' * P * x(:,i+1);
end

v_minus = V(2:end) - V(1:end-1);

figure;
grid on;
stairs(-l(1:end-1),'LineWidth', 1.5, 'DisplayName', '-l(x,u)');
hold on;
stairs(v_minus,'LineWidth', 1.5,  'DisplayName', 'V(k+1) - V(k)');
hold off;
legend('show');
xlabel('Time step');
ylabel('Cost variation');
title('Lyapunov decrease');

%% Sim LQR for comparison
x_lqr = zeros(dim.nx, T_sim+1);
u_lqr = zeros(dim.nu, T_sim);
x_lqr(:,1) = LTI.x0;

for k = 1:T_sim
    u_lqr(k) = K * x_lqr(:,k);
    x_lqr(:,k+1) = A_d * x_lqr(:,k) + B_d * u_lqr(k);
end

u_lqr = [0, u_lqr];



%% Plots to compare MPC and LQR performance
state_labels = {'e1 (m)', 'e1 dot (m/s)', 'e2 (rad)', 'e2 dot (rad/s)'};

for i = 1:4
    figure;
    stairs(time, x(i,:),'LineWidth', 1.5,'Color',  'b', 'DisplayName', 'MPC');
    hold on;
    stairs(time, x_lqr(i,:),'LineWidth', 1.5,'Color',  'r', 'DisplayName', 'LQR');

    % Add constraint lines
    yline(x_bnd(i), 'k--', 'Max bound');
    yline(-x_bnd(i), 'k--', 'Min bound');

    xlabel('Time (s)');
    ylabel(state_labels{i});
    title(['State ', num2str(i), ' - ', state_labels{i}]);
    legend('Location','best');
    grid on;
end

%%
figure;
stairs(time, u_rec,'LineWidth', 1.5,'Color',  'b', 'DisplayName', 'MPC');
hold on;
stairs(time, u_lqr,'LineWidth', 1.5,'Color',  'r', 'DisplayName', 'LQR');

yline(u_bnd, 'k--', 'Max input');
yline(u_lb, 'k--', 'Min input');

xlabel('Time (s)');
ylabel('Steering input u (rad)');
title('Control Input: MPC vs LQR');
legend('Location','best');
grid on;
