%% main.m
% This script implements an output-feedback offset-free MPC for a car model.
% The state and disturbance are not measured directly and must be estimated using a Luenberger observer.

clear all; clc; close all;

%% 1. Car and Model Parameters
m   = 1573;       Iz  = 2873;
lf  = 1.1;        lr  = 1.58;
Caf = 80000;      Car = 80000;
Vx  = 30.0;       % Longitudinal speed

%% 2. Continuous-Time Model
A = zeros(4,4);
A(1,2) = 1;
A(2,:) = [0, -(2*Caf+2*Car)/(m*Vx), (2*Caf+2*Car)/m, (-2*Caf*lf+2*Car*lr)/(m*Vx)];
A(3,4) = 1;
A(4,:) = [0, (-2*Caf*lf+2*Car*lr)/(Iz*Vx), (-2*Caf*lf+2*Car*lr)/Iz, (-2*Caf*lf^2-2*Car*lr^2)/(Iz*Vx)];
B = [0; 2*Caf/m; 0; 2*Caf*lf/Iz];
C = [1 0 0 0];   % We measure only the first state (e.g., lateral position)
D = 0;

%% 3. Discretization
Ts = 0.01;
sys = ss(A,B,C,D);
sysd = c2d(sys, Ts);
[A, B, C, D] = ssdata(sysd);

% Initial state
x0 = [0.8; 0.0; 0.0; 0.0];

%% 4. Disturbance Model
% Assume a constant disturbance d entering the model.
d = 0.5;   
Bd = zeros(4,1);
Bd(1,1) = 0.1/m;  % Small disturbance coupling on state 1
% Disturbance effect on the output
Cd = 1;          % y = C*x + Cd*d

% Reference output (we want the lateral error to be 0)
yref = 0;

%% 5. Build LTI Structure
LTI.A  = A; 
LTI.B  = B;
LTI.Bd = Bd;
LTI.C  = C;
LTI.Cd = Cd;
LTI.x0 = x0;
LTI.d  = d;
LTI.yref = yref;

%% 6. Define Dimensions, Horizon, and Weights
dim.nx = 4;    % number of states
dim.nu = 1;    % number of inputs
dim.ny = 1;    % number of outputs
dim.nd = 1;    % disturbance dimension
dim.N  = 10;   % MPC horizon

% Cost function weights for stage cost (state penalty and input penalty)
weight.Q = 10 * eye(dim.nx);    % 4x4 state penalty matrix
weight.R = 1;                   % scalar input penalty

% For the DARE terminal cost, use an output penalty as a scalar.
weight.Qout = 10;  % Scalar output penalty

% Terminal cost from DARE using output weight
[P,~,~] = idare(A, B, C' * weight.Qout * C, weight.R);
weight.P = P;

% Simulation horizon
T = 200;

%% 7. Build Extended System
% Extended state: [x; d] with d being constant.
% Extended system:
%      x+ = A*x + B*u + Bd*d
%      d+ = d (d is constant)
%
% Thus, A_e = [A, Bd; zeros(1,4), 1], 
%       B_e = [B; 0],
%       C_e = [C, Cd].
LTIe.A = [A,  Bd; zeros(dim.nd, dim.nx), 1];
LTIe.B = [B; zeros(dim.nd, dim.nu)];
LTIe.C = [C,  Cd];
LTIe.x0 = [x0; d];   % Extended initial condition
LTIe.yref = yref;

dime.nx = dim.nx + dim.nd;  % extended state dimension (5)
dime.nu = dim.nu;           % input dimension
dime.ny = dim.ny;           % output dimension
dime.N  = dim.N;            % horizon

% Weights for the extended system; here we penalize the output error and ignore the disturbance component.
weighte.Q = blkdiag(weight.Q, 0);
weighte.R = weight.R;
weighte.P = blkdiag(weight.P, 0);

%% --- New: Compute Feedback Gain, Closed-Loop Matrix, and Constraint Sets ---

% Compute a stabilizing state-feedback gain K (e.g., using DLQR)
[K, ~, ~] = dlqr(A, B, weight.Q, weight.R);
Acl = A + B*K;

% Define state and input bounds.
x_bnd = [10; 10; 2; 6];
u_bnd = deg2rad(70);

% State constraints: Fx * x <= bx, where Fx = [I; -I]
Fx = [eye(dim.nx); -eye(dim.nx)];
bx = [x_bnd; x_bnd];

% For the input constraints, one common approach (for tightening) is to 
% use the pre-stabilizing law u = K*x. Thus, define Fu using K.
F_u = [1; -1];
b_u = [u_bnd; u_bnd];


% Combine state and input constraints:
F_total = [Fx; K; -K];
b_total = [bx; repmat(u_bnd, 2, 1)];


% Compute the terminal invariant set Xf for the closed-loop system:

% Project the terminal set onto the original state space
% Compute the terminal set in the original 4D state space
% Compute the terminal invariant set Xf for the closed-loop system in the original 4-D state space:
Xf = Polyhedron('H', [F_total, b_total]);
Xf.minHRep();
T = 200;       % Try a smaller simulation horizon
maxIter = 50;  % maximum number of iterations for the invariant set loop
for i = 1:maxIter
    % Cache the current H-representation of the terminal set:
    temp_A = Xf.A; 
    temp_b = Xf.b;
    
    % Compute the one-step pre-set using the closed-loop dynamics Acl
    preXf = Polyhedron('H', [temp_A * Acl, temp_b]);
    
    % Compute the intersection between the current terminal set and its pre-set
    newXf = intersect(Xf, preXf);
    
    % Clear temporary variables to free memory
    clear temp_A temp_b preXf;
    
    % Optionally, compact the polyhedron to remove redundancies:
    newXf = newXf.minHRep();

    
    % Check if no further contraction occurs
    if newXf == Xf
        break;
    end
    
    % Update the terminal set for the next iteration.
    Xf = newXf;
    
    % Clear temporary newXf
    clear newXf;
end

% [Optional] Compute an admissible set XN is skipped to reduce memory usage.
% Uncomment the following block if you implement a more efficient version:
% N_adm = 1;
% X_bounds = [-x_bnd, x_bnd];  % Each column represents lower and upper bounds.
% U_bounds = [-u_bnd, u_bnd];
% XN = admissible_set(A, B, K, N_adm, X_bounds(:,1), X_bounds(:,2), U_bounds(1), U_bounds(2), 'lqr');
%
% if XN.contains(x0)
%     disp("✅ x0 is admissible");
% else
%     disp("❌ x0 is not in the admissible set");
% end


%% 8. Output-based Offset-Free MPC with Observer
% We measure only y and estimate the extended state (x,d) with a Luenberger observer.

% Generate the prediction model aFund cost matrices for the extended system
predmode = predmodgen(LTIe, dime);
[He, he] = costgen(predmode, weighte, dime);

% Initialize the "true" extended state trajectory
xe = zeros(dime.nx, T+1);
xe(:,1) = LTIe.x0;

% Initialize the observer (estimated extended state)
xehat = zeros(dime.nx, T+1);
xehat(:,1) = [0; 0; 0; 0; 0.5];  % initial guess

% Initialize measured output
y_meas = zeros(dime.ny, T+1);
y_meas(:,1) = LTIe.C * xe(:,1);

% Input history
u_rec = zeros(dime.nu, T);

% Place observer poles for the extended system (choose arbitrarily within the unit circle)
poles = [0.1; 0.11; 0.12; 0.13; 0.14];
%poles = [0.7; 0.6; 0.8; 0.75; 0.65];
L = place(LTIe.A', LTIe.C', poles)';

% Simulation loop
for k = 1:T
    % 8a) Get current disturbance estimate (last element of observer state)
    d_est = xehat(end,k);
    
    % 8b) Compute the steady-state target online (with the current disturbance estimate)
    eqconstraints = eqconstraintsgen(LTI, dim, d_est);
    [xr, ur] = optimalss(LTI, dim, weight, [], eqconstraints); 
    xre = [xr; d_est];   % extended steady-state target
    
    % 8c) Set up the QP for the MPC using YALMIP
    uostar = sdpvar(dime.nu * dime.N, 1);
    
    % Initialize constraint set
    Constraints = [];
    
    % Loop over the prediction horizon and add state and input constraints:
        for i = 1:dim.N
        % Predicted state at stage i:
        idx_x_start = (i * dim.nx) + 1;
        idx_x_end   = (i+1) * dim.nx;
        Xi = predmode.Phi(idx_x_start:idx_x_end, :) * xe(:,k) + predmode.Gamma(idx_x_start:idx_x_end, :) * uostar;
        Constraints = [Constraints, Fx * Xi <= bx];
        
        % Predicted input at stage i:
        idx_u_start = (i-1) * dim.nu + 1;
        idx_u_end   = i * dim.nu;
        Ui = uostar(idx_u_start:idx_u_end);
        Constraints = [Constraints, F_u * Ui <= b_u];
    end

    
    % Define the MPC objective
    Objective = 0.5 * uostar' * He * uostar + (he * (xe(:,k) - xre))' * uostar;
    
    ops = sdpsettings('verbose', 0, 'solver', 'quadprog', 'quadprog.Display', 'off');

    optimize(Constraints, Objective, ops);
    
    u_opt = value(uostar);
    if isempty(u_opt)
        warning('Infeasible QP at step %d', k);
        u_opt = zeros(dime.nu * dime.N, 1);
    end
    % Use only the first control action
    u_rec(:,k) = u_opt(1:dime.nu);
    
    % 8d) Update the "true" extended state
    xe(:,k+1) = LTIe.A * xe(:,k) + LTIe.B * u_rec(:,k);
    y_meas(:,k+1) = LTIe.C * xe(:,k+1);
    
    % 8e) Observer update: estimate the extended state using the measured y
    xehat(:,k+1) = LTIe.A * xehat(:,k) + LTIe.B * u_rec(:,k) + L * (y_meas(:,k) - LTIe.C * xehat(:,k));
end

%% 9. Plot the Results
figure;
plot(0:T, y_meas - yref, 'LineWidth', 1.2); grid on;
xlabel('Timestep k');
ylabel('Tracking Error (y - y_{ref})');
title('Offset-Free MPC Tracking Error');

figure;
plot(0:T-1, u_rec, 'LineWidth', 1.2); grid on;
xlabel('Timestep k');
ylabel('Control Input u(k)');
title('Control Input History');
