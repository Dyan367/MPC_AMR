clc; clear all; close all;

%% === Car Parameters ===
m = 1573; Iz = 2873;
lf = 1.1; lr = 1.58;
Caf = 80000; Car = 80000;
Vx = 30.0;

%% === Continuous-Time Model ===
A = zeros(4,4);
A(1,2) = 1;
A(2,:) = [0 -(2*Caf+2*Car)/(m*Vx) (2*Caf+2*Car)/m (-2*Caf*lf+2*Car*lr)/(m*Vx)];
A(3,4) = 1;
A(4,:) = [0 (-2*Caf*lf+2*Car*lr)/(Iz*Vx) (-2*Caf*lf+2*Car*lr)/Iz (-2*Caf*lf^2-2*Car*lr^2)/(Iz*Vx)];
B = [0; 2*Caf/m; 0; 2*Caf*lf/Iz];
C = [1 0 0 0];
D = [0];

%% === Discretization ===
Ts = 0.01;
sys = ss(A,B,C,D);
sysd = c2d(sys, Ts);
[A, B, C, D] = ssdata(sysd);

x0 = [0.8; 0.0; 0.0; 0.0];
d = 0.1;
yref = 0; 

%% === Disturbance Model ===
Bd = zeros(4,1);
Bd(1,1) = 0.1/m;

Cd = 1;
LTI.C = C;
LTI.Cd = Cd;

%% === LTI Structure ===
LTI.A=A; 
LTI.B=B;
LTI.C=C;
LTI.Bd=Bd;
LTI.Cd=Cd;
LTI.x0=x0;
LTI.d=d;
LTI.yref=yref;

%% === Dimensions ===
dim.nx=4;
dim.nu=1;
dim.N=50;
dim.ny=1;
dim.nd=1;
%% === LQR ===
weight.Q=diag([1 1 1 1]);
weight.R=1;
[P,K,~] = idare(A,B,weight.Q,weight.R);
weight.P =P;
K = -K;

T_sim=200;

%% === Extended System ===
LTIe.A=[LTI.A LTI.Bd; zeros(dim.nd,dim.nx) eye(dim.nd)];
LTIe.B=[LTI.B; zeros(dim.nd,dim.nu)];
LTIe.C=[LTI.C LTI.Cd];
LTIe.x0 = [LTI.x0; LTI.d];
LTIe.yref=LTI.yref;

%% === Extended Dimensions ===
dime.nx=5;
dime.nu=1;
dime.ny=1;
dime.N=50;

%% === Weights for Extended System ===
weighte.Q=blkdiag(weight.Q,zeros(dim.nd));
weighte.R=weight.R;
weighte.P=blkdiag(weight.P,zeros(dim.nd));

%% === Prediction Models ===
predmod=predmodgen(LTIe,dime);
predmodinv=predmodgen(LTI,dim);

%% === Invariant Set ===
mpt_init;
Acl = A + B * K;
% % Bounds for each state and input
% xlb = -[10; 10; 2; 6];
% xub =  [10; 10; 2; 6];
% ulb = -deg2rad(30);
% uub =  deg2rad(30);
% 
% [Xn, Z] = admissible_set(A, B, K, dim.N, xlb, xub, ulb, uub, 'lqr');
% Xf = Polyhedron('A', Xn{end}.A, 'b', Xn{end}.b);
% if Xf.contains(x0)
%     disp("x0 is inside the terminal invariant set.");
% else
%     disp("x0 is NOT inside the terminal invariant set.");
% end

%x_bnd = [4; 3; 0.5; 2.5];
%x_bnd = [2.5; 3; 0.5; 2.5]
x_bnd = [10; 10; 2; 6];
%x_bnd = ones(4,1)*5;
u_bnd = deg2rad(70);
Fx = [eye(4); -eye(4)];
bx = [x_bnd; x_bnd];
Fu = [K; -K];
bu = [u_bnd; u_bnd];

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
%% === admissible set ===
N = 1;
X_bounds = [-x_bnd, x_bnd];  % from your code
U_bounds = [-u_bnd, u_bnd];

XN = computeAdmissibleSet(A, B, K, Xf, X_bounds, U_bounds, N);

% Check x0:
if XN.contains(x0)
    disp("✅ x0 is admissible");
else
    disp("❌ x0 is not in the admissible set");
end

% % 
% figure;
% state_labels = {'e_1', 'ė_1', 'e_2', 'ė_2'};
% plot_idx = 1;
% 
% for i = 1:4
%     for j = i+1:4
%         subplot(3,2,plot_idx);
%         plot(Xf.projection([i,j]), 'color', 'b');
%         xlabel(state_labels{i});
%         ylabel(state_labels{j});
%         title(sprintf('%s vs %s', state_labels{i}, state_labels{j}));
%         grid on;
%         plot_idx = plot_idx + 1;
%     end
% end
%% === Check if x0 in Xf ===
distance = norm(Xf.chebyCenter.x - x0);
disp(['Distance from center of Xf: ', num2str(distance)]);


%% === Extended Cost Function ===
[He,he]=costgen(predmod,weighte,dime);

%% === Initialization ===
xe=zeros(dime.nx,T_sim+1);
y=zeros(dime.ny,T_sim+1);
u_rec=zeros(dime.nu,T_sim);
xehat=zeros(dime.nx,T_sim+1);

xe(:,1)=LTIe.x0;
xehat(:,1) = zeros(dime.nx,1);
xehat(:,1) = [x0; 0.1]; 
fprintf("Size of LTIe.C: %dx%d\n", size(LTIe.C));
fprintf("Size of LTIe.x0: %dx%d\n", size(LTIe.x0));

y(:,1)=LTIe.C*LTIe.x0;

%% === Observer Gain ===
Obs_check = [eye(4) - A, -Bd;
             C,         Cd];

% Now compute rank
r = rank(Obs_check);
fprintf("Rank of Obs_check = %d, expected = %d\n", r, dim.nx + dim.nd);
L = place(LTIe.A', LTIe.C', [0.65 0.75 0.8 0.85 0.9])';
A_obs = LTIe.A - L * LTIe.C;
eig_obs = eig(A_obs);
fprintf("Observer error dynamics eigenvalues (A - LC):\n");
disp(eig_obs);


%% === MPC Loop ===
for k = 1:T_sim    
    xe_0    = xe(:,k);
    xe_0hat = xehat(:,k);
    dhat    = xehat(end-dim.nd+1:end, k);
    %dhat = min(dhat, 0.1); % or tighter bounds depending on your expectations
    fprintf("dhat:\n");
    disp(dhat)
    
    eqconstraints = eqconstraintsgen(LTI, dim, dhat);
    [xr, ur] = optimalss(LTI, dim, weight, [], eqconstraints); 
    fprintf("xr:\n");
    disp(xr)
    fprintf("xr:\n");
    disp(xr)
    xre = [xr; dhat];

    % === Terminal constraint based on invariant set ===
    Xf_shifted = Xf ;

    uostar = sdpvar(dime.nu * dime.N, 1);

    % Build state constraints over horizon
    Fx = [eye(dim.nx); -eye(dim.nx)];
    bx = [x_bnd; x_bnd];
    
    % Repeat Fx and bx over the horizon
    F_state = kron(eye(dim.N), Fx);          % size: dim.N*2*nx × dim.N*nx
    b_state = repmat(bx, dim.N, 1);          % size: dim.N*2*nx × 1
    
    % Combine into one constraint: Fx * x_k <= bx → Fx*(T*x0 + S*u) <= bx
    X_constraint = F_state * predmodinv.S(1:end-dim.nx, :) * uostar ...
                 + F_state * predmodinv.T(1:end-dim.nx, :) * xe_0hat(1:dim.nx) <= b_state;
    
    % Input constraints
    Fu = [eye(dim.nu); -eye(dim.nu)];
    bu = [u_bnd; u_bnd];
    F_input = kron(eye(dim.N), Fu);
    b_input = repmat(bu, dim.N, 1);
    U_constraint = F_input * uostar <= b_input;
    
    % Terminal constraint
    % Indexes for the final state x_N in the prediction
    row_start = dim.N * dim.nx + 1;
    row_end = (dim.N + 1) * dim.nx;
    
    S_N = predmodinv.S(row_start:row_end, :);
    T_N = predmodinv.T(row_start:row_end, :);
    
    terminal_constraint = Xf_shifted.A * (S_N * uostar + T_N * xe_0hat(1:dim.nx)) <= Xf_shifted.b;

    % Final combined constraint
    % Constraint = [X_constraint; U_constraint; terminal_constraint];
    Constraint = [X_constraint; U_constraint];
    u_max = deg2rad(70);  % realistic steering bounds
    Fu = [eye(dim.nu); -eye(dim.nu)];
    bu = [u_max; u_max];
    
    % Add to your constraint set:
    %Constraint = [Fu * uostar(1:dim.nu) <= bu];

    Objective  = 0.5 * uostar' * He * uostar + (he * [xe_0hat; xre; ur])' * uostar;

    optimize(Constraint, Objective);
    uostar = value(uostar);

    % Apply first control input
    u_rec(:,k) = uostar(1:dim.nu);

    % Simulate system
    xe(:,k+1) = LTIe.A * xe_0 + LTIe.B * u_rec(:,k);
    y(:,k+1)  = LTIe.C * xe(:,k+1);

    % Update observer
    xehat(:,k+1) = LTIe.A * xehat(:,k) + LTIe.B * u_rec(:,k) + L * (y(:,k) - LTIe.C * xehat(:,k));
end


%% === Plot Results ===
u_rec = [0,u_rec];
time = 0:Ts:T_sim*Ts;

figure;
subplot(4,1,1);
plot(time, xe(1,:)); ylabel('e1');

subplot(4,1,2);
plot(time, xe(2,:)); ylabel('e1 dot');

subplot(4,1,3);
plot(time, xe(3,:)); ylabel('e2');

subplot(4,1,4);
plot(time, u_rec); ylabel('u (rad)'); xlabel('Time (s)');
%%
figure;
plot(0:T_sim, xehat(end,:), 'LineWidth', 1.5);
xlabel('Time step'); ylabel('Estimated disturbance (d̂)');
title('Observer Estimate of Disturbance');
grid on;
