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
A = zeros(4,4);
A(1,2) = 1;
A(2,:) = [0 -(2*Caf+2*Car)/(m*Vx) (2*Caf+2*Car)/m (-2*Caf*lf+2*Car*lr)/(m*Vx)];
A(3,4) = 1;
A(4,:) = [0 (-2*Caf*lf+2*Car*lr)/(Iz*Vx) (-2*Caf*lf+2*Car*lr)/Iz (-2*Caf*lf^2-2*Car*lr^2)/(Iz*Vx)];
B = [0; 2*Caf/m; 0; 2*Caf*lf/Iz];
C = [1 0 0 0];
D = [0];

%% Discretize system
Ts = 0.01;
sys = ss(A,B,C,D);
sysd = c2d(sys, Ts);
[A, B, C, D] = ssdata(sysd);

x0 = [0.8; 0.0; 0.0; 0.0];
d = 0.5;
yref = 0; 

Bd = zeros(4,1);
Bd(1,1) = 0.1/m;

Cd = 1;
LTI.C = C;
LTI.Cd = Cd;

LTI.A=A; 
LTI.B=B;
LTI.C=C;
LTI.Bd=Bd;
LTI.Cd=Cd;
LTI.x0=x0;
LTI.d=d;
LTI.yref=yref;

dim.nx=4;
dim.nu=1;
dim.N=50;
dim.ny=1;
dim.nd=1;
%% LQR
weight.Q=diag([1 1 1 1]);
weight.R=1;
[P,K,~] = idare(A,B,weight.Q,weight.R);
weight.P =P;
K = -K;

T_sim=30;

%% Extended system
LTIe.A=[LTI.A LTI.Bd; zeros(dim.nd,dim.nx) eye(dim.nd)];
LTIe.B=[LTI.B; zeros(dim.nd,dim.nu)];
LTIe.C=[LTI.C LTI.Cd];
LTIe.x0 = [LTI.x0; LTI.d];
LTIe.yref=LTI.yref;

%% Setup MPC params
dime.nx=5;
dime.nu=1;
dime.ny=1;
dime.N=50;

weighte.Q=blkdiag(weight.Q,zeros(dim.nd));
weighte.R=weight.R;
weighte.P=blkdiag(weight.P,zeros(dim.nd));

predmod=predmodgen(LTIe,dime);
predmodinv=predmodgen(LTI,dim);

%% Invariant set
mpt_init;
Acl = A + B * K;

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
%% Check if x0 in Xf 
distance = norm(Xf.chebyCenter.x - x0);
disp(['Distance from center of Xf: ', num2str(distance)]);


%% Extended costfunction
[He,he]=costgen(predmod,weighte,dime);

xe=zeros(dime.nx,T_sim+1);
y=zeros(dime.ny,T_sim+1);
u_rec=zeros(dime.nu,T_sim);
xehat=zeros(dime.nx,T_sim+1);

xe(:,1)=LTIe.x0;
xehat(:,1) = zeros(dime.nx,1);
xehat(:,1) = [x0; 0.4]; 
fprintf("Size of LTIe.C: %dx%d\n", size(LTIe.C));
fprintf("Size of LTIe.x0: %dx%d\n", size(LTIe.x0));

y(:,1)=LTIe.C*LTIe.x0;

%% Observer
Obs_check = [eye(4) - A, -Bd;
             C,         Cd];
r = rank(Obs_check);
fprintf("Rank of Obs_check = %d, expected = %d\n", r, dim.nx + dim.nd);
L = place(LTIe.A', LTIe.C',[0.65 0.7 0.75 0.8 0.9999])';
A_obs = LTIe.A - L * LTIe.C;
eig_obs = eig(A_obs);
fprintf("Observer error dynamics eigenvalues (A - LC):\n");
disp(eig_obs);


%% MPC simulation loop with disturbance
for k = 1:T_sim    
    xe_0    = xe(:,k);
    xe_0hat = xehat(:,k);
    dhat    = xehat(end-dim.nd+1:end, k);
    %dhat = min(dhat, 0.1); % or tighter bounds depending on expectations
    fprintf("dhat:\n");
    disp(dhat)
    
    eqconstraints = eqconstraintsgen(LTI, dim, dhat);
    [xr, ur] = optimalss(LTI, dim, weight, [], eqconstraints); 
    fprintf("xr:\n");
    disp(xr)
    fprintf("xr:\n");
    disp(xr)
    xre = [xr; dhat];

    Xf_shifted = Xf ;

    uostar = sdpvar(dime.nu * dime.N, 1);

    Fx = [eye(dim.nx); -eye(dim.nx)];
    bx = [x_bnd; x_bnd];
    
    F_state = kron(eye(dim.N), Fx);    
    b_state = repmat(bx, dim.N, 1);        
    
    X_constraint = F_state * predmodinv.S(1:end-dim.nx, :) * uostar ...
                 + F_state * predmodinv.T(1:end-dim.nx, :) * xe_0hat(1:dim.nx) <= b_state;
    
    Fu = [eye(dim.nu); -eye(dim.nu)];
    bu = [u_bnd; u_bnd];
    F_input = kron(eye(dim.N), Fu);
    b_input = repmat(bu, dim.N, 1);
    U_constraint = F_input * uostar <= b_input;

    row_start = dim.N * dim.nx + 1;
    row_end = (dim.N + 1) * dim.nx;
    
    S_N = predmodinv.S(row_start:row_end, :);
    T_N = predmodinv.T(row_start:row_end, :);
    
    terminal_constraint = Xf_shifted.A * ((S_N * uostar + T_N * xe_0hat(1:dim.nx)) - xr) <= Xf_shifted.b;


    if k > 5
        Constraint = [X_constraint; U_constraint; terminal_constraint];
    else
        Constraint = [X_constraint; U_constraint];
    end

    u_max = deg2rad(70); 
    Fu = [eye(dim.nu); -eye(dim.nu)];
    bu = [u_max; u_max];

    Objective  = 0.5 * uostar' * He * uostar + (he * [xe_0hat; xre; ur])' * uostar;

    optimize(Constraint, Objective);
    uostar = value(uostar);

    u_rec(:,k) = uostar(1:dim.nu);

    xe(:,k+1) = LTIe.A * xe_0 + LTIe.B * u_rec(:,k);
    y(:,k+1)  = LTIe.C * xe(:,k+1);

    xehat(:,k+1) = LTIe.A * xehat(:,k) + LTIe.B * u_rec(:,k) + L * (y(:,k) - LTIe.C * xehat(:,k));
end


%% Plots MPC results
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
%% Plot disturbance estimate
figure; hold on;
stairs(0:T_sim, xehat(end,:), 'LineWidth', 1.5, 'Color', 'b', ...
    'DisplayName', '$\hat{d}$');
plot(0:T_sim, ones(1,T_sim+1)*d, 'r--', 'LineWidth', 1.2, ...
    'DisplayName', 'true $d$');

xlabel('Time (s)');
ylabel('$\hat{d}$ (m)', 'Interpreter', 'latex');
lgd = legend('show');             
set(lgd, 'Interpreter', 'latex');  
grid on;
title('True vs estimated disturbance');




