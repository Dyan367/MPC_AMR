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
C = eye(4); D = zeros(4,1);

%% === Discretization ===
Ts = 0.01;
sys = ss(A,B,C,D);
sysd = c2d(sys, Ts);
[A, B, C, D] = ssdata(sysd);

x0 = [0.5; 0; 0.1; 0];
d = 0.01;
yref = [0.0; 0; 0; 0];

%% === Disturbance Model ===
Bd = ones(4,1)*0.01
Cd = zeros(4,1); 
Cd(1,1) = 1;
% Bd = 1e-3 * eye(4);  % Small influence on all states (or just eye(4))
% Cd = eye(4)
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
dim.N=20;
dim.ny=4;
dim.nd=1;

%% === LQR ===
weight.Q=diag([10 1 10 1]);
weight.R=1;
[P,K,~] = idare(A,B,weight.Q,weight.R);
weight.P =P;
K = -K;

T_sim=100;

%% === Extended System ===
LTIe.A=[LTI.A LTI.Bd; zeros(dim.nd,dim.nx) eye(dim.nd)];
LTIe.B=[LTI.B; zeros(dim.nd,dim.nu)];
LTIe.C=[LTI.C LTI.Cd];
LTIe.x0 = [LTI.x0; LTI.d];
LTIe.yref=LTI.yref;

%% === Extended Dimensions ===
dime.nx=5;
dime.nu=1;
dime.ny=4;
dime.N=20;

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

%x_bnd = [1.5; 3; 0.5; 2.5];
x_bnd = ones(4,1)*5;
u_bnd = deg2rad(30);
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
% 
% figure;
% plot(Xf.projection([1,3]), 'color', 'b');
% xlabel('Lateral error e_1');
% ylabel('Heading error e_2');
% title('Projection of LQR Invariant Set (e_1 vs e_2)');
% grid on;

%% === Extended Cost Function ===
[He,he]=costgen(predmod,weighte,dime);

%% === Initialization ===
xe=zeros(dime.nx,T_sim+1);
y=zeros(dime.ny,T_sim+1);
u_rec=zeros(dime.nu,T_sim);
xehat=zeros(dime.nx,T_sim+1);

xe(:,1)=LTIe.x0;
xehat(:,1) = zeros(dime.nx,1);
fprintf("Size of LTIe.C: %dx%d\n", size(LTIe.C));
fprintf("Size of LTIe.x0: %dx%d\n", size(LTIe.x0));

y(:,1)=LTIe.C*LTIe.x0;

%% === Observer Gain ===
Obs_check = [eye(4) - A, -Bd;
             C,         Cd];

% Now compute rank
r = rank(Obs_check);
fprintf("Rank of Obs_check = %d, expected = %d\n", r, dim.nx + dim.nd);
L = place(LTIe.A', LTIe.C', [0.01 0.05 0.07 0.1 0.15])';
A_obs = LTIe.A - L * LTIe.C;
eig_obs = eig(A_obs);
fprintf("Observer error dynamics eigenvalues (A - LC):\n");
disp(eig_obs);


%% === MPC Loop ===
for k=1:T_sim    
    xe_0=xe(:,k);
    xe_0hat = xehat(:,k);
    dhat=xehat(end-dim.nd+1:end,k);
%     dhat = min(max(dhat, -1), 1);
%     dhat = clamp(dhat, -1, 1);  % if needed
    %LTI.yref = LTI.Cd * dhat;   % make target reachable

    eqconstraints=eqconstraintsgen(LTI,dim,dhat);
    fprintf("eq_A size: %dx%d\n", size(eqconstraints.A));
    fprintf("eq_b size: %dx%d\n", size(eqconstraints.b));

    [xr,ur]=optimalss(LTI,dim,weight,[],eqconstraints); 
    xre=[xr;dhat];
    disp(xr)
    new_in = Xf

    uostar = sdpvar(dime.nu*dime.N,1);
    Constraint=[new_in.A*(predmodinv.S(end,:)*uostar+predmodinv.T(end)*xe_0hat(1:dim.nx))<=new_in.b];
    Objective = 0.5*uostar'*He*uostar+(he*[xe_0hat; xre; ur])'*uostar;
    optimize(Constraint,Objective);
    uostar=value(uostar);      

    u_rec(:,k)=uostar(1:dim.nu);
    xe(:,k+1)=LTIe.A*xe_0 + LTIe.B*u_rec(:,k);
    y(:,k+1)=LTIe.C*xe(:,k+1);

    xehat(:,k+1)=LTIe.A*xehat(:,k)+LTIe.B*u_rec(:,k)+L*(y(:,k)-LTIe.C*xehat(:,k));
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
