clc;
clear all;
close all;

%% System Parameters
m = 1573;
Iz = 2873;
lf = 1.1;
lr = 1.58;
Caf = 80000;
Car = 80000;
Vx = 30.0;
Ts = 0.1;
T_sim = 30;

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

sys_d = c2d(ss(A,B,eye(4),zeros(4,1)), Ts);
[A_d, B_d, ~, ~] = ssdata(sys_d);

%% Constraints
x_bnd = [0.5; 2; 0.3; 1.5];
u_bnd = deg2rad(30);
x_lb = -x_bnd; x_ub = x_bnd;
u_lb = -u_bnd; u_ub = u_bnd;
Fx = [eye(4); -eye(4)];
bx = [x_ub; -x_lb];

%% Weight matrix options
Q_set = {
    diag([10 1 10 1])
    diag([100 1 100 1])
    diag([1000 1 1000 1])
};
R_set = {
    1,
    0.1,
    0.01
};

N = 20;  % fixed horizon
x0 = [0.5; 0; 0.2; 0];
time = (0:T_sim)*Ts;
colors = lines(length(Q_set));
state_history = zeros(4, T_sim+1, length(Q_set));
u_all = zeros(T_sim+1, length(Q_set));

for idx = 1:length(Q_set)
    Q = Q_set{idx};
    R = R_set{idx};
    [P, K, ~] = idare(A_d, B_d, Q, R);
    K = -K;

    %% Terminal set
    mpt_init;
    Acl = A_d + B_d*K;
    Fu = [K; -K];
    bu = [u_ub; -u_lb];
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

    %% Prediction Model
    dim.nx = 4;
    dim.nu = 1;
    dim.N = N;
    LTI.A = A_d;
    LTI.B = B_d;
    LTI.x0 = x0;
    predmod = predmodgen(LTI, dim);

    %% Cost matrices
    Qbar = blkdiag(kron(eye(N), Q), P);
    Rbar = kron(eye(N), R);
    H = predmod.S'*Qbar*predmod.S + Rbar;
    h = predmod.S'*Qbar*predmod.T;

    %% Simulate
    x = zeros(dim.nx, T_sim+1);
    x(:,1) = x0;
    for k = 1:T_sim
        x_k = x(:,k);
        U = sdpvar(dim.nu * dim.N, 1);
        constraints = [];
        for i = 1:dim.N
            x_i = predmod.S((i-1)*dim.nx+1:i*dim.nx,:)*U + predmod.T((i-1)*dim.nx+1:i*dim.nx,:)*x_k;
            constraints = [constraints; Fx * x_i <= bx];
        end
        for i = 1:dim.N
            u_i = U((i-1)*dim.nu+1:i*dim.nu);
            constraints = [constraints; u_lb <= u_i <= u_ub];
        end
        xN = predmod.S(end-dim.nx+1:end,:) * U + predmod.T(end-dim.nx+1:end,:) * x_k;
        constraints = [constraints; Xf.A * xN <= Xf.b];

        objective = 0.5 * U' * H * U + (h * x_k)' * U;
        ops = sdpsettings('solver','quadprog','verbose',0);
        diagnostics = optimize(constraints, objective, ops);
        if diagnostics.problem ~= 0
            warning("Solver failed at step %d for Q-R combo %d", k, idx);
            break;
        end
        u_k = value(U(1));
        x(:,k+1) = A_d * x_k + B_d * u_k;
        u_all(k+1, idx) = u_k;
    end
    state_history(:,:,idx) = x;
    
end

%% Plotting states
state_labels = {'e_1 (m)', 'e_1 dot (m/s)', 'e_2 (rad)', 'e_2 dot (rad/s)'};

for i = 1:4
    figure;
    hold on;
    for idx = 1:length(Q_set)
        stairs(time, state_history(i,:,idx), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Q=%.0e, R=%.2f', Q_set{idx}(1,1), R_set{idx}));
    end
    yline(x_bnd(i), 'k--', 'Max');
    yline(-x_bnd(i), 'k--', 'Min');
    xlabel('Time (s)');
    ylabel(state_labels{i});
    title(['State ', num2str(i), ' - ', state_labels{i}]);
    legend('Location','best');
    grid on;
end
%%
% Align dimensions: Prepend u(0) = 0
u_all(1,:) = 0;

% Plot control input for each (Q,R)
figure;
hold on;
for idx = 1:length(Q_set)
    stairs(time, u_all(:,idx), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Q=%.0e, R=%.2f', Q_set{idx}(1,1), R_set{idx}));
end
yline(u_bnd, 'k--', 'Max Input');
yline(u_lb, 'k--', 'Min Input');
xlabel('Time (s)');
ylabel('Steering Input u (rad)');
title('Control Input Comparison for Different Q and R');
legend('Location','best');
grid on;
