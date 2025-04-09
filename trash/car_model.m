clear; clc; close all;

%% Vehicle parameters
m = 1500;
Iz = 3000;
lf = 1.2;
lr = 1.6;
Caf = 80000;
Car = 80000;
Vx = 10;             % constant forward velocity [m/s]
dt = 0.01;
T = 30;
N = T / dt;

%% State-space matrices
A = [0, 1, 0, 0;
     0, (-2*Caf*lf + 2*Car*lr)/(m*Vx), (2*Caf + 2*Car)/m, (-2*Caf*lf^2 + 2*Car*lr^2)/(m*Vx);
     0, 0, 0, 1;
     0, (2*Caf*lf - 2*Car*lr)/(Iz*Vx), (2*Caf*lf - 2*Car*lr)/Iz, (2*Caf*lf^2 + 2*Car*lr^2)/(Iz*Vx)];

B_delta = [0;
           2*Caf/m;
           0;
           2*Caf*lf/Iz];

B_psi_des = [0;
            (-2*Caf*lf + 2*Car*lr)/(m*Vx);
             0;
            (-2*Caf*lf^2 + 2*Car*lr^2)/(Iz*Vx)];

%% Generate figure-8 reference trajectory
a = 15;      % amplitude [m]
omega = 0.2; % angular speed [rad/s]

t = linspace(0, T, N);
ref_x = a * sin(omega * t);
ref_y = a * sin(omega * t) .* cos(omega * t);

% Compute reference yaw and yaw rate
dx = gradient(ref_x, dt);
dy = gradient(ref_y, dt);
ref_yaw = atan2(dy, dx);
ref_psi_dot = gradient(ref_yaw, dt);

%% Initialize lateral error state
x = [0.0; 0; 0; 0];  % [e1; e1_dot; e2; e2_dot]

%% Logs
X = zeros(4, N);
veh_X = zeros(1, N);
veh_Y = zeros(1, N);
veh_yaw = zeros(1, N);
veh_X(1) = ref_x(1);
veh_Y(1) = ref_y(1);
veh_yaw(1) = ref_yaw(1);

%% Placeholder steering input (zero for now)
delta = zeros(1,N);  % You can insert your MPC output here

%% Simulation
for k = 1:N-1
    u_delta = delta(k);
    psi_dot_des = ref_psi_dot(k);
    
    % Update error state
    dx_state = A*x + B_delta*u_delta + B_psi_des*psi_dot_des;
    x = x + dx_state * dt;
    X(:,k+1) = x;

    % Recover global pose
    yaw = ref_yaw(k) - x(3);
    veh_yaw(k+1) = yaw;
    veh_X(k+1) = veh_X(k) + Vx * cos(yaw) * dt;
    veh_Y(k+1) = veh_Y(k) + Vx * sin(yaw) * dt;
end

%% Plot path tracking
figure;
plot(ref_x, ref_y, 'r--', 'LineWidth', 2); hold on;
plot(veh_X, veh_Y, 'b', 'LineWidth', 2);
xlabel('X [m]');
ylabel('Y [m]');
legend('Reference Path (Figure-8)', 'Vehicle Path');
title('Vehicle Path Tracking — Figure-8 Trajectory');
axis equal;
grid on;

