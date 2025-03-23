clear; clc; close all;

%% Vehicle parameters
m = 1500;       % mass [kg]
Iz = 3000;      % yaw inertia [kg m^2]
lf = 1.2;       % distance from CG to front axle [m]
lr = 1.6;       % distance from CG to rear axle [m]
L = lf + lr;
Caf = 80000;    % front cornering stiffness [N/rad]
Car = 80000;    % rear cornering stiffness [N/rad]
Vx = 10;        % constant forward speed [m/s]
dt = 0.01;      % time step [s]
T = 15;         % total simulation time [s]
N = round(T / dt);     % number of time steps
t = 0:dt:(T - dt);     % time vector

%% Linearized continuous-time model (used only by MPC)
A = [0, 1, 0, 0;
     0, (-2*Caf*lf + 2*Car*lr)/(m*Vx), (2*Caf + 2*Car)/m, (-2*Caf*lf^2 + 2*Car*lr^2)/(m*Vx);
     0, 0, 0, 1;
     0, (2*Caf*lf - 2*Car*lr)/(Iz*Vx), (2*Caf*lf - 2*Car*lr)/Iz, (2*Caf*lf^2 + 2*Car*lr^2)/(Iz*Vx)];
B_delta = [0; 2*Caf/m; 0; 2*Caf*lf/Iz];
B_psi_des = [0;
            (-2*Caf*lf + 2*Car*lr)/(m*Vx);
             0;
            (-2*Caf*lf^2 + 2*Car*lr^2)/(Iz*Vx)];

%% Reference trajectory: straight → right turn → straight
turn_radius = 30;
seg_len = round(N / 3);
ref_x = zeros(1, N); ref_y = zeros(1, N); ref_yaw = zeros(1, N);
for i = 1:seg_len
    ref_x(i) = Vx * dt * i;
    ref_y(i) = 0;
    ref_yaw(i) = 0;
end
for i = 1:seg_len
    theta = (i / seg_len) * (pi/2);
    ref_x(seg_len+i) = ref_x(seg_len) + turn_radius * sin(theta);
    ref_y(seg_len+i) = ref_y(seg_len) - turn_radius * (1 - cos(theta));
    ref_yaw(seg_len+i) = -theta;
end
for i = 1:(N - 2*seg_len)
    idx = 2*seg_len + i;
    ref_x(idx) = ref_x(2*seg_len) + Vx * dt * i * cos(-pi/2);
    ref_y(idx) = ref_y(2*seg_len) + Vx * dt * i * sin(-pi/2);
    ref_yaw(idx) = -pi/2;
end
ref_psi_dot = [diff(ref_yaw)/dt, 0];

%% Discretize system for MPC
sys_c = ss(A, B_delta, eye(4), zeros(4,1));
sys_d = c2d(sys_c, dt);
Ad = sys_d.A; Bd = sys_d.B;

%% Set up MPC
mpc_horizon = 15;
mpc_ctrl = mpc(sys_d, dt, mpc_horizon, mpc_horizon);
mpc_ctrl.MV.Min = -0.5; mpc_ctrl.MV.Max = 0.5;
mpc_ctrl.Weights.MV = 0.1;
mpc_ctrl.Weights.MVRate = 0.1;
mpc_ctrl.Weights.OV = [1 0.1 1 0.1];

%% Initialization
x = [0.0; 0; 0; 0];
X = zeros(4, N); X(:,1) = x;
delta = zeros(1, N);
veh_X = zeros(1, N); veh_Y = zeros(1, N); veh_yaw = zeros(1, N);
veh_X(1) = ref_x(1); veh_Y(1) = ref_y(1); veh_yaw(1) = ref_yaw(1);
x_mpc = mpcstate(mpc_ctrl);

%% Simulation loop (nonlinear error model)
for k = 1:N-1
    % Get control from linear MPC
    yref = [0; 0; 0; 0];
    u = mpcmove(mpc_ctrl, x_mpc, x, yref);
    delta(k) = u;
    
    % Extract states
    e1     = x(1);
    e1_dot = x(2);
    e2     = x(3);
    e2_dot = x(4);
    
    % Tire slip angles
    alpha_f = (e1_dot + lf * e2_dot - Vx * u) / Vx;
    alpha_r = (-e1_dot + lr * e2_dot) / Vx;
    
    % Lateral forces
    F_yf = -Caf * alpha_f;
    F_yr = -Car * alpha_r;
    
    % Nonlinear lateral dynamics
    dx1 = e1_dot;
    dx2 = (F_yf + F_yr)/m + Vx * e2 + ref_psi_dot(k) * Vx;
    dx3 = e2_dot;
    dx4 = (lf * F_yf - lr * F_yr)/Iz;
    x = x + dt * [dx1; dx2; dx3; dx4];
    X(:,k+1) = x;

    % Global pose update
    yaw = ref_yaw(k) - x(3);
    veh_yaw(k+1) = yaw;
    veh_X(k+1) = veh_X(k) + Vx * cos(yaw) * dt;
    veh_Y(k+1) = veh_Y(k) + Vx * sin(yaw) * dt;
end

%% Plot path tracking with vehicle body rectangles and orientation
figure; hold on;
plot(ref_x, ref_y, 'r--', 'LineWidth', 2);
plot(veh_X, veh_Y, 'b', 'LineWidth', 2);
xlabel('X [m]'); ylabel('Y [m]');
title('MPC Path Tracking with Nonlinear Vehicle Model');
axis equal; grid on;
xlim([min(ref_x)-10, max(ref_x)+10]);
ylim([min(ref_y)-20, max(ref_y)+20]);

% Vehicle dimensions (for plotting rectangle)
car_length = 4.5;  % meters
car_width = 2.0;

skip = 100;
for k = 1:skip:N
    % Center of vehicle
    cx = veh_X(k);
    cy = veh_Y(k);
    yaw = veh_yaw(k);

    % Vehicle corners in local frame
    corners = [car_length/2,  car_width/2;
              -car_length/2,  car_width/2;
              -car_length/2, -car_width/2;
               car_length/2, -car_width/2]';

    % Rotation matrix
    R = [cos(yaw), -sin(yaw);
         sin(yaw),  cos(yaw)];

    % Rotate and translate
    world_corners = R * corners + [cx; cy];

    % Draw the car as a red box
    fill(world_corners(1,:), world_corners(2,:), 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');

    % Shortened quiver for heading direction
    q_len = 1.5;  % shorter arrow
    quiver(cx, cy, q_len*cos(yaw), q_len*sin(yaw), 0, 'k', 'LineWidth', 1.2, 'MaxHeadSize', 2);
end


%% Plot error and steering performance
figure;
subplot(3,1,1);
plot(t, X(1,:), 'LineWidth', 1.5);
ylabel('Lateral Error e₁ [m]'); grid on;

subplot(3,1,2);
plot(t, rad2deg(X(3,:)), 'LineWidth', 1.5);
ylabel('Heading Error e₂ [deg]'); grid on;

subplot(3,1,3);
plot(t, rad2deg(delta), 'LineWidth', 1.5);
ylabel('Steering \delta [deg]'); xlabel('Time [s]'); grid on;
