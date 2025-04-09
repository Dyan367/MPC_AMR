function [xr, ur] = optimalss(LTI, dim, weight, ~, eqconstraints)
% optimalss computes the steady-state targets (x_ss and u_ss) that achieve
% the desired output y_ref given the estimated disturbance.
%
% It solves the quadratic program:
%    min 0.5 * z' * Hss * z
%    s.t. Aeq * z = beq,
% where z = [x_ss; u_ss].
%
% Inputs:
%   LTI           - structure with system matrices and yref.
%   dim           - structure with dim.nx (state dimension) and dim.nu (input dimension).
%   weight        - structure with weight.Q and weight.R (for Hss).
%   eqconstraints - structure with fields Aeq and beq.
%
% Outputs:
%   xr - computed steady-state target state.
%   ur - computed steady-state target input.

    % Extract equality constraints.
    Aeq = eqconstraints.Aeq;
    beq = eqconstraints.beq;

    nx = dim.nx;
    nu = dim.nu;

    % Define weight matrices for the steady-state QP.
    % Here we require weight.Q to be a matrix of size (nx x nx)
    Q = weight.Q;
    R = weight.R;
    
    % Build block diagonal matrix for steady-state cost, dimension: (nx+nu) x (nx+nu).
    Hss = blkdiag(Q, R);
    fss = zeros(nx + nu, 1);

    % Decision variable z = [x_ss; u_ss].
    z = sdpvar(nx + nu, 1);
    Objective = 0.5 * z' * Hss * z + fss' * z;
    Constraints = [Aeq * z == beq];

    ops = sdpsettings('verbose', 0, 'solver', 'quadprog');
    sol = optimize(Constraints, Objective, ops);

    if sol.problem == 0
        z_opt = value(z);
        xr = z_opt(1:nx);
        ur = z_opt(nx+1:end);
    else
        warning('optimalss: problem infeasible, returning zeros');
        xr = zeros(nx, 1);
        ur = zeros(nu, 1);
    end
end
