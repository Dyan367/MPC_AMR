function [H, h] = costgen(predmod, weight, dim)
% costgen builds the quadratic cost matrices H and h for the MPC QP.
%
% The cost function is defined as:
%   J = 0.5 * u' H u + (h * [x0; x_target; u_target])' * u
%
% Inputs:
%   predmod - structure from predmodgen (contains Phi and Gamma).
%   weight  - structure with weight.Q, weight.R, and weight.P.
%   dim     - structure with dim.nx, dim.nu, and dim.N.
%
% Outputs:
%   H and h - matrices for the QP formulation.

    nx = dim.nx;
    nu = dim.nu;
    N  = dim.N;

    Phi   = predmod.Phi;
    Gamma = predmod.Gamma;

    % Stage and terminal cost matrices
    Q = weight.Q;
    R = weight.R;
    P = weight.P;

    % Build block-diagonal matrices Qbar and Rbar
    Qbar = [];
    for i = 1:N
        Qbar = blkdiag(Qbar, Q);
    end
    Qbar = blkdiag(Qbar, P);  % terminal cost appended

    Rbar = [];
    for i = 1:N
        Rbar = blkdiag(Rbar, R);
    end

    % The prediction relation: x_vec = Phi*x0 + Gamma*u_vec.
    % Cost function: J = (Phi*x0 + Gamma*u)'*Qbar*(Phi*x0 + Gamma*u) + u'*Rbar*u.
    H = Gamma' * Qbar * Gamma + Rbar;
    h = Gamma' * Qbar * Phi;  % term multiplying the current state (and steady-state targets)
end
