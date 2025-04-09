function eqconstraints = eqconstraintsgen(LTI, dim, d_est)
% eqconstraintsgen generates the equality constraints for computing the steady-state
% given the estimated disturbance d_est.
%
% The steady-state equations are:
%    x_ss = A*x_ss + B*u_ss + Bd*d_est,
%    y_ref = C*x_ss + Cd*d_est.
%
% Inputs:
%   LTI   - structure with fields A, B, Bd, C, Cd, and yref.
%   dim   - structure with dim.nx (state dimension) and dim.nu (input dimension).
%   d_est - estimated disturbance.
%
% Output:
%   eqconstraints - structure with fields Aeq and beq.

    A  = LTI.A;
    B  = LTI.B;
    Bd = LTI.Bd;
    C  = LTI.C;
    Cd = LTI.Cd;
    yref = LTI.yref;

    nx = dim.nx;
    nu = dim.nu;

    % Steady-state equation: (I - A)*x_ss - B*u_ss = Bd*d_est.
    Aeq1 = [eye(nx) - A, -B];
    beq1 = Bd * d_est;

    % Output equation: C*x_ss = yref - Cd*d_est.
    Aeq2 = [C, zeros(1, nu)];
    beq2 = yref - Cd * d_est;

    eqconstraints.Aeq = [Aeq1; Aeq2];
    eqconstraints.beq = [beq1; beq2];
end
