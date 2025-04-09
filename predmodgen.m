function predmod = predmodgen(LTI, dim)
% predmodgen builds the prediction model matrices for the MPC horizon.
%
% Inputs:
%   LTI  - structure with fields LTI.A and LTI.B.
%   dim  - structure with dim.nx (number of states), dim.nu (inputs),
%          and dim.N (prediction horizon).
%
% Output:
%   predmod - structure containing the prediction matrices (Phi and Gamma).

    A = LTI.A;
    B = LTI.B;
    N = dim.N;
    nx = dim.nx;
    nu = dim.nu;

    % Build the state prediction matrix Phi: 
    % [A; A^2; ...; A^(N+1)]
    Phi = A;
    for i = 2:(N+1)
        Phi = [Phi; A^i];
    end

    % Build the input matrix Gamma.
    bigGamma = zeros(nx*(N+1), nu*N);
    for i = 1:N
        for j = 1:i
            rowStart = nx*(i) + 1;
            rowEnd   = nx*(i+1);
            colStart = nu*(j-1) + 1;
            colEnd   = nu*j;
            bigGamma(rowStart:rowEnd, colStart:colEnd) = A^(i-j)*B;
        end
    end

    predmod.Phi   = Phi;
    predmod.Gamma = bigGamma;
end
