function [Xn, Z] = admissible_set(A, B, K, N, xlb, xub, ulb, uub, terminal)

% Define constraints for Z = {[x; u] | G x + H u + \psi <= 0}.
Nx = size(A, 1);
[Az, bz] = hyperrectangle([xlb; ulb], [xub; uub]); %
Z = struct('G', Az(:,1:Nx), 'H', Az(:,(Nx + 1):end), 'psi', bz);

% Decide terminal constraint.
switch terminal
case 'termeq'
    % Equality constraint at the origin.
    Xf = [0; 0; 0; 0];
    
case 'lqr'
    % Build feasible region considering x \in X and Kx \in U.
    % Control input
    [A_U, b_U] = hyperrectangle(ulb, uub);
    A_lqr = A_U*K;
    b_lqr = b_U;
    
    % State input
    [A_X, b_X] = hyperrectangle(xlb, xub);
    Acon = [A_lqr; A_X];
    bcon = [b_lqr; b_X];

    % Use LQR-invariant set.
    Xf = struct();
    ApBK = A + B*K; % LQR evolution matrix.
    [Xf.A, Xf.b] = calcOinf(ApBK, Acon, bcon);
    [~, Xf.A, Xf.b] = removeredundantcon(Xf.A, Xf.b);

end

% Now do feasible sets computation.
figure();
hold('on');

Xn = cell(N + 1, 1);
Xn{1} = Xf;
names = cell(N + 1, 1);
names{1} = 'Xf';
for n = 1:(N + 1)
    % Compute next set. Also need to prune constraints.
    nextXn = computeX1(Z, A, B, Xn{n});
    [~, nextXn.A, nextXn.b] = removeredundantcon(nextXn.A, nextXn.b);
    Xn{n + 1} = nextXn;
end

legend(names{:});

end%function