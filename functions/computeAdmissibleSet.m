function XN = computeAdmissibleSet(A, B, K, Xf, X_bounds, U_bounds, N)
    % Inputs:
    % A, B       - system matrices (discrete)
    % K          - feedback gain (optional)
    % Xf         - terminal set (Polyhedron)
    % X_bounds   - [xmin, xmax] matrix (nx x 2)
    % U_bounds   - [umin, umax] matrix (nu x 2)
    % N          - horizon length (int)

    % Sizes
    nx = size(A, 1);
    nu = size(B, 2);

    % Polyhedral state and input constraints
    X = Polyhedron('lb', X_bounds(:,1), 'ub', X_bounds(:,2));
    U = Polyhedron('lb', U_bounds(:,1), 'ub', U_bounds(:,2));

    % Initialize terminal set
    XN = Xf;

    for i = 1:N
        % Build (x,u) ∈ Z s.t. A*x + B*u ∈ XN
        Axu = [XN.A * A, XN.A * B];     % (nXN x (nx+nu))
        bxu = XN.b;

        % x ∈ X → lifted as (x,u)
        Ax_lifted = [X.A, zeros(size(X.A,1), nu)];
        bx_lifted = X.b;

        % u ∈ U → lifted as (x,u)
        Au_lifted = [zeros(size(U.A,1), nx), U.A];
        bu_lifted = U.b;

        % Final stacked constraints
        A_z = [Axu; Ax_lifted; Au_lifted];
        b_z = [bxu; bx_lifted; bu_lifted];

        % (x,u) set
        Z = Polyhedron('A', A_z, 'b', b_z);

        % Project to get x
        Pre_X = Z.projection(1:nx);

        % Intersect with state bounds
        XN = X.intersect(Pre_X);

        % Stop early if set becomes empty
        if XN.isEmptySet
            warning("⚠️ Admissible set empty at step %d", i);
            break;
        end
    end
end
