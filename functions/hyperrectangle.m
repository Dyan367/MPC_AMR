function [A, b] = hyperrectangle(lb, ub)
    n = length(lb);
    A = [eye(n); -eye(n)];
    b = [ub; -lb];
end
