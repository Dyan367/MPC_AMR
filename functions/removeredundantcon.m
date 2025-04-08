function [idx, Aout, bout] = removeredundantcon(A, b)
    P = Polyhedron('A', A, 'b', b);
    P.minHRep();  % automatically removes redundant constraints
    Aout = P.A;
    bout = P.b;
    idx = 1:size(A, 1);  % dummy output if needed
end
