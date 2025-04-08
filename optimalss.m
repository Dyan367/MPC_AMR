function[xr,ur]=optimalss(LTI,dim,weight,constraints,eqconstraints)

H=blkdiag(zeros(dim.nx),eye(dim.nu));
h=zeros(dim.nx+dim.nu,1);


options1 = optimoptions(@quadprog); 
options1.OptimalityTolerance=1e-6;
options1.ConstraintTolerance=1.0000e-6;
options1.Display='off';
[xur,~,exitflag]=quadprog(H,h,[],[],eqconstraints.A,eqconstraints.b,[],[],[],options1);
if exitflag ~= 1
    fprintf("❌ QP failed in optimalss(): exitflag = %d\n", exitflag);
    disp("eq_A = "); disp(eqconstraints.A);
    disp("eq_b = "); disp(eqconstraints.b);
    xr = NaN(dim.nx, 1);
    ur = NaN(dim.nu, 1);
    return;
end
xr=xur(1:dim.nx);
ur=xur(dim.nx+1:end);

end