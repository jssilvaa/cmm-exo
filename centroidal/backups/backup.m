%% Define symbolic variables and DH table 
nq = 6; % number of dofs 
syms q1 q2 q3 l1 l2 l3 real % joint variables and link lengths
syms dq1 dq2 dq3       real % joint velocities  
syms c1 c2 c3 m1 m2 m3 real % CoM lengths and masses
syms I1 I2 I3          real % link inertias 
syms g                 real % gravity 
syms xb zb phib        real % floating base coordinates 
syms dxb dzb dphib     real % floating base velocities
syms x0  z0  theta0    real % planar contact constraint 
syms tau1 tau2 tau3    real % actuated joints 
ddq    = sym('ddq', [nq, 1], 'real'); 
lambda = sym('lambda', [3,1], 'real'); 

% System parameters 
mtot = m1 + m2 + m3; 
qj     = [q1; q2; q3];
dqj    = [dq1; dq2; dq3]; 
qb     = [xb; zb; phib]; 
dqb    = [dxb; dzb; dphib]; 
qfull  = [qb; qj]; 
dqfull = [dqb; dqj];
tau_full = [0;0;0;tau1;tau2;tau3]; 

% Base Transform, i.e. {W}orld to {P}elvis
TB = [cos(phib) -sin(phib) 0 xb; 
      sin(phib) cos(phib)  0 zb; 
      0         0          1  0; 
      0         0          0  1]; 

% MDH Table Parameters
DH = [0, 0, 0,  q1; 
      l1, 0, 0, q2;
      l2, 0, 0, q3;
      l3, 0, 0, 0;]; 

% MDH Tables for the CoMs of each link 
% position to c1 i.e. hip CoM
DHc1 = [0, 0, 0, q1; 
        c1, 0, 0, 0]; 
% position to c2 i.e. shank CoM
DHc2 = [0, 0, 0, q1; 
        l1, 0, 0, q2; 
        c2, 0, 0, 0]; 
% position to c3 i.e. feet CoM 
DHc3 = [0, 0, 0, q1; 
        l1,0, 0, q2; 
        l2, 0, 0, q3;
        c3, 0, 0, 0]; 

% Compute Transformation Matrices
n = size(DH, 1);
T = computeFromDH(DH, n);

% hip CoM and jacobian
nc1 = size(DHc1, 1);
Tc1 = computeFromDH(DHc1, nc1);
pc1 = Tc1(1:3, 4);

% shank CoM and jacobian 
nc2 = size(DHc2, 1);
Tc2 = computeFromDH(DHc2, nc2);
pc2 = Tc2(1:3, 4);

% feet CoM and jacobian 
nc3 = size(DHc3, 1);
Tc3 = computeFromDH(DHc3, nc3);
pc3 = Tc3(1:3, 4);

% World frame positions 
pc1_global = TB * [pc1; 1]; 
pc2_global = TB * [pc2; 1];
pc3_global = TB * [pc3; 1]; 

pc1W = pc1_global(1:3);
pc2W = pc2_global(1:3); 
pc3W = pc3_global(1:3); 

pcW = (m1*pc1W + m2*pc2W + m3*pc3W) / mtot; 

% Get the velocity at each link CoM 
Jc1 = jacobian(pc1W, qfull);
Jc2 = jacobian(pc2W, qfull); 
Jc3 = jacobian(pc3W, qfull); 
 
vc1 = Jc1 * dqfull;
vc2 = Jc2 * dqfull;
vc3 = Jc3 * dqfull;

% Linear momentum 
p1 = m1*vc1; 
p2 = m2*vc2; 
p3 = m3*vc3;
l = p1 + p2 + p3;  % 3x1, [p_x; p_z; 0]

% Angular velocities
omega1 = dqfull(3) + dqfull(4);
omega2 = dqfull(3) + dqfull(4) + dqfull(5);
omega3 = dqfull(3) + dqfull(4) + dqfull(5) + dqfull(6);

% Angular momentum
r1 = pc1W - pcW; 
r2 = pc2W - pcW; 
r3 = pc3W - pcW; 

% Cross product y-components
cross1_y = r1(2)*p1(1) - r1(1)*p1(2);
cross2_y = r2(2)*p2(1) - r2(1)*p2(2);
cross3_y = r3(2)*p3(1) - r3(1)*p3(2);

h_y = I1*omega1 + cross1_y ...
    + I2*omega2 + cross2_y ...
    + I3*omega3 + cross3_y;

% Centroidal momentum vector 
h_vec = [h_y; l(1); l(2)]; 
A = simplify(jacobian(h_vec, dqfull));
hdot_kin = jacobian(h_vec, qfull)*dqfull + A*ddq;

% Compute Kinetic Energy 
K = 1/2 * m1 * (vc1.' * vc1) + 1/2 * I1 * omega1^2 ...
  + 1/2 * m2 * (vc2.' * vc2) + 1/2 * I2 * omega2^2 ... 
  + 1/2 * m3 * (vc3.' * vc3) + 1/2 * I3 * omega3^2; 

% Compute Potential Energy
V = m1 * g * pc1W(2) ... 
  + m2 * g * pc2W(2) ...
  + m3 * g * pc3W(2);

% Lagrangian Matrices
M  = sym('M' , [6,6]); 
h  = sym('h',  [6,1]); 
gq = sym('gq', [6,1]); 

% Compute the Mass-Inertia matrix
for i = 1:nq
    for j = 1:nq
        M(i,j) = simplify( diff(diff(K, dqfull(i)), dqfull(j)) ); 
    end
end

% Compute Coriolis/Centrifugal Matrix and Gravity Matrix
for i = 1:nq
    dK_dqi  = diff(K, qfull(i)); 
    dK_ddqi = diff(K, dqfull(i)); 

    dt_dK_ddqi = 0; 
   
    for k = 1:nq 
        dt_dK_ddqi = dt_dK_ddqi + diff(dK_ddqi, qfull(k)) * dqfull(k); 
    end

    h(i) = simplify(dt_dK_ddqi - dK_dqi); 
    gq(i) = simplify(diff(V, qfull(i))); 
end

% Transform from W to F 
Tfoot = TB * T; 
pfoot = Tfoot(1:3,4); % [x; z; 0] is the constraint 
phi   = [pfoot(1) - x0; 
         pfoot(2) - z0; 
         (phib + q1 + q2 + q3) - theta0]; 
Jc    =  jacobian(phi, qfull); 

% Constraint in velocity:     Jc *dqfull == 0
% Constraint in acceleration: Jc * dot dqfull + dot Jc * dqfull == 0
Jc_dot_dq = sym(zeros(size(phi,1),1));
for i = 1:size(phi,1)
    tmp = sym(0);
    for k = 1:nq
        dJ_row_dqk = diff(Jc(i,:), qfull(k));   % 1x6
        tmp = tmp + (dJ_row_dqk * dqfull) * dqfull(k);
    end
    Jc_dot_dq(i) = simplify(tmp);
end

% % Equations of motion 
% KK = [M,  -Jc.'; 
%      Jc, sym(zeros(size(Jc,1)))]; 
% 
% rhs = [tau_full - h - gq; -Jc_dot_dq]; 
% 
% sol = KK \ rhs; 
% ddq = sol(1:nq);
% lambda = sol(nq+1:end);
% 
% % static case: dq = 0, ddq = 0; 
% dq_static = zeros(nq, 1);  
% h_static  = subs(h, dqfull, dq_static); 
% 
% % Set external forces as zero: tau = 0 
% tau = sym(zeros(nq, 1));
% lambda_static = (Jc.') \ (gq - tau);

% Export Functions 
params = [l1; l2; l3; c1; c2; c3; m1; m2; m3; I1; I2; I3; g];

matlabFunction(M,    'File','M_leg_fb',    'Vars',{qfull, params},              'Outputs',{'Mq'});
matlabFunction(h,    'File','h_leg_fb',    'Vars',{qfull, dqfull, params},     'Outputs',{'hq'});
matlabFunction(gq,   'File','g_leg_fb',    'Vars',{qfull, params},              'Outputs',{'gq'});
matlabFunction(Jc,   'File','Jc_leg_fb',   'Vars',{qfull, params},              'Outputs',{'Jc_q'});
matlabFunction(Jc_dot_dq, ...
               'File','JcDotdq_leg_fb',   'Vars',{qfull, dqfull, params},      'Outputs',{'Jcdq'});
matlabFunction(h_vec,'File','h_centroidal_planar', ...
               'Vars',{qfull, dqfull, params}, 'Outputs',{'hvec'});
matlabFunction(A,    'File','A_centroidal_planar', ...
               'Vars',{qfull, params},          'Outputs',{'Aq'});


% Helper functions 
function T = computeFromDH(DH, n)
    T = eye(4); 
    for i = 1:n
       T_i = transform(DH(i, 1), DH(i, 2), DH(i, 3), DH(i, 4));
       T = T * T_i;
    end
    T = simplify(T); 
end

% General Transform 
function T = transform(a, alpha, d, theta)
    ca = cos(alpha); sa = sin(alpha);
    ct = cos(theta); st = sin(theta); 

    T = [
        ct, -st, 0, a; 
        st .* ca, ct .* ca, -sa, -sa .* d; 
        st .* sa, ct .* sa, ca, ca .* d;
        0, 0, 0, 1;
        ];
end 