%% Define symbolic variables and DH table 
nq = 6; % number of dofs 
syms q1 q2 q3 l1 l2 l3 real % joint variables and link lengths
syms dq1 dq2 dq3       real % joint velocities  
syms c1 c2 c3 m1 m2 m3 real % CoM lengths and masses
syms qF dqF cF lF      real % foot related DOFs. simple model accounts for just +1 DOF 
syms I1 I2 I3          real % link inertias 
syms g                 real % gravity 
syms xb zb phib        real % floating base coordinates 
syms dxb dzb dphib     real % floating base velocities
syms x0  z0  theta0    real % planar contact constraint. measured w.r.t. {W}orld Frame
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
RB = [ cos(phib) 0 sin(phib); 
       0          1 0; 
       -sin(phib) 0 cos(phib)];
TB = [RB [xb; 0; zb]; 0 0 0 1];

%  Joints transformations for 1, 2, 3 and foot 
% hip
T1 = Tlink(q1, c1);
pc1 = T1(1:3, 4); 
% knee
T2 = Tlink(q1, l1) * Tlink(q2, c2); 
pc2 = T2(1:3, 4); 
% ankle 
T3 = Tlink(q1, l1) * Tlink(q2, l2) * Tlink(q3, c3); 
pc3 = T3(1:3, 4); 
% foot 
Tf = Tlink(q1, l1) * Tlink(q2, l2) * Tlink(q3, l3) * Tlink(qF, cF); 
pcF = Tf(1:3, 4); 

% World frame positions 
pc1_global = TB * [pc1; 1]; 
pc2_global = TB * [pc2; 1];
pc3_global = TB * [pc3; 1];
pcF_global = TB * [pcF; 1]; 

pc1W = pc1_global(1:3);
pc2W = pc2_global(1:3); 
pc3W = pc3_global(1:3); 
pcF_W = pcF_global(1:3);

% ignore foot contribution for now. assume all the mass at the ankle. 
% use foot DOF for simple modelling. might need to account for it later in
% here.
pcW = (m1*pc1W + m2*pc2W + m3*pc3W) / mtot; 

% The following discussion does not use the foot DOF yet.
% is there need to? to investigate 

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
l = p1 + p2 + p3;  % 3x1, [p_x; 0; p_z]

% Angular velocities
omega1 = dqfull(3) + dqfull(4);
omega2 = dqfull(3) + dqfull(4) + dqfull(5);
omega3 = dqfull(3) + dqfull(4) + dqfull(5) + dqfull(6);

% Angular momentum
r1 = pc1W - pcW; 
r2 = pc2W - pcW; 
r3 = pc3W - pcW; 

% Cross product pitch components
cross1_y = r1(3)*p1(1) - r1(1)*p1(3);
cross2_y = r2(3)*p2(1) - r2(1)*p2(3);
cross3_y = r3(3)*p3(1) - r3(1)*p3(3);

h_y = I1*omega1 + cross1_y ...
    + I2*omega2 + cross2_y ...
    + I3*omega3 + cross3_y;

% Centroidal momentum vector 
h_vec = [h_y; l(1); l(3)]; % [pitch ang mom, x lin mom, z lin mom]
A        = jacobian(h_vec, dqfull);
hdot_kin = jacobian(h_vec, qfull) * dqfull + A * ddq;

% Compute Kinetic Energy 
K = 1/2 * m1 * (vc1.' * vc1) + 1/2 * I1 * omega1^2 ...
  + 1/2 * m2 * (vc2.' * vc2) + 1/2 * I2 * omega2^2 ... 
  + 1/2 * m3 * (vc3.' * vc3) + 1/2 * I3 * omega3^2; 

% Compute Potential Energy
V = m1 * g * pc1W(3) ... 
  + m2 * g * pc2W(3) ...
  + m3 * g * pc3W(3);

% Lagrangian Matrices. 
M  = sym('M' , [6,6]); 
C  = sym('h',  [6,1]); 
gq = sym('gq', [6,1]); 

% We use the chain rule to compute the dynamics matrices
% Compute the Mass-Inertia matrix
for i = 1:nq
    for j = 1:nq
        M(i,j) = diff(diff(K, dqfull(i)), dqfull(j)); 
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

    C(i) = dt_dK_ddqi - dK_dqi; 
    gq(i) = diff(V, qfull(i)); 
end

% Establish a simple, static foot constraint 
% For the single legged model  
phi   = [pcF_W(1) - x0; 
         pcF_W(3) - z0; 
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
    Jc_dot_dq(i) = tmp;
end

% % Equations of motion 
% The following solves rfor the evolution of the contact constraints and
% generalized coordinates. however matlab can't handle this symbolically.
% prefer numerical solutions, but the dynamical model is here for
% reference. 
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
params = [l1; l2; l3; c1; c2; c3; cF; qF; m1; m2; m3; I1; I2; I3; g];

matlabFunction(M,    'File','M_leg_fb',    'Vars',{qfull, params},              'Outputs',{'Mq'});
matlabFunction(C,    'File','C_leg_fb',    'Vars',{qfull, dqfull, params},     'Outputs',{'Cq'});
matlabFunction(gq,   'File','g_leg_fb',    'Vars',{qfull, params},              'Outputs',{'gq'});
matlabFunction(Jc,   'File','Jc_leg_fb',   'Vars',{qfull, params},              'Outputs',{'Jc_q'});
matlabFunction(Jc_dot_dq, ...
               'File','JcDotdq_leg_fb',   'Vars',{qfull, dqfull, params},      'Outputs',{'Jcdq'});
matlabFunction(h_vec,'File','h_centroidal_planar', ...
               'Vars',{qfull, dqfull, params}, 'Outputs',{'hvec'});
matlabFunction(A,    'File','A_centroidal_planar', ...
               'Vars',{qfull, params},          'Outputs',{'Aq'});


% Helper functions
% Compute CoB transformations between links 
% A modified modified DH
function T = Tlink(theta, a)
    c = cos(theta); s = sin(theta);
    R = [ c 0  s;
          0 1  0;
         -s 0  c ];
    p = R * [a;0;0];
    T = [R p; 0 0 0 1];
end
