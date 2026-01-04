%% demo_lip.m
% Demonstrate: (1) h = A(q) dq (CMM identity)
%              (2) LIP emerges from centroidal angular momentum regulation
%                  via  \dot h_y = (u-x)Fz + z Fx  and  \dot h_y = 0.
%              (3) CoP bounds + friction feasibility checks.

clear; clc; close all;

%% --- Make sure repo is on path ---
this = fileparts(mfilename('fullpath'));
root = fullfile(this, '..');  % from tests/ to root
models = fullfile(root, 'models/'); % from root to models/ 
addpath(genpath(root));
addpath(genpath(models));

%% --- Require generated functions ---
req = {@A_centroidal_planar, @h_centroidal_planar, @pcom_planar};
for i = 1:numel(req)
    fn = func2str(req{i});
    if exist(fn, 'file') ~= 2
        error(['Missing function "', fn, '".\n', ...
               'Generate it (or add it to MATLAB path) before running this demo.\n']);
    end
end

%% --- reasonable lip parameters (single-leg "effective mass") ---
% Geometry [m]
l1 = 0.45; l2 = 0.45; l3 = 0.10;
c1 = 0.22; c2 = 0.22; c3 = 0.05;

% Foot parameter placeholders (treated as constants)
cF = 0.00;
qF = 0.00;  % keep as PARAMETER (not in qfull)

% Masses [kg] (choose mtot ~ human+exo effective for LIP-like behavior)
m1 = 25; m2 = 25; m3 = 25;
mtot = m1 + m2 + m3;

% Link inertias about pitch axis [kg m^2] (toy values)
I1 = 1.2; I2 = 1.0; I3 = 0.6;

% Gravity
g  = 9.81;

% Contact feasibility
mu = 0.6;       % friction coefficient
L  = 0.24;      % foot length [m]
xfoot = 0.0;    % stance foot center in world (toy assumption)
umin = xfoot - L/2;
umax = xfoot + L/2;

%% --- configuration for the planar chain ---
% Convention: sagittal plane is x-z, rotation about y.
% Tlink uses R_y(theta) and translates by R*[a;0;0].
% links point upward (+z), q1=-pi/2 and q2=q3=0.
xb   = 0.00;
zb   = 1.00;    % pelvis height [m] (not CoM height)
phib = 0.00;

q1 = -pi/2;
q2 = 0.0;
q3 = 0.0;

qfull = [xb; zb; phib; q1; q2; q3];

%% --- Choose a simple generalized velocity: translate base in +x ---
dxb   = 0.25;   % m/s
dzb   = 0.00;
dphib = 0.00;

dq1 = 0.00;
dq2 = 0.00;
dq3 = 0.00;

dqfull = [dxb; dzb; dphib; dq1; dq2; dq3];

%% --- Pack params ---
% params = [l1;l2;l3;c1;c2;c3;cF;qF;m1;m2;m3;I1;I2;I3;g];
params = [l1; l2; l3; c1; c2; c3; cF; qF; m1; m2; m3; I1; I2; I3; g];

%% --- centroidal quantities ---
A = A_centroidal_planar(qfull, params);
h = h_centroidal_planar(qfull, dqfull, params);   % h = [h_y; p_x; p_z]
pcom = pcom_planar(qfull, params);                % [x; y; z]

x = pcom(1);
z = pcom(3);

hy = h(1);
px = h(2);
pz = h(3);

%% --- CMM identity check: h ?= A(q) dq ---
h_from_A = A * dqfull;
err = h - h_from_A;

fprintf('--- CMM identity check: h - A*dq ---\n');
fprintf('h        = [hy; px; pz] = [%+.6f; %+.6f; %+.6f]\n', hy, px, pz);
fprintf('A*dq     =               [%+.6f; %+.6f; %+.6f]\n', h_from_A(1), h_from_A(2), h_from_A(3));
fprintf('error    =               [%+.3e; %+.3e; %+.3e]\n', err(1), err(2), err(3));
fprintf('||error||_2 = %.3e\n\n', norm(err));

%% --- Build the LIP reduction from centroidal angular momentum regulation ---
% Centroidal wrench identity (planar):
%   \dot h_y = (u - x) Fz + z Fx
%
% If we choose \dot h_y^* = 0 (kill angular momentum about CoM),
% and assume standing: Fz ≈ mtot*g,
% then Fx = (Fz/z) (x - u).
%
% Also, \dot p_x = Fx and p_x = mtot * xdot  =>  xddot = Fx/mtot
%
% => xddot = (g/z) (x - u)   (LIP equation with z constant)

Fz = mtot * g;

% Infer CoM horizontal velocity from centroidal linear momentum:
xdot = px / mtot;

omega = sqrt(g / z);        % "effective" LIP omega from actual CoM height
xi = x + xdot/omega;        % capture point (DCM)

% Stable DCM control: set xi_dot_des = -kxi*(xi - xi_ref)
kxi = 3.0;           % [1/s] DCM convergence rate (try 1..6)
xi_ref = xfoot;

xi_dot_des = -kxi*(xi - xi_ref);
u_star = xi - (1/omega)*xi_dot_des;   % = xi + (kxi/omega)*(xi - xi_ref)
u = min(max(u_star, umin), umax);

% CoP saturation (sheet foot)
u = min(max(u_star, umin), umax);
sat = (u ~= u_star);

% Choose Fx to enforce \dot h_y = 0 (centroidal angular momentum regulation)
Fx = (Fz / z) * (x - u);

% Check implied accelerations
xdd_from_wrench = Fx / mtot;
xdd_lip_formula = (g / z) * (x - u);

% Verify centroidal identity for \dot h_y under that wrench
hdot_y = (u - x)*Fz + z*Fx;   % should be ~0 by construction

% Friction feasibility
fric_ratio = abs(Fx) / (mu * Fz);

fprintf('--- LIP as centroidal angular-momentum regulation ---\n');
fprintf('CoM: x=%.4f m, z=%.4f m\n', x, z);
fprintf('px=%.4f -> xdot=%.4f m/s\n', px, xdot);
fprintf('omega=sqrt(g/z)=%.4f 1/s\n', omega);
fprintf('DCM xi=%.4f m, u* (desired CoP)=%.4f m\n', xi, u_star);
fprintf('CoP bounds [%.4f, %.4f], u (sat)=%.4f, saturated=%d\n', umin, umax, u, sat);
fprintf('Assume Fz=mg=%.2f N\n', Fz);
fprintf('Choose Fx=(Fz/z)(x-u)=%.2f N\n', Fx);
fprintf('xdd (Fx/m)=%.4f  vs  (g/z)(x-u)=%.4f\n', xdd_from_wrench, xdd_lip_formula);
fprintf('hdot_y=(u-x)Fz+zFx = %.3e  (target 0)\n', hdot_y);
fprintf('|Fx|/(mu Fz)=%.4f  (<=1 feasible static friction)\n\n', fric_ratio);

%% --- (4) Tiny "simulation": iterate centroidal regulation to cancel h_y ---
% We iterate a reduced-order update on (x, xdot) but recompute h_y via CMM
% each step to show: "CMM-based centroidal quantity is regulated".

Ts = 0.005;      % timestep [s]
Tsim = 10.0;      % total sim time [s]
N = floor(Tsim/Ts) + 1;
t = (0:N-1)*Ts;

% Reduced state for toy propagation (LIP):
x_sim  = zeros(N,1);
xd_sim = zeros(N,1);

% Logs from CMM at each step:
hy_sim = zeros(N,1);
px_sim = zeros(N,1);
u_sim  = zeros(N,1);
Fx_sim = zeros(N,1);
sat_sim = false(N,1);

% Initial conditions come from your current snapshot:
x_sim(1)  = x;
xd_sim(1) = xdot;

% Disturbance: kick the COM velocity at t = 0.5 s
t_kick = 0.5;
dv_kick = 0.1;     % [m/s] try 0.3..1.0
kick_k = round(t_kick/Ts) + 1;

% Control gains:
k_dcm = 0.35;        % same as before
k_h   = 3.0;         % [1/s] angular momentum damping gain (hy -> 0)

% Keep CoM height constant for LIP propagation:
z_c = z;             % use your computed CoM height as constant
omega = sqrt(g / z_c);

for kstep = 1:N
    % Apply disturbance kick once
    if kstep == kick_k
        xd_sim(kstep) = xd_sim(kstep) + dv_kick;
    end

    % ---- Reconstruct q and dq for CMM evaluation ----
    % Minimal consistent "lift" from toy state to generalized coordinates:
    % - base x follows x_sim
    % - base z and angles fixed (standing)
    qk  = qfull;
    dqk = dqfull;

    qk(1) = x_sim(kstep);      % xb
    dqk(1) = xd_sim(kstep);    % dxb

    % Evaluate centroidal momentum using exported functions
    hk = h_centroidal_planar(qk, dqk, params);
    hy_sim(kstep) = hk(1);
    px_sim(kstep) = hk(2);

    % ---- Centroidal momentum-rate target (disturbance cancellation) ----
    % We want to damp centroidal angular momentum:
    %   \dot h_y^* = -k_h * h_y
    %
    % Using planar centroidal wrench identity:
    %   \dot h_y = (u - x)Fz + z_c Fx
    % Solve for Fx given u (CoP). We'll choose u via DCM plus a correction
    % term from the desired \dot h_y^*.

    hdoty_des = -k_h * hy_sim(kstep);

    % --- DCM control (must recompute xi each step) ---
    xi = x_sim(kstep) + xd_sim(kstep)/omega;
    
    kxi = 0.4;      % [1/s] start small to avoid constant saturation
    xi_ref = xfoot;
    
    xi_dot_des = -kxi*(xi - xi_ref);
    u_star = xi - (1/omega)*xi_dot_des;        % = xi + (kxi/omega)*(xi - xi_ref)
    
    u_k = min(max(u_star, umin), umax);
    sat_sim(kstep) = (u_k ~= u_star);
    u_sim(kstep) = u_k;


    % Compute Fx needed to achieve hdoty_des (given u_k)
    %   hdoty_des = (u-x)Fz + z_c Fx  =>  Fx = (hdoty_des - (u-x)Fz)/z_c
    Fx_k = (hdoty_des - (u_k - x_sim(kstep))*Fz) / z_c;
    Fx_sim(kstep) = Fx_k;

    % Optional: friction clamp (keeps it physically honest)
    Fx_max = mu * Fz;
    Fx_k = min(max(Fx_k, -Fx_max), Fx_max);

    % Propagate reduced LIP state:
    %   xdd = Fx/m  (since Fx = \dot p_x and p_x = m xdot)
    xdd = Fx_k / mtot;

    if kstep < N
        % simple semi-implicit Euler (good enough for this toy)
        xd_sim(kstep+1) = xd_sim(kstep) + Ts*xdd;
        x_sim(kstep+1)  = x_sim(kstep)  + Ts*xd_sim(kstep+1);
    end
end

%% --- Plot: CMM-measured centroidal angular momentum h_y decays ---
figure;
plot(t, hy_sim, 'LineWidth', 2); grid on;
xlabel('t [s]');
ylabel('h_y [N·m·s]');
title('CMM-based centroidal angular momentum regulation (h_y \rightarrow 0)');

figure;
plot(t, x_sim, 'LineWidth', 2); grid on;
xlabel('t [s]'); ylabel('x [m]');
title('CoM horizontal position (toy propagation)');

figure;
hold on; grid on;
plot(t, u_sim, 'LineWidth', 2);
yline(umin, '--'); yline(umax, '--');
xlabel('t [s]'); ylabel('u = x_{CoP} [m]');
title('CoP command with bounds (sheet foot)');
legend('u','u_{min}','u_{max}','Location','best');



%% --- plots ---
% figure;
% bar([hy, px, pz]);
% grid on;
% set(gca,'XTickLabel',{'h_y','p_x','p_z'});
% ylabel('SI units (N·m·s for h_y, N·s for p)');
% title('Centroidal momentum components (from exported functions)');
% 
% figure;
% hold on; grid on;
% plot([umin umax],[0 0],'k-','LineWidth',4);            % support interval
% plot(u_star, 0, 'rx','MarkerSize',12,'LineWidth',2);
% plot(u,      0, 'bo','MarkerSize',10,'LineWidth',2);
% plot(x,      0, 'g.','MarkerSize',24);
% legend('support [u_{min},u_{max}]','u* desired','u saturated','CoM x','Location','best');
% xlabel('x [m]');
% title('CoP saturation (sheet foot)');
% 
% figure;
% bar(fric_ratio);
% grid on;
% yline(1.0,'--');
% ylabel('|F_x| / (\mu F_z)');
% title('Friction usage ( > 1 means slip required )');
% 
% fprintf('Done.\n');
