function [ddq, lambda] = fb_leg_dynamics(qfull, dqfull, tau, params)
% qfull: 6x1 [xb; zb; phib; q1; q2; q3]
% dqfull: 6x1
% tau: 3x1 [tau1; tau2; tau3]
% params: {l1,l2,l3,c1,c2,c3,m1,m2,m3,I1,I2,I3,g}

    % Unpack model matrices numerically
    Mq   = M_leg_fb(qfull, params{:});          % 6x6 double
    hq   = h_leg_fb(qfull, dqfull, params{:});  % 6x1 double
    gq   = g_leg_fb(qfull, params{:});          % 6x1 double
    Jc_q = Jc_leg_fb(qfull, params{:});         % 3x6 double
    Jcdq = JcDotdq_leg_fb(qfull, dqfull, params{:}); % 3x1 double

    % Build full torque vector (base unactuated)
    tau_full = [0; 0; 0; tau(:)];               % 6x1

    % Build K and rhs numerically
    K   = [Mq,         -Jc_q.';
           Jc_q,  zeros(size(Jc_q,1))];

    rhs = [tau_full - hq - gq;
           -Jcdq];

    sol     = K \ rhs;
    ddq     = sol(1:6);
    lambda  = sol(7:end);
end
