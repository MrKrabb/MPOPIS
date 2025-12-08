function [x_hist, psi, X, Y, delta_H_hist] = simulate_vehicle( ...
                    beta_ref, a, dt, params, x0, Kp, Ki, Kd)
%SIMULATE_VEHICLE_SS_PID  Vehicle model with PID control of beta.
%
%   States: x = [beta; r; v]
%   Inputs: u = [delta_H; a]
%
%   [x_hist, psi, X, Y, delta_H_hist] = simulate_vehicle_ss_pid( ...
%           beta_ref, a, dt, params, x0, Kp, Ki, Kd)
%
%   beta_ref : [N×1] reference for beta (rad)
%   a        : [N×1] longitudinal acceleration (m/s^2)
%   dt       : sample time (s)
%   params   : struct (m, cv, ch, lv, lh, theta)
%   x0       : 3×1 initial state [beta0; r0; v0]
%   Kp, Ki, Kd : PID gains
%
%   Outputs:
%     x_hist      : [N×3] state history [beta, r, v]
%     psi         : [N×1] yaw angle history
%     X, Y        : [N×1] global position
%     delta_H_hist: [N×1] applied steering (rad)

    N = numel(beta_ref);
    if numel(a) ~= N
        error('beta_ref and a must have the same length.');
    end

    % Preallocate
    x_hist       = zeros(N, 3);
    psi          = zeros(N, 1);
    X            = zeros(N, 1);
    Y            = zeros(N, 1);
    delta_H_hist = zeros(N, 1);

    % Initial conditions
    x = x0(:);           % [r; beta; v]
    x_hist(1,:) = x.';
    psi(1) = 0;
    X(1)   = 0;
    Y(1)   = 0;

    % PID memory
    e_prev = 0;
    I      = 0;

    % Optional steering saturation (e.g. ±30°)
    delta_max = pi/4;

    for k = 1:N-1
        % ---- 1) measure output and compute error ----
        beta_k = x(2);
        e      = beta_ref(k) - beta_k;

        % ---- 2) PID update ----
        I    = I + e * dt;           % integral term
        D    = (e - e_prev) / dt;    % derivative term
        e_prev = e;

        delta_H = Kp*e + Ki*I + Kd*D;

        % optional saturation + simple anti-windup
        if delta_H > delta_max
            delta_H = delta_max;
        elseif delta_H < -delta_max
            delta_H = -delta_max;
        end

        delta_H_hist(k) = delta_H;

        % ---- 3) state-space update with u = [delta_H; a(k)] ----
        u   = [delta_H; a(k)];
        v_k = x(3);                       % current velocity
        [A,B] = AB_matrices(v_k, params); % LPV matrices

        x_dot = A * x + B * u;
        x     = x + x_dot * dt;           % Euler integration

        % store state
        x_hist(k+1,:) = x.';

        % ---- 4) yaw and position update (kinematics) ----
        r_k    = x_hist(k,1);
        beta_k = x_hist(k,2);
        v_k    = x_hist(k,3);

        psi(k+1) = psi(k) + r_k * dt;
        X(k+1)   = X(k) + v_k * cos(psi(k) + beta_k) * dt;
        Y(k+1)   = Y(k) + v_k * sin(psi(k) + beta_k) * dt;
    end

    % last steering value (just repeat previous)
    delta_H_hist(end) = delta_H_hist(end-1);
end

% -------------------------------------------------------------------------
function [A,B] = AB_matrices(v, p)
    %y not included, would just be y=I[3x3]*x
    % Avoid division by zero
    v_eff = max(v, 1e-6);

    m     = p.m;
    cv    = p.cv;
    ch    = p.ch;
    lv    = p.lv;
    lh    = p.lh;
    theta = p.theta;

    % a11 = -(cv + ch) / (m * v_eff);
    % a12 = (m * v_eff^2 - (ch*lh - cv*lv)) / (m * v_eff^2);
    % a21 = -(ch*lh - cv*lv) / theta;
    % a22 = -(ch*lh^2 + cv*lv^2) / (theta * v_eff);

    a11 = -(cv*lv^2 + ch*lh^2) / (theta * v);

    a12 = -(cv*lv - ch*lh) / theta;

    a21 = -1 - (cv*lv - ch*lh) / (m * v^2);

    a22 = -(cv + ch) / (m * v);

    b21 =  cv / (m * v_eff);
    b11 =  cv * lv / (theta);

    A = [a11, a12, 0;
         a21, a22, 0;
         0,   0,   0];

    B = [b11, 0;
         b21, 0;
         0,   1];
end
