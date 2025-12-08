%% Single lange model parameters and simulation

clear all

% Parameters
params.m     = 2120;
params.cv    = 72100;
params.ch    = 61800;
params.lv    = 1.3;
params.lh    = 1.5;
params.theta = 3862;

dt      = 0.01;
t       = 0:dt:10;
N       = numel(t);


%a= 0.5*ones(N,1);       % constant accel
a = 0*sin(t);

x0 = [0; 0.5; 5];                  % r0, beta0, v0

beta_ref = ones(N,1);       % wanted slip angle

% PID gains (start simple, then tune)
Kp = 1;
Ki = 0;
Kd = 0;


%delta_H = sin(t);


% % make sure the steering angle doesnt go over the max 
% delta_min=-pi/4;
% delta_max=pi/4;
% for f = 1:length(delta_H)
%     if (delta_H(f) > delta_max)
%         delta_H(f) = delta_max;
%     elseif (delta_H(f) < delta_min)
%         delta_H(f) = delta_min;
%     end
% end
% 
% [x_hist, psi, X, Y] = simulate_vehicle(delta_H(:), a(:), dt, params, x0);

[x_hist, psi, X, Y, delta_H] = simulate_vehicle(beta_ref, a, dt, params, x0, Kp, Ki, Kd);

r    = x_hist(:,1);
beta = x_hist(:,2);
v    = x_hist(:,3);

nexttile
plot(X, Y); axis equal; grid on
xlabel('X [m]'); ylabel('Y [m]');
title('Fahrzeugtrajektorie');

% Top plot
nexttile
plot(t,delta_H)
title('Lenkwinkel')

% Bottom plot
nexttile
plot(t,v)
title('Geschwindigkeit')

% Bottom plot
nexttile
plot(t,beta)
title('Schwimmwinkel')