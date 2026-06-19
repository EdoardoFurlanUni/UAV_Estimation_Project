function [x_new, P_new] = UKF_UAV(x, y, P, R, dtheta, dv, Delta_theta_n, Delta_v_n, wb, ab, dt, mode, alpha)
%% UKF for 16D UAV state  [ref: filters/UKF.m, Algorithm steps 1-11]
%
% Inputs:
%   x    : 16x1  x_hat_t  (state prediction at time t)
%   y    : measurement vector (6x1 GPS+baro or 3x1 flow+baro)
%   P    : 16x16 P_tilde_t  (state covariance at time t)
%   R    : measurement noise covariance DD^T
%   dtheta, dv    : IMU angular/velocity increments (1x3 each)
%   Delta_theta_n : 3x1 angular increment noise std dev
%   Delta_v_n     : 3x1 velocity increment noise std dev
%   wb            : 3x1 gyro bias noise std dev
%   ab            : 3x1 accel bias noise std dev
%   dt   : sampling interval (s)
%   mode : 'gps' or 'flow'
%   alpha : (optional) UKF spread parameter, typically 1e-4 ~ 1
%           [ref: Wan & Van der Merwe, 2001]. Defaults to 0.1 if omitted.
%
% Outputs:
%   x_new : 16x1  x_hat_{t+1}    (predicted state)
%   P_new : 16x16 P_tilde_{t+1}  (predicted covariance)

if nargin < 13
    alpha = 0.1;
end
n = 16;

%% Scaling parameters  [ref: Wan & Van der Merwe, 2001]
% alpha is now an input parameter, defaults to 0.1
kapa   = 3 - n;    
lambda = alpha^2 * (kapa + n) - n;
beta   = 2;   % for Gaussian 2 is optimal

%% Weights Wc/Wm  
Wm(1:2*n)   = 1/(2*(n+lambda));
Wm(2*n+1)   = lambda/(n+lambda);
Wc(1:2*n)   = 1/(2*(n+lambda));
Wc(2*n+1)   = lambda/(n+lambda) + 1 - alpha^2 + beta;

%% Generate sigma points X^(i) of x_hat_t  
P = (P + P')/2 + 1e-10 * eye(n);
sqrtP = chol(P);
X = zeros(n, 2*n+1);
for i = 1:n
    X(:,i)   = x + sqrt(n+lambda) * sqrtP(i,:)';
    X(:,i+n) = x - sqrt(n+lambda) * sqrtP(i,:)';
end
X(:,2*n+1) = x;

%% Compute Y^(i) = h(X^(i)) and predicted measurement y_pred  
m      = size(R, 1);
Y      = zeros(m, 2*n+1);
y_pred = zeros(m, 1);

if strcmp(mode, 'gps')
    C = [zeros(3,4), eye(3), zeros(3,9);
         zeros(3,4), zeros(3,3), eye(3), zeros(3,6)];
    for i = 1:2*n+1
        Y(:,i)  = C * X(:,i);
        y_pred  = y_pred + Wm(i) * Y(:,i);
    end
elseif strcmp(mode, 'flow')
    for i = 1:2*n+1
        Y(:,i)  = func_h(X(:,i));
        y_pred  = y_pred + Wm(i) * Y(:,i);
    end
end

%% Find the Kalman estimation gain  
P_y  = zeros(m, m);
P_xy = zeros(n, m);
for i = 1:2*n+1
    P_y  = P_y  + Wc(i) * (Y(:,i) - y_pred) * (Y(:,i) - y_pred)';
    P_xy = P_xy + Wc(i) * (X(:,i) - x)      * (Y(:,i) - y_pred)';
end
P_y = P_y + R;
L   = P_xy / P_y;

%% Compute x_hat_{t|t}
x_hat = x + L * (y - y_pred);
x_hat(1:4) = x_hat(1:4) / norm(x_hat(1:4));

%% Compute P_{t|t}
P_hat = P - L * P_y * L';

%% Generate sigma points X_hat^(i) of x_hat_{t|t}  
P_hat = (P_hat + P_hat')/2 + 1e-10 * eye(n);
sqrtP_hat = chol(P_hat);
X_hat = zeros(n, 2*n+1);
for i = 1:n
    X_hat(:,i)   = x_hat + sqrt(n+lambda) * sqrtP_hat(i,:)';
    X_hat(:,i+n) = x_hat - sqrt(n+lambda) * sqrtP_hat(i,:)';
end
X_hat(:,2*n+1) = x_hat;

%% Compute X_{t+1}^(i) = f(X_hat^(i)) and its mean x_hat_{t+1}  
X_pred = zeros(n, 2*n+1);
x_new  = zeros(n, 1);
for i = 1:2*n+1
    X_pred(:,i) = func_f(X_hat(:,i), dtheta, dv, dt);
    x_new       = x_new + Wm(i) * X_pred(:,i);
end

%% Compute the prediction variance P_{t+1}  
% Q_t time-varying: depends on quaternion of x_hat_{t|t}
q0 = x_hat(1); q1 = x_hat(2); q2 = x_hat(3); q3 = x_hat(4);
Q = calcQ16(wb, ab, Delta_theta_n, Delta_v_n, q0, q1, q2, q3);
P_new = zeros(n, n);
for i = 1:2*n+1
    P_new = P_new + Wc(i) * (X_pred(:,i) - x_new) * (X_pred(:,i) - x_new)';
end
P_new = P_new + Q;

end
