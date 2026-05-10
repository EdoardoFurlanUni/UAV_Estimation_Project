function [z] = func_f(x, dtheta, dv, dt)
%% f(x): Process Model for 16D UAV State
% Adapted from EFFI_EKF/process.m. Quaternion utilities inlined.
%
% Inputs:
%   x      : 16x1 state vector [q(4); v_ned(3); p_ned(3); wb(3); ab(3)]
%   dtheta : 3x1 IMU angular increment  (rad)
%   dv     : 3x1 IMU velocity increment (m/s)
%   dt     : sampling interval          (s)
%
% Output:
%   z  : 16x1 predicted state

x_new = zeros(16,1);

%% Quaternion update  [ref: RotToQuat, QuatMult, NormQuat from EFFI_EKF]
% Correct angular increment for estimated gyro bias
dtheta_b = dtheta(:) - x(11:13);

% Convert rotation vector to quaternion increment: dq = [cos(|v|/2); sin(|v|/2)*v/|v|]
theta = norm(dtheta_b);
if theta < 1e-6
    dq = [1; 0; 0; 0];          % identity quaternion for negligible rotation - avoid division by zero
else
    dq = [cos(theta/2); sin(theta/2)*dtheta_b/theta];
end

% Quaternion product q_new = q * dq  (Hamilton product)
q = x(1:4);
q_new = [q(1)*dq(1) - q(2:4)'*dq(2:4); ...
         q(1)*dq(2:4) + dq(1)*q(2:4) + cross(q(2:4), dq(2:4))];
q_new        = q_new / norm(q_new);   % normalize to unit quaternion
x_new(1:4)   = q_new;

%% Body-to-NED rotation matrix (Direction Cosine Matrix)  [ref: Quat2Tbn from EFFI_EKF]
% Tbn maps body-frame vectors to NED: v_ned = Tbn * v_body
q0=q_new(1); q1=q_new(2); q2=q_new(3); q3=q_new(4);
Tbn = [q0^2+q1^2-q2^2-q3^2,  2*(q1*q2-q0*q3),     2*(q1*q3+q0*q2); ...
       2*(q1*q2+q0*q3),       q0^2-q1^2+q2^2-q3^2,  2*(q2*q3-q0*q1); ...
       2*(q1*q3-q0*q2),       2*(q2*q3+q0*q1),      q0^2-q1^2-q2^2+q3^2];

%% Velocity update
% Correct velocity increment for estimated accel bias, rotate to NED,
% then add gravity (NED down convention: g = +9.8065 m/s^2 along z)
dv_b         = dv(:) - x(14:16);
v_prev       = x(5:7);
x_new(5:7)   = v_prev + Tbn*dv_b + [0; 0; 9.8065]*dt;

%% Position update (trapezoidal integration)
x_new(8:10)  = x(8:10) + 0.5*dt*(v_prev + x_new(5:7));

%% Biases: modeled as constant (random walk noise handled by Q via calcQ16)
x_new(11:16) = x(11:16);

z = x_new;
end
