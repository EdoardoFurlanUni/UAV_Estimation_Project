function [z] = func_h(x)
    %% h(x): Optical Flow Sensor + Barometer Measurement Model [Task 1]

    % x: 16x1 state vector
    % 1:4   - Quaternions (q0, q1, q2, q3)
    % 5:7   - Velocity NED (vn, ve, vd)
    % 8:10  - Position NED (pn, pe, pd)

    % z: 3x1 predicted output vector
    % 1:2   - X,Y Velocity Body (vx, vy) --> compared with optical flow measurements
    % 3     - Down Position NED (pd) --> compared with barometer measurement
    
    q = x(1:4);
    v_n = x(5:7);
    pd = x(10);
    
    % Direction Cosine Matrix (Body to NED) - R_b2n
    % We need NED to Body (R_n2b), which is the transpose
    q0 = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
    
    R_b2n = [q0^2 + q1^2 - q2^2 - q3^2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2); ...
             2*(q1*q2 + q0*q3), q0^2 - q1^2 + q2^2 - q3^2, 2*(q2*q3 - q0*q1); ...
             2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0^2 - q1^2 - q2^2 + q3^2];
    
    R_n2b = R_b2n';
    
    v_b = R_n2b * v_n;
    
    
    z = [v_b(1);
         v_b(2);
         pd];
end

