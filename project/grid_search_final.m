% % GRID SEARCH : R_gps and R_flow tuning on EKF(dataset 49) 
% =========================================================================
% Tunes measurement noise covariances using EKF only.
% R_gps RMSE is computed on GPS-available periods only.
% R_flow RMSE is computed on GPS-denied period only.
% After finding best R_gps and R_flow runs a separate alpha grid on UKF.
% Performs sequential 1D sweeps(Coordinate Descent) to change parameters
% individually.
% ========================================================================= 
clear;
clc;
close all;

data_num = '49';

project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir, '..', 'filters'));
addpath(fullfile(project_dir, '..', 'Data', 'mat'));

data_path = fullfile(project_dir, '..', 'Data', 'mat', sprintf('data_sync_%s.mat', data_num));
fprintf('Loading dataset %s...\n', data_num);
load(data_path);

N = length(t_sync);
dt = 1 / Delta;

% GPS-denied interval
T_tot = t_sync(N) - t_sync(1);
I_deny = 25;
window = round((T_tot - 2 * I_deny) / 3);
T_deny_1 = window;
T_deny_2 = 2 * window + I_deny;

gps_denied_1 = denied(gps_mea, T_deny_1, I_deny, Delta);
gps_denied   = denied(gps_denied_1, T_deny_2, I_deny, Delta);

deny_start_1 = T_deny_1 * Delta + 1;
deny_end_1   = min((T_deny_1 + I_deny) * Delta, N);
deny_start_2 = T_deny_2 * Delta + 1;
deny_end_2   = min((T_deny_2 + I_deny) * Delta, N);
mode_vec = repmat({'gps'}, N, 1);
mode_vec(deny_start_1 : deny_end_1) = {'flow'};
mode_vec(deny_start_2 : deny_end_2) = {'flow'};

% Index masks for RMSE evaluation
idx_gps = setdiff(1:N-1, [deny_start_1:deny_end_1, deny_start_2:deny_end_2]);
% %Init
x0 = [q_sync(1, :)'; ...
      gps_gt(1, 1:3)'; ...
      gps_gt(1, 4:5)'; ...
      -dist_h(1); ...
      2.7556e-6 * ones(3, 1); ...
      6.7600e-11 * ones(3, 1)];

P0 = 1e-4 * eye(16);

Delta_theta_n = [2.6e-5; 2.6e-5; 2.6e-5];
Delta_v_n = [1.66e-3; 1.66e-3; 1.66e-3];
wb = [2.6e-6; 2.6e-6; 2.6e-6];
ab = [1.66e-4; 1.66e-4; 1.66e-4];

%% EMA thresholds (precompute once, same for all grid runs)
lam = 0.98;
mu_x = abs(veh_flow_v(1, 1));
var_x = 0;
mu_y = abs(veh_flow_v(1, 2));
var_y = 0;
VEL_THR_x = zeros(N, 1);
VEL_THR_y = zeros(N, 1);
VEL_THR_x(1) = 5.0;
VEL_THR_y(1) = 5.0;

burn_in = 100;
for k = 2 : N 
    xk = veh_flow_v(k - 1, 1);
    yk = veh_flow_v(k - 1, 2);
    if k < burn_in || abs(xk) < VEL_THR_x(k - 1)
        mu_x = lam * mu_x + (1 - lam) * abs(xk);
        var_x = lam * var_x + (1 - lam) * (abs(xk) - mu_x) ^ 2;
    end
    if k < burn_in || abs(yk) < VEL_THR_y(k - 1)
        mu_y = lam * mu_y + (1 - lam) * abs(yk);
        var_y = lam * var_y + (1 - lam) * (abs(yk) - mu_y) ^ 2;
    end
    VEL_THR_x(k) = max(mu_x + 3 * sqrt(var_x), 1.0);
    VEL_THR_y(k) = max(mu_y + 3 * sqrt(var_y), 1.0);
end

idx = 1:N-1;
gt_vel = gps_gt(idx, 1:3);
gt_pos = [gps_gt(1:N-1, 4:5), -dist_h(1:N-1)];
calc_rmse = @(est, gt) sqrt(mean(sum((est - gt).^2, 2)));

% Baseline covariance for Optical Flow
R_flow = diag([0.5^2; 0.4^2; 0.4^2]);

%% ========================================================================
%% PHASE 1: R_gps tuning (Sequential 1D Sweeps on GPS zones)
%% ========================================================================
fprintf('\n=== PHASE 1: R_gps tuning (1D Sweeps) ===\n');

% Base values for GPS measurement noise std devs (from task34)
bv_h = 0.05; bv_d = 0.1;   % velocity std (m/s)
bp_h = 0.3;  bp_d = 0.4;   % position std (m)

% Define the search grid for the multipliers
gps_scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
n_scales = length(gps_scales);

% Initialize multipliers to 1.0
m_gps = [1.0, 1.0, 1.0, 1.0, 1.0];

% --- Sweep 1: Tune v_N velocity multiplier m_gps(1) ---
fprintf('\n--- Sweeping v_N velocity multiplier ---\n');
rmse_p_gps_vn = zeros(n_scales, 1);
for i = 1:n_scales
    s = gps_scales(i);
    R_gps_test = diag([(s*bv_h)^2; (m_gps(2)*bv_h)^2; (m_gps(3)*bv_d)^2; ...
                       (m_gps(4)*bp_h)^2; (m_gps(5)*bp_h)^2; bp_d^2]);
    
    % Run EKF
    X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_test;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md);
    end
    
    rmse_p_gps_vn(i) = calc_rmse(X_ekf(8:10, idx_gps)', gt_pos(idx_gps, :));
    fprintf('  multiplier = %g => Pos RMSE (GPS) = %.4f m\n', s, rmse_p_gps_vn(i));
end
[~, best_i] = min(rmse_p_gps_vn);
m_gps(1) = gps_scales(best_i);
fprintf('>>> Best v_N velocity multiplier: %g\n', m_gps(1));


% --- Sweep 2: Tune v_E velocity multiplier m_gps(2) ---
fprintf('\n--- Sweeping v_E velocity multiplier ---\n');
rmse_p_gps_ve = zeros(n_scales, 1);
for i = 1:n_scales
    s = gps_scales(i);
    R_gps_test = diag([(m_gps(1)*bv_h)^2; (s*bv_h)^2; (m_gps(3)*bv_d)^2; ...
                       (m_gps(4)*bp_h)^2; (m_gps(5)*bp_h)^2; bp_d^2]);
    
    % Run EKF
    X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_test;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md);
    end
    
    rmse_p_gps_ve(i) = calc_rmse(X_ekf(8:10, idx_gps)', gt_pos(idx_gps, :));
    fprintf('  multiplier = %g => Pos RMSE (GPS) = %.4f m\n', s, rmse_p_gps_ve(i));
end
[~, best_i] = min(rmse_p_gps_ve);
m_gps(2) = gps_scales(best_i);
fprintf('>>> Best v_E velocity multiplier: %g\n', m_gps(2));


% --- Sweep 3: Tune v_D velocity multiplier m_gps(3) ---
fprintf('\n--- Sweeping v_D velocity multiplier ---\n');
rmse_p_gps_vd = zeros(n_scales, 1);
for i = 1:n_scales
    s = gps_scales(i);
    R_gps_test = diag([(m_gps(1)*bv_h)^2; (m_gps(2)*bv_h)^2; (s*bv_d)^2; ...
                       (m_gps(4)*bp_h)^2; (m_gps(5)*bp_h)^2; bp_d^2]);
    
    % Run EKF
    X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_test;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md);
    end
    
    rmse_p_gps_vd(i) = calc_rmse(X_ekf(8:10, idx_gps)', gt_pos(idx_gps, :));
    fprintf('  multiplier = %g => Pos RMSE (GPS) = %.4f m\n', s, rmse_p_gps_vd(i));
end
[~, best_i] = min(rmse_p_gps_vd);
m_gps(3) = gps_scales(best_i);
fprintf('>>> Best v_D velocity multiplier: %g\n', m_gps(3));


% --- Sweep 4: Tune p_N position multiplier m_gps(4) ---
fprintf('\n--- Sweeping p_N position multiplier ---\n');
rmse_p_gps_pn = zeros(n_scales, 1);
for i = 1:n_scales
    s = gps_scales(i);
    R_gps_test = diag([(m_gps(1)*bv_h)^2; (m_gps(2)*bv_h)^2; (m_gps(3)*bv_d)^2; ...
                       (s*bp_h)^2; (m_gps(5)*bp_h)^2; bp_d^2]);
    
    % Run EKF
    X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_test;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md);
    end
    
    rmse_p_gps_pn(i) = calc_rmse(X_ekf(8:10, idx_gps)', gt_pos(idx_gps, :));
    fprintf('  multiplier = %g => Pos RMSE (GPS) = %.4f m\n', s, rmse_p_gps_pn(i));
end
[~, best_i] = min(rmse_p_gps_pn);
m_gps(4) = gps_scales(best_i);
fprintf('>>> Best p_N position multiplier: %g\n', m_gps(4));


% --- Sweep 5: Tune p_E position multiplier m_gps(5) ---
fprintf('\n--- Sweeping p_E position multiplier ---\n');
rmse_p_gps_pe = zeros(n_scales, 1);
for i = 1:n_scales
    s = gps_scales(i);
    R_gps_test = diag([(m_gps(1)*bv_h)^2; (m_gps(2)*bv_h)^2; (m_gps(3)*bv_d)^2; ...
                       (m_gps(4)*bp_h)^2; (s*bp_h)^2; bp_d^2]);
    
    % Run EKF
    X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_test;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md);
    end
    
    rmse_p_gps_pe(i) = calc_rmse(X_ekf(8:10, idx_gps)', gt_pos(idx_gps, :));
    fprintf('  multiplier = %g => Pos RMSE (GPS) = %.4f m\n', s, rmse_p_gps_pe(i));
end
[~, best_i] = min(rmse_p_gps_pe);
m_gps(5) = gps_scales(best_i);
fprintf('>>> Best p_E position multiplier: %g\n', m_gps(5));

% Build best R_gps for Phase 2
R_gps_best = diag([(m_gps(1)*bv_h)^2; (m_gps(2)*bv_h)^2; (m_gps(3)*bv_d)^2; ...
                   (m_gps(4)*bp_h)^2; (m_gps(5)*bp_h)^2; bp_d^2]);

% % =========================================================================
% % PHASE 2 : R_flow tuning (Sequential 1D Sweeps on GPS-denied zone)
% % =========================================================================

fprintf('\n=== PHASE 2: R_flow tuning (1D Sweeps) ===\n');

bf_vx = 0.5; bf_vy = 0.4; bf_pd = 0.4;
flow_scales_x = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
flow_scales_y = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];

idx_flow = setdiff(1:N-1, idx_gps);

% --- Sweep 1: Tune m_flow_x (with m_flow_y fixed at 1.0) ---
fprintf('\n--- Sweeping m_flow_x (m_flow_y = 1.0) ---\n');
n_fx = length(flow_scales_x);
rmse_p_flow_x = zeros(n_fx, 1);

for ix = 1:n_fx
    mx = flow_scales_x(ix);
    R_flow_test = diag([(mx * bf_vx)^2; (1.0 * bf_vy)^2; bf_pd^2]);
    
    % Run EKF
    X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_best;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow_test;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md);
    end
    
    rmse_p_flow_x(ix) = calc_rmse(X_ekf(8:10, idx_flow)', gt_pos(idx_flow, :));
    fprintf('  m_flow_x = %.2f => Pos RMSE (denied) = %.4f m\n', mx, rmse_p_flow_x(ix));
end

[~, best_ix] = min(rmse_p_flow_x);
best_m_flow_x = flow_scales_x(best_ix);
fprintf('>>> Best m_flow_x = %.2f\n', best_m_flow_x);

% Sweep 2: Tune m_flow_y (with m_flow_x fixed)
fprintf('\n--- Sweeping m_flow_y (m_flow_x = %.2f) ---\n', best_m_flow_x);
n_fy = length(flow_scales_y);
rmse_p_flow_y = zeros(n_fy, 1);

for iy = 1:n_fy
    my = flow_scales_y(iy);
    R_flow_test = diag([(best_m_flow_x * bf_vx)^2; (my * bf_vy)^2; bf_pd^2]);
    
    % Run EKF
    X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps_best;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow_test;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md);
    end
    
    rmse_p_flow_y(iy) = calc_rmse(X_ekf(8:10, idx_flow)', gt_pos(idx_flow, :));
    fprintf('  m_flow_y = %.2f => Pos RMSE (denied) = %.4f m\n', my, rmse_p_flow_y(iy));
end

[~, best_iy] = min(rmse_p_flow_y);
best_m_flow_y = flow_scales_y(best_iy);
fprintf('>>> Best m_flow_y = %.2f\n', best_m_flow_y);

m_flow = [best_m_flow_x, best_m_flow_y];
R_flow_best = diag([(best_m_flow_x * bf_vx)^2; (best_m_flow_y * bf_vy)^2; bf_pd^2]);

% % =========================================================================
% % PHASE 3 : Alpha grid (test on full timeline with UKF)
% % =========================================================================

fprintf('\n=== PHASE 3: Alpha tuning (RMSE on full timeline, UKF) ===\n');

alpha_scales = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
n_alpha = length(alpha_scales);
rmse_ukf_alpha = zeros(n_alpha, 1);

for ia = 1 : n_alpha
    a = alpha_scales(ia);
    X_ukf = run_ukf_inner(N, x0, P0, mode_vec, gps_denied, veh_flow_v, baro_h, dtheta, dv, Delta_theta_n, Delta_v_n, wb, ab, dt, R_gps_best, R_flow_best, VEL_THR_x, VEL_THR_y, a);

    rmse_ukf_alpha(ia) = calc_rmse(X_ukf(8:10, 1:N-1)', gt_pos(1:N-1,:));
    fprintf('  alpha=%.4f => UKF Pos RMSE = %.4f m\n', a, rmse_ukf_alpha(ia));
end

[~, mi_a] = min(rmse_ukf_alpha);
best_alpha = alpha_scales(mi_a);
fprintf('\n>>> BEST alpha: %.2f (RMSE=%.4f m)\n\n', best_alpha, rmse_ukf_alpha(mi_a));

%% ========================================================================
%% SUMMARY
%% ========================================================================
fprintf('==========================================================\n');
fprintf('          TUNING RESULTS - Dataset %s\n', data_num);
fprintf('==========================================================\n');
fprintf('Best R_gps multipliers:  [%.3f, %.3f, %.3f, %.3f, %.3f] (v_N, v_E, v_D, p_N, p_E)\n', m_gps);
fprintf('  = diag([%.4f, %.4f, %.4f, %.4f, %.4f, %.4f])\n', diag(R_gps_best)');
fprintf('Best R_flow multipliers: [%.3f, %.3f] (v_X, v_Y)\n', m_flow);
fprintf('  = diag([%.4f, %.4f, %.4f])\n', diag(R_flow_best)');
fprintf('Best alpha (UKF): %.2f\n', best_alpha);
fprintf('==========================================================\n');

%% ========================================================================
%% Local function: run EKF
%% ========================================================================
function X = run_ekf_inner(N, x0, P0, mode_vec, gps_denied, ...
    veh_flow_v, baro_h, dtheta, dv, Delta_theta_n, Delta_v_n, wb, ab, dt, ...
    R_gps, R_flow, VEL_THR_x, VEL_THR_y)

    X = zeros(16, N); X(:,1) = x0; P = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X(:,k+1), P] = EKF_UAV(X(:,k), y_k, P, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md);
    end
end

%% Local function: run UKF
function X = run_ukf_inner(N, x0, P0, mode_vec, gps_denied, ...
    veh_flow_v, baro_h, dtheta, dv, Delta_theta_n, Delta_v_n, wb, ab, dt, ...
    R_gps, R_flow, VEL_THR_x, VEL_THR_y, alpha)

    X = zeros(16, N); X(:,1) = x0; P = P0;
    for k = 1:N-1
        md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
        if strcmp(md, 'gps')
            y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps;
        else
            y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
            if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
            if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
        end
        [X(:,k+1), P] = UKF_UAV(X(:,k), y_k, P, R_k, dth, dvk, ...
            Delta_theta_n, Delta_v_n, wb, ab, dt, md, alpha);
    end
end