%% GRID SEARCH: R_gps and R_flow tuning on EKF (dataset 49)
% =========================================================================
%   Tunes measurement noise covariances using EKF only.
%   R_gps RMSE is computed on GPS-available periods only.
%   R_flow RMSE is computed on GPS-denied period only.
%   After finding best R_gps and R_flow, runs a separate alpha grid on UKF.
%   Performs sequential 1D sweeps (Coordinate Descent) to change parameters 
%   individually.
% =========================================================================
clear; clc; close all;

data_num = '49';

%% Setup
%project_dir = fileparts(mfilename('fullpath'));
project_dir = pwd;
addpath(fullfile(project_dir, '..', 'filters'));
addpath(fullfile(project_dir, '..', 'Data', 'mat'));

data_path = fullfile(project_dir, '..', 'Data', 'mat', sprintf('data_sync_%s.mat', data_num));
fprintf('Loading dataset %s...\n', data_num);
load(data_path);

N  = length(t_sync);
dt = 1 / Delta;

%% GPS-denied interval
T_deny = 100; I_deny = 40;
gps_denied = denied(gps_mea, T_deny, I_deny, Delta);

deny_start = T_deny * Delta + 1;
deny_end   = min((T_deny + I_deny) * Delta, N);
mode_vec   = repmat({'gps'}, N, 1);
mode_vec(deny_start:deny_end) = {'flow'};

% Index masks for RMSE evaluation
idx_gps  = setdiff(1:N-1, deny_start:deny_end);   % GPS-available steps
idx_flow = deny_start:min(deny_end, N-1);         % GPS-denied steps

%% Init
x0 = [q_sync(1,:)'; gps_gt(1,1:3)'; gps_gt(1,4:5)'; -dist_h(1);
      2.7556e-6*ones(3,1); 6.7600e-11*ones(3,1)];
P0 = 1e-4 * eye(16);

Delta_theta_n = [2.6e-5; 2.6e-5; 2.6e-5];
Delta_v_n     = [1.66e-3; 1.66e-3; 1.66e-3];
wb            = [2.6e-6; 2.6e-6; 2.6e-6];
ab            = [1.66e-4; 1.66e-4; 1.66e-4];

%% EMA thresholds (precompute once, same for all grid runs)
lam = 0.98;
mu_x = abs(veh_flow_v(1,1)); var_x = 0;
mu_y = abs(veh_flow_v(1,2)); var_y = 0;
VEL_THR_x = zeros(N,1); VEL_THR_y = zeros(N,1);
VEL_THR_x(1) = 5.0; VEL_THR_y(1) = 5.0;

burn_in = 100;
for k = 2:N
    xk = veh_flow_v(k-1,1); yk = veh_flow_v(k-1,2);
    if k < burn_in || abs(xk) < VEL_THR_x(k-1)
        mu_x = lam*mu_x + (1-lam)*abs(xk);
        var_x = lam*var_x + (1-lam)*(xk-mu_x)^2;
    end
    if k < burn_in || abs(yk) < VEL_THR_y(k-1)
        mu_y = lam*mu_y + (1-lam)*abs(yk);
        var_y = lam*var_y + (1-lam)*(yk-mu_y)^2;
    end
    
    VEL_THR_x(k) = max(mu_x + 3 * sqrt(var_x), 1.0);
    VEL_THR_y(k) = max(mu_y + 3 * sqrt(var_y), 1.0);
end

%% Ground truth
gt_vel = gps_gt(1:N-1, 1:3);
gt_pos = [gps_gt(1:N-1, 4:5), -dist_h(1:N-1)];
calc_rmse = @(est, gt) sqrt(mean(sum((est - gt).^2, 2)));

%% Helper: run EKF with given R_gps and R_flow
run_ekf = @(R_gps, R_flow) run_ekf_inner(N, x0, P0, mode_vec, gps_denied, ...
    veh_flow_v, baro_h, dtheta, dv, Delta_theta_n, Delta_v_n, wb, ab, dt, ...
    R_gps, R_flow, VEL_THR_x, VEL_THR_y);

%% ========================================================================
%% PHASE 1: R_gps tuning (Sequential 1D Sweeps on GPS zones)
%% ========================================================================
fprintf('\n=== PHASE 1: R_gps tuning (1D Sweeps) ===\n');

% Base values
bv_h = 0.05; bv_d = 0.1;   % velocity std (m/s)
bp_h = 0.3;  bp_d = 0.4;   % position std (m)

gps_scales = []; % Skipped
m_gps = [0.5, 0.5, 0.001, 0.05, 0.05]; % Locked to optimal tuned values

% Bypass Phase 1 search to focus on Phase 2 & 3
fprintf('  => Phase 1 bypassed. Using optimal m_gps = [%.3f, %.3f, %.3f, %.3f, %.3f]\n', m_gps);

R_gps_best = diag([(m_gps(1)*bv_h)^2; (m_gps(2)*bv_h)^2; (m_gps(3)*bv_d)^2;
                   (m_gps(4)*bp_h)^2; (m_gps(5)*bp_h)^2; bp_d^2]);

%% ========================================================================
%% PHASE 2: R_flow tuning (Sequential 1D Sweeps on GPS-denied zone)
%% ========================================================================
fprintf('\n=== PHASE 2: R_flow tuning (1D Sweeps) ===\n');

flow_scales_list = {}; % Skipped
m_flow = [3.0, 0.25]; % Locked to optimal tuned values

% Bypass Phase 2 search to focus on Phase 3
fprintf('  => Phase 2 bypassed. Using optimal m_flow = [%.3f, %.3f]\n', m_flow);

bf_vx = 0.5; bf_vy = 0.4; bf_pd = 0.4;
R_flow_best = diag([(m_flow(1)*bf_vx)^2; (m_flow(2)*bf_vy)^2; bf_pd^2]);

%% ========================================================================
%% PHASE 3: Alpha grid (test on full timeline with UKF)
%% ========================================================================
fprintf('\n=== PHASE 3: Alpha tuning (RMSE on full timeline, UKF) ===\n');

alpha_scales = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
n_alpha = length(alpha_scales);
rmse_ukf_alpha = zeros(n_alpha, 1);

for ia = 1:n_alpha
    a = alpha_scales(ia);
    X_ukf = run_ukf_inner(N, x0, P0, mode_vec, gps_denied, ...
        veh_flow_v, baro_h, dtheta, dv, Delta_theta_n, Delta_v_n, wb, ab, dt, ...
        R_gps_best, R_flow_best, VEL_THR_x, VEL_THR_y, a);
    
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
