%% Task 3,4: EKF, UKF, REKF and RUKF with Optical Flow Measurement Model (GPS-Denied)
% =========================================================================
%   Runs filters on the full dataset (SELECT dataset 48 or 49).
%   A GPS-denied interval is simulated via denied.m: during that window
%   the filter switches from 'gps' to 'flow'.
%   GPS mode  : y = [vn; ve; vd; pn; pe; pd_baro]  R = R_gps (6x6)
%   Flow mode : y = [v_bx; v_by; pd_baro]          R = R_flow (3x3)
%   Optical Flow Level 2: vehicle_optical_flow
%   Outliers are rejected using Channel-wise Measurement Gating (Dynamic R).
%   Performance is evaluated using FULL-FLIGHT 3D RMSE.
% =========================================================================

clear; clc; close all;

data_num  = '49'; % SELECT dataset 48 or 49 --> change c_rekf and c_rukf

% Robustness parameters (TUNED: optimized for minimum 3D Position RMSE)
% c_grid = [1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05] for tuning
c_rekf = 1e-06; % 1e-08 IF dataset 48, 1e-06 IF dataset 49
c_rukf = 1e-07; % 1e-07 IF dataset 48, 1e-07 IF dataset 49

%% 1. Setup paths and load data  
project_dir = fileparts(mfilename('fullpath'));
filters_dir = fullfile(project_dir, '..', 'filters');
data_dir    = fullfile(project_dir, '..', 'Data', 'mat');
addpath(filters_dir);
addpath(data_dir);
data_path = fullfile(project_dir, '..', 'Data', 'mat', sprintf('data_sync_%s.mat', data_num));

if ~exist(data_path, 'file')
    error('Please run Data/mat/DATA_PROCESS.m first to generate data_sync_%s.mat', data_num);
end
fprintf('Loading data from %s...\n', data_path);
load(data_path);
% Variables: t_sync, Delta, dtheta, dv, gps_gt, gps_mea, baro_h, dist_h,
%            q_sync, veh_flow_v, ...

N  = length(t_sync);
dt = 1 / Delta;

%% 2. GPS-denied interval
% T_deny: start of denial (s from beginning), I_deny: duration (s)
T_deny = 100;
I_deny = 100;

gps_denied = denied(gps_mea, T_deny, I_deny, Delta);

% Mode vector: 'gps' outside denial window, 'flow' inside
deny_start = T_deny * Delta + 1;
deny_end   = min((T_deny + I_deny) * Delta, N);
mode_vec   = repmat({'gps'}, N, 1);
mode_vec(deny_start:deny_end) = {'flow'};

%% 3. Initialization
start_idx = 1;
x0 = [q_sync(start_idx,:)'; ...
      gps_gt(start_idx,1:3)'; ...
      gps_gt(start_idx,4:5)'; ...
      -dist_h(start_idx); ...
      2.7556e-6*ones(3,1); ... % [ref: paper]
      6.7600e-11*ones(3,1)];   % [ref: paper]

P0 = 1e-4*eye(16); 

%% 4. Noise parameters & Gating Threshold
% Process noise [ref: calcQ16]
Delta_theta_n = [2.6e-5; 2.6e-5; 2.6e-5;];     % angular increment noise std dev
Delta_v_n     = [1.66e-3; 1.66e-3; 1.66e-3;];  % velocity increment noise std dev
wb            = [2.6e-6; 2.6e-6; 2.6e-6;];     % gyro bias random walk std dev
ab            = [1.66e-4; 1.66e-4; 1.66e-4;];  % accel bias random walk std dev 

% Measurement noise
R_gps  = diag([0.05^2*ones(2,1); 0.1^2; 0.3^2*ones(2,1); 0.4^2]);  % 6x6: vel (m/s), pos (m)
R_flow = diag([0.5^2; 0.4^2; 0.4^2]);  % (task2)                   % 3x3: vbx, vby (m/s), pd (m)

% Dynamic outlier thresholds: mean 3 sigma computed from data
mean_x = mean(abs(veh_flow_v(:,1)), 'omitnan');
std_x  = std(veh_flow_v(:,1), 'omitnan');
mean_y = mean(abs(veh_flow_v(:,2)), 'omitnan');
std_y  = std(veh_flow_v(:,2), 'omitnan');

VEL_THRESHOLD_x = mean_x + 3 * std_x;
VEL_THRESHOLD_y = mean_y + 3 * std_y;

fprintf('====================================================\n');
fprintf('Optical Flow Gating Thresholds (mean + 3*sigma):\n');
fprintf('  X-axis: mean=%.3f, std=%.3f -> threshold=%.3f m/s\n', mean_x, std_x, VEL_THRESHOLD_x);
fprintf('  Y-axis: mean=%.3f, std=%.3f -> threshold=%.3f m/s\n', mean_y, std_y, VEL_THRESHOLD_y);
fprintf('====================================================\n');

%% 5. Filter Loops (EKF, UKF, REKF, RUKF)
fprintf('Running all filters...\n');

X_ekf = zeros(16, N); X_ekf(:,1) = x0; P_ekf = P0;
X_ukf = zeros(16, N); X_ukf(:,1) = x0; P_ukf = P0;
X_rekf= zeros(16, N); X_rekf(:,1)= x0; P_rekf= P0; theta_rekf = zeros(N, 1);
X_rukf= zeros(16, N); X_rukf(:,1)= x0; P_rukf= P0; theta_rukf = zeros(N, 1);

% EKF
tic;
gated_x = 0; gated_y = 0; flow_total = 0;
for k = 1:N-1
    md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);

    if strcmp(md, 'gps'), y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps;
    else
        y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
        flow_total = flow_total + 1;
        if abs(y_k(1)) > VEL_THRESHOLD_x, R_k(1,1) = R_k(1,1) * 1e6; gated_x = gated_x + 1; end
        if abs(y_k(2)) > VEL_THRESHOLD_y, R_k(2,2) = R_k(2,2) * 1e6; gated_y = gated_y + 1; end
    end
    
    [X_ekf(:,k+1), P_ekf] = EKF_UAV(X_ekf(:,k), y_k, P_ekf, R_k, dth, dvk, Delta_theta_n, Delta_v_n, wb, ab, dt, md);
end
time_ekf = toc;
fprintf('EKF_UAV done in %.2f s\n', time_ekf);
fprintf('  Gating stats: %d/%d X-axis rejected, %d/%d Y-axis rejected\n', gated_x, flow_total, gated_y, flow_total);

% UKF
tic;
for k = 1:N-1
    md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);

    if strcmp(md, 'gps'), y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps;
    else
        y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
        if abs(y_k(1)) > VEL_THRESHOLD_x, R_k(1,1) = R_k(1,1) * 1e6; end
        if abs(y_k(2)) > VEL_THRESHOLD_y, R_k(2,2) = R_k(2,2) * 1e6; end
    end

    [X_ukf(:,k+1), P_ukf] = UKF_UAV(X_ukf(:,k), y_k, P_ukf, R_k, dth, dvk, Delta_theta_n, Delta_v_n, wb, ab, dt, md);
end
time_ukf = toc;
fprintf('UKF_UAV done in %.2f s\n', time_ukf);

% REKF
tic;
for k = 1:N-1
    md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);

    if strcmp(md, 'gps'), y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps;
    else
        y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
        if abs(y_k(1)) > VEL_THRESHOLD_x, R_k(1,1) = R_k(1,1) * 1e6; end
        if abs(y_k(2)) > VEL_THRESHOLD_y, R_k(2,2) = R_k(2,2) * 1e6; end
    end

    [X_rekf(:,k+1), P_rekf, theta_rekf(k)] = REKF_UAV(X_rekf(:,k), y_k, P_rekf, R_k, dth, dvk, Delta_theta_n, Delta_v_n, wb, ab, dt, md, c_rekf);
end
time_rekf = toc;
fprintf('REKF_UAV done in %.2f s\n', time_rekf);

% RUKF
tic;
for k = 1:N-1
    md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);

    if strcmp(md, 'gps'), y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps;
    else
        y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
        if abs(y_k(1)) > VEL_THRESHOLD_x, R_k(1,1) = R_k(1,1) * 1e6; end
        if abs(y_k(2)) > VEL_THRESHOLD_y, R_k(2,2) = R_k(2,2) * 1e6; end
    end

    [X_rukf(:,k+1), P_rukf, theta_rukf(k)] = RUKF_UAV(X_rukf(:,k), y_k, P_rukf, R_k, dth, dvk, Delta_theta_n, Delta_v_n, wb, ab, dt, md, c_rukf);
end
time_rukf = toc;
fprintf('RUKF_UAV done in %.2f s\n', time_rukf);

%% 6. Compute 3D RMSE (FULL TIMELINE)
% Use 1:N-1 to align with filter simulation steps
idx = 1:N-1;

gt_vel_full = gps_gt(idx, 1:3);
gt_pos_full = [gps_gt(idx, 4:5), -dist_h(idx)]; 

calc_rmse = @(est, gt) sqrt(mean(sum((est - gt).^2, 2)));

% 3D RMSE (Velocity)
rmse_v_3d = [calc_rmse(X_ekf(5:7, idx)', gt_vel_full), ...
             calc_rmse(X_ukf(5:7, idx)', gt_vel_full), ...
             calc_rmse(X_rekf(5:7, idx)', gt_vel_full), ...
             calc_rmse(X_rukf(5:7, idx)', gt_vel_full)];

% 3D RMSE (Position)
rmse_p_3d = [calc_rmse(X_ekf(8:10, idx)', gt_pos_full), ...
             calc_rmse(X_ukf(8:10, idx)', gt_pos_full), ...
             calc_rmse(X_rekf(8:10, idx)', gt_pos_full), ...
             calc_rmse(X_rukf(8:10, idx)', gt_pos_full)];

fprintf('\n==========================================================\n');
fprintf('           PERFORMANCE SUMMARY (FULL TIMELINE)\n');
fprintf('==========================================================\n');
fprintf('Filter | Exec Time (s) | 3D Pos RMSE (m) | 3D Vel RMSE (m/s)\n');
fprintf('----------------------------------------------------------\n');
fprintf('EKF    | %12.2f  | %14.4f  | %16.4f\n', time_ekf, rmse_p_3d(1), rmse_v_3d(1));
fprintf('UKF    | %12.2f  | %14.4f  | %16.4f\n', time_ukf, rmse_p_3d(2), rmse_v_3d(2));
fprintf('REKF   | %12.2f  | %14.4f  | %16.4f\n', time_rekf, rmse_p_3d(3), rmse_v_3d(3));
fprintf('RUKF   | %12.2f  | %14.4f  | %16.4f\n', time_rukf, rmse_p_3d(4), rmse_v_3d(4));
fprintf('==========================================================\n\n');

%% 7. Shared Plot Configurations
lw_ref = 1.5;
lw_est = 1.2;

labels_v = {'v_N (m/s)', 'v_E (m/s)', 'v_D (m/s)'};
labels_p = {'p_N (m)',   'p_E (m)',   'p_D (m)'};

rmse_text = sprintf('3D Position RMSE:\n EKF: %.3fm\n UKF: %.3fm\n REKF: %.3fm\n RUKF: %.3fm\n\n3D Velocity RMSE:\n EKF: %.3fm/s\n UKF: %.3fm/s\n REKF: %.3fm/s\n RUKF: %.3fm/s', ...
    rmse_p_3d(1), rmse_p_3d(2), rmse_p_3d(3), rmse_p_3d(4), rmse_v_3d(1), rmse_v_3d(2), rmse_v_3d(3), rmse_v_3d(4));

%% 8. FIGURE 1: VELOCITY (Separate Figure)
fig_vel = figure('Name', 'Task 3,4 - Velocity', 'NumberTitle', 'off', 'Units', 'normalized', 'OuterPosition', [0 0.05 1 0.9]);
set(fig_vel, 'DefaultAxesPosition', [0.15 0.1 0.7 0.85]); % Margins for text box and legend

for j = 1:3
    subplot(3, 1, j); hold on; grid on;
    plot(t_sync, gps_gt(:,j),      'k--', 'LineWidth', lw_ref, 'DisplayName', 'Ground truth');
    plot(t_sync, X_ekf(4+j,:)',    'r-',  'LineWidth', lw_est, 'DisplayName', 'EKF');
    plot(t_sync, X_ukf(4+j,:)',    'b-',  'LineWidth', lw_est, 'DisplayName', 'UKF');
    plot(t_sync, X_rekf(4+j,:)',   'r--', 'LineWidth', lw_ref, 'DisplayName', 'REKF');
    plot(t_sync, X_rukf(4+j,:)',   'b--', 'LineWidth', lw_ref, 'DisplayName', 'RUKF');
    xregion(t_sync(deny_start), t_sync(deny_end), 'FaceColor', [0.9 0.9 0], 'FaceAlpha', 0.2, 'DisplayName', 'GPS denied');
    ylabel(labels_v{j}, 'FontWeight', 'normal');
    if j == 3, xlabel('Time (s)'); end
    xlim([t_sync(1) t_sync(end)]);
    
    if j == 1
        title('Task 3,4 - Velocity EKF vs UKF vs REKF vs RUKF: GPS-denied with Optical Flow', 'FontSize', 18);
        % MATLAB Trick to put legend outside without shrinking the axis
        pos = get(gca, 'Position'); 
        legend('Location', 'northeastoutside');
        set(gca, 'Position', pos);
    end
end
annotation('textbox', [0.01 0.75 0.1 0.2], 'String', rmse_text, 'EdgeColor', 'k', 'BackgroundColor', 'w', 'FitBoxToText', 'on', 'FontSize', 10);

%% 9. FIGURE 2: POSITION (Separate Figure)
fig_pos = figure('Name', 'Task 3,4 - Position', 'NumberTitle', 'off', 'Units', 'normalized', 'OuterPosition', [0 0.05 1 0.9]);
set(fig_pos, 'DefaultAxesPosition', [0.15 0.1 0.7 0.85]); % Margins for text box and legend

for j = 1:3
    subplot(3, 1, j); hold on; grid on;
    ref_p = (j < 3) * gps_gt(:, 3+j) + (j == 3) * (-dist_h);
    plot(t_sync, ref_p,            'k--', 'LineWidth', lw_ref, 'DisplayName', 'Ground truth');
    plot(t_sync, X_ekf(7+j,:)',    'r-',  'LineWidth', lw_est, 'DisplayName', 'EKF');
    plot(t_sync, X_ukf(7+j,:)',    'b-',  'LineWidth', lw_est, 'DisplayName', 'UKF');
    plot(t_sync, X_rekf(7+j,:)',   'r--', 'LineWidth', lw_ref, 'DisplayName', 'REKF');
    plot(t_sync, X_rukf(7+j,:)',   'b--', 'LineWidth', lw_ref, 'DisplayName', 'RUKF');
    xregion(t_sync(deny_start), t_sync(deny_end), 'FaceColor', [0.9 0.9 0], 'FaceAlpha', 0.2, 'DisplayName', 'GPS denied');
    ylabel(labels_p{j}, 'FontWeight', 'normal');
    if j == 3, xlabel('Time (s)'); end
    xlim([t_sync(1) t_sync(end)]);
    
    if j == 1
        title('Task 3,4 - Position EKF vs UKF vs REKF vs RUKF: GPS-denied with Optical Flow', 'FontSize', 18);
        % MATLAB Trick to put legend outside without shrinking the axis
        pos = get(gca, 'Position'); 
        legend('Location', 'northeastoutside');
        set(gca, 'Position', pos);
    end
end
annotation('textbox', [0.01 0.75 0.1 0.2], 'String', rmse_text, 'EdgeColor', 'k', 'BackgroundColor', 'w', 'FitBoxToText', 'on', 'FontSize', 10);

%% 10. FIGURE 3: THETA EVOLUTION
fig_theta = figure('Name', 'Task 3 - Theta Parameter', 'NumberTitle', 'off', 'Units', 'normalized', 'OuterPosition', [0 0.3 1 0.4]);
hold on; grid on;
plot(t_sync(1:N-1), theta_rekf(1:N-1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'REKF \theta');
plot(t_sync(1:N-1), theta_rukf(1:N-1), 'b--', 'LineWidth', 1.5, 'DisplayName', 'RUKF \theta');
xregion(t_sync(deny_start), t_sync(deny_end), 'FaceColor', [0.9 0.9 0], 'FaceAlpha', 0.2, 'DisplayName', 'GPS denied');
ylabel('\theta', 'FontWeight', 'normal', 'FontSize', 12); 
xlabel('Time (s)');
title('Robust Parameter (\theta) Evolution over Time', 'FontSize', 12);
legend('Location', 'northeastoutside');
xlim([t_sync(1) t_sync(end)]);

% %% 10. FIGURE 3: THETA EVOLUTION
% fig_theta = figure('Name', 'Task 3 - Theta Parameter', 'NumberTitle', 'off', 'Units', 'normalized', 'OuterPosition', [0 0.3 1 0.4]);
% ax_theta = axes(fig_theta);
% hold(ax_theta, 'on'); grid(ax_theta, 'on');
% plot(ax_theta, t_sync(1:N-1), theta_rekf(1:N-1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'REKF \theta');
% plot(ax_theta, t_sync(1:N-1), theta_rukf(1:N-1), 'b--', 'LineWidth', 1.5, 'DisplayName', 'RUKF \theta');
% xregion(ax_theta, t_sync(deny_start), t_sync(deny_end), 'FaceColor', [0.9 0.9 0], 'FaceAlpha', 0.2, 'DisplayName', 'GPS denied');  % <-- ax_theta aggiunto
% ylabel(ax_theta, '\theta', 'FontSize', 12);
% xlabel(ax_theta, 'Time (s)');
% title(ax_theta, 'Robust Parameter (\theta) Evolution over Time', 'FontSize', 12);
% legend(ax_theta, 'Location', 'northeastoutside');
% xlim(ax_theta, [t_sync(1) t_sync(end)]);

%% 11. Save Figures
set(fig_vel, 'PaperPositionMode', 'auto');
set(fig_pos, 'PaperPositionMode', 'auto');
set(fig_theta, 'PaperPositionMode', 'auto');
print(fig_vel, fullfile(project_dir, sprintf('Task34_Velocity_%s', data_num)), '-dpng', '-r600');
print(fig_pos, fullfile(project_dir, sprintf('Task34_Position_%s', data_num)), '-dpng', '-r600');
figure(fig_theta);
print(fig_theta, fullfile(project_dir, sprintf('Task34_Theta_%s', data_num)), '-dpng', '-r600');
fprintf('Figures saved successfully.\n');