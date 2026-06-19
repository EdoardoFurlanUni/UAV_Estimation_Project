%% Task 2: Accuracy Verification of the Optical Flow Measurement Model
% =========================================================================
%   GPS NED velocity (ground truth) is fed through the measurement model h(x)
%   (which rotates v from n-frame to b-frame via quaternion attitude) to
%   produce the predicted body-frame velocities, compared against three
%   optical flow measurement levels.
%
%   Level 1 | Raw (sensor_optical_flow)
%              vel = (pixel_flow / timespan_s) * distance_m
%              Raw data from the sensor chip at variable sampling rate.
%              Not gyro-compensated.
%
%   Level 2 | Processed (vehicle_optical_flow)
%              Same velocity formula as Level 1, but data is resampled
%              by PX4 to a fixed sampling time. Not gyro-compensated.
%
%   Level 3 | PX4 EKF output (estimator_optical_flow_vel)
%              Gyro-compensated body/NE velocity from PX4 internal EKF
%      
%   Robust Standard Deviation (via MAD) is used instead of RMSE
%   to compare the accuracy of the three optical flow levels
%   against the model prediction, without being inflated by outliers.
% =========================================================================

clear; clc; close all;

%% 1. Load synchronized data

folder_num = input('Please enter the data number to be read:','s');
% 46_2025-10-18-10-11-28
% 47_2025-10-18-10-28-26
% 48_2025-10-18-10-40-54
% 49_2025-10-18-10-53-38
% 50_2025-10-18-11-09-00

project_dir = fileparts(mfilename('fullpath'));
filters_dir = fullfile(project_dir, '..', 'filters');
plots_dir   = fullfile(project_dir, 'PLOTS');
if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end
addpath(filters_dir);
num_only   = strtok(folder_num, '_');
data_path  = fullfile(project_dir, '..', 'Data', 'mat', sprintf('data_sync_%s.mat', num_only));

if ~exist(data_path, 'file')
    error('Please run Data/mat/DATA_PROCESS.m first to generate data_sync_%s.mat', num_only);
end
fprintf('Loading data from %s...\n', data_path);
load(data_path);
% Variables loaded: t_sync, gps_gt, q_sync, flow_v, raw_flow_v, veh_flow_v, ...

%% 2. Model prediction: h(GPS_velocity, attitude) -> predicted body velocity
N      = length(t_sync);
y_pred = zeros(N, 2);

fprintf('Running model prediction over %d samples...\n', N);
for k = 1 : N
    q     = q_sync(k, :)';          % quaternion from vehicle_attitude log
    v_ned = gps_gt(k, 1:3)';        % GPS NED velocity [vn; ve; vd] 

    % State vector for func_h (h(x)):
    %   [q0,q1,q2,q3 | vn,ve,vd | pn,pe,pd | wb | ab]
    x = [q; v_ned; zeros(9,1)];

    y_full      = func_h(x);   % returns [v_body_x; v_body_y; pd]
    y_pred(k,:) = y_full(1:2)';
end

%% 3. Measurement vectors 
y_raw = raw_flow_v(:, 1:2);   % Lv1 - body velocity from sensor_optical_flow [vx, vy]
y_veh = veh_flow_v(:, 1:2);   % Lv2 - body velocity from vehicle_optical_flow [vx, vy]
y_ekf = flow_v(:, 1:2);       % Lv3 - body velocity from estimator_optical_flow_vel [vx, vy]

%% 4. Robust Standard Deviation (via MAD)
% Scale factor to convert Median Absolute Deviation (MAD) to Gaussian standard deviation
scale_factor = 1.4826;

sigma_raw = median(abs(y_raw - y_pred)) * scale_factor;
sigma_veh = median(abs(y_veh - y_pred)) * scale_factor;
sigma_ekf = median(abs(y_ekf - y_pred)) * scale_factor;

fprintf('\n=== Robust Std Dev (sigma): h(x)|GPS vs Optical Flow measurements ===\n');
fprintf('  Level 1 - sensor_optical_flow        : sigma_vx=%.4f m/s  sigma_vy=%.4f m/s\n', sigma_raw(1), sigma_raw(2));
fprintf('  Level 2 - vehicle_optical_flow       : sigma_vx=%.4f m/s  sigma_vy=%.4f m/s\n', sigma_veh(1), sigma_veh(2));
fprintf('  Level 3 - estimator_optical_flow_vel : sigma_vx=%.4f m/s  sigma_vy=%.4f m/s\n', sigma_ekf(1), sigma_ekf(2));

%% Figure - All levels comparison (2 rows x 3 columns)
fig = figure('Name', 'Task2 - All levels comparison', ...
             'NumberTitle', 'off', 'Units', 'normalized', 'OuterPosition', [0 0.35 1 0.65]);

lw_meas = 0.6;  
lw_pred = 1.2;

% Colour scheme
c_lv1 = [0.15 0.50 0.85]; % actual measurement
c_lv2 = [0.85 0.45 0.05]; %    "        "
c_lv3 = [0.15 0.70 0.35]; %    "        "
c_pred = [0.05 0.05 0.05]; % model prediction (GPS)

col_names = {'Level 1 - sensor\_optical\_flow', ...
             'Level 2 - vehicle\_optical\_flow', ...
             'Level 3 - estimator\_optical\_flow\_vel'};

plot_sigma_vx = [sigma_raw(1), sigma_veh(1), sigma_ekf(1)];
plot_sigma_vy = [sigma_raw(2), sigma_veh(2), sigma_ekf(2)];

meas_x = {y_raw(:,1), y_veh(:,1), y_ekf(:,1)};
meas_y = {y_raw(:,2), y_veh(:,2), y_ekf(:,2)};
colors = {c_lv1, c_lv2, c_lv3};

ax = gobjects(2, 3);
for col = 1:3
    ax(1,col) = subplot(2, 3, col);
    hold on; grid on;
    plot(t_sync, meas_x{col}, '-', 'Color', [colors{col} 0.6], 'LineWidth', lw_meas, 'DisplayName', 'Measurement');
    plot(t_sync, y_pred(:,1), '-', 'Color', c_pred, 'LineWidth', lw_pred, 'DisplayName', 'Prediction (h(x)|GPS)');
    xlabel('Time (s)'); ylabel('v_{body,x}  (m/s)');
    title(col_names{col}, 'FontSize', 13);
    subtitle(sprintf('Robust \\sigma_{vx} = %.4f m/s', plot_sigma_vx(col)), 'FontSize', 10);
    legend('Location', 'northeast', 'FontSize', 6);
    xlim([t_sync(1), t_sync(end)]);

    ax(2,col) = subplot(2, 3, col+3);
    hold on; grid on;
    plot(t_sync, meas_y{col}, '-', 'Color', [colors{col} 0.6], 'LineWidth', lw_meas, 'DisplayName', 'Measurement');
    plot(t_sync, y_pred(:,2), '-', 'Color', c_pred, 'LineWidth', lw_pred, 'DisplayName', 'Prediction (h(x)|GPS)');
    xlabel('Time (s)'); ylabel('v_{body,y}  (m/s)');
    subtitle(sprintf('Robust \\sigma_{vy} = %.4f m/s', plot_sigma_vy(col)), 'FontSize', 10);
    legend('Location', 'northeast', 'FontSize', 6);
    xlim([t_sync(1), t_sync(end)]);
end

linkaxes(ax(1,:), 'y');   % same y-scale across all vx subplots
linkaxes(ax(2,:), 'y');   % same y-scale across all vy subplots

sgtitle('Task 2 - Optical Flow Measurement Model Accuracy Verification', 'FontWeight', 'bold');

% -------------------------------------------------------------------------
%% Save all figures
set(fig, 'PaperPositionMode', 'auto');
print(fig, fullfile(plots_dir, sprintf('Task2_%s', num_only)), '-dpng', '-r600');
fprintf('\nFigure saved: PLOTS/Task2_%s.png\n', num_only);

