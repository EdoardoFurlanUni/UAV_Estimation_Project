%% TUNE_ROBUST: Script to find the optimal robustness parameter (c)
% =========================================================================
% Performs a 1D grid search on the robustness parameter 'c' for both 
% REKF and RUKF across multiple datasets (48 and 49).
% Uses the previously tuned R_gps, R_flow, and alpha parameters.
% =========================================================================
clear; clc; close all;

project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir, '..', 'filters'));
addpath(fullfile(project_dir, '..', 'Data', 'mat'));

datasets = {'46', '47', '48', '49', '50'};
c_scales = logspace(-12, -4, 9);
n_c = length(c_scales);

best_c_rekf = zeros(length(datasets), 1);
best_c_rukf = zeros(length(datasets), 1);

%% Tuned Parameters (From grid_search & alpha tuning)
R_gps  = diag([(0.5*0.05)^2; (0.5*0.05)^2; (0.001*0.1)^2; (0.05*0.3)^2; (0.05*0.3)^2; 0.4^2]); 
R_flow = diag([(3.0*0.5)^2; (0.25*0.4)^2; 0.4^2]);
best_alpha = 5e-4;

fprintf('==========================================================\n');
fprintf('          STARTING ROBUSTNESS TUNING (c)                  \n');
fprintf('==========================================================\n');

for d = 1:length(datasets)
    data_num = datasets{d};
    fprintf('\n>>> Processing Dataset %s...\n', data_num);
    
    %% Setup & Load
    project_dir = fileparts(mfilename('fullpath'));
    data_path = fullfile(project_dir, '..', 'Data', 'mat', sprintf('data_sync_%s.mat', data_num));
    load(data_path);
    
    N  = length(t_sync);
    dt = 1 / Delta;
    
    %% GPS-denied interval
    T_tot = t_sync(N) - t_sync(1);
    I_deny = 25;
    
    % two gps_denied window equidistributed
    window = round((T_tot - 2 * I_deny) / 3);
    
    T_deny_1 = window;
    T_deny_2 = 2 * window + I_deny;
    
    gps_denied_1 = denied(gps_mea, T_deny_1, I_deny, Delta);
    gps_denied = denied(gps_denied_1, T_deny_2, I_deny, Delta);
    
    % Mode vector : 'gps' outside denial window, 'flow' inside
    deny_start_1 = T_deny_1 * Delta + 1;
    deny_end_1   = min((T_deny_1 + I_deny) * Delta, N);
    deny_start_2 = T_deny_2 * Delta + 1;
    deny_end_2   = min((T_deny_2 + I_deny) * Delta, N);
    mode_vec = repmat({'gps'}, N, 1);
    mode_vec(deny_start_1 : deny_end_1) = {'flow'};
    mode_vec(deny_start_2 : deny_end_2) = {'flow'};
    
    %% EMA thresholds
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
            var_x = lam*var_x + (1-lam)*(abs(xk)-mu_x)^2;
        end
        if k < burn_in || abs(yk) < VEL_THR_y(k-1)
            mu_y = lam*mu_y + (1-lam)*abs(yk);
            var_y = lam*var_y + (1-lam)*(abs(yk)-mu_y)^2;
        end
        VEL_THR_x(k) = max(mu_x + 3 * sqrt(var_x), 1.0);
        VEL_THR_y(k) = max(mu_y + 3 * sqrt(var_y), 1.0);
    end
    
    %% Init
    x0 = [q_sync(1,:)'; gps_gt(1,1:3)'; gps_gt(1,4:5)'; -dist_h(1);
          2.7556e-6*ones(3,1); 6.7600e-11*ones(3,1)];
    P0 = 1e-4 * eye(16);
    
    Delta_theta_n = [2.6e-5; 2.6e-5; 2.6e-5];
    Delta_v_n     = [1.66e-3; 1.66e-3; 1.66e-3];
    wb            = [2.6e-6; 2.6e-6; 2.6e-6];
    ab            = [1.66e-4; 1.66e-4; 1.66e-4];
    
    gt_pos = [gps_gt(1:N-1, 4:5), -dist_h(1:N-1)];
    calc_rmse = @(est, gt) sqrt(mean(sum((est - gt).^2, 2)));
    
    %% REKF Tuning
    fprintf('--- Tuning REKF ---\n');
    rmse_rekf = zeros(n_c, 1);
    for ic = 1:n_c
        c = c_scales(ic);
        X_rekf = zeros(16, N); X_rekf(:,1) = x0; P_rekf = P0;
        for k = 1:N-1
            md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
            if strcmp(md, 'gps'), y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps;
            else
                y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
                if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
                if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
            end
            [X_rekf(:,k+1), P_rekf, ~] = REKF_UAV(X_rekf(:,k), y_k, P_rekf, R_k, dth, dvk, ...
                Delta_theta_n, Delta_v_n, wb, ab, dt, md, c);
        end
        rmse_rekf(ic) = calc_rmse(X_rekf(8:10, 1:N-1)', gt_pos);
        fprintf('  c = %1.0e => Pos RMSE = %.4f m\n', c, rmse_rekf(ic));
    end
    [~, mi_rekf] = min(rmse_rekf);
    best_c_rekf(d) = c_scales(mi_rekf);
    fprintf('  >> Best REKF c: %1.0e\n\n', best_c_rekf(d));
    
    %% RUKF Tuning
    fprintf('--- Tuning RUKF ---\n');
    rmse_rukf = zeros(n_c, 1);
    for ic = 1:n_c
        c = c_scales(ic);
        X_rukf = zeros(16, N); X_rukf(:,1) = x0; P_rukf = P0;
        for k = 1:N-1
            md = mode_vec{k}; dth = dtheta(k,:); dvk = dv(k,:);
            if strcmp(md, 'gps'), y_k = [gps_denied(k,1:5)'; baro_h(k,1)]; R_k = R_gps;
            else
                y_k = [veh_flow_v(k,1:2)'; baro_h(k,1)]; R_k = R_flow;
                if abs(y_k(1)) > VEL_THR_x(k), R_k(1,1) = R_k(1,1) * 1e6; end
                if abs(y_k(2)) > VEL_THR_y(k), R_k(2,2) = R_k(2,2) * 1e6; end
            end
            [X_rukf(:,k+1), P_rukf, ~] = RUKF_UAV(X_rukf(:,k), y_k, P_rukf, R_k, dth, dvk, ...
                Delta_theta_n, Delta_v_n, wb, ab, dt, md, c, best_alpha);
        end
        rmse_rukf(ic) = calc_rmse(X_rukf(8:10, 1:N-1)', gt_pos);
        fprintf('  c = %1.0e => Pos RMSE = %.4f m\n', c, rmse_rukf(ic));
    end
    [~, mi_rukf] = min(rmse_rukf);
    best_c_rukf(d) = c_scales(mi_rukf);
    fprintf('  >> Best RUKF c: %1.0e\n\n', best_c_rukf(d));
    
end

%% ========================================================================
%% SUMMARY
%% ========================================================================
fprintf('==========================================================\n');
fprintf('          FINAL ROBUSTNESS PARAMETERS TO USE IN task34.m  \n');
fprintf('==========================================================\n');
fprintf('data_ids = {''46''; ''47'';''48''; ''49''; ''50''};\n');
fprintf('c_rekf_vals = [%1.0e; %1.0e; %1.0e; %1.0e; %1.0e];\n', best_c_rekf(1), best_c_rekf(2), best_c_rekf(3), best_c_rekf(4), best_c_rekf(5));
fprintf('c_rukf_vals = [%1.0e; %1.0e; %1.0e; %1.0e; %1.0e];\n', best_c_rukf(1), best_c_rukf(2), best_c_rukf(3), best_c_rukf(4), best_c_rukf(5));
fprintf('==========================================================\n');
