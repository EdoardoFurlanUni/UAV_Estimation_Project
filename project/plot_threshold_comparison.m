%% Plot Optical Flow with Static & EMA thresholds + Quality overlay
% Shows all data as scatter, draws both threshold types, highlights outliers.
clc; clear; close all;

base_path = fullfile(fileparts(mfilename('fullpath')), '..', 'Data', 'mat');
lam = 0.98;  % EMA forgetting factor (same as task34)

folders = {
    'log_46_2025-10-18-10-11-28', '46';
    'log_47_2025-10-18-10-28-26', '47';
    'log_48_2025-10-18-10-40-54', '48';
    'log_49_2025-10-18-10-53-38', '49';
    'log_50_2025-10-18-11-09-00', '50';
};

for f = 1:size(folders, 1)
    folder = folders{f, 1};
    label  = folders{f, 2};
    
    csv_file = fullfile(base_path, folder, [folder '_vehicle_optical_flow_0.csv']);
    if ~exist(csv_file, 'file'), continue; end
    
    tbl = readtable(csv_file);
    pf_x   = tbl.pixel_flow_0_;
    pf_y   = tbl.pixel_flow_1_;
    dt_us  = tbl.integration_timespan_us;
    qual   = tbl.quality;
    dist_m = tbl.distance_m;
    t_us   = tbl.timestamp;
    
    dt_s = dt_us * 1e-6;
    vx = pf_x ./ dt_s .* dist_m;
    vy = pf_y ./ dt_s .* dist_m;
    t_sec = (t_us - t_us(1)) * 1e-6;
    Ns = length(vx);
    
    % Static threshold (global mean+3sigma)
    thr_x_static = mean(abs(vx), 'omitnan') + 3 * std(vx, 'omitnan');
    thr_y_static = mean(abs(vy), 'omitnan') + 3 * std(vy, 'omitnan');
    
    % EMA threshold (online, causal)
    ema_thr_x = zeros(Ns, 1); ema_thr_y = zeros(Ns, 1);
    mu_x = abs(vx(1)); var_x = 0;
    mu_y = abs(vy(1)); var_y = 0;
    ema_thr_x(1) = 5.0; ema_thr_y(1) = 5.0;
    
    burn_in = 500;
    
    for k = 2:Ns
        % Robust EMA: only update if sample is NOT an outlier
        % Always update during burn-in to initialize variance
        if k < burn_in || abs(vx(k-1)) < ema_thr_x(k-1)
            mu_x  = lam * mu_x  + (1 - lam) * abs(vx(k-1));
            var_x = lam * var_x + (1 - lam) * (vx(k-1) - mu_x)^2;
        end
        if k < burn_in || abs(vy(k-1)) < ema_thr_y(k-1)
            mu_y  = lam * mu_y  + (1 - lam) * abs(vy(k-1));
            var_y = lam * var_y + (1 - lam) * (vy(k-1) - mu_y)^2;
        end
        ema_thr_x(k) = max(mu_x + 3 * sqrt(var_x), 1.0);
        ema_thr_y(k) = max(mu_y + 3 * sqrt(var_y), 1.0);
    end
    
    % Quality classification
    idx_mid = (qual > 0 & qual < 245);
    idx_bad = (qual == 0);
    
    ema_rej_x = sum(abs(vx) > ema_thr_x);
    ema_rej_y = sum(abs(vy) > ema_thr_y);
    
    fprintf('Dataset %s:  static thr=[%.2f, %.2f],  EMA thr range X=[%.2f,%.2f] Y=[%.2f,%.2f]\n', ...
        label, thr_x_static, thr_y_static, ...
        min(ema_thr_x(100:end)), max(ema_thr_x(100:end)), ...
        min(ema_thr_y(100:end)), max(ema_thr_y(100:end)));
    fprintf('  EMA Rejected: X=%d/%d (%.1f%%), Y=%d/%d (%.1f%%)\n', ...
        ema_rej_x, Ns, 100*ema_rej_x/Ns, ema_rej_y, Ns, 100*ema_rej_y/Ns);
    
    fig = figure('Visible', 'off', 'Name', sprintf('Dataset %s — Thresholds', label), ...
           'NumberTitle', 'off', 'Units', 'normalized', ...
           'OuterPosition', [0.05 0.05 0.9 0.9]);
    
    % --- VX ---
    subplot(2,1,1); hold on;
    scatter(t_sec, vx, 4, [0.7 0.7 0.7], 'filled', 'HandleVisibility', 'off');
    yline( thr_x_static, '--r', 'LineWidth', 1.2, 'DisplayName', sprintf('Static +/-%.2f', thr_x_static));
    yline(-thr_x_static, '--r', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    plot(t_sec,  ema_thr_x, '-', 'Color', [0 0.7 0.3], 'LineWidth', 1.5, 'DisplayName', 'EMA +thr');
    plot(t_sec, -ema_thr_x, '-', 'Color', [0 0.7 0.3], 'LineWidth', 1.5, 'HandleVisibility', 'off');
    if any(idx_mid)
        scatter(t_sec(idx_mid), vx(idx_mid), 60, [1 0.6 0], 'o', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('q~120 (%d)', sum(idx_mid)));
    end
    if any(idx_bad)
        scatter(t_sec(idx_bad), vx(idx_bad), 80, [1 0 0], 'x', 'LineWidth', 2, ...
            'DisplayName', sprintf('q=0 (%d)', sum(idx_bad)));
    end
    ylabel('v_x body (m/s)');
    title(sprintf('Dataset %s - v_x', label));
    legend('Location', 'best'); grid on; hold off;
    
    % --- VY ---
    subplot(2,1,2); hold on;
    scatter(t_sec, vy, 4, [0.7 0.7 0.7], 'filled', 'HandleVisibility', 'off');
    yline( thr_y_static, '--r', 'LineWidth', 1.2, 'DisplayName', sprintf('Static +/-%.2f', thr_y_static));
    yline(-thr_y_static, '--r', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    plot(t_sec,  ema_thr_y, '-', 'Color', [0 0.7 0.3], 'LineWidth', 1.5, 'DisplayName', 'EMA +thr');
    plot(t_sec, -ema_thr_y, '-', 'Color', [0 0.7 0.3], 'LineWidth', 1.5, 'HandleVisibility', 'off');
    if any(idx_mid)
        scatter(t_sec(idx_mid), vy(idx_mid), 60, [1 0.6 0], 'o', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('q~120 (%d)', sum(idx_mid)));
    end
    if any(idx_bad)
        scatter(t_sec(idx_bad), vy(idx_bad), 80, [1 0 0], 'x', 'LineWidth', 2, ...
            'DisplayName', sprintf('q=0 (%d)', sum(idx_bad)));
    end
    ylabel('v_y body (m/s)'); xlabel('Time (s)');
    title(sprintf('Dataset %s - v_y', label));
    legend('Location', 'best'); grid on; hold off;
    
    % Save figure
    project_dir = fileparts(mfilename('fullpath'));
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, fullfile(project_dir, sprintf('Threshold_Comparison_%s', label)), '-dpng', '-r600');
    fprintf('Saved: Threshold_Comparison_%s.png\n', label);
    close(fig);
end

fprintf('\nLegend:\n');
fprintf('  Grey dots     = all samples\n');
fprintf('  Red dashed    = STATIC mean+3sigma (global, uses all data)\n');
fprintf('  Green solid   = EMA threshold (online, lambda=%.2f)\n', lam);
fprintf('  Orange/Red    = low quality samples\n');
