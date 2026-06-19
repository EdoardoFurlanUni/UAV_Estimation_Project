# UAV Estimation Project (GPS-Denied Navigation)

This repository focuses on **GPS-Denied UAV State Estimation** using **Optical Flow** and **Barometric Altitude** data, based on the [PX4-Autopilot v1.16.0](https://github.com/PX4/PX4-Autopilot) model. 

It implements, tunes, and compares different Kalman Filtering architectures to estimate the UAV's 3D position and velocity from synchronized flight datasets.

---

## 📂 Repository Structure

*   **`filters/`**: Core filter implementations and helper functions:
    *   `EKF_UAV.m` / `UKF_UAV.m`: Extended Kalman Filter & Unscented Kalman Filter.
    *   `REKF_UAV.m` / `RUKF_UAV.m`: Robust versions of EKF and UKF, with adaptive covariance update controlled by parameter `c`.
    *   `func_f.m` / `func_h.m`: Non-linear state transition and measurement functions.
    *   `calcC_h.m`: Analytical Jacobian of `func_h` (3×16), used by EKF and REKF.
    *   `calcF16.m`: State transition Jacobian F (16×16), used by EKF and REKF.
    *   `calcQ16.m`: Time-varying process noise covariance matrix Q (16×16).
    *   `linearized_process.m`: Computes linearized process matrices A and Q for EKF and REKF.
*   **`project/`**: Scripts for execution, verification, and parameter tuning:
    *   `task2.m`: Accuracy verification of the optical flow measurement model against GPS ground truth, using robust standard deviation (MAD) across the three optical flow data levels.
    *   `task34.m`: Main simulation script running and comparing EKF, UKF, REKF, and RUKF with EMA-based outlier gating across all datasets.
    *   `report_tuning.pdf`: Report describing the tuning methodology and filter performance results.
    *   `results_filters.txt`: Performance logs (3D/2D position RMSE, velocity RMSE, execution time) for all filters across all datasets.
    *   `results_tuning_c.txt`: Optimal robustness parameter $c$ for REKF and RUKF, per dataset.
    *   `PLOTS/`: All generated figures (Task2, Task34 velocity/position/attitude, threshold comparisons) for all 5 datasets.
*   **`Data/mat/`**: Data processing scripts and synchronized sensor data:
    *   `DATA_PROCESS.m`: Loads raw CSV logs and generates `data_sync_XX.mat` files.
    *   `sync_all_sensors.m`: Synchronizes all sensor data (IMU, GPS, barometer, optical flow, distance sensor) to the highest-rate sensor timestamp grid.
    *   `GPS_NED.m`: Converts GPS lat/lon/alt to NED coordinates using the first sample as origin.
    *   `denied.m`: Simulates GPS-denied windows by holding the last valid GPS measurement.
    *   `data_sync_XX.mat`: Synchronized data files for each dataset (git-ignored).
*   **`presentation/`**: Project presentation slides (`UAV.pdf`).

---

## Estimators

1.  **Extended Kalman Filter (EKF)**: Linearized estimation using pre-computed analytical Jacobians (`calcF16.m`, `calcC_h.m`).
2.  **Unscented Kalman Filter (UKF)**: Sigma-point-based estimation that avoids analytical linearization.
3.  **Robust EKF (REKF) & Robust UKF (RUKF)**: Robust versions that adaptively inflate the predicted covariance at each step via parameter `c`, making the filter more conservative when measurement uncertainty is high.

---