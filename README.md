# UAV Estimation Project (GPS-Denied Navigation)

This repository focuses on **GPS-Denied UAV State Estimation** using **Optical Flow** and **Barometric Altitude** data, based on the [PX4-Autopilot v1.16.0](https://github.com/PX4/PX4-Autopilot) model. 

It implements, tunes, and compares different Kalman Filtering architectures to estimate the UAV's 3D position and velocity from synchronized flight datasets.

---

## 📂 Repository Structure

*   **`filters/`**: Core implementations of the Kalman Filters:
    *   `EKF_UAV.m` / `UKF_UAV.m`: Extended Kalman Filter & Unscented Kalman Filter.
    *   `REKF_UAV.m` / `RUKF_UAV.m`: Robust Extended and Unscented Kalman Filters with measurement gating and fault-protection.
    *   `func_f.m` / `func_h.m`: Non-linear state transition and measurement functions.
*   **`project/`**: Scripts for execution, verification, and parameter tuning:
    *   `task34.m`: Main simulation running and comparing EKF, UKF, REKF, and RUKF.
    *   `grid_search.m`: Coordinate descent search for optimal measurement noise covariance matrices ($R_{gps}$ and $R_{flow}$) and UKF parameter $\alpha$.
    *   `tune_robust.m`: Grid search for optimal robust parameter $c$ in REKF/RUKF.
    *   `results.txt`: Performance logs and 3D RMSE summary for all datasets.
*   **`Data/`**: Dynamic datasets containing sensor flows, GPS ground truth, and IMU measurements (git-ignored to keep the repository lightweight).
*   **`presentation/`**: Course presentation slides and academic reports.

---

## Estimators

1.  **Extended Kalman Filter (EKF)**: High-speed estimation using first-order Taylor linearization. Bypasses MATLAB symbolic substitutions using compiled functions to maintain execution speeds within fractions of a second.
2.  **Unscented Kalman Filter (UKF)**: Accurate estimation utilizing the Unscented Transform to propagate mean and covariance without analytical linearizations.
3.  **Robust EKF (REKF) & Robust UKF (RUKF)**: Protected filters designed to resist sensor outliers and spikes (e.g., lens scale mismatches, glitched rangefinders) using channel-wise measurement gating, SVD-bounding constraints to ensure positive-definiteness, and algebraic matrix optimization.

---