# Filters-for-IMU-Attitude-Estimation

The repository contains materials of the course (Perception in robotics, Skoltech) project. In this work we compare several attitude estimation filters in case of 3D movement. The motivation is to estimate a Huawei X3 Mate smartphone orientation in world frame using only IMU data. We compare Particle Filter (PF), Madgwick filter and Invariant Extended Kalman Filter (IEKF) perfomance. 

## Repository Structure
```bash
├── data
│ ├── Mocap_simple              # motion capture data for Huawei X3 Mate
│ ├── TUM-VI                    # TUM-VI data (motion capture + IMU)
│ ├── X3_simple                 # IMU data for Huawei X3 Mate
├── src
│ ├── filters                   # contains files with madgwick, PF, IEKF implementation
│ ├── tools                     # contains useful tools utilized in the project
│ ├── PF_tum.ipynb              # contains experiments with particle filter separately
│ ├── filters.ipynb             # MAIN! the notebook contains the whole pipeline - from data loading to performance evaluation
```

## Requirements

```bash
pip install ahrs mrob prettytable numpy matplotlib
```

## Example output for IEKF
