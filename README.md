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
├── README.md
├── Report.pdf
```

## Requirements

```bash
pip install ahrs mrob prettytable numpy matplotlib
```

## Example output for IEKF
![Image](https://github.com/user-attachments/assets/a7ff6050-2a13-4e6b-99da-632c4ad2fbde)
![Image](https://github.com/user-attachments/assets/d0cc7a6c-96b0-488e-b16b-2645113375b1)
![Image](https://github.com/user-attachments/assets/ce71a93e-b328-4f7e-baa3-7c0f75b7d3b8)
