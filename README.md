# Subsea AUV Navigation Without GPS — Mathematics Assignment (CAT II & III)

**Live Interactive Simulator:** [https://dhrgupta096.github.io/maths-assignment/](https://dhrgupta096.github.io/maths-assignment/)

An interactive simulation and 5-module mathematical study on **Autonomous Underwater Vehicle (AUV) Dead Reckoning and Acoustic Kalman Filtering**.

## Problem Statement

Autonomous underwater vehicles cannot use GPS because radio signals are absorbed within metres of the ocean surface. They estimate position and orientation by **dead reckoning**: measuring heading and speed, rotating the body-frame velocity into the Earth frame with an orthogonal rotation matrix, and integrating over time. Because sensor noise and compass bias are integrated alongside motion, position errors grow without bound unless corrected by periodic acoustic beacon fixes.

## Five-Module Mathematical Architecture

| Module | Topic | Implementation in Project |
| :--- | :--- | :--- |
| **Module 1** | **Gauss-Jordan Elimination & Matrix Rank** | Solves linearized 3D acoustic beacon trilateration $A\mathbf{p} = \mathbf{b}$ ($\text{Rank}(A) = 3$) and inverts the Kalman innovation covariance $S^{-1} = (\Sigma + R_{\text{fix}})^{-1}$. |
| **Module 2** | **Vector Spaces, Basis & Rotation Matrices** | Transforms local body-frame velocity $\mathbf{v}_b$ into Earth coordinates via $2\text{D}$ and $3\text{D}$ rotation matrices $R_z(\psi)\,R_y(\theta)\,R_x(\phi)$. |
| **Module 3** | **Eigenvalues, Eigenvectors & Diagonalization** | Diagonalizes the sample error covariance matrix $\Sigma = P D P^T$ to find the major/minor semi-axes ($\sqrt{\lambda_1}, \sqrt{\lambda_2}$) and cross-track drift direction ($\mathbf{v}_1$). |
| **Module 4** | **Probability, Monte Carlo & Kalman Filter** | Models Gaussian sensor noise $\mathcal{N}(0, \sigma^2)$, confirms **H1** (heading error RMSE $3.70\text{ m}$ dominates speed error RMSE $1.04\text{ m}$), and bounds drift via Bayesian Kalman updates (**H2**). |
| **Module 5** | **Ordinary Least Squares Linear Regression** | Fits steady-state Kalman error against acoustic fix interval: $\widehat{\text{RMSE}} = 1.085 + 0.0303\,\Delta T_{\text{fix}}$ ($R^2 = 0.9671$, $\text{SE} = 0.0028$). |

## Files

- `index.html` — 60fps interactive AUV & Sonar web simulator with live rotation matrix, covariance ellipse, and sawtooth Kalman bound chart.
- `auv_project.py` — Zero-dependency Python mathematical engine and 300-run Monte Carlo simulator with built-in verification assertions.

## Quick Start

```bash
# Run the 5-module mathematical verification & Monte Carlo simulation
python3 auv_project.py

# Open the interactive visual simulator
open index.html
```
